# tsconfig.json 完全ガイド

> TypeScript コンパイラオプションの全体像を把握し、プロジェクトに最適な設定を選択する

## この章で学ぶこと

1. **tsconfig.json の構造** -- ファイル構成、extends による継承、プロジェクト参照の仕組み
2. **主要コンパイラオプション** -- strict系、module系、target系、path系の全オプションと推奨設定
3. **ユースケース別設定** -- フロントエンド、バックエンド、ライブラリ、モノレポそれぞれの最適構成

---

## 1. tsconfig.json の基本構造

### 1-1. ファイル構成

```
tsconfig.json の主要セクション:

+------------------------------------------+
| {                                        |
|   "compilerOptions": {                   |
|     // コンパイラの動作設定              |
|   },                                     |
|   "include": [...],  // コンパイル対象   |
|   "exclude": [...],  // 除外パターン     |
|   "extends": "...",  // ベース設定の継承  |
|   "references": [...] // プロジェクト参照 |
| }                                        |
+------------------------------------------+
```

```typescript
// 基本的な tsconfig.json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
```

### 1-2. extends による継承

```
extends チェーン:

  @tsconfig/node20/tsconfig.json   (コミュニティベース)
       |
       v
  tsconfig.base.json               (プロジェクト共通)
       |
   +---+---+
   |       |
   v       v
  tsconfig.json   tsconfig.test.json
  (アプリ用)       (テスト用)
```

```json
// tsconfig.base.json -- 共通設定
{
  "compilerOptions": {
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "verbatimModuleSyntax": true
  }
}

// tsconfig.json -- アプリ用（継承 + 上書き）
{
  "extends": "./tsconfig.base.json",
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "outDir": "./dist",
    "rootDir": "./src"
  },
  "include": ["src/**/*"]
}

// tsconfig.test.json -- テスト用
{
  "extends": "./tsconfig.base.json",
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "types": ["vitest/globals"]
  },
  "include": ["src/**/*", "tests/**/*"]
}
```

---

## 2. strict 系オプション

### 2-1. strict フラグの内訳

```
strict: true は以下のフラグ全てを有効化:

+----------------------------------+--------+
| オプション                        | 効果   |
+----------------------------------+--------+
| strictNullChecks                 | null/  |
|                                  | undef  |
|                                  | チェック|
+----------------------------------+--------+
| strictFunctionTypes              | 関数型 |
|                                  | の厳密 |
|                                  | チェック|
+----------------------------------+--------+
| strictBindCallApply              | bind,  |
|                                  | call,  |
|                                  | apply  |
+----------------------------------+--------+
| strictPropertyInitialization     | class  |
|                                  | プロパ |
|                                  | ティ初 |
|                                  | 期化   |
+----------------------------------+--------+
| noImplicitAny                    | 暗黙   |
|                                  | any    |
|                                  | 禁止   |
+----------------------------------+--------+
| noImplicitThis                   | 暗黙   |
|                                  | this   |
|                                  | 禁止   |
+----------------------------------+--------+
| alwaysStrict                     | "use   |
|                                  | strict"|
|                                  | 出力   |
+----------------------------------+--------+
| useUnknownInCatchVariables       | catch  |
|                                  | 変数が |
|                                  | unknown|
+----------------------------------+--------+
```

```typescript
// strictNullChecks: true の効果
function getUser(id: string): User | null {
  // ...
  return null;
}

const user = getUser("1");
// user.name;  // エラー: Object is possibly 'null'
if (user) {
  user.name; // OK: null チェック後
}

// strictFunctionTypes: true の効果
type Handler = (event: MouseEvent) => void;
const handler: Handler = (event: Event) => {}; // エラー: 反変
// MouseEvent は Event のサブタイプだが、Handler は MouseEvent を要求

// noImplicitAny: true の効果
function processData(data) {} // エラー: Parameter 'data' implicitly has an 'any' type
function processData(data: unknown) {} // OK: 明示的に型指定
```

### 2-2. 追加の厳密性オプション

```typescript
// noUncheckedIndexedAccess: true（strict に含まれない）
const arr: string[] = ["a", "b", "c"];
const item = arr[5]; // 型: string | undefined（true の場合）
                      // 型: string（false の場合）

// exactOptionalPropertyTypes: true（strict に含まれない）
interface Config {
  debug?: boolean;
}
const config: Config = { debug: undefined };
// エラー: Type 'undefined' is not assignable to type 'boolean'
// undefined を明示的に代入するなら debug?: boolean | undefined と書く

// noPropertyAccessFromIndexSignature: true
interface Dict {
  [key: string]: string;
  knownKey: string;
}
declare const dict: Dict;
dict.knownKey;      // OK
dict.unknownKey;    // エラー: use dict["unknownKey"] instead
dict["unknownKey"]; // OK
```

---

## 3. module / moduleResolution

### 3-1. module オプション

```
module オプションと出力形式:

  TypeScript ソース          module          出力
  +-----------+         +-----------+    +-----------+
  | import x  |  -----> | ESNext    | -> | import x  |
  | from "y"  |         +-----------+    | from "y"  |
  +-----------+         +-----------+    +-----------+
                        | CommonJS  | -> | const x = |
                        +-----------+    | require() |
                        +-----------+    +-----------+
                        | NodeNext  | -> | ESM or CJS|
                        +-----------+    | (自動判別) |
```

```json
// フロントエンド（バンドラー使用）
{
  "compilerOptions": {
    "module": "ESNext",
    "moduleResolution": "bundler"
  }
}

// Node.js バックエンド（ESM）
{
  "compilerOptions": {
    "module": "NodeNext",
    "moduleResolution": "NodeNext"
  }
}

// ライブラリ（両対応）
{
  "compilerOptions": {
    "module": "ESNext",
    "moduleResolution": "bundler",
    "declaration": true,
    "declarationMap": true
  }
}
```

### 3-2. moduleResolution の違い

| 値 | import 解決 | 拡張子 | package.json exports | 用途 |
|---|---|---|---|---|
| `bundler` | バンドラー準拠 | 省略可 | サポート | Vite, webpack |
| `NodeNext` | Node.js ESM 準拠 | 必須(.js) | サポート | Node.js |
| `node` (旧) | CJS 準拠 | 省略可 | 非サポート | レガシー |
| `classic` | TypeScript 独自 | 省略可 | 非サポート | 非推奨 |

---

## 4. パスエイリアス

```typescript
// tsconfig.json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@components/*": ["src/components/*"],
      "@utils/*": ["src/utils/*"],
      "@types/*": ["src/types/*"]
    }
  }
}

// 使用例
import { Button } from "@components/Button";
import { formatDate } from "@utils/date";
import type { User } from "@types/user";

// 注意: paths は型チェックのみ。実行時の解決には
// バンドラー設定（Vite: resolve.alias）や
// tsconfig-paths が別途必要
```

---

## 5. プロジェクト参照（Project References）

```
モノレポでのプロジェクト参照:

  packages/
  +-- shared/           ← 共有ライブラリ
  |   +-- tsconfig.json (composite: true)
  |   +-- src/
  +-- frontend/         ← フロントエンド
  |   +-- tsconfig.json (references: [shared])
  |   +-- src/
  +-- backend/          ← バックエンド
      +-- tsconfig.json (references: [shared])
      +-- src/

  tsc --build で依存順にビルド
```

```json
// packages/shared/tsconfig.json
{
  "compilerOptions": {
    "composite": true,
    "declaration": true,
    "declarationMap": true,
    "outDir": "./dist",
    "rootDir": "./src"
  },
  "include": ["src/**/*"]
}

// packages/frontend/tsconfig.json
{
  "compilerOptions": {
    "outDir": "./dist",
    "rootDir": "./src"
  },
  "references": [
    { "path": "../shared" }
  ],
  "include": ["src/**/*"]
}

// ルートの tsconfig.json
{
  "files": [],
  "references": [
    { "path": "packages/shared" },
    { "path": "packages/frontend" },
    { "path": "packages/backend" }
  ]
}
```

---

## ユースケース別推奨設定

### Next.js (App Router)

```json
{
  "compilerOptions": {
    "target": "ES2017",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./src/*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

### Node.js API サーバー

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "declaration": true,
    "sourceMap": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

---

## 比較表

### target 値と対応 ECMAScript 機能

| target | 主な機能 | Node.js | ブラウザ |
|--------|---------|---------|---------|
| ES2020 | `?.`, `??`, BigInt | 14+ | モダン全て |
| ES2021 | `&&=`, `\|\|=`, `??=` | 16+ | モダン全て |
| ES2022 | Top-level await, `.at()`, `cause` | 18+ | モダン全て |
| ES2023 | Array `.findLast()`, `#private` | 20+ | 最新 |
| ESNext | 最新の Stage 4 提案 | 最新 | 最新 |

### lib 値の選択ガイド

| 環境 | lib | 説明 |
|------|-----|------|
| ブラウザ | `["dom", "dom.iterable", "esnext"]` | DOM API + 最新 JS |
| Node.js | `["esnext"]` | JS のみ（DOM なし） |
| Web Worker | `["webworker", "esnext"]` | Worker API |
| 共有ライブラリ | `["esnext"]` | 環境非依存 |

---

## アンチパターン

### AP-1: strict: false のまま放置

```json
// NG: strict を無効にして型安全性を放棄
{
  "compilerOptions": {
    "strict": false,
    "noImplicitAny": false
  }
}
// any が蔓延し、TypeScript を使う意味が薄れる

// OK: 段階的に strict 化
{
  "compilerOptions": {
    "strict": true,
    // 移行中は個別に緩める
    "strictPropertyInitialization": false
  }
}
```

### AP-2: skipLibCheck と型チェックの混同

```json
// NG: skipLibCheck を型チェック全体の無効化と勘違い
// skipLibCheck は .d.ts ファイルのチェックをスキップするだけ
// 自分のコードの型チェックには影響しない

// OK: skipLibCheck: true は推奨設定
// .d.ts 間の型衝突を回避し、ビルド速度が向上する
{
  "compilerOptions": {
    "skipLibCheck": true
  }
}
```

---

## FAQ

### Q1: `verbatimModuleSyntax` は有効にすべきですか？

TypeScript 5.0 以降で推奨されます。このオプションは `import type` と `import` を明確に区別し、型のみのインポートが実行時に残らないことを保証します。`isolatedModules` と `esModuleInterop` の一部を置き換える上位互換です。

### Q2: `moduleResolution: "bundler"` はいつ使うべきですか？

Vite, webpack, esbuild などのバンドラーを使用する場合に使います。Node.js で直接実行する場合は `NodeNext` を使ってください。`bundler` は拡張子の省略やindex ファイルの暗黙的な解決をサポートし、バンドラーの動作に合致します。

### Q3: `composite` と `references` の違いは何ですか？

`composite: true` はプロジェクトが他のプロジェクトから参照可能であることを宣言します。`references` は依存先プロジェクトを指定します。モノレポで `tsc --build` を使う場合に両方が必要です。`composite` を有効にすると `declaration: true` が強制されます。

---

## まとめ表

| 概念 | 要点 |
|------|------|
| strict | 常に `true`。個別フラグで一時的に緩める |
| target | 実行環境の最低バージョンに合わせる |
| module | バンドラー → ESNext、Node.js → NodeNext |
| moduleResolution | バンドラー → bundler、Node.js → NodeNext |
| paths | 型チェック用。実行時にはバンドラー設定が別途必要 |
| composite | モノレポのプロジェクト参照で使用 |

---

## 次に読むべきガイド

- [ビルドツール](./01-build-tools.md) -- tsc, esbuild, SWC, Vite の使い分け
- [ESLint + TypeScript](./04-eslint-typescript.md) -- コンパイラ設定と lint の連携
- [JS→TS 移行](./03-migration-guide.md) -- tsconfig の段階的な厳密化

---

## 参考文献

1. **TypeScript TSConfig Reference**
   https://www.typescriptlang.org/tsconfig

2. **@tsconfig/bases** -- Community-maintained tsconfig bases
   https://github.com/tsconfig/bases

3. **Matt Pocock - TSConfig Cheat Sheet**
   https://www.totaltypescript.com/tsconfig-cheat-sheet
