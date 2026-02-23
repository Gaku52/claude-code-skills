# tsconfig.json 完全ガイド

> TypeScript コンパイラオプションの全体像を把握し、プロジェクトに最適な設定を選択する

## この章で学ぶこと

1. **tsconfig.json の構造** -- ファイル構成、extends による継承、プロジェクト参照の仕組み
2. **主要コンパイラオプション** -- strict系、module系、target系、path系の全オプションと推奨設定
3. **ユースケース別設定** -- フロントエンド、バックエンド、ライブラリ、モノレポそれぞれの最適構成
4. **パフォーマンスチューニング** -- 大規模プロジェクトでのビルド高速化手法
5. **トラブルシューティング** -- よくある設定ミスとその解決策

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
|   "files": [...],    // 明示的なファイル  |
|   "watchOptions": {} // ウォッチ設定      |
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

### 1-2. include / exclude / files の詳細

```typescript
// include: glob パターンで対象を指定
{
  "include": [
    "src/**/*",        // src 配下の全ファイル
    "src/**/*.ts",     // .ts ファイルのみ（明示的）
    "src/**/*.tsx",    // .tsx ファイルを含む
    "types/**/*.d.ts"  // 型定義ファイル
  ]
}

// exclude: include で指定した中から除外
// ※ exclude はデフォルトで node_modules, bower_components, jspm_packages, outDir を除外
{
  "exclude": [
    "node_modules",      // デフォルトで除外されるが明示推奨
    "dist",              // ビルド出力先
    "**/*.test.ts",      // テストファイル
    "**/*.spec.ts",      // スペックファイル
    "**/__tests__/**",   // テストディレクトリ
    "coverage",          // カバレッジ出力先
    "scripts"            // ビルドスクリプト等
  ]
}

// files: 個別のファイルを直接指定（glob パターン不可）
// 少数の特定ファイルだけをコンパイルしたい場合に使用
{
  "files": [
    "src/index.ts",
    "src/global.d.ts"
  ]
}
```

```
include / exclude / files の優先順位:

  files > include > exclude

  1. files に指定されたファイルは exclude で除外できない
  2. include と exclude が競合する場合、exclude が優先
  3. files が指定されている場合、files のみがコンパイル対象
     （include と併用する場合は両方の合計が対象）
```

### 1-3. extends による継承

```
extends チェーン:

  @tsconfig/node20/tsconfig.json   (コミュニティベース)
       |
       v
  tsconfig.base.json               (プロジェクト共通)
       |
   +---+---+---+
   |       |   |
   v       v   v
  tsconfig.json   tsconfig.test.json   tsconfig.build.json
  (アプリ用)       (テスト用)            (ビルド用)
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
    "rootDir": "./src",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "jsx": "react-jsx"
  },
  "include": ["src/**/*"]
}

// tsconfig.test.json -- テスト用
{
  "extends": "./tsconfig.base.json",
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "types": ["vitest/globals"],
    "noEmit": true
  },
  "include": ["src/**/*", "tests/**/*"]
}

// tsconfig.build.json -- ビルド専用（テスト除外）
{
  "extends": "./tsconfig.json",
  "compilerOptions": {
    "noEmit": false,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  },
  "exclude": ["**/*.test.ts", "**/*.spec.ts", "**/__tests__/**"]
}
```

### 1-4. コミュニティベースの利用

```bash
# コミュニティが提供する推奨設定をインストール
npm install -D @tsconfig/node20
npm install -D @tsconfig/strictest
npm install -D @tsconfig/vite-react
```

```json
// @tsconfig/node20 を使った例
{
  "extends": "@tsconfig/node20/tsconfig.json",
  "compilerOptions": {
    "outDir": "./dist",
    "rootDir": "./src"
  },
  "include": ["src/**/*"]
}

// @tsconfig/strictest を使って最大限の厳密性を確保
{
  "extends": "@tsconfig/strictest/tsconfig.json",
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler"
  }
}
```

```
@tsconfig/strictest が有効化するオプション:
  - strict: true
  - noUncheckedIndexedAccess: true
  - noImplicitOverride: true
  - noPropertyAccessFromIndexSignature: true
  - noFallthroughCasesInSwitch: true
  - exactOptionalPropertyTypes: true
  - forceConsistentCasingInFileNames: true
  - verbatimModuleSyntax: true
  - isolatedModules: true
```

### 1-5. watchOptions の設定

```json
{
  "watchOptions": {
    // ファイル監視の方法
    "watchFile": "useFsEvents",
    // ディレクトリ監視の方法
    "watchDirectory": "useFsEvents",
    // ポーリングのフォールバック間隔（ms）
    "fallbackPolling": "dynamicPriority",
    // 同時変更の待機時間
    "synchronousWatchDirectory": false,
    // 監視対象から除外
    "excludeDirectories": ["**/node_modules", "dist"],
    // ファイル監視対象の追加
    "excludeFiles": []
  }
}
```

```
watchFile のオプション:
  useFsEvents       -- macOS / Windows で最も効率的（デフォルト）
  fixedPollingInterval -- 一定間隔でポーリング
  priorityPollingInterval -- 変更頻度に応じてポーリング
  dynamicPriorityPolling -- 動的にポーリング間隔を調整
  useFsEventsOnParentDirectory -- 親ディレクトリのイベントを使用

推奨:
  macOS/Windows → useFsEvents（デフォルト）
  Linux → useFsEvents（inotify 利用可能な場合）
  NFS/Docker → fixedPollingInterval（ネットワークFS対策）
```

---

## 2. strict 系オプション

### 2-1. strict フラグの内訳

```
strict: true は以下のフラグ全てを有効化:

+----------------------------------+-----------------------------------------+
| オプション                        | 効果                                    |
+----------------------------------+-----------------------------------------+
| strictNullChecks                 | null / undefined チェック               |
| strictFunctionTypes              | 関数型の厳密チェック（反変性）          |
| strictBindCallApply              | bind, call, apply の型チェック         |
| strictPropertyInitialization     | class プロパティの初期化チェック        |
| noImplicitAny                    | 暗黙の any を禁止                      |
| noImplicitThis                   | 暗黙の this を禁止                     |
| alwaysStrict                     | "use strict" を出力                    |
| useUnknownInCatchVariables       | catch 変数を unknown 型に              |
+----------------------------------+-----------------------------------------+
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

// Optional Chaining との組み合わせ
const name = user?.name; // 型: string | undefined
const upper = user?.name?.toUpperCase() ?? "Unknown";

// Non-null Assertion（注意して使用）
const forcedName = user!.name; // 型: string（null の可能性を無視）
// ↑ ランタイムで null だとクラッシュするため、確実な場合のみ使用

// strictFunctionTypes: true の効果（反変性チェック）
type Handler = (event: MouseEvent) => void;
const handler: Handler = (event: Event) => {}; // エラー: 反変
// MouseEvent は Event のサブタイプだが、Handler は MouseEvent を要求
// Event の handler を MouseEvent の handler として使うと
// MouseEvent 固有のプロパティにアクセスできない可能性がある

// strictFunctionTypes の実践例
interface Animal {
  name: string;
}
interface Dog extends Animal {
  breed: string;
}

type Comparer<T> = (a: T, b: T) => number;
const animalComparer: Comparer<Animal> = (a, b) => a.name.localeCompare(b.name);
// const dogComparer: Comparer<Dog> = animalComparer; // strict: true ではエラー

// strictBindCallApply: true の効果
function greet(name: string, age: number): string {
  return `Hello ${name}, age ${age}`;
}
greet.call(undefined, "Alice", 30);       // OK
// greet.call(undefined, "Alice", "thirty"); // エラー: string は number に代入不可
greet.bind(undefined, "Alice")(30);       // OK: 部分適用
// greet.bind(undefined, "Alice")("thirty"); // エラー

// strictPropertyInitialization: true の効果
class User {
  name: string;        // エラー: 初期化されていない
  email: string = "";  // OK: 初期値あり
  age?: number;        // OK: optional
  id!: string;         // OK: definite assignment assertion

  constructor(name: string) {
    this.name = name;  // OK: コンストラクタで初期化
  }
}

// noImplicitAny: true の効果
function processData(data) {} // エラー: Parameter 'data' implicitly has an 'any' type
function processData(data: unknown) {} // OK: 明示的に型指定

// useUnknownInCatchVariables: true の効果
try {
  throw new Error("boom");
} catch (error) {
  // error の型は unknown（true の場合）
  // error の型は any（false の場合）
  if (error instanceof Error) {
    console.log(error.message); // OK: Error 型に絞り込み
  }
}
```

### 2-2. 追加の厳密性オプション（strict に含まれない）

```typescript
// noUncheckedIndexedAccess: true（非常に推奨）
// 配列やオブジェクトのインデックスアクセスに undefined を追加
const arr: string[] = ["a", "b", "c"];
const item = arr[5]; // 型: string | undefined（true の場合）
                      // 型: string（false の場合）

// 安全な使用パターン
if (item !== undefined) {
  console.log(item.toUpperCase()); // OK: undefined チェック後
}

// for...of や forEach は影響を受けない
for (const item of arr) {
  console.log(item.toUpperCase()); // OK: item は string
}

// Record 型のインデックスアクセスにも適用
const dict: Record<string, number> = { a: 1, b: 2 };
const value = dict["unknown"]; // 型: number | undefined
// これにより辞書型の安全なアクセスが保証される

// exactOptionalPropertyTypes: true
// undefined の明示的な代入と省略を区別
interface Config {
  debug?: boolean;
}
const config1: Config = {};                        // OK: debug を省略
const config2: Config = { debug: true };           // OK
const config3: Config = { debug: undefined };      // エラー!
// undefined を明示的に代入するなら debug?: boolean | undefined と書く

// この区別が重要な理由:
// Object.hasOwn(config, "debug") の結果が異なる
// 省略: false、undefined代入: true

// noPropertyAccessFromIndexSignature: true
// インデックスシグネチャへのドットアクセスを禁止
interface Dict {
  [key: string]: string;
  knownKey: string;
}
declare const dict: Dict;
dict.knownKey;      // OK: 既知のプロパティ
dict.unknownKey;    // エラー: use dict["unknownKey"] instead
dict["unknownKey"]; // OK: ブラケットアクセスを強制

// noImplicitOverride: true
// オーバーライドを明示的にする
class Base {
  greet() { return "hello"; }
}

class Derived extends Base {
  greet() { return "hi"; } // エラー: override キーワードが必要
  override greet() { return "hi"; } // OK
}

// noFallthroughCasesInSwitch: true
// switch のフォールスルーを禁止
function process(status: string) {
  switch (status) {
    case "active":
      console.log("Active");
      // エラー: break がない（フォールスルー）
    case "inactive":
      console.log("Inactive");
      break;
  }
}

// allowUnreachableCode: false
// 到達不可能なコードをエラーにする
function example(x: number) {
  return x;
  console.log("unreachable"); // エラー: Unreachable code detected
}

// allowUnusedLabels: false
// 未使用のラベルをエラーにする
function search(matrix: number[][]) {
  loop: // エラー: 使用されていないラベル
  for (const row of matrix) {
    for (const cell of row) {
      if (cell === 0) break;
    }
  }
}
```

### 2-3. strict 化の段階的アプローチ

```json
// Step 1: 最小限の strict オプション（既存プロジェクト移行時）
{
  "compilerOptions": {
    "strict": false,
    "noImplicitAny": true
  }
}

// Step 2: null チェックの追加
{
  "compilerOptions": {
    "strict": false,
    "noImplicitAny": true,
    "strictNullChecks": true
  }
}

// Step 3: 関数型の厳密化
{
  "compilerOptions": {
    "strict": false,
    "noImplicitAny": true,
    "strictNullChecks": true,
    "strictFunctionTypes": true,
    "strictBindCallApply": true
  }
}

// Step 4: 全ての strict を有効化
{
  "compilerOptions": {
    "strict": true
  }
}

// Step 5: 追加の厳密性
{
  "compilerOptions": {
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "exactOptionalPropertyTypes": true,
    "noImplicitOverride": true,
    "noPropertyAccessFromIndexSignature": true,
    "noFallthroughCasesInSwitch": true
  }
}
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
                        +-----------+    +-----------+
                        | Preserve  | -> | そのまま   |
                        +-----------+    | 保持       |
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

// ライブラリ（ESM出力 + 型定義）
{
  "compilerOptions": {
    "module": "ESNext",
    "moduleResolution": "bundler",
    "declaration": true,
    "declarationMap": true
  }
}

// Node.js バックエンド（CJS -- レガシー）
{
  "compilerOptions": {
    "module": "CommonJS",
    "moduleResolution": "node"
  }
}
```

### 3-2. moduleResolution の違い

| 値 | import 解決 | 拡張子 | package.json exports | 用途 |
|---|---|---|---|---|
| `bundler` | バンドラー準拠 | 省略可 | サポート | Vite, webpack |
| `NodeNext` | Node.js ESM 準拠 | 必須(.js) | サポート | Node.js |
| `node` (旧) | CJS 準拠 | 省略可 | 非サポート | レガシー |
| `node16` | Node.js 16 準拠 | 必須(.js) | サポート | Node.js 16 |
| `classic` | TypeScript 独自 | 省略可 | 非サポート | 非推奨 |

```typescript
// moduleResolution: "bundler" の挙動
// 拡張子省略が可能（バンドラーが解決する前提）
import { utils } from "./utils";       // OK: ./utils.ts を発見
import { config } from "./config";     // OK: ./config.ts を発見
import { Button } from "@/components"; // OK: パスエイリアスもサポート

// moduleResolution: "NodeNext" の挙動
// 拡張子 .js が必須（Node.js ESM の仕様）
import { utils } from "./utils.js";    // OK: コンパイル後の拡張子を指定
import { config } from "./config.js";  // OK
// import { utils } from "./utils";    // エラー: 拡張子が必要

// NodeNext での package.json "type" フィールドとの関係
// package.json に "type": "module" → .ts ファイルは ESM として扱われる
// package.json に "type": "commonjs" → .ts ファイルは CJS として扱われる
// .mts ファイル → 常に ESM
// .cts ファイル → 常に CJS
```

### 3-3. module: "Preserve" (TypeScript 5.4+)

```typescript
// module: "Preserve" は入力のモジュール構文をそのまま保持
// moduleResolution: "bundler" と同等の解決を行いつつ、
// import/require を書いたとおりに出力する

// tsconfig.json
{
  "compilerOptions": {
    "module": "Preserve",
    "moduleResolution": "bundler"
  }
}

// 入力
import { foo } from "./foo";
const bar = require("./bar");

// 出力（そのまま保持）
import { foo } from "./foo";
const bar = require("./bar");
// バンドラーが最終的な解決を行う前提の設定
```

### 3-4. package.json の exports フィールドとの連携

```json
// ライブラリの package.json
{
  "name": "my-lib",
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
    "./utils": {
      "import": {
        "types": "./dist/utils.d.ts",
        "default": "./dist/utils.js"
      }
    }
  },
  "main": "./dist/index.cjs",
  "module": "./dist/index.js",
  "types": "./dist/index.d.ts"
}
```

```
exports フィールドの解決順序:

  import { x } from "my-lib"
       |
       v
  package.json の "exports" を確認
       |
  +----+----+
  |         |
  ESM       CJS
  |         |
  "import"  "require"
  |         |
  types →   types →
  default   default
```

---

## 4. パスエイリアス

### 4-1. 基本設定

```typescript
// tsconfig.json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@components/*": ["src/components/*"],
      "@utils/*": ["src/utils/*"],
      "@types/*": ["src/types/*"],
      "@lib/*": ["src/lib/*"],
      "@hooks/*": ["src/hooks/*"],
      "@services/*": ["src/services/*"],
      "@config": ["src/config/index.ts"]
    }
  }
}

// 使用例
import { Button } from "@components/Button";
import { formatDate } from "@utils/date";
import type { User } from "@types/user";
import { api } from "@lib/api";
import { useAuth } from "@hooks/useAuth";
import { UserService } from "@services/UserService";
import { config } from "@config";

// 注意: paths は型チェックのみ。実行時の解決には
// バンドラー設定（Vite: resolve.alias）や
// tsconfig-paths が別途必要
```

### 4-2. バンドラーとの連携設定

```typescript
// Vite: vite.config.ts
import { defineConfig } from "vite";
import path from "path";

export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
      "@components": path.resolve(__dirname, "src/components"),
      "@utils": path.resolve(__dirname, "src/utils"),
    },
  },
});

// webpack: webpack.config.js
module.exports = {
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
      "@components": path.resolve(__dirname, "src/components"),
    },
    extensions: [".ts", ".tsx", ".js", ".jsx"],
  },
};

// Jest: jest.config.ts
const config = {
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1",
    "^@components/(.*)$": "<rootDir>/src/components/$1",
  },
};

// Vitest: vitest.config.ts（vite.config.ts から自動取得）
// Vite の resolve.alias がそのまま使われる

// Node.js 直接実行: tsconfig-paths を使用
// node --import tsconfig-paths/register src/index.ts
```

### 4-3. 複数の候補パスによるフォールバック

```json
{
  "compilerOptions": {
    "baseUrl": ".",
    "paths": {
      // 複数のパスを指定すると、順番に解決を試みる
      "@shared/*": [
        "packages/shared/src/*",
        "packages/shared/dist/*"
      ],
      // ワイルドカードなしの完全一致
      "config": [
        "src/config/production.ts",
        "src/config/default.ts"
      ]
    }
  }
}
```

---

## 5. プロジェクト参照（Project References）

### 5-1. 基本構成

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
  |   +-- tsconfig.json (references: [shared])
  |   +-- src/
  +-- e2e/              ← E2E テスト
      +-- tsconfig.json (references: [frontend, backend])
      +-- tests/

  tsc --build で依存順にビルド
  → shared → frontend & backend (並列) → e2e
```

```json
// packages/shared/tsconfig.json
{
  "compilerOptions": {
    "composite": true,
    "declaration": true,
    "declarationMap": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true
  },
  "include": ["src/**/*"]
}

// packages/frontend/tsconfig.json
{
  "compilerOptions": {
    "outDir": "./dist",
    "rootDir": "./src",
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "jsx": "react-jsx"
  },
  "references": [
    { "path": "../shared" }
  ],
  "include": ["src/**/*"]
}

// packages/backend/tsconfig.json
{
  "compilerOptions": {
    "outDir": "./dist",
    "rootDir": "./src",
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "strict": true
  },
  "references": [
    { "path": "../shared" }
  ],
  "include": ["src/**/*"]
}

// ルートの tsconfig.json（ソリューションファイル）
{
  "files": [],
  "references": [
    { "path": "packages/shared" },
    { "path": "packages/frontend" },
    { "path": "packages/backend" },
    { "path": "packages/e2e" }
  ]
}
```

### 5-2. ビルドコマンド

```bash
# プロジェクト参照付きビルド（依存順に自動ビルド）
tsc --build

# ウォッチモード
tsc --build --watch

# クリーンビルド
tsc --build --clean

# verbose でビルド過程を確認
tsc --build --verbose

# 特定のパッケージのみビルド（依存も含む）
tsc --build packages/frontend

# 並列ビルドを強制（--build で自動的に並列化される場合がある）
tsc --build --force

# ドライラン（実際にはビルドしない）
tsc --build --dry
```

### 5-3. composite の制約と効果

```
composite: true を設定すると:

  1. declaration: true が強制される
     → .d.ts ファイルが必ず生成される

  2. rootDir が自動的に tsconfig.json の
     ディレクトリに設定される（未指定時）

  3. すべてのソースファイルが include/files に
     マッチする必要がある

  4. .tsbuildinfo ファイルが生成される
     → インクリメンタルビルドが可能に

  5. 参照元プロジェクトは .d.ts を介して
     型情報を取得する（ソースを直接見ない）
```

---

## 6. 出力関連オプション

### 6-1. declaration 系

```json
{
  "compilerOptions": {
    // 型定義ファイル(.d.ts)を出力
    "declaration": true,

    // .d.ts.map ファイルを出力（ソースへのジャンプ用）
    "declarationMap": true,

    // 型定義ファイルの出力先（outDir と分けたい場合）
    "declarationDir": "./types",

    // .js を出力せず .d.ts のみ出力
    "emitDeclarationOnly": true,

    // ソースマップを出力
    "sourceMap": true,

    // .js にインラインソースマップを埋め込む
    "inlineSourceMap": false,

    // ソースマップにソースコード自体を埋め込む
    "inlineSources": false
  }
}
```

### 6-2. ビルド出力の設定

```json
{
  "compilerOptions": {
    // 出力先ディレクトリ
    "outDir": "./dist",

    // ソースのルートディレクトリ（出力の構造に影響）
    "rootDir": "./src",

    // 複数のルートディレクトリを指定
    "rootDirs": ["src", "generated"],

    // 単一ファイルに出力（module: "system" or "amd" の場合のみ）
    "outFile": "./dist/bundle.js",

    // BOM (Byte Order Mark) を出力に含めない
    "emitBOM": false,

    // ファイルを出力しない（型チェックのみ）
    "noEmit": true,

    // エラーがあっても出力する
    "noEmitOnError": true,

    // ヘルパー関数のインポートを使用（出力サイズ削減）
    "importHelpers": true,

    // ダウンレベル時のヘルパー関数をインライン化しない
    "noEmitHelpers": false,

    // 改行コード
    "newLine": "lf",

    // コメントを出力に含めない
    "removeComments": false,

    // 各ファイルを独立してトランスパイル
    "isolatedModules": true
  }
}
```

### 6-3. isolatedModules の重要性

```typescript
// isolatedModules: true の場合に禁止される構文

// 1. const enum のエクスポート（cross-file inlining が不可能）
// NG:
export const enum Direction {
  Up,
  Down,
  Left,
  Right,
}

// OK: 通常の enum を使用
export enum Direction {
  Up,
  Down,
  Left,
  Right,
}

// 2. 型のみの re-export
// NG:
import { User } from "./types";
export { User }; // User が型なのか値なのか不明

// OK: type キーワードを明示
import type { User } from "./types";
export type { User };
// もしくは
export { type User } from "./types";

// 3. 宣言だけのファイル（値のエクスポートがない）
// NG: このファイルには値のエクスポートがない
declare const x: number;

// OK: 何らかの値をエクスポート
export {};
declare const x: number;

// isolatedModules を有効にすべき理由:
// esbuild, SWC, Babel などのトランスパイラはファイル単位で変換するため
// cross-file の情報が必要な構文を使うとビルドが壊れる
```

---

## 7. jsx オプション

```
jsx オプションの出力:

  入力: <div>Hello</div>

  "jsx": "preserve"      → <div>Hello</div>        (.jsx)
  "jsx": "react"         → React.createElement(...) (.js)
  "jsx": "react-jsx"     → _jsx("div", ...)         (.js)
  "jsx": "react-jsxdev"  → _jsxDEV("div", ...)      (.js)
  "jsx": "react-native"  → <div>Hello</div>         (.js)
```

```json
// React 17+ (自動 JSX トランスフォーム)
{
  "compilerOptions": {
    "jsx": "react-jsx",
    "jsxImportSource": "react"
  }
}

// Preact
{
  "compilerOptions": {
    "jsx": "react-jsx",
    "jsxImportSource": "preact"
  }
}

// Solid.js（Vite で変換するため preserve）
{
  "compilerOptions": {
    "jsx": "preserve",
    "jsxImportSource": "solid-js"
  }
}

// Emotion CSS-in-JS
{
  "compilerOptions": {
    "jsx": "react-jsx",
    "jsxImportSource": "@emotion/react"
  }
}
```

---

## 8. パフォーマンス最適化

### 8-1. インクリメンタルビルド

```json
{
  "compilerOptions": {
    // インクリメンタルビルドを有効化
    "incremental": true,
    // ビルド情報ファイルの保存先
    "tsBuildInfoFile": "./.tsbuildinfo"
  }
}
```

```
インクリメンタルビルドの効果:

  初回ビルド:
  src/ (100ファイル) → tsc → dist/ + .tsbuildinfo

  2回目以降（3ファイル変更）:
  src/ (3ファイル変更) → tsc → 3ファイルのみ再コンパイル

  効果:
  - 大規模プロジェクトで 2回目以降のビルドが 50-80% 高速化
  - .tsbuildinfo はキャッシュとして機能
  - CI では .tsbuildinfo をキャッシュすると効果的
```

### 8-2. 大規模プロジェクトでの高速化

```json
// パフォーマンス最適化設定
{
  "compilerOptions": {
    // .d.ts ファイルの型チェックをスキップ
    "skipLibCheck": true,

    // インクリメンタルビルド
    "incremental": true,

    // ファイルを出力しない（型チェックのみの場合）
    "noEmit": true,

    // import type の自動検出
    "verbatimModuleSyntax": true,
    // ↑ isolatedModules + preserveValueImports + importsNotUsedAsValues
    //   を1つのオプションで置き換え
  },
  // 対象ファイルを最小化
  "include": ["src/**/*"],
  "exclude": [
    "node_modules",
    "dist",
    "**/*.test.ts",
    "**/*.spec.ts",
    "**/__tests__/**"
  ]
}
```

```bash
# tsc のパフォーマンス診断
tsc --extendedDiagnostics

# 出力例:
# Files:           1234
# Lines:           89012
# Nodes:           345678
# Identifiers:     123456
# Symbols:         67890
# Types:           34567
# Instantiations:  234567
# Memory used:     456MB
# I/O Read:        0.12s
# Parse time:      1.23s
# Bind time:       0.45s
# Check time:      5.67s
# Emit time:       0.89s
# Total time:      8.36s

# traceResolution でモジュール解決をデバッグ
tsc --traceResolution > trace.txt 2>&1

# generateTrace でパフォーマンスプロファイルを生成
tsc --generateTrace ./trace-output
# Chrome の chrome://tracing で trace.json を開いて分析
```

### 8-3. CI/CD での最適化

```yaml
# GitHub Actions での TypeScript ビルド最適化
name: TypeCheck
on: [push, pull_request]

jobs:
  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'npm'

      - run: npm ci

      # .tsbuildinfo のキャッシュ
      - uses: actions/cache@v4
        with:
          path: .tsbuildinfo
          key: tsbuildinfo-${{ hashFiles('tsconfig.json') }}-${{ github.sha }}
          restore-keys: |
            tsbuildinfo-${{ hashFiles('tsconfig.json') }}-

      # 型チェックのみ実行
      - run: tsc --noEmit --incremental
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
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "verbatimModuleSyntax": true,
    "incremental": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

### React SPA (Vite)

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "jsx": "react-jsx",
    "skipLibCheck": true,
    "noEmit": true,
    "isolatedModules": true,
    "verbatimModuleSyntax": true,
    "resolveJsonModule": true,
    "forceConsistentCasingInFileNames": true,
    "paths": {
      "@/*": ["./src/*"]
    }
  },
  "include": ["src/**/*", "vite-env.d.ts"],
  "exclude": ["node_modules"]
}
```

### npm ライブラリ

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "exactOptionalPropertyTypes": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "isolatedModules": true,
    "verbatimModuleSyntax": true,
    "noImplicitOverride": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
```

### Cloudflare Workers / Edge Runtime

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "noEmit": true,
    "lib": ["ES2022"],
    "types": ["@cloudflare/workers-types"],
    "isolatedModules": true,
    "verbatimModuleSyntax": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "jsx": "react-jsx"
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules"]
}
```

### Electron アプリ

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "CommonJS",
    "moduleResolution": "node",
    "strict": true,
    "esModuleInterop": true,
    "outDir": "./dist",
    "rootDir": "./src",
    "declaration": true,
    "sourceMap": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true
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
| ES2018 | async iteration, rest/spread | 10+ | モダン全て |
| ES2019 | `Array.flat()`, `Object.fromEntries()` | 12+ | モダン全て |
| ES2020 | `?.`, `??`, BigInt, `import()` | 14+ | モダン全て |
| ES2021 | `&&=`, `\|\|=`, `??=`, WeakRef | 16+ | モダン全て |
| ES2022 | Top-level await, `.at()`, `cause`, `#private` | 18+ | モダン全て |
| ES2023 | Array `.findLast()`, hashbang | 20+ | 最新 |
| ES2024 | `Object.groupBy()`, `Promise.withResolvers()` | 22+ | 最新 |
| ESNext | 最新の Stage 4 提案 | 最新 | 最新 |

### lib 値の選択ガイド

| 環境 | lib | 説明 |
|------|-----|------|
| ブラウザ | `["dom", "dom.iterable", "esnext"]` | DOM API + 最新 JS |
| Node.js | `["esnext"]` | JS のみ（DOM なし） |
| Web Worker | `["webworker", "esnext"]` | Worker API |
| Service Worker | `["webworker", "esnext"]` | SW API |
| 共有ライブラリ | `["esnext"]` | 環境非依存 |
| Deno | `["deno.ns", "esnext"]` | Deno 名前空間 |

### コンパイラオプション一覧（カテゴリ別）

| カテゴリ | オプション | デフォルト | 推奨 |
|---------|-----------|----------|------|
| 厳密性 | strict | false | true |
| 厳密性 | noUncheckedIndexedAccess | false | true |
| 厳密性 | exactOptionalPropertyTypes | false | 検討 |
| モジュール | module | - | 環境に応じて |
| モジュール | moduleResolution | - | bundler or NodeNext |
| モジュール | verbatimModuleSyntax | false | true |
| モジュール | isolatedModules | false | true |
| 出力 | outDir | - | ./dist |
| 出力 | declaration | false | ライブラリで true |
| 出力 | sourceMap | false | true |
| 出力 | noEmit | false | バンドラー使用時 true |
| 互換性 | esModuleInterop | false | true |
| 互換性 | skipLibCheck | false | true |
| 互換性 | forceConsistentCasingInFileNames | false | true |
| パフォーマンス | incremental | false | true |

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
// バグの早期発見ができず、リファクタリングの安全性も損なわれる

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
// 特に @types パッケージ間で型の衝突が起きた場合に有効
// 自分のコードの安全性は維持される
```

### AP-3: moduleResolution: "node" を新規プロジェクトで使用

```json
// NG: 旧式の moduleResolution を使用
{
  "compilerOptions": {
    "moduleResolution": "node"
  }
}
// package.json の "exports" フィールドをサポートしない
// ESM の正しい解決ができない

// OK: 用途に応じた最新の設定
// バンドラー使用時:
{
  "compilerOptions": {
    "moduleResolution": "bundler"
  }
}
// Node.js 直接実行時:
{
  "compilerOptions": {
    "moduleResolution": "NodeNext"
  }
}
```

### AP-4: paths を設定してバンドラー設定を忘れる

```typescript
// tsconfig.json で paths を設定
{
  "compilerOptions": {
    "paths": {
      "@/*": ["src/*"]
    }
  }
}

// NG: Vite の resolve.alias を設定し忘れる
// → 型チェックは通るが、実行時にモジュールが見つからない

// OK: 必ずバンドラー側にも同じエイリアスを設定
// vite.config.ts
import { defineConfig } from "vite";
import path from "path";

export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
});
```

### AP-5: 巨大な include 範囲

```json
// NG: プロジェクト全体を include
{
  "include": ["**/*"]
}
// node_modules 以外の全ファイルがスキャンされ、パフォーマンス悪化

// OK: ソースディレクトリのみを指定
{
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

---

## トラブルシューティング

### よくあるエラーと解決策

```
エラー: Cannot find module './utils' or its corresponding type declarations.

原因: moduleResolution の不一致
解決:
  1. moduleResolution: "bundler" の場合
     → 拡張子なしでOK、バンドラーの設定を確認
  2. moduleResolution: "NodeNext" の場合
     → import "./utils.js" のように .js 拡張子を追加
  3. ファイルが include パターンに含まれているか確認
```

```
エラー: Type 'X' is not assignable to type 'Y'.
       Type 'undefined' is not assignable to type 'string'.

原因: strictNullChecks が有効
解決:
  1. null/undefined チェックを追加
     if (value !== undefined) { ... }
  2. Optional chaining を使用
     const name = user?.name ?? "default";
  3. 型定義を修正して undefined を許容
     let name: string | undefined;
```

```
エラー: 'X' is declared but its value is never read.

原因: noUnusedLocals / noUnusedParameters が有効
解決:
  1. 変数名の先頭に _ を付ける
     const _unused = someValue;
  2. 本当に不要なら削除
  3. tsconfig でオプションを調整（非推奨）
```

```
エラー: File 'X' is not listed within the file list of project 'Y'.
       Projects must list all files or use an 'include' pattern.

原因: composite: true のプロジェクトで include に含まれないファイルを参照
解決:
  1. include パターンを修正して対象ファイルを含める
  2. files に直接追加
  3. 参照先プロジェクトの tsconfig を確認
```

```
エラー: Cannot use JSX unless the '--jsx' flag is provided.

原因: jsx オプションが未設定
解決:
  {
    "compilerOptions": {
      "jsx": "react-jsx"  // React 17+
    }
  }
```

---

## FAQ

### Q1: `verbatimModuleSyntax` は有効にすべきですか？

TypeScript 5.0 以降で推奨されます。このオプションは `import type` と `import` を明確に区別し、型のみのインポートが実行時に残らないことを保証します。`isolatedModules` と `esModuleInterop` の一部を置き換える上位互換です。有効にすると以下の効果があります:

- `import type { X }` は必ず出力から除去される
- 値として使われないインポートに `type` キーワードを強制する
- esbuild, SWC などのトランスパイラとの互換性が向上する
- バンドルサイズの最適化に貢献する

### Q2: `moduleResolution: "bundler"` はいつ使うべきですか？

Vite, webpack, esbuild などのバンドラーを使用する場合に使います。Node.js で直接実行する場合は `NodeNext` を使ってください。`bundler` は拡張子の省略やindex ファイルの暗黙的な解決をサポートし、バンドラーの動作に合致します。具体的には:

- 拡張子なしのインポートが許可される（`import "./utils"` が有効）
- `index.ts` の暗黙的な解決がサポートされる
- `package.json` の `exports` フィールドが正しく解決される
- `import.meta.url` などの ESM 固有の構文がサポートされる

### Q3: `composite` と `references` の違いは何ですか？

`composite: true` はプロジェクトが他のプロジェクトから参照可能であることを宣言します。`references` は依存先プロジェクトを指定します。モノレポで `tsc --build` を使う場合に両方が必要です。`composite` を有効にすると `declaration: true` が強制され、`.tsbuildinfo` ファイルが生成されます。

### Q4: `noEmit: true` と `emitDeclarationOnly: true` の使い分けは？

- `noEmit: true`: 一切のファイルを出力しない（型チェックのみ）。バンドラーが JS を生成し、tsc は型チェック専任にする場合に使用
- `emitDeclarationOnly: true`: `.d.ts` ファイルのみ出力。esbuild/SWC で JS を生成し、tsc で型定義を生成する場合に使用

### Q5: `resolveJsonModule` と Import Attributes の関係は？

`resolveJsonModule: true` は `.json` ファイルのインポートを型チェックします。TypeScript 5.3+ では Import Attributes（`import data from "./data.json" with { type: "json" }`）も使えます。`resolveJsonModule` は TypeScript 独自の機能で、Import Attributes は ECMAScript 標準です。将来的には Import Attributes が推奨されますが、現時点ではどちらも有効です。

### Q6: `exactOptionalPropertyTypes` は有効にすべきですか？

型の厳密性を高めたい場合に推奨しますが、既存のコードベースでは多くの修正が必要になる可能性があります。特に `{ prop?: T }` と `{ prop?: T | undefined }` を区別するため、`undefined` を明示的に代入しているコードがエラーになります。新規プロジェクトでは有効化を検討してください。

### Q7: tsc の型チェックが遅い場合の対処法は？

1. `skipLibCheck: true` で `.d.ts` のチェックをスキップ
2. `incremental: true` でインクリメンタルビルドを有効化
3. `tsc --extendedDiagnostics` でボトルネックを特定
4. `include` の範囲を最小化
5. プロジェクト参照で分割し、変更の影響範囲を限定
6. `tsc --generateTrace` でプロファイルを取得し、型のインスタンス化が多いコードを特定

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
| incremental | 大規模プロジェクトで必須。CI でもキャッシュ活用 |
| verbatimModuleSyntax | 5.0+ で推奨。import type の一貫性を保証 |
| isolatedModules | バンドラー使用時は必須。ファイル単位のトランスパイル互換性 |
| skipLibCheck | 推奨。.d.ts のチェックをスキップしてビルド高速化 |
| noUncheckedIndexedAccess | 推奨。配列・辞書アクセスの安全性を向上 |

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

4. **TypeScript Performance** -- Wiki on TypeScript Performance
   https://github.com/microsoft/TypeScript/wiki/Performance

5. **TypeScript Module Resolution**
   https://www.typescriptlang.org/docs/handbook/modules/reference.html
