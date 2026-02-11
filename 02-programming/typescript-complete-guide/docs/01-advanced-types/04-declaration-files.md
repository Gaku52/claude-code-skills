# 宣言ファイル（Declaration Files）

> .d.ts ファイルの仕組みと書き方。DefinitelyTyped、ambient declarations、モジュール拡張による既存ライブラリの型付けを学ぶ。

## この章で学ぶこと

1. **宣言ファイルの基本** -- .d.ts ファイルの役割、declare キーワード、型のみの世界
2. **DefinitelyTyped** -- @types パッケージ、型定義の検索と利用
3. **モジュール拡張** -- 既存の型定義の拡張、グローバル型の追加、ambient modules

---

## 1. 宣言ファイルの基本

### コード例1: .d.ts ファイルの構造

```typescript
// types/math-utils.d.ts

// 関数の宣言
declare function add(a: number, b: number): number;
declare function multiply(a: number, b: number): number;

// 変数の宣言
declare const PI: number;
declare let debugMode: boolean;

// クラスの宣言
declare class Calculator {
  constructor(initial?: number);
  add(n: number): this;
  subtract(n: number): this;
  result(): number;
}

// インターフェース（declare 不要）
interface MathOptions {
  precision: number;
  roundingMode: "ceil" | "floor" | "round";
}
```

### 宣言ファイルの役割

```
  JavaScript ライブラリ          TypeScript プロジェクト
+-------------------------+    +-------------------------+
| math-utils.js           |    | app.ts                  |
| (実装コード)            |    | import { add } from ... |
+-------------------------+    +-------------------------+
         |                              |
         v                              v
+-------------------------+    +-------------------------+
| math-utils.d.ts         |<==>| TypeScript コンパイラ   |
| (型情報のみ)            |    | 型チェック実行          |
+-------------------------+    +-------------------------+

  .d.ts は「型の契約書」
  実装コードなし、型情報のみ
```

### コード例2: モジュールの宣言

```typescript
// types/my-library.d.ts

// モジュール全体の型を宣言
declare module "my-library" {
  export interface Config {
    host: string;
    port: number;
  }

  export function createClient(config: Config): Client;

  export class Client {
    connect(): Promise<void>;
    disconnect(): Promise<void>;
    query<T>(sql: string): Promise<T[]>;
  }

  export default createClient;
}
```

### コード例3: グローバル変数・型の宣言

```typescript
// types/global.d.ts

// グローバル変数の宣言
declare const __APP_VERSION__: string;
declare const __DEV__: boolean;

// グローバルインターフェースの拡張
interface Window {
  analytics: {
    track(event: string, data?: Record<string, unknown>): void;
    identify(userId: string): void;
  };
}

// グローバル型の追加
declare global {
  interface Array<T> {
    customMethod(): T[];
  }
}

export {}; // モジュールスコープにするために必要
```

---

## 2. DefinitelyTyped と @types

### コード例4: @types パッケージの利用

```bash
# 型定義のインストール
npm install --save-dev @types/node
npm install --save-dev @types/express
npm install --save-dev @types/lodash
npm install --save-dev @types/jest

# 型定義の検索
npx typesearch express
```

```typescript
// @types/express がインストールされていれば型が効く
import express, { Request, Response } from "express";

const app = express();

app.get("/users/:id", (req: Request, res: Response) => {
  const id = req.params.id; // 型: string
  res.json({ id });
});
```

### 型定義の解決順序

```
  TypeScript が型を探す順序:

  1. ローカルの .d.ts ファイル
     └── tsconfig.json の include/files で指定

  2. パッケージ自体の型定義
     └── package.json の "types" / "typings" フィールド

  3. @types パッケージ
     └── node_modules/@types/ ディレクトリ

  4. paths / typeRoots の設定
     └── tsconfig.json で指定されたパス

  見つからない場合 → 暗黙的に any（noImplicitAny で禁止可能）
```

### 型定義の提供パターン比較

| パターン | 例 | メリット | デメリット |
|----------|-----|---------|-----------|
| バンドル型 | axios, zod | インストール不要 | ライブラリ作者の負担 |
| @types | @types/express | コミュニティメンテナンス | バージョン不一致のリスク |
| 自前 .d.ts | カスタム宣言 | 完全な制御 | メンテナンスコスト |
| 型なし | 古いライブラリ | なし | 型安全性なし |

---

## 3. モジュール拡張

### コード例5: 既存モジュールの型拡張

```typescript
// types/express-extension.d.ts

// Express の Request にカスタムプロパティを追加
import "express";

declare module "express" {
  interface Request {
    userId?: string;
    role?: "admin" | "user" | "guest";
    startTime: number;
  }
}

// 使用側
import { Request, Response, NextFunction } from "express";

function authMiddleware(req: Request, res: Response, next: NextFunction) {
  req.userId = "user-123";  // OK: 拡張されたプロパティ
  req.role = "admin";       // OK
  req.startTime = Date.now();
  next();
}
```

### コード例6: 環境変数の型定義

```typescript
// types/env.d.ts

// process.env の型を拡張
declare global {
  namespace NodeJS {
    interface ProcessEnv {
      NODE_ENV: "development" | "production" | "test";
      PORT: string;
      DATABASE_URL: string;
      JWT_SECRET: string;
      REDIS_URL?: string;
    }
  }
}

export {};

// 使用側
const port = parseInt(process.env.PORT);  // 型: string（常に存在する）
const redis = process.env.REDIS_URL;      // 型: string | undefined
```

### コード例7: Wildcard モジュール宣言

```typescript
// types/assets.d.ts

// 画像ファイルのインポートに型を付ける
declare module "*.png" {
  const src: string;
  export default src;
}

declare module "*.svg" {
  import { FC, SVGProps } from "react";
  const SVGComponent: FC<SVGProps<SVGSVGElement>>;
  export default SVGComponent;
}

declare module "*.css" {
  const classes: Record<string, string>;
  export default classes;
}

declare module "*.json" {
  const value: unknown;
  export default value;
}

// 使用側
import logo from "./logo.png";       // 型: string
import Icon from "./icon.svg";       // 型: FC<SVGProps<SVGSVGElement>>
import styles from "./app.css";      // 型: Record<string, string>
```

---

## 4. 宣言ファイルの書き方ベストプラクティス

### コード例8: ライブラリの型定義を書く

```typescript
// types/analytics-sdk.d.ts
declare module "analytics-sdk" {
  // 設定型
  export interface AnalyticsConfig {
    apiKey: string;
    endpoint?: string;
    debug?: boolean;
    batchSize?: number;
  }

  // イベント型
  export interface TrackEvent {
    name: string;
    properties?: Record<string, string | number | boolean>;
    timestamp?: Date;
  }

  // メインクラス
  export class Analytics {
    constructor(config: AnalyticsConfig);

    /** イベントを送信 */
    track(event: TrackEvent): Promise<void>;

    /** ユーザーを識別 */
    identify(userId: string, traits?: Record<string, unknown>): Promise<void>;

    /** バッファをフラッシュ */
    flush(): Promise<void>;

    /** 接続を閉じる */
    close(): void;
  }

  // ファクトリ関数
  export function createAnalytics(config: AnalyticsConfig): Analytics;

  // デフォルトエクスポート
  export default createAnalytics;
}
```

### tsconfig.json での型定義設定

```
  tsconfig.json
  {
    "compilerOptions": {
      "typeRoots": ["./types", "./node_modules/@types"],
      "types": ["node", "jest"],      // 特定の@typesのみ使用
      "paths": {
        "*": ["./types/*"]            // カスタム型のパス
      }
    },
    "include": [
      "src/**/*",
      "types/**/*.d.ts"              // 型定義ファイルをインクルード
    ]
  }
```

---

## declare の種類比較

| 宣言 | 用途 | スコープ |
|------|------|---------|
| `declare function` | 関数の型宣言 | グローバル/モジュール |
| `declare const/let/var` | 変数の型宣言 | グローバル/モジュール |
| `declare class` | クラスの型宣言 | グローバル/モジュール |
| `declare module "x"` | モジュール全体の型 | モジュール |
| `declare global { }` | グローバル型の追加 | グローバル |
| `declare namespace` | 名前空間の型 | グローバル/モジュール |
| `declare enum` | 列挙型の宣言 | グローバル/モジュール |

---

## アンチパターン

### アンチパターン1: any で型定義を誤魔化す

```typescript
// BAD: 面倒だからanyで逃げる
declare module "some-library" {
  const lib: any;
  export default lib;
}

// GOOD: 最低限の型を書く（段階的に充実させる）
declare module "some-library" {
  export function doSomething(input: string): Promise<unknown>;
  export function configure(options: Record<string, unknown>): void;
}
```

### アンチパターン2: 不要なトリプルスラッシュディレクティブ

```typescript
// BAD: 現代のTypeScriptでは不要なケースが多い
/// <reference types="node" />
/// <reference path="./types.d.ts" />

// GOOD: tsconfig.json で管理する
// tsconfig.json の "types" と "include" で制御
```

---

## FAQ

### Q1: .d.ts ファイルはどこに置くべきですか？

**A:** プロジェクトルートに `types/` ディレクトリを作成し、そこに配置するのが一般的です。`tsconfig.json` の `include` に `"types/**/*.d.ts"` を追加してください。ライブラリを公開する場合は `package.json` の `"types"` フィールドでエントリポイントを指定します。

### Q2: @types パッケージとライブラリのバージョンが合わない場合は？

**A:** `@types/xxx` のメジャーバージョンは対応するライブラリのメジャーバージョンに合わせるのが慣例です（例: express@4.x に対して @types/express@4.x）。パッチレベルの不一致は通常問題ありません。大きな不一致がある場合はモジュール拡張で差分を補います。

### Q3: ライブラリに型定義が全くない場合はどうしますか？

**A:** 以下の優先順位で対応します:
1. DefinitelyTyped に型定義がないか確認
2. 自前の `.d.ts` ファイルを作成（最低限使うAPIだけ）
3. `declare module "library-name";` で暫定的に any にする（最終手段）
長期的には DefinitelyTyped に型定義をコントリビュートすることを検討してください。

---

## まとめ

| 項目 | 内容 |
|------|------|
| .d.ts ファイル | 型情報のみを記述するファイル。実装コードなし |
| declare | 外部に存在する値の型をコンパイラに伝えるキーワード |
| DefinitelyTyped | コミュニティ管理の型定義リポジトリ (@types) |
| モジュール拡張 | `declare module` で既存ライブラリの型を拡張 |
| グローバル拡張 | `declare global` でグローバル型を追加 |
| Wildcard宣言 | `declare module "*.png"` で非JSファイルに型を付与 |
| 解決順序 | ローカル → パッケージ内蔵 → @types → typeRoots |

---

## 次に読むべきガイド

- [../02-patterns/00-error-handling.md](../02-patterns/00-error-handling.md) -- エラーハンドリング
- [../03-tooling/00-tsconfig.md](../03-tooling/00-tsconfig.md) -- tsconfig.json の詳細

---

## 参考文献

1. **TypeScript Handbook: Declaration Files** -- https://www.typescriptlang.org/docs/handbook/declaration-files/introduction.html
2. **DefinitelyTyped** -- https://github.com/DefinitelyTyped/DefinitelyTyped
3. **TypeSearch** -- https://www.typescriptlang.org/dt/search
