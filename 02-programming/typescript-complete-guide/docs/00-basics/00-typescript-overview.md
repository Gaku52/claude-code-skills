# TypeScript概要

> JavaScriptの完全な上位互換（スーパーセット）として設計された静的型付き言語。型システムによりコードの安全性・保守性・開発体験を劇的に向上させる。

## この章で学ぶこと

1. **TypeScriptとは何か** -- JavaScriptとの関係、スーパーセットの意味、コンパイルの仕組み
2. **型システムがもたらす価値** -- バグの早期発見、リファクタリング安全性、IDE支援
3. **歴史とエコシステム** -- 誕生の背景、バージョンの変遷、主要ツールチェーン
4. **プロジェクト構成と設定** -- tsconfig.json の詳細、モジュールシステム、ビルド構成
5. **実務での導入戦略** -- 新規プロジェクト、既存プロジェクト移行、段階的導入
6. **TypeScriptの内部動作** -- コンパイラの仕組み、型推論エンジン、型消去
7. **パフォーマンスと最適化** -- コンパイル速度の改善、プロジェクト参照、インクリメンタルビルド

---

## 1. TypeScriptとは何か

### JavaScriptのスーパーセット

TypeScriptはMicrosoftが2012年に公開したオープンソース言語である。全ての正しいJavaScriptコードはそのままTypeScriptとしても有効である。TypeScriptはそこに**静的型付け**を追加する。

この「スーパーセット」という概念は非常に重要である。C言語とC++の関係に似ているが、TypeScriptの場合はコンパイル後にJavaScriptに変換される点が異なる。つまり、TypeScriptの型情報は実行時には一切存在しない。これを「型消去（Type Erasure）」と呼ぶ。

```
+------------------------------------------+
|            TypeScript                     |
|  +------------------------------------+  |
|  |          JavaScript                |  |
|  |  +------------------------------+  |  |
|  |  |       ECMAScript仕様         |  |  |
|  |  +------------------------------+  |  |
|  +------------------------------------+  |
|  + 型アノテーション                     |  |
|  + インターフェース                     |  |
|  + ジェネリクス                         |  |
|  + 列挙型                               |  |
|  + その他の型機能                       |  |
+------------------------------------------+
```

### スーパーセットであることの実践的意味

スーパーセットであることは、以下のような実践的な利点をもたらす。

```typescript
// 1. 既存のJavaScriptファイルをそのまま .ts に変更できる
// rename: utils.js → utils.ts
// 型エラーが出る箇所を段階的に修正していけばよい

// 2. JavaScriptライブラリをそのまま利用できる
import _ from "lodash"; // JavaScriptライブラリ
// @types/lodash をインストールすれば型補完も効く

// 3. JSDocコメントによる段階的な型付けも可能
/**
 * @param {string} name
 * @returns {string}
 */
function greetJS(name) {
  return `Hello, ${name}!`;
}
// TypeScriptコンパイラはJSDocの型情報も認識する
```

### コード例1: JavaScriptがそのままTypeScript

```typescript
// これは有効なJavaScriptであり、同時に有効なTypeScriptでもある
const greet = (name) => `Hello, ${name}!`;
console.log(greet("World"));
```

### コード例2: 型アノテーションの追加

```typescript
// 型アノテーションを追加すると、TypeScriptの力を活用できる
const greet = (name: string): string => `Hello, ${name}!`;

// コンパイルエラー: Argument of type 'number' is not assignable to parameter of type 'string'
// greet(42);

console.log(greet("World")); // OK
```

### コンパイルフロー

```
  TypeScript ソースコード (.ts / .tsx)
         |
         v
  +-------------------+
  | TypeScript        |
  | コンパイラ (tsc)   |
  +-------------------+
         |
    +----+----+
    |         |
    v         v
 JavaScript  型エラー
 (.js)       レポート
```

### コンパイルプロセスの詳細

TypeScriptコンパイラ（tsc）の内部処理は複数のフェーズに分かれている。

```
  ソースコード (.ts)
       |
       v
  +-------------+
  | Scanner     |  テキスト → トークン列
  +-------------+
       |
       v
  +-------------+
  | Parser      |  トークン列 → AST (抽象構文木)
  +-------------+
       |
       v
  +-------------+
  | Binder      |  AST → シンボルテーブル構築
  +-------------+
       |
       v
  +-------------+
  | Checker     |  型チェック実行（最も重い処理）
  +-------------+
       |
       v
  +-------------+
  | Emitter     |  AST → JavaScript出力
  +-------------+
       |
       v
  JavaScript (.js) + 宣言ファイル (.d.ts) + ソースマップ (.js.map)
```

```typescript
// コンパイラの各フェーズが何をするかの具体例

// Scanner: テキストを解析してトークン列を生成
// "const x: number = 42;" → [const, x, :, number, =, 42, ;]

// Parser: トークン列からASTを構築
// VariableStatement
//   └── VariableDeclaration
//       ├── Identifier: x
//       ├── TypeAnnotation: NumberKeyword
//       └── Initializer: NumericLiteral(42)

// Binder: シンボルテーブルを構築
// Symbol "x" → { type: number, flags: const, declarations: [...] }

// Checker: 型チェックを実行
// x の型 (number) と初期値 (42: number) が互換 → OK

// Emitter: JavaScriptを出力
// "const x = 42;" （型アノテーションが除去される）
```

### コード例3: コンパイル実行

```bash
# TypeScriptコンパイラのインストール
npm install -g typescript

# コンパイル
tsc hello.ts        # -> hello.js が生成される

# コンパイル（型チェックのみ、出力なし）
tsc --noEmit hello.ts

# 宣言ファイルの生成
tsc --declaration hello.ts  # -> hello.d.ts も生成される

# ソースマップの生成
tsc --sourceMap hello.ts    # -> hello.js.map も生成される

# ウォッチモード（ファイル変更を監視して自動コンパイル）
tsc --watch

# 特定のターゲットバージョンでコンパイル
tsc --target ES2020 hello.ts

# 複数ファイルのコンパイル
tsc src/**/*.ts --outDir dist/
```

### 型消去（Type Erasure）の具体例

```typescript
// TypeScriptソース
interface User {
  id: number;
  name: string;
}

function getUser(id: number): User {
  return { id, name: "Alice" };
}

const user: User = getUser(1);
console.log(user.name);
```

```javascript
// コンパイル後のJavaScript（型情報が完全に消去される）
"use strict";
function getUser(id) {
  return { id, name: "Alice" };
}
const user = getUser(1);
console.log(user.name);

// interface User は完全に消えている
// 関数の引数型・返り値型も消えている
// 変数の型アノテーションも消えている
```

### 型消去されないTypeScript構文

一部のTypeScript固有の構文は、コンパイル後もJavaScriptコードとして残る。

```typescript
// 1. enum はJavaScriptオブジェクトに変換される
enum Direction {
  Up = "UP",
  Down = "DOWN",
  Left = "LEFT",
  Right = "RIGHT",
}
// ↓ コンパイル後
// var Direction;
// (function (Direction) {
//     Direction["Up"] = "UP";
//     Direction["Down"] = "DOWN";
//     Direction["Left"] = "LEFT";
//     Direction["Right"] = "RIGHT";
// })(Direction || (Direction = {}));

// 2. const enum はインライン展開される
const enum StatusCode {
  OK = 200,
  NotFound = 404,
  ServerError = 500,
}
const code = StatusCode.OK;
// ↓ コンパイル後
// const code = 200; // 直接値が埋め込まれる

// 3. namespace はIIFEに変換される
namespace MathUtils {
  export function add(a: number, b: number): number {
    return a + b;
  }
}
// ↓ コンパイル後
// var MathUtils;
// (function (MathUtils) {
//     function add(a, b) { return a + b; }
//     MathUtils.add = add;
// })(MathUtils || (MathUtils = {}));

// 4. デコレータ（experimentalDecorators）はヘルパー関数に変換される
// 5. パラメータプロパティはコンストラクタ代入に変換される
class Point {
  constructor(
    public x: number,
    public y: number
  ) {}
}
// ↓ コンパイル後
// class Point {
//     constructor(x, y) {
//         this.x = x;
//         this.y = y;
//     }
// }
```

---

## 2. 型システムがもたらす価値

### コード例4: 型がバグを防ぐ

```typescript
// JavaScript（実行時まで気づかない）
function calculateArea(width, height) {
  return width * height;
}
calculateArea("10", 20); // "1020" -- 意図しない文字列連結

// TypeScript（コンパイル時に検出）
function calculateArea(width: number, height: number): number {
  return width * height;
}
// calculateArea("10", 20); // コンパイルエラー！
calculateArea(10, 20); // 200 -- 正しい結果
```

### 型によるバグ防止の詳細パターン

```typescript
// パターン1: null/undefined アクセスの防止
function getLength(str: string | null): number {
  // str.length; // エラー: Object is possibly 'null'
  if (str === null) return 0;
  return str.length; // OK: null チェック後は安全
}

// パターン2: 存在しないプロパティへのアクセス防止
interface Config {
  host: string;
  port: number;
}

function createConnection(config: Config) {
  // config.hostname; // エラー: Property 'hostname' does not exist
  return `${config.host}:${config.port}`; // OK
}

// パターン3: 配列操作の型安全性
const numbers: number[] = [1, 2, 3];
// numbers.push("4"); // エラー: Argument of type 'string' is not assignable
numbers.push(4); // OK

// パターン4: switch文の網羅性チェック
type Shape = "circle" | "square" | "triangle";

function getArea(shape: Shape, size: number): number {
  switch (shape) {
    case "circle":
      return Math.PI * size * size;
    case "square":
      return size * size;
    case "triangle":
      return (Math.sqrt(3) / 4) * size * size;
    // "triangle" を忘れた場合、コンパイラが警告する
    // （exhaustive check を有効にしている場合）
  }
}

// パターン5: 関数の戻り値の型チェック
function divide(a: number, b: number): number {
  if (b === 0) {
    // return "Error"; // エラー: Type 'string' is not assignable to type 'number'
    throw new Error("Division by zero");
  }
  return a / b;
}

// パターン6: 暗黙の型変換の防止
const value: number = 42;
// const result: string = value; // エラー: Type 'number' is not assignable to type 'string'
const result: string = String(value); // OK: 明示的な変換
```

### コード例5: IDEの自動補完

```typescript
interface User {
  id: number;
  name: string;
  email: string;
  createdAt: Date;
}

function displayUser(user: User) {
  // user. と入力した時点で id, name, email, createdAt が候補に表示される
  console.log(`${user.name} <${user.email}>`);
}
```

### IDEサポートの詳細

TypeScriptの型情報は、IDE（VSCode, WebStorm等）で以下のような高度な開発支援を可能にする。

```typescript
// 1. 自動補完（IntelliSense）
interface ApiResponse<T> {
  data: T;
  status: number;
  headers: Record<string, string>;
  pagination: {
    page: number;
    perPage: number;
    total: number;
    totalPages: number;
  };
}

// response. と入力するだけで data, status, headers, pagination が候補に表示
// response.pagination. と入力すれば page, perPage, total, totalPages が表示
// 階層の深いプロパティまで正確に補完される

// 2. シグネチャヘルプ（関数の引数情報表示）
function createUser(
  name: string,
  email: string,
  options?: {
    role?: "admin" | "user" | "guest";
    active?: boolean;
    metadata?: Record<string, unknown>;
  }
): User {
  // ...
  return {} as User;
}
// createUser( と入力した時点で、3つの引数の型情報が表示される
// 第3引数のオブジェクト構造も表示される

// 3. クイック情報（ホバー時の型情報表示）
const users = [
  { id: 1, name: "Alice", age: 30 },
  { id: 2, name: "Bob", age: 25 },
];
// users にホバーすると { id: number; name: string; age: number; }[] と表示

const names = users.map(u => u.name);
// names にホバーすると string[] と表示

const eldest = users.reduce((prev, curr) =>
  prev.age > curr.age ? prev : curr
);
// eldest にホバーすると { id: number; name: string; age: number } と表示

// 4. エラーのインライン表示
const config = {
  host: "localhost",
  port: 3000,
};
// config.port = "8080"; // 赤い波線でエラー表示: Type 'string' is not assignable to type 'number'

// 5. リファクタリング支援
// - シンボル名の一括変更（F2キー）
// - 関数の抽出
// - インターフェースの自動抽出
// - import文の自動追加・整理

// 6. コードナビゲーション
// - 定義へのジャンプ（F12）
// - 参照の検索（Shift+F12）
// - 実装へのジャンプ（Ctrl+F12）
// - 型定義へのジャンプ
```

### 型システムの利点比較

| 観点 | JavaScript (型なし) | TypeScript (型あり) |
|------|---------------------|---------------------|
| バグ検出タイミング | 実行時（本番含む） | コンパイル時（開発中） |
| リファクタリング | 手動で全箇所確認 | コンパイラが影響範囲を自動検出 |
| IDE補完 | 推測ベース（不正確） | 型情報ベース（正確） |
| ドキュメント | コメントで記述（陳腐化しやすい） | 型が生きたドキュメントになる |
| チーム開発 | 口頭・ドキュメント依存 | 型がコントラクトとして機能 |
| 学習コスト | 低い | やや高い（投資価値あり） |
| デバッグ時間 | 長い（型起因のバグが多い） | 短い（型エラーは開発中に解消） |
| コードレビュー | 型の意図を口頭で確認 | 型が意図を明示する |
| 新メンバーのオンボーディング | コードを読んで型を推測 | 型が入口ガイドになる |

### コード例6: リファクタリング安全性

```typescript
// 大規模リファクタリングのシナリオ
// Before: price フィールドが円単位
interface Product {
  id: number;
  name: string;
  price: number; // 円単位
}

function formatPrice(product: Product): string {
  return `¥${product.price.toLocaleString()}`;
}

function calculateTotal(products: Product[]): number {
  return products.reduce((sum, p) => sum + p.price, 0);
}

function applyDiscount(product: Product, rate: number): number {
  return product.price * (1 - rate);
}

// After: price を priceInCents にリネーム（銭単位に変更）
interface Product {
  id: number;
  name: string;
  priceInCents: number; // 銭単位
}

// TypeScriptコンパイラが以下の全箇所でエラーを出す:
// - formatPrice 内の product.price
// - calculateTotal 内の p.price
// - applyDiscount 内の product.price
// → 修正漏れが絶対に起きない

// さらに、型の変更による影響範囲を「エラー一覧」として確認できる
// IDE上では赤い波線として視覚的に表示される
```

### 大規模プロジェクトでのTypeScriptの効果

```typescript
// 実際のプロジェクトで型がどのように安全性を担保するかの例

// API レスポンスの型定義
interface ApiResponse<T> {
  success: boolean;
  data: T;
  error?: {
    code: string;
    message: string;
    details?: Record<string, string[]>;
  };
  meta?: {
    requestId: string;
    timestamp: number;
  };
}

// ユーザー関連の型定義
interface User {
  id: string;
  email: string;
  profile: {
    firstName: string;
    lastName: string;
    avatar?: string;
    bio?: string;
  };
  settings: {
    theme: "light" | "dark" | "system";
    language: "ja" | "en" | "zh";
    notifications: {
      email: boolean;
      push: boolean;
      sms: boolean;
    };
  };
  createdAt: string;
  updatedAt: string;
}

// APIクライアント
async function fetchUser(id: string): Promise<ApiResponse<User>> {
  const response = await fetch(`/api/users/${id}`);
  return response.json();
}

// 呼び出し側で型安全にデータにアクセスできる
async function displayUserProfile(userId: string): Promise<void> {
  const result = await fetchUser(userId);

  if (result.success) {
    // result.data は User 型として認識される
    const { profile, settings } = result.data;
    console.log(`${profile.firstName} ${profile.lastName}`);
    console.log(`Theme: ${settings.theme}`);
    console.log(`Language: ${settings.language}`);

    // settings.notifications.email は boolean として認識
    if (settings.notifications.email) {
      console.log("Email notifications are enabled");
    }
  } else {
    // result.error にアクセスできる
    console.error(`Error: ${result.error?.message}`);
  }
}
```

---

## 3. 歴史とエコシステム

### TypeScriptの歴史年表

```
2012  v0.8   初回リリース（Microsoft）
  |          Anders Hejlsberg（C#設計者）が主導
  |
2013  v0.9   ジェネリクスの導入
  |
2014  v1.0   安定版リリース
  |          Angular 2がTypeScriptを採用（大きな転機）
  |
2015  v1.5   ES2015モジュール対応、デコレータ（実験的）
  |
2016  v2.0   Non-nullable types, Tagged Unions, readonly
  |   v2.1   keyof, Mapped Types, Lookup Types
  |
2017  v2.3   --strict フラグ導入
  |   v2.4   String enums
  |
2018  v3.0   Project References, unknown型
  |   v3.1   Mapped types on tuples and arrays
  |
2019  v3.7   Optional Chaining (?.), Nullish Coalescing (??)
  |   v3.8   Type-Only Imports/Exports
  |
2020  v4.0   Variadic Tuple Types, Labeled Tuples
  |   v4.1   Template Literal Types, Key Remapping
  |
2021  v4.5   Awaited型, ESM対応強化
  |   v4.7   Node.js ESM 対応, instantiation expressions
  |
2022  v4.9   satisfies 演算子
  |
2023  v5.0   Decorators (Stage 3), const型パラメータ
  |   v5.2   using宣言, デコレータメタデータ
  |
2024  v5.4   NoInfer, Object.groupBy型
  |   v5.5   型述語の推論、正規表現チェック
  |   v5.6   --noUncheckedSideEffectImports
  |
2025  v5.7   --erasableSyntaxOnly, 最新機能
  |          Node.js が --experimental-strip-types で直接実行対応
```

### 各バージョンの注目機能の詳細

```typescript
// TypeScript 2.0: Non-nullable types
// strictNullChecks を有効にすることで、null/undefined を厳密に型チェック
let name: string;
// name = null;  // エラー（strictNullChecks有効時）
let nullableName: string | null = null; // OK

// TypeScript 2.1: keyof と Mapped Types
interface Person {
  name: string;
  age: number;
}
type PersonKeys = keyof Person; // "name" | "age"
type ReadonlyPerson = { readonly [K in keyof Person]: Person[K] };

// TypeScript 3.0: unknown型
// any よりも安全な「何でも受け入れるが使う前にチェックが必要」な型
function processValue(value: unknown): string {
  // value.toString(); // エラー: Object is of type 'unknown'
  if (typeof value === "string") {
    return value.toUpperCase(); // OK: string に絞り込まれた
  }
  return String(value);
}

// TypeScript 3.7: Optional Chaining
interface Company {
  name: string;
  address?: {
    street: string;
    city: string;
    country?: string;
  };
}
function getCountry(company: Company): string | undefined {
  return company.address?.country; // address が undefined でも安全
}

// TypeScript 4.1: Template Literal Types
type HTTPMethod = "GET" | "POST" | "PUT" | "DELETE";
type Endpoint = "/users" | "/posts" | "/comments";
type Route = `${HTTPMethod} ${Endpoint}`;
// "GET /users" | "GET /posts" | ... | "DELETE /comments" の24通りの組み合わせ

// TypeScript 4.9: satisfies 演算子
// 型チェックしつつ、推論された型を維持する
const palette = {
  red: [255, 0, 0],
  green: "#00FF00",
  blue: [0, 0, 255],
} satisfies Record<string, string | number[]>;

// palette.red は number[] として推論される（Record<string, string | number[]> ではない）
const redValue = palette.red[0]; // number（satisfies なしでは string | number になる）

// TypeScript 5.0: const型パラメータ
function createRoute<const T extends readonly string[]>(routes: T): T {
  return routes;
}
const routes = createRoute(["home", "about", "contact"]);
// routes の型は readonly ["home", "about", "contact"]（as const 不要）

// TypeScript 5.2: using宣言（Explicit Resource Management）
class FileHandle {
  [Symbol.dispose]() {
    console.log("File closed");
  }
}
function processFile() {
  using file = new FileHandle();
  // file を使う処理
  // スコープを抜けると自動的に [Symbol.dispose]() が呼ばれる
}
```

### エコシステム全体像

| カテゴリ | 主要ツール | 役割 |
|----------|-----------|------|
| コンパイラ | tsc | 型チェック + トランスパイル |
| バンドラ | esbuild, SWC, Vite | 高速ビルド |
| リンター | typescript-eslint | コード品質チェック |
| フォーマッター | Prettier, dprint | コード整形 |
| テスト | Vitest, Jest | 型安全なテスト |
| スキーマ | Zod, io-ts, Valibot | ランタイムバリデーション |
| ORM | Prisma, Drizzle, Kysely | 型安全なDB操作 |
| API | tRPC, GraphQL Code Generator | 型安全なAPI通信 |
| フレームワーク | Next.js, Remix, Astro, Hono | フルスタック開発 |
| ランタイム | Node.js, Deno, Bun | TypeScript実行環境 |
| モノレポ | Turborepo, Nx | 大規模プロジェクト管理 |

### エコシステムの詳細解説

```typescript
// ===== Zod: ランタイムバリデーション =====
import { z } from "zod";

// スキーマ定義 = 型定義 + バリデーション
const UserSchema = z.object({
  id: z.string().uuid(),
  name: z.string().min(1).max(100),
  email: z.string().email(),
  age: z.number().int().min(0).max(150),
  role: z.enum(["admin", "user", "guest"]),
});

// スキーマから型を自動生成
type User = z.infer<typeof UserSchema>;
// → { id: string; name: string; email: string; age: number; role: "admin" | "user" | "guest" }

// ランタイムでバリデーション
function createUser(input: unknown): User {
  return UserSchema.parse(input); // 不正なデータは例外をスロー
}

// ===== Prisma: 型安全なDB操作 =====
// schema.prisma から自動生成される型を使用
// const user = await prisma.user.findUnique({
//   where: { id: "..." },
//   select: {
//     name: true,
//     email: true,
//     posts: {
//       select: { title: true },
//     },
//   },
// });
// user の型: { name: string; email: string; posts: { title: string }[] } | null

// ===== tRPC: 型安全なAPI通信 =====
// サーバー側の型定義がクライアント側に自動伝播
// APIのスキーマ変更時にクライアント側でコンパイルエラーが出る
// → フロントエンドとバックエンドの型不整合を完全に防止

// ===== Vitest: 型安全なテスト =====
import { describe, it, expect } from "vitest";

describe("User", () => {
  it("should create a valid user", () => {
    const user: User = {
      id: "550e8400-e29b-41d4-a716-446655440000",
      name: "Alice",
      email: "alice@example.com",
      age: 30,
      role: "admin",
    };
    expect(user.name).toBe("Alice");
  });
});
```

### コード例7: 最小限のTypeScriptプロジェクト構成

```bash
# プロジェクト初期化
mkdir my-ts-project && cd my-ts-project
npm init -y
npm install typescript --save-dev
npx tsc --init

# ディレクトリ構成
# my-ts-project/
# ├── src/
# │   └── index.ts
# ├── dist/           # コンパイル出力
# ├── tsconfig.json
# └── package.json
```

---

## 4. プロジェクト構成と設定

### tsconfig.json の詳細

tsconfig.json はTypeScriptプロジェクトの設定ファイルであり、コンパイラの動作を制御する。以下に実務で頻繁に使用する設定を網羅的に解説する。

```jsonc
{
  "compilerOptions": {
    // ===== 型チェック関連 =====
    "strict": true,                    // 全てのstrictチェックを有効化（推奨）
    // strict: true は以下を全て有効にする:
    // - strictNullChecks: null/undefined の厳密チェック
    // - strictFunctionTypes: 関数型の厳密チェック
    // - strictBindCallApply: bind/call/apply の厳密チェック
    // - strictPropertyInitialization: クラスプロパティの初期化チェック
    // - noImplicitAny: 暗黙の any を禁止
    // - noImplicitThis: 暗黙の this を禁止
    // - alwaysStrict: "use strict" を出力
    // - useUnknownInCatchVariables: catch 変数を unknown 型に

    "noUncheckedIndexedAccess": true,  // インデックスアクセスに undefined を追加
    "noUnusedLocals": true,            // 未使用ローカル変数をエラーに
    "noUnusedParameters": true,        // 未使用パラメータをエラーに
    "noImplicitReturns": true,         // 暗黙のreturnをエラーに
    "noFallthroughCasesInSwitch": true, // switchのフォールスルーをエラーに
    "exactOptionalPropertyTypes": true, // オプショナルプロパティの厳密チェック
    "noPropertyAccessFromIndexSignature": true, // インデックスシグネチャへのドットアクセスを禁止

    // ===== モジュール関連 =====
    "module": "ESNext",                // モジュールシステム
    "moduleResolution": "bundler",     // モジュール解決戦略（bundler推奨）
    "esModuleInterop": true,           // CommonJS/ESM相互運用
    "allowImportingTsExtensions": true, // .ts拡張子でのimportを許可
    "resolveJsonModule": true,         // JSONファイルのimportを許可
    "isolatedModules": true,           // ファイル単位のトランスパイルを保証

    // ===== 出力関連 =====
    "target": "ES2022",                // 出力するJavaScriptのバージョン
    "outDir": "./dist",                // 出力先ディレクトリ
    "declaration": true,               // .d.ts ファイルを生成
    "declarationMap": true,            // .d.ts のソースマップを生成
    "sourceMap": true,                 // .js のソースマップを生成
    "removeComments": false,           // コメントを維持

    // ===== パス関連 =====
    "rootDir": "./src",                // ソースのルートディレクトリ
    "baseUrl": "./src",                // パス解決のベース
    "paths": {                         // パスエイリアス
      "@/*": ["./*"],
      "@components/*": ["./components/*"],
      "@utils/*": ["./utils/*"],
      "@types/*": ["./types/*"]
    },

    // ===== その他 =====
    "skipLibCheck": true,              // .d.ts のチェックをスキップ（ビルド高速化）
    "forceConsistentCasingInFileNames": true, // ファイル名の大文字小文字を厳密チェック
    "lib": ["ES2022", "DOM", "DOM.Iterable"] // 使用するライブラリの型定義
  },

  "include": ["src/**/*"],             // コンパイル対象
  "exclude": [                         // コンパイル除外
    "node_modules",
    "dist",
    "**/*.test.ts",
    "**/*.spec.ts"
  ]
}
```

### プロジェクト種別ごとの推奨tsconfig

```jsonc
// ===== Node.js バックエンドプロジェクト =====
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

```jsonc
// ===== React フロントエンドプロジェクト =====
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true  // Vite/esbuildがビルドするため、tscは型チェックのみ
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

```jsonc
// ===== ライブラリプロジェクト =====
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    // ライブラリは幅広い環境で使えるよう、低めのtargetを設定
    "lib": ["ES2020"]
  },
  "include": ["src"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
```

### モジュールシステムの選択

```typescript
// ===== CommonJS (Node.js 伝統的なモジュールシステム) =====
// tsconfig: "module": "CommonJS"
const express = require("express");  // require を使用
module.exports = { myFunction };     // module.exports を使用

// ===== ESModules (現代の標準モジュールシステム) =====
// tsconfig: "module": "ESNext" or "NodeNext"
import express from "express";       // import を使用
export { myFunction };               // export を使用

// ===== moduleResolution の選択指針 =====
// "node10" (= "node"): 古いNode.jsスタイル（非推奨）
// "node16" / "nodenext": Node.js 16+のESM対応
// "bundler": Vite, webpack等のバンドラー使用時（推奨）

// moduleResolution による動作の違い
// "bundler" の場合:
import { utils } from "./utils";     // 拡張子省略OK
import data from "./data.json";      // JSON import OK

// "nodenext" の場合:
import { utils } from "./utils.js";  // 拡張子必須（.ts → .js）
import data from "./data.json" with { type: "json" }; // import assertions
```

### ディレクトリ構成パターン

```
# ===== 小規模プロジェクト =====
my-app/
├── src/
│   ├── index.ts          # エントリポイント
│   ├── types.ts          # 型定義
│   ├── utils.ts          # ユーティリティ
│   └── config.ts         # 設定
├── tests/
│   └── index.test.ts
├── tsconfig.json
├── package.json
└── .gitignore

# ===== 中規模プロジェクト（機能ごとに分割） =====
my-app/
├── src/
│   ├── index.ts
│   ├── types/
│   │   ├── index.ts
│   │   ├── user.ts
│   │   └── product.ts
│   ├── services/
│   │   ├── user.service.ts
│   │   └── product.service.ts
│   ├── repositories/
│   │   ├── user.repository.ts
│   │   └── product.repository.ts
│   ├── controllers/
│   │   ├── user.controller.ts
│   │   └── product.controller.ts
│   ├── middleware/
│   │   ├── auth.ts
│   │   └── validation.ts
│   └── utils/
│       ├── logger.ts
│       └── helpers.ts
├── tests/
│   ├── services/
│   ├── repositories/
│   └── controllers/
├── tsconfig.json
├── tsconfig.test.json    # テスト用設定
├── package.json
└── .gitignore

# ===== 大規模プロジェクト（モノレポ） =====
my-monorepo/
├── packages/
│   ├── shared/           # 共通型定義・ユーティリティ
│   │   ├── src/
│   │   ├── tsconfig.json
│   │   └── package.json
│   ├── api/              # バックエンド
│   │   ├── src/
│   │   ├── tsconfig.json
│   │   └── package.json
│   ├── web/              # フロントエンド
│   │   ├── src/
│   │   ├── tsconfig.json
│   │   └── package.json
│   └── mobile/           # モバイルアプリ
│       ├── src/
│       ├── tsconfig.json
│       └── package.json
├── tsconfig.base.json    # 共通設定
├── turbo.json            # Turborepo設定
├── package.json
└── pnpm-workspace.yaml
```

---

## 5. 実務での導入戦略

### 新規プロジェクトでの導入

```bash
# ===== 方法1: 手動セットアップ =====
mkdir new-project && cd new-project
npm init -y
npm install typescript @types/node --save-dev
npx tsc --init

# package.json にスクリプトを追加
# {
#   "scripts": {
#     "build": "tsc",
#     "dev": "tsc --watch",
#     "typecheck": "tsc --noEmit"
#   }
# }

# ===== 方法2: フレームワークのスキャフォールディング =====
# Next.js
npx create-next-app@latest --typescript

# Vite + React
npm create vite@latest my-app -- --template react-ts

# Hono（バックエンド）
npm create hono@latest

# Astro
npm create astro@latest
```

### 既存JavaScriptプロジェクトからの段階的移行

```typescript
// ===== ステップ1: TypeScriptの導入（最小限） =====
// tsconfig.json を作成（permissiveな設定から開始）
// {
//   "compilerOptions": {
//     "allowJs": true,          // .jsファイルも含める
//     "checkJs": false,         // .jsの型チェックは無効
//     "strict": false,          // strictモードは後で有効化
//     "noImplicitAny": false,   // 暗黙のanyを許容
//     "target": "ES2020",
//     "module": "ESNext",
//     "moduleResolution": "bundler",
//     "outDir": "./dist",
//     "esModuleInterop": true,
//     "skipLibCheck": true
//   },
//   "include": ["src/**/*"]
// }

// ===== ステップ2: ファイルを段階的に .ts に変更 =====
// 依存関係の少ない末端ファイルから変更していく
// utils.js → utils.ts
// constants.js → constants.ts

// ===== ステップ3: 型定義を追加 =====
// まず @types パッケージをインストール
// npm install @types/express @types/lodash --save-dev

// ===== ステップ4: strictモードを段階的に有効化 =====
// 1. noImplicitAny: true   （暗黙のanyを禁止）
// 2. strictNullChecks: true （null/undefinedを厳密チェック）
// 3. strictFunctionTypes: true
// 4. strict: true           （全てのstrictチェックを有効化）

// ===== ステップ5: 残りの .js ファイルを変換 =====
// 優先度の高い順に変換していく
// 1. 型定義ファイル（types, interfaces）
// 2. ユーティリティ関数
// 3. ビジネスロジック
// 4. API層
// 5. UI層
```

### 移行時の実践的テクニック

```typescript
// テクニック1: JSDocで型を付ける（.tsに変更せずに型チェック）
// config: "checkJs": true, "allowJs": true

// utils.js
/**
 * @param {string} name
 * @param {number} age
 * @returns {{ name: string, age: number, greeting: string }}
 */
function createPerson(name, age) {
  return {
    name,
    age,
    greeting: `Hi, I'm ${name}`,
  };
}

// テクニック2: 一時的な型アサーションで移行を進める
// 完全な型付けが難しい場合の暫定措置
const legacyConfig = getLegacyConfig() as any; // 暫定的にany
// TODO: getLegacyConfig の戻り値型を定義する

// テクニック3: 宣言ファイルで既存モジュールに型を付ける
// legacy-module.d.ts
declare module "legacy-module" {
  export function doSomething(input: string): Promise<number>;
  export interface LegacyResult {
    status: "ok" | "error";
    data: unknown;
  }
}

// テクニック4: @ts-expect-error で既知の型エラーを一時的に抑制
// @ts-expect-error: Legacy code, will be fixed in #1234
const result = legacyFunction(untypedData);

// テクニック5: 型安全な移行用ユーティリティ
// 安全にunknownからの型変換を行うヘルパー
function isString(value: unknown): value is string {
  return typeof value === "string";
}

function isNumber(value: unknown): value is number {
  return typeof value === "number" && !Number.isNaN(value);
}

function assertNonNull<T>(value: T | null | undefined, message?: string): T {
  if (value === null || value === undefined) {
    throw new Error(message ?? "Unexpected null or undefined");
  }
  return value;
}
```

### TypeScript導入の効果測定

```typescript
// 導入効果を測定するための指標

// 1. バグ検出率の変化
// - TypeScript導入前: 本番環境で発見されるバグの件数
// - TypeScript導入後: コンパイル時に発見されるバグの件数
// 一般的に、型関連のバグの80%以上がコンパイル時に検出される

// 2. 開発速度の変化
// - 初期: 型定義の作成で速度が10-20%低下
// - 中期: IDE支援により速度が回復
// - 長期: リファクタリング速度が50-100%向上

// 3. コードレビュー効率
// - 型の意図が明確になり、レビュー時間が20-30%短縮
// - 「この関数の引数は何型？」という質問が激減

// 4. 新メンバーのオンボーディング時間
// - 型定義がドキュメントとして機能し、コード理解が加速
// - 一般的に30-50%の時間短縮が報告されている
```

---

## 6. TypeScriptの内部動作

### 型推論エンジンの仕組み

TypeScriptの型推論は「構造的型付け（Structural Typing）」に基づいている。

```typescript
// 構造的型付け vs 名前的型付け

// TypeScriptは構造的型付け（Duck Typing）
// 同じ構造を持つ型は互換性がある
interface Point {
  x: number;
  y: number;
}

interface Coordinate {
  x: number;
  y: number;
}

const point: Point = { x: 1, y: 2 };
const coord: Coordinate = point; // OK: 構造が同じなので互換性あり

// Java/C#などは名前的型付け（Nominal Typing）
// 同じ構造でも型名が異なれば互換性がない
// TypeScriptでこれを再現するにはブランド型を使う

// ブランド型（Nominal Typing のエミュレーション）
type USD = number & { __brand: "USD" };
type EUR = number & { __brand: "EUR" };

function createUSD(amount: number): USD {
  return amount as USD;
}

function createEUR(amount: number): EUR {
  return amount as EUR;
}

const dollars: USD = createUSD(100);
const euros: EUR = createEUR(85);
// const mixed: USD = euros; // エラー: EUR は USD に代入できない
```

### 型推論のフロー制御（Control Flow Analysis）

```typescript
// TypeScriptは制御フローを分析して型を自動的に絞り込む

function processInput(input: string | number | null | undefined) {
  // この時点: string | number | null | undefined

  if (input === null || input === undefined) {
    return; // この分岐後: string | number
  }

  // この時点: string | number (null | undefined は除外)

  if (typeof input === "string") {
    // この分岐内: string
    console.log(input.toUpperCase()); // OK
  } else {
    // この分岐内: number
    console.log(input.toFixed(2)); // OK
  }
}

// instanceof による型の絞り込み
class Dog {
  bark() { console.log("Woof!"); }
}
class Cat {
  meow() { console.log("Meow!"); }
}

function makeSound(animal: Dog | Cat) {
  if (animal instanceof Dog) {
    animal.bark(); // OK: Dog に絞り込まれた
  } else {
    animal.meow(); // OK: Cat に絞り込まれた
  }
}

// in 演算子による型の絞り込み
interface Fish {
  swim: () => void;
}
interface Bird {
  fly: () => void;
}

function move(animal: Fish | Bird) {
  if ("swim" in animal) {
    animal.swim(); // OK: Fish に絞り込まれた
  } else {
    animal.fly(); // OK: Bird に絞り込まれた
  }
}

// ユーザー定義型ガード
interface ApiError {
  type: "error";
  code: number;
  message: string;
}

interface ApiSuccess<T> {
  type: "success";
  data: T;
}

type ApiResult<T> = ApiSuccess<T> | ApiError;

function isApiError<T>(result: ApiResult<T>): result is ApiError {
  return result.type === "error";
}

function handleResult<T>(result: ApiResult<T>): T {
  if (isApiError(result)) {
    // result は ApiError に絞り込まれた
    throw new Error(`API Error ${result.code}: ${result.message}`);
  }
  // result は ApiSuccess<T> に絞り込まれた
  return result.data;
}
```

### 型の互換性チェック

```typescript
// TypeScriptの型互換性ルール

// 1. Excess Property Check（余剰プロパティチェック）
interface Options {
  width: number;
  height: number;
}

// オブジェクトリテラルを直接代入する場合は余剰プロパティがエラーになる
// const opts: Options = { width: 100, height: 200, color: "red" }; // エラー

// 変数経由で代入する場合は余剰プロパティが許容される
const rawOpts = { width: 100, height: 200, color: "red" };
const opts: Options = rawOpts; // OK（余剰プロパティは無視される）

// 2. 関数の互換性
type Handler = (event: MouseEvent) => void;
type GeneralHandler = (event: Event) => void;

// 関数パラメータは反変（contravariant）
// より広い型のパラメータを受け入れる関数は、より狭い型のパラメータを持つ関数に代入できない
// let handler: Handler = generalHandler; // strictFunctionTypes 有効時はエラー

// 3. 共変（covariant）な戻り値
interface Animal { name: string; }
interface Dog extends Animal { breed: string; }

type GetAnimal = () => Animal;
type GetDog = () => Dog;

const getDog: GetDog = () => ({ name: "Buddy", breed: "Labrador" });
const getAnimal: GetAnimal = getDog; // OK: Dog は Animal のサブタイプ
```

---

## 7. パフォーマンスと最適化

### コンパイル速度の改善

```jsonc
// tsconfig.json でのパフォーマンス最適化

{
  "compilerOptions": {
    // 1. skipLibCheck: .d.ts ファイルの型チェックをスキップ
    "skipLibCheck": true,   // ビルド時間を30-50%短縮できることがある

    // 2. incremental: インクリメンタルコンパイル
    "incremental": true,    // .tsbuildinfo ファイルで差分ビルド
    "tsBuildInfoFile": "./dist/.tsbuildinfo",

    // 3. isolatedModules: ファイル単位でのトランスパイルを保証
    "isolatedModules": true // esbuild/SWC との互換性確保
  }
}
```

### プロジェクト参照（Project References）

```jsonc
// 大規模プロジェクトでビルド時間を劇的に短縮する仕組み

// tsconfig.json (ルート)
{
  "references": [
    { "path": "./packages/shared" },
    { "path": "./packages/api" },
    { "path": "./packages/web" }
  ],
  "files": []  // ルートはファイルをコンパイルしない
}

// packages/shared/tsconfig.json
{
  "compilerOptions": {
    "composite": true,      // プロジェクト参照に必須
    "declaration": true,    // .d.ts を出力
    "declarationMap": true, // ソースへのジャンプを可能に
    "outDir": "./dist"
  },
  "include": ["src/**/*"]
}

// packages/api/tsconfig.json
{
  "compilerOptions": {
    "composite": true,
    "outDir": "./dist"
  },
  "references": [
    { "path": "../shared" }  // shared パッケージへの依存
  ],
  "include": ["src/**/*"]
}
```

```bash
# プロジェクト参照を使ったビルド
tsc --build              # 全パッケージをビルド（依存関係順に）
tsc --build --watch      # ウォッチモード
tsc --build --clean      # ビルド成果物を削除
tsc --build --verbose    # 詳細ログ出力
```

### ビルドツールとの組み合わせ

```typescript
// ===== esbuild: 超高速バンドラー =====
// TypeScriptの型チェックは行わず、トランスパイルのみ
// tsc の 10-100倍高速

// esbuild.config.ts
import { build } from "esbuild";

await build({
  entryPoints: ["src/index.ts"],
  bundle: true,
  outfile: "dist/index.js",
  platform: "node",
  target: "node20",
  format: "esm",
});

// 型チェックは tsc --noEmit で別途実行
// package.json:
// {
//   "scripts": {
//     "build": "esbuild src/index.ts --bundle --outdir=dist",
//     "typecheck": "tsc --noEmit",
//     "ci": "npm run typecheck && npm run build"
//   }
// }

// ===== SWC: Rustベースの高速トランスパイラ =====
// Next.js, Vite で内部的に使用される

// ===== Vite: 開発サーバー + ビルドツール =====
// 開発時: esbuild でトランスパイル（型チェックなし、高速HMR）
// ビルド時: Rollup + esbuild でバンドル
// 型チェック: vite-plugin-checker で別スレッド実行

// vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import checker from "vite-plugin-checker";

export default defineConfig({
  plugins: [
    react(),
    checker({
      typescript: true,  // TypeScript 型チェックを別スレッドで実行
    }),
  ],
});
```

### CI/CDパイプラインでの型チェック

```yaml
# GitHub Actions での型チェック例
# .github/workflows/typecheck.yml
name: TypeScript Check

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  typecheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npx tsc --noEmit  # 型チェックのみ（出力なし）

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npx eslint src/ --ext .ts,.tsx

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
      - run: npm ci
      - run: npx vitest run
```

---

## 8. TypeScriptランタイム環境

### Node.js での TypeScript 実行

```bash
# ===== 方法1: ts-node（従来の方法） =====
npm install ts-node --save-dev
npx ts-node src/index.ts

# ===== 方法2: tsx（高速な ts-node 代替） =====
npm install tsx --save-dev
npx tsx src/index.ts
npx tsx watch src/index.ts  # ウォッチモード付き

# ===== 方法3: Node.js 22+ のネイティブ TypeScript サポート =====
# --experimental-strip-types フラグで直接実行
node --experimental-strip-types src/index.ts

# Node.js 23+ では --experimental-transform-types も利用可能
# enum や namespace などの TypeScript 固有構文もサポート
```

### Deno での TypeScript 実行

```bash
# Deno は TypeScript をネイティブサポート
# 設定ファイルなしで .ts ファイルを直接実行できる
deno run src/index.ts

# パーミッション付きで実行
deno run --allow-net --allow-read src/server.ts

# deno.json で TypeScript コンパイラオプションを設定可能
# {
#   "compilerOptions": {
#     "strict": true,
#     "lib": ["deno.window"]
#   }
# }
```

### Bun での TypeScript 実行

```bash
# Bun は TypeScript をネイティブサポート
# 非常に高速な実行が可能
bun run src/index.ts

# テスト実行
bun test

# パッケージインストール（npmの5-10倍高速）
bun install
```

---

## アンチパターン

### アンチパターン1: any の濫用

```typescript
// BAD: anyを使うと型システムの恩恵がゼロになる
function processData(data: any): any {
  return data.map((item: any) => item.value);
}

// GOOD: 適切な型を定義する
interface DataItem {
  value: string;
}
function processData(data: DataItem[]): string[] {
  return data.map((item) => item.value);
}

// BETTER: ジェネリクスで汎用的にする
function processData<T, K extends keyof T>(data: T[], key: K): T[K][] {
  return data.map((item) => item[key]);
}
```

### アンチパターン2: TypeScriptを「ただのJavaScript + 拡張子変更」として使う

```typescript
// BAD: .ts にしただけで型を一切書かない
// tsconfig.json で strict: false にする
// → JavaScriptと変わらず、移行コストだけ発生

// GOOD: strict: true を有効にし、段階的に型をつける
// tsconfig.json
{
  "compilerOptions": {
    "strict": true  // 全てのstrictチェックを有効化
  }
}
```

### アンチパターン3: 過度に複雑な型定義

```typescript
// BAD: 読めない型定義
type DeepPartial<T> = T extends object
  ? { [P in keyof T]?: DeepPartial<T[P]> extends infer U
      ? U extends never ? never : U : never }
  : T;

// GOOD: コメントで意図を説明し、テストで動作を保証
/**
 * オブジェクトの全プロパティを再帰的にオプショナルにする。
 * 深いネスト構造の部分更新に使用。
 */
type DeepPartial<T> = T extends object
  ? { [P in keyof T]?: DeepPartial<T[P]> }
  : T;

// 型のテスト（型レベルのテスト）
type _TestDeepPartial = DeepPartial<{
  a: { b: { c: string } };
}>;
// 期待: { a?: { b?: { c?: string } } }
```

### アンチパターン4: @ts-ignore の乱用

```typescript
// BAD: エラーを握りつぶす
// @ts-ignore
const result = someFunction(invalidArg);

// BETTER: @ts-expect-error で理由を明記（エラーが解消されたら検知できる）
// @ts-expect-error: Legacy API requires string but we pass number. Fix in #1234
const result = someFunction(invalidArg);

// BEST: 型を正しく修正する
const result = someFunction(validArg as ExpectedType);
```

### アンチパターン5: 型アサーション（as）の過剰使用

```typescript
// BAD: 型アサーションで型チェックを迂回
const data = JSON.parse(response) as UserData;
// → response が不正な形式でもコンパイルエラーにならない

// GOOD: ランタイムバリデーションを組み合わせる
import { z } from "zod";

const UserDataSchema = z.object({
  id: z.number(),
  name: z.string(),
  email: z.string().email(),
});

type UserData = z.infer<typeof UserDataSchema>;

function parseUserData(response: string): UserData {
  const parsed = JSON.parse(response);
  return UserDataSchema.parse(parsed); // 実行時にも型チェック
}
```

### アンチパターン6: 型定義の重複

```typescript
// BAD: 同じ構造を複数箇所で定義
interface CreateUserRequest {
  name: string;
  email: string;
  age: number;
}

interface UpdateUserRequest {
  name: string;     // 重複！
  email: string;    // 重複！
  age: number;      // 重複！
}

// GOOD: ユーティリティ型で派生させる
interface User {
  id: string;
  name: string;
  email: string;
  age: number;
  createdAt: Date;
  updatedAt: Date;
}

type CreateUserRequest = Omit<User, "id" | "createdAt" | "updatedAt">;
type UpdateUserRequest = Partial<CreateUserRequest>;
type UserResponse = Pick<User, "id" | "name" | "email">;
```

### アンチパターン7: 不適切なenum使用

```typescript
// BAD: 数値enumは意図しない挙動を起こしやすい
enum Status {
  Active,   // 0
  Inactive, // 1
  Pending,  // 2
}
const status: Status = 999; // エラーにならない！（数値enumの落とし穴）

// GOOD: 文字列enumを使う
enum Status {
  Active = "ACTIVE",
  Inactive = "INACTIVE",
  Pending = "PENDING",
}

// BETTER: union typeを使う（多くの場合こちらが推奨）
type Status = "active" | "inactive" | "pending";

// union type の方が:
// - Tree-shakingが効く
// - JavaScript出力がシンプル
// - 型の推論が自然
// - as const と組み合わせやすい

const STATUS = {
  Active: "active",
  Inactive: "inactive",
  Pending: "pending",
} as const;

type Status = (typeof STATUS)[keyof typeof STATUS];
// "active" | "inactive" | "pending"
```

---

## 9. TypeScriptの設計哲学

### ゴール（公式Design Goals より）

TypeScriptの設計は以下の目標に基づいている。

```
1. 静的に型付けされたコードの構造的な不整合を検出する
2. 大規模プログラムの構造化メカニズムを提供する
3. 実行時に追加のオーバーヘッドを課さない
4. 出力されるJavaScriptは明快で慣用的なものにする
5. 一貫性があり完全に消去可能な型システムを使用する
6. 現在と将来のECMAScript提案に沿った言語とする
7. JavaScript のランタイム動作を変更しない（型の世界と値の世界を厳密に分離）
```

### 非ゴール（Non-Goals）

```
1. 音響的(sound)な型システムは追求しない（実用性を優先）
2. JavaScript プログラムの高速化は目的としない
3. プログラムの正しさの証明は目的としない
4. TypeScript固有のランタイム機能の提供（型消去の原則を守る）
```

```typescript
// 音響性（Soundness）を追求しない例
// TypeScriptは意図的に安全でない操作を許容している

// 例1: 配列のインデックスアクセス
const arr: string[] = ["a", "b", "c"];
const item: string = arr[10]; // undefined が返るが、型は string
// noUncheckedIndexedAccess: true で改善可能

// 例2: any 型の存在
// any は型システムの「脱出口」として意図的に用意されている

// 例3: 型アサーション
const value = "hello" as unknown as number; // 任意の型変換が可能

// これらは「実用的であること」を優先した設計判断
// 100%安全な型システムは使いにくくなりがちで、
// TypeScriptは実用性と安全性のバランスを取っている
```

---

## 10. よくある開発パターン

### 環境変数の型安全な読み込み

```typescript
// 環境変数の型定義
interface EnvConfig {
  NODE_ENV: "development" | "staging" | "production";
  PORT: number;
  DATABASE_URL: string;
  REDIS_URL: string;
  JWT_SECRET: string;
  LOG_LEVEL: "debug" | "info" | "warn" | "error";
}

// 環境変数の読み込みとバリデーション
function loadEnvConfig(): EnvConfig {
  const requiredVars = [
    "NODE_ENV",
    "PORT",
    "DATABASE_URL",
    "REDIS_URL",
    "JWT_SECRET",
  ] as const;

  const missing = requiredVars.filter((key) => !process.env[key]);
  if (missing.length > 0) {
    throw new Error(
      `Missing required environment variables: ${missing.join(", ")}`
    );
  }

  return {
    NODE_ENV: process.env.NODE_ENV as EnvConfig["NODE_ENV"],
    PORT: parseInt(process.env.PORT!, 10),
    DATABASE_URL: process.env.DATABASE_URL!,
    REDIS_URL: process.env.REDIS_URL!,
    JWT_SECRET: process.env.JWT_SECRET!,
    LOG_LEVEL: (process.env.LOG_LEVEL ?? "info") as EnvConfig["LOG_LEVEL"],
  };
}

// Zodを使ったよりロバストな方法
import { z } from "zod";

const envSchema = z.object({
  NODE_ENV: z.enum(["development", "staging", "production"]),
  PORT: z.coerce.number().int().positive(),
  DATABASE_URL: z.string().url(),
  REDIS_URL: z.string().url(),
  JWT_SECRET: z.string().min(32),
  LOG_LEVEL: z.enum(["debug", "info", "warn", "error"]).default("info"),
});

export const env = envSchema.parse(process.env);
// env は完全に型安全で、ランタイムバリデーション済み
```

### エラーハンドリングパターン

```typescript
// Result型パターン（例外を使わないエラーハンドリング）
type Result<T, E = Error> =
  | { success: true; data: T }
  | { success: false; error: E };

function ok<T>(data: T): Result<T, never> {
  return { success: true, data };
}

function err<E>(error: E): Result<never, E> {
  return { success: false, error };
}

// 使用例
interface ValidationError {
  field: string;
  message: string;
}

function validateEmail(email: string): Result<string, ValidationError> {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) {
    return err({ field: "email", message: "Invalid email format" });
  }
  return ok(email.toLowerCase().trim());
}

function validateAge(age: number): Result<number, ValidationError> {
  if (age < 0 || age > 150) {
    return err({ field: "age", message: "Age must be between 0 and 150" });
  }
  return ok(age);
}

// Result型のチェーン
function registerUser(email: string, age: number): Result<{ id: string }, ValidationError> {
  const emailResult = validateEmail(email);
  if (!emailResult.success) return emailResult;

  const ageResult = validateAge(age);
  if (!ageResult.success) return ageResult;

  return ok({ id: crypto.randomUUID() });
}
```

### 設定の型安全な管理

```typescript
// アプリケーション設定の型安全な管理
interface AppConfig {
  server: {
    host: string;
    port: number;
    cors: {
      origins: string[];
      methods: ("GET" | "POST" | "PUT" | "DELETE" | "PATCH")[];
      credentials: boolean;
    };
  };
  database: {
    host: string;
    port: number;
    name: string;
    pool: {
      min: number;
      max: number;
      idleTimeoutMs: number;
    };
  };
  auth: {
    jwtSecret: string;
    tokenExpiresIn: string;
    refreshTokenExpiresIn: string;
    bcryptRounds: number;
  };
  logging: {
    level: "debug" | "info" | "warn" | "error";
    format: "json" | "text";
    destination: "stdout" | "file";
  };
}

// デフォルト設定と環境固有設定のマージ
const defaultConfig: AppConfig = {
  server: {
    host: "0.0.0.0",
    port: 3000,
    cors: {
      origins: ["http://localhost:3000"],
      methods: ["GET", "POST", "PUT", "DELETE"],
      credentials: true,
    },
  },
  database: {
    host: "localhost",
    port: 5432,
    name: "myapp",
    pool: { min: 2, max: 10, idleTimeoutMs: 30000 },
  },
  auth: {
    jwtSecret: "change-me",
    tokenExpiresIn: "15m",
    refreshTokenExpiresIn: "7d",
    bcryptRounds: 12,
  },
  logging: {
    level: "info",
    format: "json",
    destination: "stdout",
  },
};

// DeepPartial でオーバーライドを型安全に
type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

function createConfig(overrides: DeepPartial<AppConfig>): AppConfig {
  // deep merge 実装
  return deepMerge(defaultConfig, overrides) as AppConfig;
}

function deepMerge<T extends Record<string, unknown>>(
  base: T,
  overrides: DeepPartial<T>
): T {
  const result = { ...base };
  for (const key in overrides) {
    const override = overrides[key];
    if (
      override !== undefined &&
      typeof override === "object" &&
      !Array.isArray(override) &&
      override !== null
    ) {
      (result as any)[key] = deepMerge(
        (base as any)[key] ?? {},
        override as any
      );
    } else if (override !== undefined) {
      (result as any)[key] = override;
    }
  }
  return result;
}
```

---

## FAQ

### Q1: TypeScriptは実行時に型チェックを行いますか？

**A:** いいえ。TypeScriptの型情報はコンパイル時に全て消去（erasure）されます。実行時はただのJavaScriptです。実行時バリデーションが必要な場合は Zod や io-ts などのライブラリを併用します。

### Q2: TypeScriptを学ぶにはJavaScriptを先に覚えるべきですか？

**A:** はい、推奨します。TypeScriptはJavaScriptの上に構築されているため、JavaScriptの基礎（関数、オブジェクト、プロトタイプ、非同期処理）を理解していると学習がスムーズです。ただし、最初からTypeScriptで学ぶアプローチも増えています。

### Q3: TypeScriptのデメリットは何ですか？

**A:** 主なデメリットは以下の通りです:
- **学習コスト**: 型システムの概念を学ぶ必要がある
- **ビルドステップ**: コンパイルが必要（ただしesbuild等で高速化可能）
- **型定義の保守**: 複雑な型は保守コストが発生する
- **サードパーティ型**: 一部のライブラリは型定義が不完全
とはいえ、中〜大規模プロジェクトではこれらのコストを大きく上回るメリットがあります。

### Q4: TypeScriptとFlowの違いは何ですか？

**A:** FlowはMeta（旧Facebook）が開発した型チェッカーで、TypeScriptと同様にJavaScriptに静的型付けを追加する。主な違いは以下の通り。

| 比較項目 | TypeScript | Flow |
|----------|-----------|------|
| 開発元 | Microsoft | Meta |
| 言語 vs ツール | 言語（独自コンパイラ） | ツール（型チェッカーのみ） |
| エコシステム | 圧倒的に大きい | 縮小傾向 |
| IDE支援 | VSCode等で標準的 | 限定的 |
| 型定義の共有 | DefinitelyTyped | 独自のflow-typed |
| コミュニティ | 非常に活発 | Meta社内中心 |
| 採用状況(2025) | デファクトスタンダード | Reactのコードベース等 |

現在はTypeScriptが事実上の標準であり、新規プロジェクトではTypeScriptを選択するのが一般的である。

### Q5: .d.ts ファイルとは何ですか？

**A:** `.d.ts` ファイルは「型宣言ファイル（Declaration File）」であり、JavaScriptライブラリの型情報を提供する。

```typescript
// math-lib.d.ts -- JavaScriptライブラリ math-lib の型宣言
declare module "math-lib" {
  export function add(a: number, b: number): number;
  export function multiply(a: number, b: number): number;

  export interface MathConfig {
    precision: number;
    rounding: "ceil" | "floor" | "round";
  }

  export class Calculator {
    constructor(config?: MathConfig);
    evaluate(expression: string): number;
  }
}

// 使用例
import { add, Calculator } from "math-lib";
const result = add(1, 2); // number
const calc = new Calculator({ precision: 2, rounding: "round" });
```

### Q6: DefinitelyTypedとは何ですか？

**A:** DefinitelyTypedは、TypeScriptの型定義を集めたコミュニティリポジトリ（GitHub上）である。`@types/xxx` パッケージとしてnpmで公開されている。

```bash
# DefinitelyTyped からの型定義インストール
npm install @types/express --save-dev
npm install @types/lodash --save-dev
npm install @types/react --save-dev
npm install @types/node --save-dev

# パッケージ自体に型定義が含まれている場合は @types 不要
# 例: axios, zod, prisma, date-fns 等
```

### Q7: strict: true にすべきですか？

**A:** 新規プロジェクトでは必ず `strict: true` にすべきです。既存プロジェクトの移行では段階的に有効化することを推奨します。strictモードが有効にする個別のフラグとその効果は以下の通り。

```typescript
// strictNullChecks: null/undefined の厳密チェック
let name: string;
// name = null; // エラー
let nullableName: string | null = null; // OK

// noImplicitAny: 暗黙の any を禁止
// function process(data) {} // エラー: 'data' パラメータには暗黙の 'any' 型があります
function process(data: unknown) {} // OK

// strictFunctionTypes: 関数パラメータの反変チェック
// strictBindCallApply: bind/call/apply の型チェック
// strictPropertyInitialization: クラスプロパティの初期化チェック

class User {
  // name: string; // エラー: 初期化されていない
  name: string = ""; // OK
  // or
  // name!: string; // 明示的にアサーション（非推奨だが使える）
}
```

### Q8: TypeScript の Node.js ネイティブサポートとは？

**A:** Node.js 22.6.0 以降で `--experimental-strip-types` フラグが追加され、TypeScriptファイルを直接実行できるようになった。これはTypeScriptの型アノテーションを単純に除去（strip）してJavaScriptとして実行する仕組みである。

```bash
# Node.js 22.6.0+
node --experimental-strip-types src/index.ts

# Node.js 23.6.0+ ではフラグなしで実行可能（デフォルト有効）
node src/index.ts

# 注意: enum, namespace, パラメータプロパティなどの
# TypeScript固有の構文はstrip-typesでは対応しない
# --experimental-transform-types が必要
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| TypeScriptとは | JavaScriptのスーパーセットで、静的型付けを追加する言語 |
| 開発元 | Microsoft（2012年公開、オープンソース） |
| コンパイル | .ts → .js に変換。型情報は実行時に消去される |
| 主な利点 | バグ早期発見、IDE支援、リファクタリング安全性 |
| 主なコスト | 学習曲線、ビルドステップ、型定義保守 |
| エコシステム | tsc, esbuild, Vitest, Zod, Prisma, tRPC など充実 |
| strict モード | 推奨。型チェックの恩恵を最大化する |
| 構造的型付け | 同じ構造を持つ型は互換性がある（Duck Typing） |
| 型消去 | コンパイル時に型情報は完全に消去される |
| ランタイム | Node.js, Deno, Bun が直接実行をサポート |
| 設計哲学 | 実用性と安全性のバランス、JavaScript互換性の維持 |
| 導入戦略 | 新規は strict:true、既存は段階的移行が推奨 |

---

## 次に読むべきガイド

- [01-type-basics.md](./01-type-basics.md) -- 型の基礎（プリミティブ型、リテラル型、配列、タプル）
- [02-functions-and-objects.md](./02-functions-and-objects.md) -- 関数とオブジェクト型

---

## 参考文献

1. **TypeScript公式ドキュメント** -- https://www.typescriptlang.org/docs/
2. **TypeScript Deep Dive (日本語版)** -- https://typescript-jp.gitbook.io/deep-dive/
3. **Programming TypeScript (Boris Cherny著, O'Reilly)** -- 型システムの理論と実践を網羅した書籍
4. **TypeScript GitHub リポジトリ** -- https://github.com/microsoft/TypeScript
5. **TypeScript Design Goals** -- https://github.com/microsoft/TypeScript/wiki/TypeScript-Design-Goals
6. **DefinitelyTyped** -- https://github.com/DefinitelyTyped/DefinitelyTyped
7. **TypeScript Playground** -- https://www.typescriptlang.org/play
8. **Effective TypeScript (Dan Vanderkam著, O'Reilly)** -- 実務で使える62のベストプラクティス
