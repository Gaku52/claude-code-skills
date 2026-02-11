# TypeScript 5.x 新機能完全ガイド

> TypeScript 5.0〜5.7 の主要な新機能を網羅し、モダン TypeScript の最新パターンを習得する

## この章で学ぶこと

1. **5.0〜5.2 の主要機能** -- デコレータ、const 型パラメータ、satisfies の活用パターン
2. **5.3〜5.5 の改善** -- Import Attributes、正規表現型チェック、型述語の推論改善
3. **5.6〜5.7 の最新機能** -- Iterator ヘルパー、--noUncheckedSideEffectImports 等の最新動向

---

## 1. TypeScript 5.0

### 1-1. ECMAScript デコレータ（Stage 3）

```
デコレータの適用順序:

  @log           3番目に適用
  @validate      2番目に適用
  @injectable    1番目に適用（最も内側）
  class UserService {
    @measure     メソッドデコレータ
    getUser() {}
  }
```

```typescript
// ECMAScript 標準デコレータ（Stage 3）
// TypeScript 5.0+ の experimentalDecorators: false（デフォルト）

// クラスデコレータ
function sealed(target: Function, context: ClassDecoratorContext) {
  Object.seal(target);
  Object.seal(target.prototype);
}

// メソッドデコレータ
function log<T extends (...args: any[]) => any>(
  target: T,
  context: ClassMethodDecoratorContext
): T {
  const methodName = String(context.name);

  return function (this: any, ...args: any[]) {
    console.log(`Calling ${methodName} with`, args);
    const result = target.apply(this, args);
    console.log(`${methodName} returned`, result);
    return result;
  } as T;
}

// フィールドデコレータ
function bound<T extends (...args: any[]) => any>(
  _target: undefined,
  context: ClassFieldDecoratorContext
) {
  return function (this: any, value: T): T {
    return value.bind(this) as T;
  };
}

// 使用例
@sealed
class UserService {
  @log
  getUser(id: string): User {
    return { id, name: "Alice" } as User;
  }

  @bound
  handleClick = () => {
    // this が常にインスタンスにバインドされる
    console.log(this);
  };
}
```

### 1-2. const 型パラメータ

```typescript
// const 型パラメータで配列リテラル型を保持

// const なし: string[] に拡大される
function routes<T extends readonly string[]>(paths: T): T {
  return paths;
}
const r1 = routes(["home", "about", "contact"]);
// 型: string[]（リテラル型が失われる）

// const あり: リテラル型タプルが保持される
function routes<const T extends readonly string[]>(paths: T): T {
  return paths;
}
const r2 = routes(["home", "about", "contact"]);
// 型: readonly ["home", "about", "contact"]

// 実践的な使用例: 型安全なルーター
function createRouter<const T extends Record<string, () => unknown>>(
  routes: T
): { navigate: (path: keyof T) => ReturnType<T[keyof T]> } {
  return {
    navigate: (path) => routes[path]() as ReturnType<T[keyof T]>,
  };
}

const router = createRouter({
  "/home": () => ({ page: "home" }),
  "/about": () => ({ page: "about", version: 2 }),
});

router.navigate("/home");   // OK
router.navigate("/unknown"); // エラー!
```

---

## 2. TypeScript 5.1〜5.2

### 2-1. 関連のない型の getter/setter（5.1）

```typescript
class Resource {
  private _value: string | undefined;

  // getter と setter で異なる型を使用可能に
  get value(): string {
    if (this._value === undefined) {
      throw new Error("Not initialized");
    }
    return this._value;
  }

  set value(val: string | undefined) {
    this._value = val;
  }
}

const r = new Resource();
r.value = "hello";       // setter: string | undefined
const v: string = r.value; // getter: string
r.value = undefined;      // OK: setter は undefined を受け付ける
```

### 2-2. using 宣言（5.2）

```
using によるリソース管理:

  {
    using file = openFile("data.txt");
    // file を使用
    // ...
  } ← スコープを抜けると自動的に file[Symbol.dispose]() が呼ばれる

  {
    await using db = connectToDatabase();
    // db を使用
    // ...
  } ← 自動的に await db[Symbol.asyncDispose]() が呼ばれる
```

```typescript
// Disposable インターフェース
class FileHandle implements Disposable {
  private handle: number;

  constructor(path: string) {
    this.handle = openFileSync(path);
    console.log(`Opened: ${path}`);
  }

  read(): string {
    return readFromHandle(this.handle);
  }

  [Symbol.dispose](): void {
    closeFileSync(this.handle);
    console.log("File closed");
  }
}

// using で自動リソース解放
function processFile(path: string): string {
  using file = new FileHandle(path);
  return file.read();
  // スコープ終了時に自動で [Symbol.dispose]() が呼ばれる
}

// AsyncDisposable
class DatabaseConnection implements AsyncDisposable {
  static async create(url: string): Promise<DatabaseConnection> {
    const conn = new DatabaseConnection();
    await conn.connect(url);
    return conn;
  }

  async [Symbol.asyncDispose](): Promise<void> {
    await this.disconnect();
    console.log("DB disconnected");
  }

  private async connect(url: string): Promise<void> { /* ... */ }
  private async disconnect(): Promise<void> { /* ... */ }
}

// await using
async function queryUsers(): Promise<User[]> {
  await using db = await DatabaseConnection.create(process.env.DB_URL!);
  return db.query("SELECT * FROM users");
  // 自動的に await db[Symbol.asyncDispose]() が呼ばれる
}
```

### 2-3. 名前付きタプル要素のラベル

```typescript
type Coordinate = [x: number, y: number, z?: number];

function distance(from: Coordinate, to: Coordinate): number {
  const [x1, y1] = from;
  const [x2, y2] = to;
  return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
}
```

---

## 3. TypeScript 5.3〜5.4

### 3-1. Import Attributes（5.3）

```typescript
// Import Attributes（旧称: Import Assertions）
import config from "./config.json" with { type: "json" };
// config の型が正しく推論される

import styles from "./app.css" with { type: "css" };

// 動的インポート
const data = await import("./data.json", {
  with: { type: "json" },
});
```

### 3-2. switch (true) の型絞り込み（5.3）

```typescript
function classify(value: string | number | boolean): string {
  switch (true) {
    case typeof value === "string":
      // value は string に絞り込まれる
      return value.toUpperCase();
    case typeof value === "number":
      // value は number に絞り込まれる
      return value.toFixed(2);
    case typeof value === "boolean":
      // value は boolean に絞り込まれる
      return value ? "yes" : "no";
    default:
      return assertNever(value);
  }
}
```

### 3-3. NoInfer ユーティリティ型（5.4）

```typescript
// NoInfer: 型推論の候補から除外

// NoInfer なし: defaultValue からも T が推論される
function getOrDefault<T>(value: T | null, defaultValue: T): T {
  return value ?? defaultValue;
}
getOrDefault("hello", 42); // T は string | number に推論（望ましくない）

// NoInfer あり: defaultValue からは T を推論しない
function getOrDefault<T>(value: T | null, defaultValue: NoInfer<T>): T {
  return value ?? defaultValue;
}
getOrDefault("hello", 42);
// エラー: number は string に代入不可
// T は "hello" の型 string からのみ推論される
```

### 3-4. クロージャでの型絞り込み保持（5.4）

```typescript
// 5.4 以前: クロージャ内で絞り込みが失われる
function process(value: string | number) {
  if (typeof value === "string") {
    // 5.4 以前: value は string | number に戻る
    // 5.4 以降: value は string のまま保持
    const fn = () => value.toUpperCase(); // OK in 5.4+
  }
}
```

---

## 4. TypeScript 5.5〜5.7

### 4-1. 型述語の推論（5.5）

```
型述語の推論:

  5.5 以前:
  const isString = (x: unknown) => typeof x === "string";
  // 型: (x: unknown) => boolean  ← 型述語にならない!

  5.5 以降:
  const isString = (x: unknown) => typeof x === "string";
  // 型: (x: unknown) => x is string  ← 自動推論!
```

```typescript
// 5.5: 型述語が自動推論される
const isNonNull = <T>(value: T | null | undefined) =>
  value != null;
// 自動推論: (value: T | null | undefined) => value is T

const values = [1, null, 2, undefined, 3];
const filtered = values.filter(isNonNull);
// 型: number[]（5.5 以降。以前は (number | null | undefined)[]）

// 配列メソッドとの組み合わせ
const users: (User | null)[] = await fetchUsers();
const validUsers = users.filter((u) => u !== null);
// 5.5: 型は User[]（自動的に型述語が推論される）
```

### 4-2. 正規表現の型チェック（5.5）

```typescript
// 5.5: 正規表現リテラルの基本的な構文チェック
const regex1 = /hello/;        // OK
const regex2 = /(\d+)/g;      // OK
const regex3 = /[unclosed/;   // エラー: Invalid regular expression
const regex4 = /(?<name>\w+)/; // OK: 名前付きキャプチャグループ
```

### 4-3. Iterator ヘルパーメソッド（5.6）

```typescript
// 5.6: Iterator / IteratorObject 型の改善

function* fibonacci(): Generator<number> {
  let [a, b] = [0, 1];
  while (true) {
    yield a;
    [a, b] = [b, a + b];
  }
}

// Iterator helper methods（TC39 Stage 3）
// TypeScript 5.6 で型定義が追加
const fib = fibonacci();

// .take() -- 最初のN個を取得
const first10 = fib.take(10);

// .map() -- 各要素を変換
const doubled = fib.take(10).map((n) => n * 2);

// .filter() -- フィルタリング
const evens = fib.take(20).filter((n) => n % 2 === 0);

// .toArray() -- 配列に変換
const array = fib.take(10).toArray();
// 型: number[]
```

### 4-4. --noUncheckedSideEffectImports（5.6）

```typescript
// 副作用のみのインポートのチェック
// tsconfig: "noUncheckedSideEffectImports": true

import "./nonexistent-module"; // エラー! モジュールが存在しない
import "reflect-metadata";     // OK: node_modules に存在
import "./setup.js";           // OK: ファイルが存在

// 副作用インポートの存在確認が厳密になる
```

### 4-5. satisfies と型推論の改善（5.0〜5.7 の横断的機能）

```typescript
// satisfies: 型の検証 + リテラル型の保持

// as const vs satisfies の違い
const config1 = {
  port: 3000,
  host: "localhost",
} as const;
// 型: { readonly port: 3000; readonly host: "localhost" }
// → Record<string, unknown> の形に準拠しているかチェックされない

type Config = {
  port: number;
  host: string;
  debug?: boolean;
};

const config2 = {
  port: 3000,
  host: "localhost",
  unknown: true, // エラー! Config にないプロパティ
} satisfies Config;

const config3 = {
  port: 3000,
  host: "localhost",
} satisfies Config;
// 型: { port: number; host: string }
// port は number（3000 ではない）

const config4 = {
  port: 3000,
  host: "localhost",
} as const satisfies Config;
// 型: { readonly port: 3000; readonly host: "localhost" }
// リテラル型 + Config 準拠チェック
```

---

## バージョン別新機能サマリー

```
TypeScript 5.x 新機能タイムライン:

  5.0  ├── ECMAScript デコレータ
       ├── const 型パラメータ
       └── enum / union の改善

  5.1  ├── getter/setter の型不一致許可
       ├── undefined 返り値の暗黙許可
       └── JSX 改善

  5.2  ├── using 宣言 (Explicit Resource Management)
       ├── デコレータメタデータ
       └── タプルの名前付きラベル

  5.3  ├── Import Attributes
       ├── switch(true) 型絞り込み
       └── インライン型ナローイング改善

  5.4  ├── NoInfer ユーティリティ型
       ├── クロージャ内の型絞り込み保持
       └── Object.groupBy / Map.groupBy 型

  5.5  ├── 型述語の推論
       ├── 正規表現の型チェック
       └── 新しい Set メソッドの型

  5.6  ├── Iterator ヘルパーメソッド
       ├── --noUncheckedSideEffectImports
       └── Disallow Nullish / Truthy checks

  5.7  ├── --noUncheckedIndexedAccess 改善
       ├── パフォーマンス改善
       └── Node.js 22 サポート強化
```

---

## 比較表

### TypeScript バージョン比較

| バージョン | リリース | 主要機能 | 破壊的変更 |
|-----------|---------|---------|-----------|
| 5.0 | 2023/03 | デコレータ, const型パラメータ | decorators構文変更 |
| 5.1 | 2023/06 | getter/setter型分離 | 小 |
| 5.2 | 2023/08 | using宣言, デコレータメタデータ | 小 |
| 5.3 | 2023/11 | Import Attributes | Assertions→Attributes |
| 5.4 | 2024/03 | NoInfer, クロージャ絞り込み | 小 |
| 5.5 | 2024/06 | 型述語推論, 正規表現チェック | filter推論変更 |
| 5.6 | 2024/09 | Iterator, SideEffectImport | 小 |
| 5.7 | 2024/12 | パフォーマンス改善 | 小 |

### satisfies vs as vs as const

| 機能 | satisfies | as | as const |
|------|-----------|-----|----------|
| 型チェック | あり | なし（上書き） | なし |
| 型の狭さ | 推論された型 | 指定した型 | リテラル型 |
| 余分なプロパティ | エラー | 無視 | 保持 |
| readonly | なし | なし | 自動付与 |
| 用途 | 検証+推論保持 | 型アサーション | リテラル保持 |

---

## アンチパターン

### AP-1: 古い experimentalDecorators を使い続ける

```typescript
// NG: 旧式デコレータ（experimentalDecorators: true）
// TypeScript 5.0+ では ECMAScript 標準デコレータが利用可能

// tsconfig.json
{
  "compilerOptions": {
    "experimentalDecorators": true  // 旧式
  }
}

// OK: ECMAScript 標準デコレータ（5.0+）
// experimentalDecorators を削除するか false にする
// 注意: Angular, NestJS 等は旧式デコレータに依存しているため
//       フレームワークの対応状況を確認すること
```

### AP-2: satisfies を使わずに型注釈で妥協する

```typescript
// NG: 型注釈で型を広げてしまう
const routes: Record<string, { path: string; component: string }> = {
  home: { path: "/", component: "Home" },
  about: { path: "/about", component: "About" },
};
routes.home.path; // 型: string（"/" ではない）

// OK: satisfies でリテラル型を保持
const routes = {
  home: { path: "/", component: "Home" },
  about: { path: "/about", component: "About" },
} satisfies Record<string, { path: string; component: string }>;
routes.home.path; // 型: "/"（リテラル型が保持される）
```

---

## FAQ

### Q1: TypeScript のバージョンアップはどのくらいの頻度で行うべきですか？

マイナーバージョン（5.x）ごとに追従することを推奨します。TypeScript は 3 ヶ月サイクルでリリースされ、各バージョンの破壊的変更は比較的小さいです。CI で `typescript@next` をテストする nightly テストジョブを追加すると、問題を早期発見できます。

### Q2: satisfies はどのような場面で使うべきですか？

(1) オブジェクトリテラルが特定の型に適合することを検証しつつ、リテラル型を保持したい場合。(2) 設定オブジェクトやルーティングテーブルなど、キーと値の組み合わせを型チェックしたい場合。(3) `as const` と組み合わせて、readonly + 型チェックを両立したい場合。

### Q3: using 宣言はいつから実際に使えますか？

TypeScript 5.2+ で構文サポートされています。ランタイムでは `Symbol.dispose` / `Symbol.asyncDispose` のポリフィルが必要な場合があります。Node.js 22+ とモダンブラウザではネイティブサポートが進んでいます。

---

## まとめ表

| 概念 | 要点 |
|------|------|
| ECMAScript デコレータ | Stage 3 標準、5.0+ でサポート |
| const 型パラメータ | ジェネリクスでリテラル型を保持 |
| satisfies | 型チェック + リテラル型保持の両立 |
| using | RAII パターンのリソース管理 |
| NoInfer | 型推論の候補から特定の位置を除外 |
| 型述語推論 | filter 等で自動的に型が絞り込まれる |

---

## 次に読むべきガイド

- [tsconfig.json](../03-tooling/00-tsconfig.md) -- 新バージョンの設定オプション
- [判別共用体](../02-patterns/02-discriminated-unions.md) -- 5.x で改善された型絞り込みの活用
- [ビルドツール](../03-tooling/01-build-tools.md) -- 新バージョンへのビルドツール対応

---

## 参考文献

1. **TypeScript Release Notes**
   https://www.typescriptlang.org/docs/handbook/release-notes/overview.html

2. **TypeScript Blog**
   https://devblogs.microsoft.com/typescript/

3. **TC39 Proposals** -- ECMAScript の提案一覧
   https://github.com/tc39/proposals
