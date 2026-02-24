# TypeScript 5.x 新機能完全ガイド

> TypeScript 5.0〜5.7 の主要な新機能を網羅し、モダン TypeScript の最新パターンを習得する

## この章で学ぶこと

1. **TypeScript 5.0** -- ECMAScript 標準デコレータ、const 型パラメータ、enum 改善
2. **TypeScript 5.1** -- getter/setter の型非対称、暗黙 undefined 返り値
3. **TypeScript 5.2** -- using 宣言（明示的リソース管理）、デコレータメタデータ
4. **TypeScript 5.3** -- Import Attributes、switch(true) ナローイング
5. **TypeScript 5.4** -- NoInfer ユーティリティ型、クロージャでの型絞り込み保持
6. **TypeScript 5.5** -- 型述語の推論、正規表現チェック、isolatedDeclarations
7. **TypeScript 5.6** -- Iterator ヘルパー型、Disallowed Nullish/Truthy Checks
8. **TypeScript 5.7** -- パフォーマンス改善、Node.js 22 サポート強化
9. **マイグレーションガイド** -- 各バージョンへの段階的移行戦略
10. **ベストプラクティス** -- 実務で活用できる実践的パターン

---

## 1. TypeScript 5.0: ECMAScript 標準デコレータとリテラル型保持

TypeScript 5.0（2023年3月）は ECMAScript Stage 3 デコレータのサポート、const 型パラメータ、enum の改善など、型システムの大幅な強化を実現しました。

### 1-1. ECMAScript デコレータ（Stage 3）

ECMAScript 標準デコレータは、従来の実験的デコレータ（`experimentalDecorators: true`）から移行する正式な仕様です。

```
デコレータの適用順序:

  @log           3番目に適用（外側から）
  @validate      2番目に適用
  @injectable    1番目に適用（最も内側、最初に実行）
  class UserService {
    @measure     メソッドデコレータ（メソッド定義時に適用）
    getUser() {}
  }

実行順序:
  1. @injectable (クラス定義を受け取る)
  2. @validate (前のデコレータの結果を受け取る)
  3. @log (最後に適用、最終的なクラスを返す)
```

#### クラスデコレータの実装

```typescript
// ECMAScript 標準デコレータ（Stage 3）
// TypeScript 5.0+ の experimentalDecorators: false（デフォルト）

// クラスデコレータ: クラス定義を受け取り、新しいクラスまたは void を返す
function sealed<T extends { new (...args: any[]): {} }>(
  target: T,
  context: ClassDecoratorContext
): T | void {
  console.log(`Sealing class: ${context.name}`);
  Object.seal(target);
  Object.seal(target.prototype);
  return target;
}

// ロギング機能を追加するデコレータ
function logged<T extends { new (...args: any[]): {} }>(
  target: T,
  context: ClassDecoratorContext
) {
  const className = String(context.name);

  return class extends target {
    constructor(...args: any[]) {
      console.log(`[${className}] Creating instance with args:`, args);
      super(...args);
      console.log(`[${className}] Instance created`);
    }
  };
}

// メソッドデコレータ: メソッドをラップして拡張
function log<T extends (...args: any[]) => any>(
  target: T,
  context: ClassMethodDecoratorContext
): T {
  const methodName = String(context.name);

  return function (this: any, ...args: any[]) {
    console.log(`Calling ${methodName} with`, args);
    const start = performance.now();
    const result = target.apply(this, args);
    const duration = performance.now() - start;
    console.log(`${methodName} returned`, result, `in ${duration.toFixed(2)}ms`);
    return result;
  } as T;
}

// 非同期メソッドデコレータ
function asyncLog<T extends (...args: any[]) => Promise<any>>(
  target: T,
  context: ClassMethodDecoratorContext
): T {
  const methodName = String(context.name);

  return async function (this: any, ...args: any[]) {
    console.log(`[Async] Calling ${methodName} with`, args);
    try {
      const result = await target.apply(this, args);
      console.log(`[Async] ${methodName} resolved with`, result);
      return result;
    } catch (error) {
      console.error(`[Async] ${methodName} rejected with`, error);
      throw error;
    }
  } as T;
}

// フィールドデコレータ: this を自動バインド
function bound<T extends (...args: any[]) => any>(
  _target: undefined,
  context: ClassFieldDecoratorContext
) {
  return function (this: any, value: T): T {
    return value.bind(this) as T;
  };
}

// アクセサデコレータ
function validate(
  target: any,
  context: ClassAccessorDecoratorContext | ClassSetterDecoratorContext
) {
  if (context.kind === "setter") {
    return function (this: any, value: any) {
      if (typeof value !== "string" || value.length === 0) {
        throw new Error(`Invalid value for ${String(context.name)}`);
      }
      target.call(this, value);
    };
  }
}

// 使用例
interface User {
  id: string;
  name: string;
}

@sealed
@logged
class UserService {
  private users: Map<string, User> = new Map();

  @log
  getUser(id: string): User | undefined {
    return this.users.get(id);
  }

  @asyncLog
  async fetchUser(id: string): Promise<User> {
    // 模擬的な非同期処理
    await new Promise((resolve) => setTimeout(resolve, 100));
    const user = this.users.get(id);
    if (!user) {
      throw new Error(`User ${id} not found`);
    }
    return user;
  }

  @bound
  handleClick = () => {
    // this が常にインスタンスにバインドされる
    console.log("UserService instance:", this);
  };
}

// 実行例
const service = new UserService();
// [UserService] Creating instance with args: []
// Sealing class: UserService
// [UserService] Instance created

service.getUser("user-1");
// Calling getUser with ['user-1']
// getUser returned undefined in 0.02ms

const button = { onClick: service.handleClick };
button.onClick(); // this は UserService のインスタンスを指す
```

#### デコレータメタデータの活用

```typescript
// メタデータを使った依存性注入パターン
const METADATA_KEY = Symbol("metadata");

type Metadata = {
  injectable?: boolean;
  dependencies?: Function[];
  singleton?: boolean;
};

function Injectable(options: { singleton?: boolean } = {}) {
  return function <T extends { new (...args: any[]): {} }>(
    target: T,
    context: ClassDecoratorContext
  ) {
    // メタデータの追加（context.metadata を使用）
    context.metadata[METADATA_KEY] = {
      injectable: true,
      singleton: options.singleton ?? false,
    } as Metadata;

    return target;
  };
}

function Inject(dependency: Function) {
  return function (
    _target: undefined,
    context: ClassFieldDecoratorContext
  ) {
    const metadata = context.metadata[METADATA_KEY] as Metadata;
    if (!metadata.dependencies) {
      metadata.dependencies = [];
    }
    metadata.dependencies.push(dependency);
  };
}

@Injectable({ singleton: true })
class DatabaseService {
  connect() {
    console.log("Database connected");
  }
}

@Injectable()
class UserRepository {
  @Inject(DatabaseService)
  db!: DatabaseService;

  findAll() {
    this.db.connect();
    return ["user1", "user2"];
  }
}
```

### 1-2. const 型パラメータ

const 型パラメータは、ジェネリック関数でリテラル型を保持するための機能です。

```typescript
// const 型パラメータの基本

// 通常のジェネリック: リテラル型が失われる
function identity<T>(value: T): T {
  return value;
}
const result1 = identity(["a", "b", "c"]);
// 型: string[]（リテラル型が失われる）

// const 型パラメータ: リテラル型が保持される
function identityConst<const T>(value: T): T {
  return value;
}
const result2 = identityConst(["a", "b", "c"]);
// 型: readonly ["a", "b", "c"]（リテラル型が保持される）
```

#### 実践例: 型安全なルーター

```typescript
// const 型パラメータを使った型安全なルーター実装

type RouteConfig<T> = {
  readonly path: string;
  readonly handler: () => T;
};

type Router<T extends Record<string, RouteConfig<any>>> = {
  readonly routes: T;
  navigate<K extends keyof T>(path: K): ReturnType<T[K]["handler"]>;
  getPaths(): ReadonlyArray<keyof T>;
};

function createRouter<const T extends Record<string, RouteConfig<any>>>(
  routes: T
): Router<T> {
  return {
    routes,
    navigate(path) {
      const route = routes[path];
      if (!route) {
        throw new Error(`Route ${String(path)} not found`);
      }
      return route.handler();
    },
    getPaths() {
      return Object.keys(routes) as Array<keyof T>;
    },
  };
}

// 使用例
const router = createRouter({
  "/home": {
    path: "/home",
    handler: () => ({ page: "home", title: "Home Page" }),
  },
  "/about": {
    path: "/about",
    handler: () => ({ page: "about", version: 2 }),
  },
  "/contact": {
    path: "/contact",
    handler: () => ({ page: "contact", email: "info@example.com" }),
  },
});

// 型安全なナビゲーション
const homeResult = router.navigate("/home");
// 型: { page: string; title: string }

const aboutResult = router.navigate("/about");
// 型: { page: string; version: number }

// @ts-expect-error: 存在しないパス
router.navigate("/unknown");

// パスの一覧を取得
const paths = router.getPaths();
// 型: readonly ("/home" | "/about" | "/contact")[]
```

#### const 型パラメータと satisfies の組み合わせ

```typescript
// const 型パラメータと satisfies を組み合わせた高度なパターン

type Action = {
  type: string;
  payload?: unknown;
};

type ActionMap = Record<string, Action>;

function createActions<const T extends ActionMap>(actions: T): T {
  return actions;
}

// 使用例: Redux-like なアクション定義
const actions = createActions({
  INCREMENT: { type: "INCREMENT" },
  DECREMENT: { type: "DECREMENT" },
  SET_VALUE: { type: "SET_VALUE", payload: 0 },
} satisfies ActionMap);

// 型: {
//   readonly INCREMENT: { type: "INCREMENT" };
//   readonly DECREMENT: { type: "DECREMENT" };
//   readonly SET_VALUE: { type: "SET_VALUE"; payload: number };
// }

type ActionsType = typeof actions;
type ActionTypes = ActionsType[keyof ActionsType];
// ActionTypes =
//   | { type: "INCREMENT" }
//   | { type: "DECREMENT" }
//   | { type: "SET_VALUE"; payload: number }

function reducer(state: number, action: ActionTypes): number {
  switch (action.type) {
    case "INCREMENT":
      return state + 1;
    case "DECREMENT":
      return state - 1;
    case "SET_VALUE":
      // action.payload の型が number と推論される
      return action.payload;
    default:
      return state;
  }
}
```

### 1-3. enum の改善

TypeScript 5.0 では、enum と union 型の相互運用性が改善されました。

```typescript
// enum の改善: union 型との統一性向上

enum Color {
  Red = "red",
  Green = "green",
  Blue = "blue",
}

// 5.0 以前: この代入はエラーになることがあった
const colors: Color[] = ["red", "green", "blue"] as Color[];

// 5.0 以降: より柔軟に扱える
type ColorValue = `${Color}`;
const colorValue: ColorValue = "red"; // OK

// enum の値から union 型を生成
type ColorUnion = Color.Red | Color.Green | Color.Blue;

function processColor(color: ColorUnion): void {
  console.log(color);
}

processColor(Color.Red); // OK
processColor("red" as Color.Red); // OK
```

---

## 2. TypeScript 5.1: getter/setter の型非対称と返り値の改善

TypeScript 5.1（2023年6月）は、getter と setter で異なる型を使用できるようになり、より柔軟なクラス設計が可能になりました。

### 2-1. 関連のない型の getter/setter

```typescript
// getter と setter で異なる型を使用可能に

class Resource {
  private _value: string | undefined;

  // getter は non-null な string を返す
  get value(): string {
    if (this._value === undefined) {
      throw new Error("Value not initialized");
    }
    return this._value;
  }

  // setter は string | undefined を受け付ける
  set value(val: string | undefined) {
    this._value = val;
  }
}

const r = new Resource();
r.value = "hello";         // setter: string | undefined を受け付ける
const v: string = r.value; // getter: string を返す（undefined でない）
r.value = undefined;       // OK: setter は undefined を受け付ける

// r.value を読むと string が返る（undefined を投げる可能性がある）
```

#### 実践例: Lazy Initialization パターン

```typescript
// Lazy Initialization with getter/setter type asymmetry

class LazyValue<T> {
  private _value: T | undefined;
  private _initializer: () => T;

  constructor(initializer: () => T) {
    this._initializer = initializer;
  }

  // getter: 必ず T を返す（初期化されていなければ初期化）
  get value(): T {
    if (this._value === undefined) {
      console.log("Initializing lazy value...");
      this._value = this._initializer();
    }
    return this._value;
  }

  // setter: T | undefined を受け付ける（リセット可能）
  set value(val: T | undefined) {
    this._value = val;
  }

  reset(): void {
    this._value = undefined;
  }
}

// 使用例
const expensiveComputation = new LazyValue(() => {
  console.log("Computing...");
  return Array.from({ length: 1000000 }, (_, i) => i).reduce((a, b) => a + b, 0);
});

console.log("Before access");
const result = expensiveComputation.value; // "Initializing lazy value..." → "Computing..."
console.log(result); // 499999500000
const cachedResult = expensiveComputation.value; // "Computing..." は表示されない（キャッシュされている）

expensiveComputation.reset();
const recomputed = expensiveComputation.value; // 再度 "Computing..." が表示される
```

#### 型変換を伴う getter/setter

```typescript
// 型変換を伴う getter/setter パターン

class TemperatureConverter {
  private _celsius: number = 0;

  // Celsius で保持、Fahrenheit で取得
  get fahrenheit(): number {
    return (this._celsius * 9) / 5 + 32;
  }

  set fahrenheit(f: number) {
    this._celsius = ((f - 32) * 5) / 9;
  }

  get celsius(): number {
    return this._celsius;
  }

  set celsius(c: number) {
    this._celsius = c;
  }
}

const temp = new TemperatureConverter();
temp.celsius = 0;
console.log(temp.fahrenheit); // 32

temp.fahrenheit = 100;
console.log(temp.celsius); // 37.77777777777778
```

### 2-2. 暗黙的な undefined 返り値

TypeScript 5.1 では、関数の返り値型が `T | undefined` の場合、明示的に `return undefined` を書かなくても良くなりました。

```typescript
// 暗黙的な undefined 返り値の改善

// 5.1 以前: return undefined が必須
function findUser(id: string): User | undefined {
  const user = database.get(id);
  if (!user) {
    return undefined; // 明示的に必要だった
  }
  return user;
}

// 5.1 以降: return を省略可能
function findUser2(id: string): User | undefined {
  const user = database.get(id);
  if (!user) {
    return; // undefined が暗黙的に返される
  }
  return user;
}

// より自然な記述が可能に
function getConfig(key: string): string | undefined {
  if (key === "port") return "3000";
  if (key === "host") return "localhost";
  // 暗黙的に undefined が返される
}
```

---

## 3. TypeScript 5.2: Explicit Resource Management（using 宣言）

TypeScript 5.2（2023年8月）は、ECMAScript の Explicit Resource Management 提案に基づく `using` 宣言をサポートしました。

### 3-1. using 宣言の基本

```
using によるリソース管理:

  {
    using file = openFile("data.txt");
    // file を使用
    const content = file.read();
    // ...
  } ← スコープを抜けると自動的に file[Symbol.dispose]() が呼ばれる

  {
    await using db = await connectToDatabase();
    // db を使用
    const users = await db.query("SELECT * FROM users");
    // ...
  } ← 自動的に await db[Symbol.asyncDispose]() が呼ばれる
```

#### Disposable の実装

```typescript
// Disposable インターフェースの実装

class FileHandle implements Disposable {
  private handle: number | null = null;
  private path: string;

  constructor(path: string) {
    this.path = path;
    this.handle = this.openFileSync(path);
    console.log(`Opened: ${path}`);
  }

  private openFileSync(path: string): number {
    // 模擬的なファイルハンドル
    return Math.floor(Math.random() * 1000);
  }

  read(): string {
    if (this.handle === null) {
      throw new Error("File is closed");
    }
    return `Content of ${this.path}`;
  }

  write(content: string): void {
    if (this.handle === null) {
      throw new Error("File is closed");
    }
    console.log(`Writing to ${this.path}:`, content);
  }

  [Symbol.dispose](): void {
    if (this.handle !== null) {
      console.log(`Closing file: ${this.path}`);
      this.handle = null;
    }
  }
}

// using で自動リソース解放
function processFile(path: string): string {
  using file = new FileHandle(path);
  const content = file.read();
  file.write("Updated content");
  return content;
  // スコープ終了時に自動で [Symbol.dispose]() が呼ばれる
}

// 実行例
const result = processFile("data.txt");
// Opened: data.txt
// Writing to data.txt: Updated content
// Closing file: data.txt
```

#### AsyncDisposable の実装

```typescript
// AsyncDisposable インターフェース

class DatabaseConnection implements AsyncDisposable {
  private connected: boolean = false;
  private url: string;

  private constructor(url: string) {
    this.url = url;
  }

  static async create(url: string): Promise<DatabaseConnection> {
    const conn = new DatabaseConnection(url);
    await conn.connect();
    return conn;
  }

  private async connect(): Promise<void> {
    console.log(`Connecting to ${this.url}...`);
    await new Promise((resolve) => setTimeout(resolve, 100));
    this.connected = true;
    console.log("Connected!");
  }

  async query<T>(sql: string): Promise<T[]> {
    if (!this.connected) {
      throw new Error("Not connected");
    }
    console.log(`Executing query: ${sql}`);
    await new Promise((resolve) => setTimeout(resolve, 50));
    return [] as T[];
  }

  async [Symbol.asyncDispose](): Promise<void> {
    if (this.connected) {
      console.log("Disconnecting from database...");
      await new Promise((resolve) => setTimeout(resolve, 50));
      this.connected = false;
      console.log("Disconnected!");
    }
  }
}

// await using で自動リソース解放
async function queryUsers(): Promise<User[]> {
  await using db = await DatabaseConnection.create("postgresql://localhost:5432/mydb");
  const users = await db.query<User>("SELECT * FROM users");
  return users;
  // 自動的に await db[Symbol.asyncDispose]() が呼ばれる
}

// 実行例
await queryUsers();
// Connecting to postgresql://localhost:5432/mydb...
// Connected!
// Executing query: SELECT * FROM users
// Disconnecting from database...
// Disconnected!
```

#### using のエラーハンドリング

```typescript
// using のエラーハンドリングと SuppressedError

class Transaction implements Disposable {
  private committed: boolean = false;

  commit(): void {
    console.log("Committing transaction...");
    this.committed = true;
  }

  [Symbol.dispose](): void {
    if (!this.committed) {
      console.log("Rolling back transaction...");
      // ロールバック処理
    }
  }
}

function processTransaction(shouldFail: boolean): void {
  using tx = new Transaction();

  if (shouldFail) {
    throw new Error("Transaction failed");
  }

  tx.commit();
}

try {
  processTransaction(true);
} catch (error) {
  console.error("Error:", error);
  // 先に tx[Symbol.dispose]() が呼ばれ、その後エラーが投げられる
}
// Rolling back transaction...
// Error: Error: Transaction failed

// dispose 中にもエラーが発生した場合
class ProblematicResource implements Disposable {
  [Symbol.dispose](): void {
    throw new Error("Dispose failed");
  }
}

function useProblematicResource(): void {
  using resource = new ProblematicResource();
  throw new Error("Operation failed");
}

try {
  useProblematicResource();
} catch (error) {
  console.error(error);
  // SuppressedError が投げられる（元のエラーと dispose のエラーを両方含む）
}
```

### 3-2. デコレータメタデータ

TypeScript 5.2 では、デコレータメタデータの型定義が改善されました。

```typescript
// デコレータメタデータの活用

type MetadataMap = {
  validation?: {
    required?: boolean;
    minLength?: number;
    maxLength?: number;
  };
  serialization?: {
    name?: string;
    ignore?: boolean;
  };
};

function Required() {
  return function (
    _target: undefined,
    context: ClassFieldDecoratorContext
  ) {
    const metadata = context.metadata as MetadataMap;
    if (!metadata.validation) {
      metadata.validation = {};
    }
    metadata.validation.required = true;
  };
}

function MinLength(length: number) {
  return function (
    _target: undefined,
    context: ClassFieldDecoratorContext
  ) {
    const metadata = context.metadata as MetadataMap;
    if (!metadata.validation) {
      metadata.validation = {};
    }
    metadata.validation.minLength = length;
  };
}

function SerializedName(name: string) {
  return function (
    _target: undefined,
    context: ClassFieldDecoratorContext
  ) {
    const metadata = context.metadata as MetadataMap;
    if (!metadata.serialization) {
      metadata.serialization = {};
    }
    metadata.serialization.name = name;
  };
}

class UserDto {
  @Required()
  @MinLength(3)
  @SerializedName("user_name")
  username!: string;

  @Required()
  email!: string;
}

// メタデータを使ったバリデーション関数
function validate<T extends object>(obj: T): boolean {
  const metadata = (obj.constructor as any)[Symbol.metadata] as MetadataMap;
  // メタデータを使った検証ロジック
  return true;
}
```

---

## 4. TypeScript 5.3: Import Attributes と型絞り込みの改善

TypeScript 5.3（2023年11月）は、Import Attributes（旧 Import Assertions）のサポートと、型絞り込みの大幅な改善をもたらしました。

### 4-1. Import Attributes

```typescript
// Import Attributes（旧称: Import Assertions）

// JSON のインポート
import config from "./config.json" with { type: "json" };
// config の型が自動的に推論される

// CSS のインポート（CSS Modules）
import styles from "./app.css" with { type: "css" };

// 動的インポート
const data = await import("./data.json", {
  with: { type: "json" },
});

// カスタム属性
import wasmModule from "./module.wasm" with { type: "webassembly" };

// 型定義の例
// config.json
{
  "port": 3000,
  "host": "localhost",
  "debug": true
}

// TypeScript が自動推論する型:
// type Config = {
//   port: number;
//   host: string;
//   debug: boolean;
// }
```

#### Import Attributes の実践例

```typescript
// Import Attributes を使った設定管理

// config/development.json
import devConfig from "./config/development.json" with { type: "json" };

// config/production.json
import prodConfig from "./config/production.json" with { type: "json" };

type Config = {
  database: {
    host: string;
    port: number;
    name: string;
  };
  api: {
    baseUrl: string;
    timeout: number;
  };
  features: {
    enableAnalytics: boolean;
    enableDebug: boolean;
  };
};

function getConfig(): Config {
  const env = process.env.NODE_ENV || "development";

  switch (env) {
    case "production":
      return prodConfig;
    case "development":
    default:
      return devConfig;
  }
}

export const config = getConfig();
```

### 4-2. switch (true) の型絞り込み

```typescript
// switch (true) の型絞り込み（5.3 の改善）

function classify(value: string | number | boolean | null): string {
  switch (true) {
    case value === null:
      // value は null に絞り込まれる
      return "null value";

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
      const _exhaustive: never = value;
      return _exhaustive;
  }
}

// より複雑な条件での絞り込み
type Shape =
  | { kind: "circle"; radius: number }
  | { kind: "rectangle"; width: number; height: number }
  | { kind: "triangle"; base: number; height: number };

function getArea(shape: Shape): number {
  switch (true) {
    case shape.kind === "circle":
      // shape は { kind: "circle"; radius: number } に絞り込まれる
      return Math.PI * shape.radius ** 2;

    case shape.kind === "rectangle":
      // shape は { kind: "rectangle"; width: number; height: number } に絞り込まれる
      return shape.width * shape.height;

    case shape.kind === "triangle":
      // shape は { kind: "triangle"; base: number; height: number } に絞り込まれる
      return (shape.base * shape.height) / 2;

    default:
      const _exhaustive: never = shape;
      return _exhaustive;
  }
}
```

### 4-3. インライン型ナローイングの改善

```typescript
// インライン型ナローイングの改善

type Response<T> =
  | { success: true; data: T }
  | { success: false; error: string };

async function fetchUser(id: string): Promise<Response<User>> {
  try {
    const user = await api.getUser(id);
    return { success: true, data: user };
  } catch (error) {
    return { success: false, error: String(error) };
  }
}

// 5.3 の改善: インライン条件での絞り込み
async function processUser(id: string): Promise<void> {
  const response = await fetchUser(id);

  // 5.3 以降: response.success の確認で型が絞り込まれる
  if (response.success) {
    // response は { success: true; data: User } に絞り込まれる
    console.log(response.data.name);
  } else {
    // response は { success: false; error: string } に絞り込まれる
    console.error(response.error);
  }

  // 三項演算子内でも絞り込みが機能
  const message = response.success
    ? `User: ${response.data.name}`  // data が利用可能
    : `Error: ${response.error}`;    // error が利用可能
}
```

---

## 5. TypeScript 5.4: NoInfer とクロージャでの型絞り込み保持

TypeScript 5.4（2024年3月）は、NoInfer ユーティリティ型の追加と、クロージャ内での型絞り込み保持を実現しました。

### 5-1. NoInfer ユーティリティ型

NoInfer は、型推論の候補から特定の位置を除外するユーティリティ型です。

```typescript
// NoInfer: 型推論の候補から除外

// NoInfer なし: defaultValue からも T が推論される
function getOrDefault<T>(
  value: T | null | undefined,
  defaultValue: T
): T {
  return value ?? defaultValue;
}

const result1 = getOrDefault("hello", 42);
// T は string | number に推論される（望ましくない）

// NoInfer あり: defaultValue からは T を推論しない
function getOrDefaultFixed<T>(
  value: T | null | undefined,
  defaultValue: NoInfer<T>
): T {
  return value ?? defaultValue;
}

const result2 = getOrDefaultFixed("hello", 42);
// エラー: number は string に代入不可
// T は "hello" の型 string からのみ推論される

const result3 = getOrDefaultFixed("hello", "world");
// OK: T は string に推論され、defaultValue も string
```

#### NoInfer の実践例

```typescript
// NoInfer を使った型安全な API

// ダメな例: 両方の引数から型が推論される
function createPair<T>(first: T, second: T): [T, T] {
  return [first, second];
}

const pair1 = createPair(1, "hello");
// T は number | string に推論される（望ましくない）
// pair1 の型: [number | string, number | string]

// 良い例: 最初の引数からのみ型を推論
function createPairFixed<T>(first: T, second: NoInfer<T>): [T, T] {
  return [first, second];
}

const pair2 = createPairFixed(1, "hello");
// エラー: string は number に代入不可

const pair3 = createPairFixed(1, 2);
// OK: pair3 の型は [number, number]

// 実践的な使用例: イベントハンドラー
type EventMap = {
  click: { x: number; y: number };
  keypress: { key: string };
  submit: { data: FormData };
};

function addEventListener<K extends keyof EventMap>(
  event: K,
  handler: (payload: NoInfer<EventMap[K]>) => void
): void {
  // イベントリスナーの登録
}

// K が event から推論され、handler の型がそれに従う
addEventListener("click", (payload) => {
  console.log(payload.x, payload.y); // OK
});

addEventListener("keypress", (payload) => {
  console.log(payload.key); // OK
  // @ts-expect-error: x は keypress にない
  console.log(payload.x);
});
```

#### NoInfer とデフォルト値パターン

```typescript
// NoInfer を使ったデフォルト値パターン

type Options<T> = {
  value: T;
  fallback?: NoInfer<T>;
  transform?: (val: T) => NoInfer<T>;
};

function process<T>(options: Options<T>): T {
  const { value, fallback, transform } = options;
  const processed = transform ? transform(value) : value;
  return processed ?? fallback ?? value;
}

// 使用例
const result = process({
  value: "hello",
  fallback: "default", // OK: string
  transform: (s) => s.toUpperCase(), // OK: (string) => string
});

const invalid = process({
  value: "hello",
  fallback: 123, // エラー: number は string に代入不可
});
```

### 5-2. クロージャでの型絞り込み保持

TypeScript 5.4 では、クロージャ内で型絞り込みが保持されるようになりました。

```typescript
// クロージャでの型絞り込み保持

function processValue(value: string | number) {
  if (typeof value === "string") {
    // 5.4 以前: クロージャ内で value は string | number に戻る
    // 5.4 以降: value は string のまま保持される

    const fn = () => {
      return value.toUpperCase(); // OK in 5.4+
    };

    return fn();
  } else {
    const fn = () => {
      return value.toFixed(2); // OK: value は number
    };

    return fn();
  }
}

// より複雑な例
type User = { type: "user"; name: string; email: string };
type Admin = { type: "admin"; name: string; permissions: string[] };
type Person = User | Admin;

function processPerson(person: Person): void {
  if (person.type === "admin") {
    // person は Admin に絞り込まれる

    const logPermissions = () => {
      // 5.4+: person.permissions にアクセス可能
      console.log(person.permissions.join(", "));
    };

    logPermissions();

    // 非同期関数内でも保持される
    setTimeout(() => {
      console.log(person.permissions); // OK
    }, 1000);
  }
}
```

#### 配列メソッドとクロージャの組み合わせ

```typescript
// 配列メソッドとクロージャの組み合わせ

type Item = { id: number; name: string; category?: string };

function filterAndMap(items: Item[]): string[] {
  // filter で category が存在するものだけを抽出
  return items
    .filter((item) => item.category !== undefined)
    .map((item) => {
      // 5.4+: item.category は string に絞り込まれている
      return item.category.toUpperCase();
    });
}

// より複雑な例
function processItems(items: (string | number)[]): void {
  items.forEach((item) => {
    if (typeof item === "string") {
      // item は string に絞り込まれる

      const delayed = () => {
        // 5.4+: item は string のまま
        console.log(item.toUpperCase());
      };

      setTimeout(delayed, 100);
    }
  });
}
```

---

## 6. TypeScript 5.5: 型述語の推論と正規表現チェック

TypeScript 5.5（2024年6月）は、型述語の自動推論、正規表現の構文チェック、isolatedDeclarations など、大きな改善がありました。

### 6-1. 型述語の推論

TypeScript 5.5 では、型述語（type predicate）が自動的に推論されるようになりました。

```
型述語の推論:

  5.5 以前:
  const isString = (x: unknown): x is string => typeof x === "string";
  // 明示的に `x is string` を書く必要があった

  5.5 以降:
  const isString = (x: unknown) => typeof x === "string";
  // 自動的に `x is string` が推論される!
```

```typescript
// 型述語の自動推論

// 5.5 以前: 明示的な型述語が必要
function isNonNullOld<T>(value: T | null | undefined): value is T {
  return value != null;
}

// 5.5 以降: 自動推論される
const isNonNull = <T>(value: T | null | undefined) => value != null;
// 推論される型: <T>(value: T | null | undefined) => value is T

const values = [1, null, 2, undefined, 3];
const filtered = values.filter(isNonNull);
// 5.5+: filtered の型は number[]
// 5.4 以前: filtered の型は (number | null | undefined)[]

// より複雑な例
const isUser = (value: unknown) =>
  typeof value === "object" &&
  value !== null &&
  "id" in value &&
  "name" in value;
// 自動推論: (value: unknown) => value is { id: unknown; name: unknown }

interface User {
  id: number;
  name: string;
  email: string;
}

const users: unknown[] = await fetchData();
const validUsers = users.filter(isUser);
// validUsers の型: { id: unknown; name: unknown }[]
```

#### filter と型述語の組み合わせ

```typescript
// filter と型述語の自動推論

type Shape =
  | { kind: "circle"; radius: number }
  | { kind: "rectangle"; width: number; height: number }
  | { kind: "triangle"; base: number; height: number };

const shapes: Shape[] = [
  { kind: "circle", radius: 5 },
  { kind: "rectangle", width: 10, height: 20 },
  { kind: "triangle", base: 8, height: 12 },
];

// 5.5: 型述語が自動推論される
const circles = shapes.filter((s) => s.kind === "circle");
// circles の型: { kind: "circle"; radius: number }[]

const rectangles = shapes.filter((s) => s.kind === "rectangle");
// rectangles の型: { kind: "rectangle"; width: number; height: number }[]

// 複数条件の組み合わせ
const bigShapes = shapes.filter((s) => {
  if (s.kind === "circle") return s.radius > 10;
  if (s.kind === "rectangle") return s.width > 10 || s.height > 10;
  return s.base > 10;
});
// bigShapes の型: Shape[]（型は絞り込まれない）
```

#### カスタム型ガードの簡略化

```typescript
// カスタム型ガードの簡略化

// 5.5 以前: 明示的な型述語
function isErrorOld(value: unknown): value is Error {
  return value instanceof Error;
}

// 5.5 以降: 自動推論
const isError = (value: unknown) => value instanceof Error;
// 推論される型: (value: unknown) => value is Error

// 配列の型ガード
const isStringArray = (value: unknown) =>
  Array.isArray(value) && value.every((item) => typeof item === "string");
// 推論される型: (value: unknown) => value is string[]

const data: unknown = JSON.parse('["a", "b", "c"]');
if (isStringArray(data)) {
  // data は string[] に絞り込まれる
  data.forEach((s) => console.log(s.toUpperCase()));
}
```

### 6-2. 正規表現の型チェック

TypeScript 5.5 では、正規表現リテラルの構文チェックが強化されました。

```typescript
// 正規表現の型チェック

// OK: 正しい正規表現
const emailRegex = /^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$/;
const phoneRegex = /^\+?[1-9]\d{1,14}$/;
const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

// エラー: 閉じ括弧がない
// @ts-expect-error
const invalidRegex1 = /[unclosed/;

// エラー: 不正なエスケープシーケンス
// @ts-expect-error
const invalidRegex2 = /\k/;

// OK: 名前付きキャプチャグループ
const dateRegex = /(?<year>\d{4})-(?<month>\d{2})-(?<day>\d{2})/;

// OK: 後方参照
const duplicateRegex = /(\w+)\s+\1/;

// OK: Unicode プロパティエスケープ
const emojiRegex = /\p{Emoji}/u;
```

#### 正規表現の実践例

```typescript
// 正規表現を使ったバリデーション

type Validator<T extends string> = {
  pattern: RegExp;
  validate: (value: string) => value is T;
  format: (value: T) => string;
};

// Email バリデーター
const emailValidator: Validator<`${string}@${string}.${string}`> = {
  pattern: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
  validate: (value): value is `${string}@${string}.${string}` => {
    return emailValidator.pattern.test(value);
  },
  format: (value) => value.toLowerCase(),
};

// URL バリデーター
const urlValidator: Validator<`https://${string}` | `http://${string}`> = {
  pattern: /^https?:\/\/.+$/,
  validate: (value): value is `https://${string}` | `http://${string}` => {
    return urlValidator.pattern.test(value);
  },
  format: (value) => value,
};

// 使用例
const email = "user@example.com";
if (emailValidator.validate(email)) {
  // email は `${string}@${string}.${string}` 型
  const formatted = emailValidator.format(email);
}

const url = "https://example.com";
if (urlValidator.validate(url)) {
  // url は `https://${string}` | `http://${string}` 型
  const formatted = urlValidator.format(url);
}
```

### 6-3. isolatedDeclarations

TypeScript 5.5 では、`--isolatedDeclarations` フラグが追加され、型定義の分離が改善されました。

```typescript
// isolatedDeclarations の有効化
// tsconfig.json
{
  "compilerOptions": {
    "isolatedDeclarations": true,
    "declaration": true
  }
}

// isolatedDeclarations 有効時: 戻り値の型を明示する必要がある

// NG: 戻り値の型が明示されていない
export function getUser(id: string) {
  return { id, name: "Alice" };
}

// OK: 戻り値の型を明示
export function getUser(id: string): { id: string; name: string } {
  return { id, name: "Alice" };
}

// または型エイリアスを使用
type UserData = { id: string; name: string };

export function getUser(id: string): UserData {
  return { id, name: "Alice" };
}

// ジェネリック関数でも同様
// NG
export function map<T, U>(arr: T[], fn: (item: T) => U) {
  return arr.map(fn);
}

// OK
export function map<T, U>(arr: T[], fn: (item: T) => U): U[] {
  return arr.map(fn);
}
```

---

## 7. TypeScript 5.6: Iterator ヘルパーと厳格化

TypeScript 5.6（2024年9月）は、Iterator ヘルパーメソッドの型定義、Disallowed Nullish/Truthy Checks など、さらなる型安全性の向上をもたらしました。

### 7-1. Iterator ヘルパーメソッド

```typescript
// Iterator ヘルパーメソッド（TC39 Stage 3）

function* fibonacci(): Generator<number> {
  let [a, b] = [0, 1];
  while (true) {
    yield a;
    [a, b] = [b, a + b];
  }
}

const fib = fibonacci();

// .take() -- 最初の N 個を取得
const first10 = fib.take(10);
// 型: IteratorObject<number, void, undefined>

// .map() -- 各要素を変換
const doubled = fib.take(10).map((n) => n * 2);
// 型: IteratorObject<number, void, undefined>

// .filter() -- フィルタリング
const evens = fib.take(20).filter((n) => n % 2 === 0);
// 型: IteratorObject<number, void, undefined>

// .toArray() -- 配列に変換
const array = fib.take(10).toArray();
// 型: number[]

// チェーン可能
const result = fib
  .take(100)
  .filter((n) => n % 2 === 0)
  .map((n) => n * 2)
  .take(10)
  .toArray();
// result: number[] = [0, 4, 8, 16, 24, 40, 56, 88, 136, 216]
```

#### Iterator ヘルパーの実践例

```typescript
// Iterator ヘルパーを使ったデータ処理

function* range(start: number, end: number, step: number = 1): Generator<number> {
  for (let i = start; i < end; i += step) {
    yield i;
  }
}

// 1 から 100 までの奇数の二乗
const oddSquares = range(1, 100)
  .filter((n) => n % 2 === 1)
  .map((n) => n ** 2)
  .toArray();
// [1, 9, 25, 49, 81, ...]

// 無限ストリーム処理
function* naturals(): Generator<number> {
  let n = 1;
  while (true) {
    yield n++;
  }
}

const first20Primes = naturals()
  .filter((n) => {
    if (n < 2) return false;
    for (let i = 2; i <= Math.sqrt(n); i++) {
      if (n % i === 0) return false;
    }
    return true;
  })
  .take(20)
  .toArray();
// [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]

// ファイル行の遅延処理
async function* readLines(path: string): AsyncGenerator<string> {
  const file = await openFile(path);
  try {
    for await (const line of file) {
      yield line;
    }
  } finally {
    await file.close();
  }
}

// 非同期 Iterator ヘルパー
const longLines = await readLines("data.txt")
  .filter((line) => line.length > 100)
  .map((line) => line.trim())
  .take(10)
  .toArray();
```

### 7-2. Disallowed Nullish/Truthy Checks

TypeScript 5.6 では、常に真または常に偽になる条件式を検出します。

```typescript
// Disallowed Nullish/Truthy Checks

// エラー: この条件は常に true
const value = "hello";
if (value) {
  // 警告: value は常に truthy
}

// エラー: この条件は常に false
const num = 42;
if (!num) {
  // 警告: num は常に truthy
}

// エラー: nullish coalescing が不要
const str = "hello";
const result = str ?? "default";
// 警告: str は常に non-nullish

// OK: nullable な値
const nullable: string | null = getValue();
if (nullable) {
  // 問題なし
}

const withDefault = nullable ?? "default";
// 問題なし

// 関数パラメータの場合
function process(value: string): void {
  // エラー: value は常に truthy（undefined/null でない）
  if (value) {
    console.log(value);
  }

  // OK: 空文字列チェック
  if (value.length > 0) {
    console.log(value);
  }
}

// オプショナルパラメータは OK
function processOptional(value?: string): void {
  // OK: value は string | undefined
  if (value) {
    console.log(value);
  }
}
```

#### 実践的な修正例

```typescript
// Disallowed Nullish/Truthy Checks の修正例

// NG: 不要な nullish チェック
function getUserName(user: User): string {
  return user.name ?? "Anonymous";
  // 警告: user.name は string 型なので ?? は不要
}

// OK: 修正版
function getUserName(user: User): string {
  return user.name || "Anonymous"; // 空文字列もカバーする場合
  // または
  return user.name.length > 0 ? user.name : "Anonymous";
}

// NG: 常に真の条件
function isValid(config: Config): boolean {
  if (config) {
    // 警告: config は常に truthy
    return true;
  }
  return false;
}

// OK: 修正版（適切なプロパティチェック）
function isValid(config: Config): boolean {
  return config.apiKey.length > 0 && config.endpoint.length > 0;
}
```

### 7-3. --noUncheckedSideEffectImports

```typescript
// 副作用のみのインポートのチェック
// tsconfig.json
{
  "compilerOptions": {
    "noUncheckedSideEffectImports": true
  }
}

// エラー: モジュールが存在しない
import "./nonexistent-module";

// OK: node_modules に存在
import "reflect-metadata";

// OK: ファイルが存在
import "./setup.js";

// OK: package.json の sideEffects フィールドに記載
import "polyfills";

// 実践例
// setup.ts（副作用のみのモジュール）
console.log("Initializing application...");

// polyfills.ts
if (!Array.prototype.at) {
  Array.prototype.at = function (index: number) {
    if (index < 0) {
      return this[this.length + index];
    }
    return this[index];
  };
}

// main.ts
import "./setup"; // OK: setup.ts が存在
import "./polyfills"; // OK: polyfills.ts が存在
```

---

## 8. TypeScript 5.7: パフォーマンス改善と最適化

TypeScript 5.7（2024年12月）は、パフォーマンスの大幅な改善と、新しい最適化機能を導入しました。

### 8-1. パフォーマンス改善

```typescript
// TypeScript 5.7 のパフォーマンス改善

// 大規模な union 型の処理が高速化
type HttpMethod = "GET" | "POST" | "PUT" | "DELETE" | "PATCH" | "OPTIONS" | "HEAD";
type StatusCode = 200 | 201 | 204 | 400 | 401 | 403 | 404 | 500 | 502 | 503;
type ContentType = "application/json" | "text/html" | "text/plain" | "application/xml";

type Response = {
  method: HttpMethod;
  status: StatusCode;
  contentType: ContentType;
  body: unknown;
};

// 5.7 では、このような複雑な型の推論が高速化されている

// mapped types の最適化
type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
};

type ComplexObject = {
  users: {
    id: number;
    profile: {
      name: string;
      settings: {
        theme: string;
        notifications: boolean;
      };
    };
  }[];
};

type ReadonlyComplex = DeepReadonly<ComplexObject>;
// 5.7 では処理が高速化
```

### 8-2. --erasableSyntaxOnly（実験的機能）

```typescript
// --erasableSyntaxOnly フラグ（5.7 の実験的機能）
// tsconfig.json
{
  "compilerOptions": {
    "erasableSyntaxOnly": true
  }
}

// このフラグを有効にすると、トランスパイル時に削除可能な構文のみを使用できる

// OK: 型アノテーションは削除可能
const name: string = "Alice";

// OK: インターフェース定義は削除可能
interface User {
  id: number;
  name: string;
}

// NG: enum は JavaScript コードを生成するため不可
enum Color {
  Red,
  Green,
  Blue,
}

// OK の代替: const オブジェクトまたは union 型を使用
const Color = {
  Red: "red",
  Green: "green",
  Blue: "blue",
} as const;

type ColorValue = typeof Color[keyof typeof Color];

// NG: namespace は JavaScript コードを生成
namespace Utils {
  export function log(message: string) {
    console.log(message);
  }
}

// OK の代替: 通常のモジュールを使用
export function log(message: string) {
  console.log(message);
}
```

### 8-3. Node.js 22 サポート強化

```typescript
// Node.js 22 の新機能サポート

// AsyncLocalStorage の型改善
import { AsyncLocalStorage } from "async_hooks";

const storage = new AsyncLocalStorage<{ requestId: string }>();

async function handleRequest(requestId: string) {
  await storage.run({ requestId }, async () => {
    // storage.getStore() の型が正しく推論される
    const context = storage.getStore();
    if (context) {
      console.log(`Request ID: ${context.requestId}`);
    }
  });
}

// Import Attributes のネイティブサポート
import packageJson from "./package.json" with { type: "json" };
// Node.js 22 では --experimental-json-modules フラグが不要

// 新しい Array メソッドの型定義
const numbers = [1, 2, 3, 4, 5];

// Array.prototype.toReversed()
const reversed = numbers.toReversed();
// 型: number[]（元の配列は変更されない）

// Array.prototype.toSorted()
const sorted = numbers.toSorted((a, b) => b - a);
// 型: number[]

// Array.prototype.toSpliced()
const spliced = numbers.toSpliced(2, 1, 10);
// 型: number[]

// Array.prototype.with()
const replaced = numbers.with(2, 100);
// 型: number[]
```

---

## 9. バージョン別マイグレーションガイド

### 9-1. TypeScript 5.0 への移行

```typescript
// 5.0 へのマイグレーション

// ステップ 1: experimentalDecorators の確認
// tsconfig.json
{
  "compilerOptions": {
    // 旧式デコレータを使っている場合は true のまま
    "experimentalDecorators": true,
    // または、新しいデコレータに移行
    "experimentalDecorators": false
  }
}

// ステップ 2: enum の使用箇所を確認
// 5.0 で enum の振る舞いが変更された場合がある
enum Status {
  Pending = "PENDING",
  Success = "SUCCESS",
  Error = "ERROR",
}

// union 型への移行を検討
type Status = "PENDING" | "SUCCESS" | "ERROR";

// ステップ 3: const 型パラメータの活用
// Before
function createConfig<T>(config: T) {
  return config;
}

// After
function createConfig<const T>(config: T) {
  return config;
}
```

### 9-2. TypeScript 5.2 への移行

```typescript
// 5.2 へのマイグレーション: using 宣言の導入

// Before: 手動でのリソース管理
async function processData() {
  const db = await connectToDatabase();
  try {
    const result = await db.query("SELECT * FROM users");
    return result;
  } finally {
    await db.disconnect();
  }
}

// After: using 宣言を使用
async function processData() {
  await using db = await DatabaseConnection.create(process.env.DB_URL!);
  const result = await db.query("SELECT * FROM users");
  return result;
  // 自動的に disconnect される
}

// AsyncDisposable の実装
class DatabaseConnection implements AsyncDisposable {
  static async create(url: string): Promise<DatabaseConnection> {
    const conn = new DatabaseConnection();
    await conn.connect(url);
    return conn;
  }

  async [Symbol.asyncDispose](): Promise<void> {
    await this.disconnect();
  }

  private async connect(url: string): Promise<void> { /* ... */ }
  private async disconnect(): Promise<void> { /* ... */ }
  async query(sql: string): Promise<any[]> { /* ... */ }
}
```

### 9-3. TypeScript 5.4 への移行

```typescript
// 5.4 へのマイグレーション: NoInfer の活用

// Before: 型推論が意図しない方向に進む
function merge<T>(defaults: T, overrides: T): T {
  return { ...defaults, ...overrides };
}

const config = merge(
  { port: 3000, host: "localhost" },
  { port: 8080, unknown: true } // unknown は許容されてしまう
);

// After: NoInfer で defaults からのみ型を推論
function merge<T>(defaults: T, overrides: NoInfer<Partial<T>>): T {
  return { ...defaults, ...overrides };
}

const config = merge(
  { port: 3000, host: "localhost" },
  { port: 8080, unknown: true } // エラー: unknown は T にない
);
```

### 9-4. TypeScript 5.5 への移行

```typescript
// 5.5 へのマイグレーション: 型述語の簡略化

// Before: 明示的な型述語
function isString(value: unknown): value is string {
  return typeof value === "string";
}

function isNonNull<T>(value: T | null | undefined): value is T {
  return value != null;
}

// After: 型述語は自動推論される
const isString = (value: unknown) => typeof value === "string";
const isNonNull = <T>(value: T | null | undefined) => value != null;

// filter の改善を活用
const mixed: (string | null | number)[] = ["a", null, 1, "b", 2];

// Before: 型が絞り込まれない
const strings1 = mixed.filter((x) => typeof x === "string");
// 型: (string | null | number)[]

// After: 型が自動的に絞り込まれる
const strings2 = mixed.filter((x) => typeof x === "string");
// 型: string[]（5.5+）
```

---

## 10. 図解: TypeScript 5.x の進化

```
TypeScript 5.x の型システム進化:

  ┌─────────────────────────────────────────────────────────────┐
  │ TypeScript 5.0 (2023/03)                                    │
  ├─────────────────────────────────────────────────────────────┤
  │ • ECMAScript デコレータ（Stage 3）                           │
  │ • const 型パラメータ                                         │
  │ • enum と union の相互運用性改善                             │
  └─────────────────────────────────────────────────────────────┘
                            ↓
  ┌─────────────────────────────────────────────────────────────┐
  │ TypeScript 5.1 (2023/06)                                    │
  ├─────────────────────────────────────────────────────────────┤
  │ • getter/setter の型非対称                                  │
  │ • 暗黙的 undefined 返り値                                   │
  └─────────────────────────────────────────────────────────────┘
                            ↓
  ┌─────────────────────────────────────────────────────────────┐
  │ TypeScript 5.2 (2023/08)                                    │
  ├─────────────────────────────────────────────────────────────┤
  │ • using 宣言（Explicit Resource Management）                │
  │ • デコレータメタデータ                                       │
  └─────────────────────────────────────────────────────────────┘
                            ↓
  ┌─────────────────────────────────────────────────────────────┐
  │ TypeScript 5.3 (2023/11)                                    │
  ├─────────────────────────────────────────────────────────────┤
  │ • Import Attributes                                         │
  │ • switch(true) の型絞り込み                                 │
  └─────────────────────────────────────────────────────────────┘
                            ↓
  ┌─────────────────────────────────────────────────────────────┐
  │ TypeScript 5.4 (2024/03)                                    │
  ├─────────────────────────────────────────────────────────────┤
  │ • NoInfer ユーティリティ型                                  │
  │ • クロージャでの型絞り込み保持                               │
  └─────────────────────────────────────────────────────────────┘
                            ↓
  ┌─────────────────────────────────────────────────────────────┐
  │ TypeScript 5.5 (2024/06)                                    │
  ├─────────────────────────────────────────────────────────────┤
  │ • 型述語の推論                                              │
  │ • 正規表現の構文チェック                                     │
  │ • isolatedDeclarations                                      │
  └─────────────────────────────────────────────────────────────┘
                            ↓
  ┌─────────────────────────────────────────────────────────────┐
  │ TypeScript 5.6 (2024/09)                                    │
  ├─────────────────────────────────────────────────────────────┤
  │ • Iterator ヘルパーメソッド                                  │
  │ • Disallowed Nullish/Truthy Checks                          │
  │ • --noUncheckedSideEffectImports                            │
  └─────────────────────────────────────────────────────────────┘
                            ↓
  ┌─────────────────────────────────────────────────────────────┐
  │ TypeScript 5.7 (2024/12)                                    │
  ├─────────────────────────────────────────────────────────────┤
  │ • パフォーマンス改善                                         │
  │ • Node.js 22 サポート強化                                   │
  │ • --erasableSyntaxOnly（実験的）                            │
  └─────────────────────────────────────────────────────────────┘
```

```
デコレータの進化:

  ┌────────────────────────────────────────────────────────────┐
  │ 実験的デコレータ（experimentalDecorators: true）            │
  │ TypeScript 独自仕様、Angular/NestJS で使用                 │
  └────────────────────────────────────────────────────────────┘
                            ↓
  ┌────────────────────────────────────────────────────────────┐
  │ ECMAScript デコレータ（TypeScript 5.0+）                   │
  │ TC39 Stage 3 標準仕様                                      │
  │ - context オブジェクトによるメタデータアクセス              │
  │ - より明確な適用順序                                        │
  │ - 関数の返り値による置き換え                                │
  └────────────────────────────────────────────────────────────┘
```

```
型推論の進化:

  5.0-5.3: 手動での型述語定義
  ┌──────────────────────────────────────────┐
  │ const isString = (x: unknown): x is string => │
  │   typeof x === "string";                │
  └──────────────────────────────────────────┘
                ↓
  5.4: クロージャ内での型絞り込み保持
  ┌──────────────────────────────────────────┐
  │ if (typeof value === "string") {        │
  │   const fn = () => value.toUpperCase(); │
  │   // value は string のまま            │
  │ }                                       │
  └──────────────────────────────────────────┘
                ↓
  5.5+: 型述語の自動推論
  ┌──────────────────────────────────────────┐
  │ const isString = (x: unknown) =>        │
  │   typeof x === "string";                │
  │ // 自動的に x is string が推論される     │
  └──────────────────────────────────────────┘
```

---

## 11. 比較表

### 11-1. TypeScript バージョン比較

| バージョン | リリース | 主要機能 | 破壊的変更 | 推奨度 |
|-----------|---------|---------|-----------|-------|
| 5.0 | 2023/03 | デコレータ, const型パラメータ | decorators構文変更 | ★★★★☆ |
| 5.1 | 2023/06 | getter/setter型分離 | 小 | ★★★☆☆ |
| 5.2 | 2023/08 | using宣言, デコレータメタデータ | 小 | ★★★★★ |
| 5.3 | 2023/11 | Import Attributes | Assertions→Attributes | ★★★☆☆ |
| 5.4 | 2024/03 | NoInfer, クロージャ絞り込み | 小 | ★★★★☆ |
| 5.5 | 2024/06 | 型述語推論, 正規表現チェック | filter推論変更 | ★★★★★ |
| 5.6 | 2024/09 | Iterator, SideEffectImport | 小 | ★★★★☆ |
| 5.7 | 2024/12 | パフォーマンス改善 | 小 | ★★★★★ |

### 11-2. satisfies vs as vs as const

| 機能 | satisfies | as | as const | satisfies + as const |
|------|-----------|-----|----------|---------------------|
| 型チェック | ✅ あり | ❌ なし（上書き） | ❌ なし | ✅ あり |
| 型の狭さ | 推論された型 | 指定した型 | リテラル型 | リテラル型 |
| 余分なプロパティ | ❌ エラー | ✅ 無視 | ✅ 保持 | ❌ エラー |
| readonly | ❌ なし | ❌ なし | ✅ 自動付与 | ✅ 自動付与 |
| 用途 | 検証+推論保持 | 型アサーション | リテラル保持 | 最強の型安全性 |

### 11-3. デコレータ比較

| 特徴 | 実験的デコレータ | ECMAScript デコレータ |
|------|----------------|---------------------|
| 仕様 | TypeScript 独自 | TC39 Stage 3 標準 |
| フラグ | `experimentalDecorators: true` | デフォルト（5.0+） |
| context | ❌ なし | ✅ あり |
| メタデータ | ✅ reflect-metadata | ✅ context.metadata |
| 適用順序 | あいまい | 明確 |
| フレームワーク対応 | Angular, NestJS | 移行中 |
| 将来性 | ⚠️ 非推奨へ | ✅ 標準化 |

### 11-4. リソース管理パターン比較

| パターン | コード例 | 欠点 | using 宣言 |
|---------|---------|------|-----------|
| try-finally | `try { use(); } finally { cleanup(); }` | 冗長、ネストが深い | ✅ 自動化 |
| callback | `withResource(res => use(res))` | コールバック地獄 | ✅ 直感的 |
| AsyncIterator | `for await (const r of resources)` | 用途限定 | ✅ 汎用的 |
| using 宣言 | `using res = acquire();` | なし | ✅ 推奨 |

---

## 12. エッジケース分析

### エッジケース 1: const 型パラメータと関数オーバーロード

```typescript
// const 型パラメータと関数オーバーロードの組み合わせ

// オーバーロードシグネチャ
function createRecord<const T extends readonly string[]>(keys: T, values: string[]): Record<T[number], string>;
function createRecord<const T extends readonly string[]>(keys: T): Record<T[number], undefined>;

// 実装シグネチャ
function createRecord<const T extends readonly string[]>(
  keys: T,
  values?: string[]
): Record<T[number], string | undefined> {
  const result: any = {};
  keys.forEach((key, index) => {
    result[key] = values?.[index];
  });
  return result;
}

// 使用例
const record1 = createRecord(["a", "b", "c"], ["1", "2", "3"]);
// 型: Record<"a" | "b" | "c", string>

const record2 = createRecord(["x", "y"]);
// 型: Record<"x" | "y", undefined>

// エッジケース: 空配列
const emptyRecord = createRecord([]);
// 型: Record<never, undefined>（never は存在しないキー）

// エッジケース: readonly タプル
const tuple = ["foo", "bar"] as const;
const tupleRecord = createRecord(tuple);
// 型: Record<"foo" | "bar", undefined>
```

### エッジケース 2: using 宣言とエラーハンドリング

```typescript
// using 宣言とエラーハンドリングのエッジケース

class Resource implements Disposable {
  constructor(public id: string) {
    console.log(`Resource ${id} acquired`);
  }

  [Symbol.dispose](): void {
    console.log(`Resource ${this.id} disposed`);
  }
}

// エッジケース 1: 複数の using 宣言
function multipleResources(): void {
  using r1 = new Resource("A");
  using r2 = new Resource("B");
  using r3 = new Resource("C");

  throw new Error("Something went wrong");

  // 実行順序:
  // 1. Resource C disposed
  // 2. Resource B disposed
  // 3. Resource A disposed
  // 4. Error: Something went wrong
}

// エッジケース 2: dispose 中にエラーが発生
class ProblematicResource implements Disposable {
  constructor(public shouldFail: boolean) {}

  [Symbol.dispose](): void {
    if (this.shouldFail) {
      throw new Error("Dispose failed");
    }
  }
}

function problematicDispose(): void {
  using r1 = new ProblematicResource(false);
  using r2 = new ProblematicResource(true);  // dispose でエラー
  using r3 = new ProblematicResource(false);

  // r3, r2, r1 の順で dispose される
  // r2 の dispose でエラーが発生すると SuppressedError が投げられる
}

// エッジケース 3: 条件付き using
function conditionalUsing(useResource: boolean): void {
  if (useResource) {
    using resource = new Resource("conditional");
    // resource はこのブロック内でのみ有効
  }
  // resource はここでは使用できない
}

// エッジケース 4: using と early return
function earlyReturn(shouldReturn: boolean): string {
  using resource = new Resource("early");

  if (shouldReturn) {
    return "early return";
    // resource は return 前に dispose される
  }

  return "normal return";
  // resource は return 前に dispose される
}
```

### エッジケース 3: NoInfer と複雑なジェネリック

```typescript
// NoInfer と複雑なジェネリックのエッジケース

// エッジケース 1: ネストしたジェネリック
function deepMerge<T extends object>(
  base: T,
  override: NoInfer<DeepPartial<T>>
): T {
  // 実装...
  return { ...base, ...override } as T;
}

type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

interface Config {
  database: {
    host: string;
    port: number;
    credentials: {
      username: string;
      password: string;
    };
  };
}

const config = deepMerge<Config>(
  {
    database: {
      host: "localhost",
      port: 5432,
      credentials: { username: "admin", password: "secret" },
    },
  },
  {
    database: {
      port: 3306, // OK
      credentials: {
        username: "user", // OK
      },
    },
  }
);

// エッジケース 2: NoInfer と union 型
function select<T>(
  options: T[],
  defaultValue: NoInfer<T>
): T {
  return options.length > 0 ? options[0] : defaultValue;
}

const result = select([1, 2, 3], "default");
// エラー: "default" は number に代入不可

// エッジケース 3: NoInfer と条件型
type ExtractValue<T> = T extends { value: infer V } ? V : never;

function transform<T extends { value: any }>(
  obj: T,
  transformer: (value: ExtractValue<T>) => NoInfer<ExtractValue<T>>
): T {
  return {
    ...obj,
    value: transformer(obj.value),
  };
}

const transformed = transform(
  { value: 10 },
  (v) => v * 2 // OK: (number) => number
);

const invalid = transform(
  { value: 10 },
  (v) => String(v) // エラー: (number) => string は NoInfer<number> に代入不可
);
```

---

## 13. アンチパターン

### AP-1: 古い experimentalDecorators を使い続ける

```typescript
// ❌ アンチパターン: 旧式デコレータ（experimentalDecorators: true）
// TypeScript 5.0+ では ECMAScript 標準デコレータが利用可能

// tsconfig.json
{
  "compilerOptions": {
    "experimentalDecorators": true  // ❌ 旧式
  }
}

// 旧式デコレータ
function OldDecorator(target: any) {
  // context オブジェクトがない
  console.log(target);
}

@OldDecorator
class OldClass {}

// ✅ 推奨: ECMAScript 標準デコレータ（5.0+）
// experimentalDecorators を削除するか false にする

// tsconfig.json
{
  "compilerOptions": {
    "experimentalDecorators": false  // ✅ または削除
  }
}

// ECMAScript 標準デコレータ
function NewDecorator(target: Function, context: ClassDecoratorContext) {
  // context オブジェクトでメタデータにアクセス
  console.log(context.name, context.kind);
}

@NewDecorator
class NewClass {}

// ⚠️ 注意: Angular, NestJS 等は旧式デコレータに依存
// フレームワークの対応状況を確認すること
```

### AP-2: satisfies を使わずに型注釈で妥協する

```typescript
// ❌ アンチパターン: 型注釈で型を広げてしまう
const routes1: Record<string, { path: string; component: string }> = {
  home: { path: "/", component: "Home" },
  about: { path: "/about", component: "About" },
};

routes1.home.path; // 型: string（リテラル型 "/" が失われる）

// ✅ 推奨: satisfies でリテラル型を保持
const routes2 = {
  home: { path: "/", component: "Home" },
  about: { path: "/about", component: "About" },
} satisfies Record<string, { path: string; component: string }>;

routes2.home.path; // 型: "/"（リテラル型が保持される）

// さらに良い: as const と satisfies の組み合わせ
const routes3 = {
  home: { path: "/", component: "Home" },
  about: { path: "/about", component: "About" },
} as const satisfies Record<string, { path: string; component: string }>;

routes3.home.path; // 型: "/"
// routes3 全体が readonly になる
```

### AP-3: 手動リソース管理を続ける

```typescript
// ❌ アンチパターン: 手動での try-finally
async function oldWay() {
  const db = await connectToDatabase();
  try {
    const users = await db.query("SELECT * FROM users");
    return users;
  } finally {
    await db.disconnect(); // 手動でクリーンアップ
  }
}

// ✅ 推奨: using 宣言を使用（5.2+）
async function newWay() {
  await using db = await DatabaseConnection.create(process.env.DB_URL!);
  const users = await db.query("SELECT * FROM users");
  return users;
  // 自動的にクリーンアップされる
}

// ❌ アンチパターン: ネストした try-finally
async function nestedOldWay() {
  const conn = await openConnection();
  try {
    const file = await openFile("data.txt");
    try {
      const lock = await acquireLock();
      try {
        // 処理...
      } finally {
        await releaseLock(lock);
      }
    } finally {
      await closeFile(file);
    }
  } finally {
    await closeConnection(conn);
  }
}

// ✅ 推奨: 複数の using 宣言
async function nestedNewWay() {
  await using conn = await openConnection();
  await using file = await openFile("data.txt");
  await using lock = await acquireLock();

  // 処理...

  // lock, file, conn の順で自動的にクリーンアップされる
}
```

### AP-4: 型述語を手動で定義し続ける

```typescript
// ❌ アンチパターン: 5.5+ で不要な明示的型述語
function isStringOld(value: unknown): value is string {
  return typeof value === "string";
}

function isNumberOld(value: unknown): value is number {
  return typeof value === "number";
}

// ✅ 推奨: 型述語は自動推論される（5.5+）
const isString = (value: unknown) => typeof value === "string";
const isNumber = (value: unknown) => typeof value === "number";

// ❌ アンチパターン: filter で手動キャスト
const mixed: (string | null)[] = ["a", null, "b"];
const strings1 = mixed.filter((x) => x !== null) as string[];

// ✅ 推奨: 型述語が自動推論される
const strings2 = mixed.filter((x) => x !== null);
// 型: string[]（5.5+ で自動的に絞り込まれる）
```

---

## 14. 演習問題

### 基礎レベル

#### 問題 1: const 型パラメータを使った関数

以下の関数を、const 型パラメータを使ってリテラル型を保持するように修正してください。

```typescript
// 現在の実装
function createTuple<T, U>(first: T, second: U): [T, U] {
  return [first, second];
}

const result = createTuple("hello", 42);
// 型: [string, number]
// 期待: [string, 42] または ["hello", 42]

// TODO: const 型パラメータを使って修正してください
```

<details>
<summary>解答例</summary>

```typescript
function createTuple<const T, const U>(first: T, second: U): [T, U] {
  return [first, second];
}

const result = createTuple("hello", 42);
// 型: ["hello", 42]
```
</details>

#### 問題 2: using 宣言を使ったリソース管理

以下の FileReader クラスに Disposable インターフェースを実装し、using 宣言で使用できるようにしてください。

```typescript
class FileReader {
  private content: string = "";

  constructor(private path: string) {
    this.content = this.readFile(path);
    console.log(`File opened: ${path}`);
  }

  private readFile(path: string): string {
    // 模擬的な実装
    return `Content of ${path}`;
  }

  getContent(): string {
    return this.content;
  }

  close(): void {
    this.content = "";
    console.log(`File closed: ${this.path}`);
  }

  // TODO: Disposable インターフェースを実装してください
}

// TODO: using 宣言を使って FileReader を使用してください
```

<details>
<summary>解答例</summary>

```typescript
class FileReader implements Disposable {
  private content: string = "";

  constructor(private path: string) {
    this.content = this.readFile(path);
    console.log(`File opened: ${path}`);
  }

  private readFile(path: string): string {
    return `Content of ${path}`;
  }

  getContent(): string {
    return this.content;
  }

  close(): void {
    this.content = "";
    console.log(`File closed: ${this.path}`);
  }

  [Symbol.dispose](): void {
    this.close();
  }
}

// 使用例
function processFile(path: string): string {
  using reader = new FileReader(path);
  return reader.getContent();
  // 自動的に close() が呼ばれる
}
```
</details>

### 応用レベル

#### 問題 3: NoInfer を使った型安全な API

以下の API を、NoInfer を使ってより型安全にしてください。

```typescript
function createState<T>(
  initialValue: T,
  validator?: (value: T) => boolean
): {
  get: () => T;
  set: (value: T) => void;
} {
  let state = initialValue;

  return {
    get: () => state,
    set: (value) => {
      if (validator && !validator(value)) {
        throw new Error("Validation failed");
      }
      state = value;
    },
  };
}

// 問題: validator の型が initialValue と一致しない場合がある
const state = createState(10, (v) => typeof v === "string");
// エラーが出ない（出るべき）

// TODO: NoInfer を使って validator の型を厳密にしてください
```

<details>
<summary>解答例</summary>

```typescript
function createState<T>(
  initialValue: T,
  validator?: (value: NoInfer<T>) => boolean
): {
  get: () => T;
  set: (value: T) => void;
} {
  let state = initialValue;

  return {
    get: () => state,
    set: (value) => {
      if (validator && !validator(value)) {
        throw new Error("Validation failed");
      }
      state = value;
    },
  };
}

// 修正後
const state = createState(10, (v) => typeof v === "string");
// エラー: (v: number) => boolean が期待されるが、(v: number) => typeof v === "string" が与えられた
```
</details>

#### 問題 4: 型述語の推論を活用したフィルタリング

以下の配列から、特定の条件を満たす要素をフィルタリングする関数を、型述語の推論を活用して実装してください。

```typescript
type Result<T> =
  | { success: true; data: T }
  | { success: false; error: string };

const results: Result<number>[] = [
  { success: true, data: 1 },
  { success: false, error: "Error 1" },
  { success: true, data: 2 },
  { success: false, error: "Error 2" },
  { success: true, data: 3 },
];

// TODO: 成功した結果のみを取得し、data の配列を返す関数を実装
// 型述語の推論を活用して、型安全に実装してください
```

<details>
<summary>解答例</summary>

```typescript
type Result<T> =
  | { success: true; data: T }
  | { success: false; error: string };

const results: Result<number>[] = [
  { success: true, data: 1 },
  { success: false, error: "Error 1" },
  { success: true, data: 2 },
  { success: false, error: "Error 2" },
  { success: true, data: 3 },
];

// 型述語が自動推論される（5.5+）
const successResults = results.filter((r) => r.success);
// 型: { success: true; data: number }[]

const data = successResults.map((r) => r.data);
// 型: number[]
// data = [1, 2, 3]

// または、1行で
const data2 = results
  .filter((r) => r.success)
  .map((r) => r.data);
// 型: number[]
```
</details>

### 発展レベル

#### 問題 5: ECMAScript デコレータを使った依存性注入

ECMAScript 標準デコレータを使って、簡易的な依存性注入システムを実装してください。

```typescript
// TODO: Injectable デコレータを実装
// TODO: Inject デコレータを実装
// TODO: Container クラスを実装

// 使用例:
@Injectable()
class Logger {
  log(message: string) {
    console.log(`[LOG] ${message}`);
  }
}

@Injectable()
class UserService {
  @Inject(Logger)
  logger!: Logger;

  getUser(id: string) {
    this.logger.log(`Getting user ${id}`);
    return { id, name: "Alice" };
  }
}

const container = new Container();
const userService = container.resolve(UserService);
userService.getUser("123");
// 期待される出力: [LOG] Getting user 123
```

<details>
<summary>解答例</summary>

```typescript
// メタデータキー
const INJECTABLE_KEY = Symbol("injectable");
const DEPENDENCIES_KEY = Symbol("dependencies");

// Injectable デコレータ
function Injectable() {
  return function <T extends { new (...args: any[]): {} }>(
    target: T,
    context: ClassDecoratorContext
  ) {
    context.metadata[INJECTABLE_KEY] = true;
    return target;
  };
}

// Inject デコレータ
function Inject(dependency: any) {
  return function (
    _target: undefined,
    context: ClassFieldDecoratorContext
  ) {
    const metadata = context.metadata;
    if (!metadata[DEPENDENCIES_KEY]) {
      metadata[DEPENDENCIES_KEY] = new Map();
    }
    (metadata[DEPENDENCIES_KEY] as Map<string | symbol, any>).set(
      context.name,
      dependency
    );
  };
}

// Container クラス
class Container {
  private instances = new Map<any, any>();

  resolve<T>(target: new (...args: any[]) => T): T {
    // すでにインスタンスがある場合は返す
    if (this.instances.has(target)) {
      return this.instances.get(target);
    }

    // インスタンスを作成
    const instance = new target();

    // メタデータから依存関係を取得
    const metadata = (target as any)[Symbol.metadata];
    const dependencies = metadata?.[DEPENDENCIES_KEY] as Map<
      string | symbol,
      any
    >;

    // 依存関係を注入
    if (dependencies) {
      for (const [fieldName, dependency] of dependencies) {
        (instance as any)[fieldName] = this.resolve(dependency);
      }
    }

    // インスタンスをキャッシュ
    this.instances.set(target, instance);

    return instance;
  }
}

// 使用例
@Injectable()
class Logger {
  log(message: string) {
    console.log(`[LOG] ${message}`);
  }
}

@Injectable()
class UserService {
  @Inject(Logger)
  logger!: Logger;

  getUser(id: string) {
    this.logger.log(`Getting user ${id}`);
    return { id, name: "Alice" };
  }
}

const container = new Container();
const userService = container.resolve(UserService);
userService.getUser("123");
// 出力: [LOG] Getting user 123
```
</details>

#### 問題 6: 複雑な型推論とリソース管理の組み合わせ

以下の要件を満たす Transaction クラスを実装してください。

要件:
1. AsyncDisposable を実装
2. ジェネリック型パラメータで操作の結果型を指定可能
3. commit/rollback の状態管理
4. 型述語を使った状態の絞り込み

```typescript
// TODO: Transaction クラスを実装

// 使用例:
async function transferMoney(from: string, to: string, amount: number) {
  await using tx = await Transaction.begin<{ newBalance: number }>();

  await tx.execute(async () => {
    // 送金処理
    return { newBalance: 1000 };
  });

  await tx.commit();

  return tx.getResult(); // 型: { newBalance: number } | undefined
}
```

<details>
<summary>解答例</summary>

```typescript
type TransactionState = "pending" | "committed" | "rolledback";

class Transaction<T> implements AsyncDisposable {
  private state: TransactionState = "pending";
  private result: T | undefined;

  private constructor() {}

  static async begin<T>(): Promise<Transaction<T>> {
    console.log("Transaction started");
    return new Transaction<T>();
  }

  async execute(operation: () => Promise<T>): Promise<void> {
    if (this.state !== "pending") {
      throw new Error("Transaction is not in pending state");
    }

    try {
      this.result = await operation();
    } catch (error) {
      await this.rollback();
      throw error;
    }
  }

  async commit(): Promise<void> {
    if (this.state !== "pending") {
      throw new Error("Transaction is not in pending state");
    }

    console.log("Transaction committed");
    this.state = "committed";
  }

  async rollback(): Promise<void> {
    if (this.state !== "pending") {
      return; // すでに完了している場合は何もしない
    }

    console.log("Transaction rolled back");
    this.state = "rolledback";
    this.result = undefined;
  }

  getResult(): T | undefined {
    return this.result;
  }

  isCommitted(): boolean {
    return this.state === "committed";
  }

  async [Symbol.asyncDispose](): Promise<void> {
    if (this.state === "pending") {
      await this.rollback();
    }
  }
}

// 使用例
async function transferMoney(from: string, to: string, amount: number) {
  await using tx = await Transaction.begin<{ newBalance: number }>();

  await tx.execute(async () => {
    // 模擬的な送金処理
    console.log(`Transferring ${amount} from ${from} to ${to}`);
    await new Promise((resolve) => setTimeout(resolve, 100));
    return { newBalance: 1000 - amount };
  });

  await tx.commit();

  return tx.getResult();
}

// 実行例
const result = await transferMoney("Alice", "Bob", 100);
// Transaction started
// Transferring 100 from Alice to Bob
// Transaction committed

console.log(result); // { newBalance: 900 }
```
</details>

---

## 15. tsconfig.json 推奨設定（バージョン別）

### TypeScript 5.0 推奨設定

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022"],
    "module": "ESNext",
    "moduleResolution": "bundler",

    // 型チェック
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "noPropertyAccessFromIndexSignature": true,

    // デコレータ（ECMAScript 標準）
    "experimentalDecorators": false,
    "emitDecoratorMetadata": false,

    // 出力
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true,
    "outDir": "./dist",
    "removeComments": true,

    // インポート
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "resolveJsonModule": true,

    // その他
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

### TypeScript 5.2 推奨設定（using 宣言対応）

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "ESNext.Disposable"],
    "module": "ESNext",
    "moduleResolution": "bundler",

    "strict": true,
    "noUncheckedIndexedAccess": true,

    // using 宣言のサポート
    // ランタイムで Symbol.dispose のポリフィルが必要な場合あり

    "declaration": true,
    "sourceMap": true,
    "outDir": "./dist"
  }
}
```

### TypeScript 5.5 推奨設定（isolatedDeclarations 対応）

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "ESNext.Disposable"],
    "module": "ESNext",
    "moduleResolution": "bundler",

    "strict": true,
    "noUncheckedIndexedAccess": true,

    // 型定義の分離（ライブラリ開発時に推奨）
    "isolatedDeclarations": true,
    "declaration": true,

    // 正規表現チェック
    // デフォルトで有効（無効化する必要はない）

    "sourceMap": true,
    "outDir": "./dist"
  }
}
```

### TypeScript 5.6 推奨設定（最新機能フル活用）

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2023", "ESNext.Disposable", "ESNext.Iterator"],
    "module": "ESNext",
    "moduleResolution": "bundler",

    // 厳格な型チェック
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "noPropertyAccessFromIndexSignature": true,
    "exactOptionalPropertyTypes": true,

    // 副作用インポートのチェック
    "noUncheckedSideEffectImports": true,

    // 型定義
    "isolatedDeclarations": true,
    "declaration": true,
    "declarationMap": true,

    // 出力
    "sourceMap": true,
    "outDir": "./dist",
    "removeComments": true,

    // インポート
    "esModuleInterop": true,
    "allowSyntheticDefaultImports": true,
    "resolveJsonModule": true,

    // その他
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist", "**/*.test.ts"]
}
```

---

## 16. FAQ

### Q1: TypeScript のバージョンアップはどのくらいの頻度で行うべきですか?

**A:** マイナーバージョン（5.x）ごとに追従することを推奨します。TypeScript は約 3 ヶ月サイクルでリリースされ、各バージョンの破壊的変更は比較的小さいです。

具体的な戦略:
1. **nightly テスト**: CI で `typescript@next` をテストするジョブを追加し、問題を早期発見
2. **段階的移行**: 新バージョンリリース後 1-2 週間で検証環境に導入
3. **本番適用**: 問題がなければ 1 ヶ月以内に本番環境に適用
4. **パッチバージョン**: セキュリティ修正を含むため即座に適用

```json
// package.json での管理
{
  "devDependencies": {
    "typescript": "^5.7.0",  // マイナーバージョンまで自動更新
    "typescript-next": "npm:typescript@next"  // nightly テスト用
  },
  "scripts": {
    "test": "tsc --noEmit && vitest",
    "test:next": "tsc --noEmit && vitest"  // next バージョンでのテスト
  }
}
```

### Q2: satisfies はどのような場面で使うべきですか?

**A:** 以下のような場面で satisfies が有効です:

1. **オブジェクトリテラルの型検証 + リテラル型保持**
```typescript
const config = {
  port: 3000,
  host: "localhost",
} satisfies Config;
// config.port の型は number（リテラル型は保持されない）

const config2 = {
  port: 3000,
  host: "localhost",
} as const satisfies Config;
// config2.port の型は 3000（リテラル型が保持される）
```

2. **ルーティングテーブルやマッピングオブジェクト**
```typescript
const routes = {
  "/home": HomePage,
  "/about": AboutPage,
  "/contact": ContactPage,
} satisfies Record<string, ComponentType>;
// 余分なプロパティがあればエラー、型も保持
```

3. **設定オブジェクトの検証**
```typescript
const featureFlags = {
  enableNewUI: true,
  enableBeta: false,
} satisfies Record<string, boolean>;
// キーのタイポを防ぎつつ、具体的なキー名を保持
```

### Q3: using 宣言はいつから実際に使えますか?

**A:** TypeScript 5.2+ で構文サポートされています。ランタイム対応状況:

**ネイティブサポート:**
- Node.js 22+ (2024年4月リリース)
- Chrome 125+ (2024年5月リリース)
- Safari 18+ (2024年9月リリース)

**ポリフィルが必要な環境:**
```typescript
// Symbol.dispose のポリフィル
if (!Symbol.dispose) {
  (Symbol as any).dispose = Symbol.for("Symbol.dispose");
}

if (!Symbol.asyncDispose) {
  (Symbol as any).asyncDispose = Symbol.for("Symbol.asyncDispose");
}
```

**推奨:**
- 新規プロジェクト: Node.js 22+ を使用してネイティブサポートを活用
- 既存プロジェクト: ポリフィルを導入してから段階的に using 宣言に移行

### Q4: 型述語の自動推論（5.5）は常に正しく動作しますか?

**A:** ほとんどの場合は正しく動作しますが、複雑な条件では明示的な型述語が必要な場合があります:

```typescript
// 自動推論が機能する例
const isString = (x: unknown) => typeof x === "string";
// 推論: (x: unknown) => x is string ✅

// 自動推論が不十分な例
const isValidUser = (x: unknown) => {
  if (typeof x !== "object" || x === null) return false;
  return "id" in x && "name" in x;
};
// 推論: (x: unknown) => boolean ❌
// 期待: (x: unknown) => x is { id: unknown; name: unknown }

// 明示的な型述語が必要
interface User {
  id: number;
  name: string;
}

function isValidUser(x: unknown): x is User {
  if (typeof x !== "object" || x === null) return false;
  return (
    "id" in x &&
    typeof (x as any).id === "number" &&
    "name" in x &&
    typeof (x as any).name === "string"
  );
}
```

### Q5: NoInfer はいつ使うべきですか?

**A:** 以下の状況で NoInfer が有効です:

1. **デフォルト値やフォールバック値の型を制限**
```typescript
function getOrDefault<T>(value: T | null, fallback: NoInfer<T>): T {
  return value ?? fallback;
}
```

2. **イベントハンドラーの型を固定**
```typescript
function on<K extends keyof Events>(
  event: K,
  handler: (payload: NoInfer<Events[K]>) => void
): void;
```

3. **変換関数の戻り値型を制限**
```typescript
function map<T>(
  items: T[],
  transform: (item: T) => NoInfer<T>
): T[] {
  return items.map(transform);
}
```

**使わない方が良い場合:**
- 両方の引数から型を推論したい場合
- 柔軟な型推論が必要な場合

---

## 17. まとめ表

| 概念 | 要点 | バージョン |
|------|------|----------|
| ECMAScript デコレータ | Stage 3 標準、context オブジェクト | 5.0+ |
| const 型パラメータ | ジェネリクスでリテラル型を保持 | 5.0+ |
| getter/setter 型非対称 | 異なる型を使用可能 | 5.1+ |
| using 宣言 | RAII パターンのリソース管理 | 5.2+ |
| Import Attributes | JSON/CSS 等のインポート | 5.3+ |
| switch(true) 絞り込み | switch 文での型ナローイング | 5.3+ |
| NoInfer | 型推論の候補から除外 | 5.4+ |
| クロージャ絞り込み | クロージャ内で型絞り込み保持 | 5.4+ |
| 型述語推論 | filter 等で自動絞り込み | 5.5+ |
| 正規表現チェック | リテラルの構文エラー検出 | 5.5+ |
| isolatedDeclarations | 型定義の分離 | 5.5+ |
| Iterator ヘルパー | take/map/filter 等のメソッド | 5.6+ |
| Nullish/Truthy チェック | 不要な条件式を警告 | 5.6+ |
| パフォーマンス改善 | 大規模プロジェクトの高速化 | 5.7+ |

---

## 18. 次に読むべきガイド

- **[tsconfig.json](../03-tooling/00-tsconfig.md)** -- 新バージョンの設定オプション詳細
- **[判別共用体](../02-patterns/02-discriminated-unions.md)** -- 5.x で改善された型絞り込みの活用
- **[ビルドツール](../03-tooling/01-build-tools.md)** -- 新バージョンへのビルドツール対応
- **[デコレータパターン](../02-patterns/05-decorators.md)** -- ECMAScript デコレータの実践的パターン
- **[エラーハンドリング](../02-patterns/03-error-handling.md)** -- using 宣言を活用したリソース管理

---

## 19. 参考文献

1. **TypeScript Release Notes**
   https://www.typescriptlang.org/docs/handbook/release-notes/overview.html
   各バージョンの詳細な変更内容と移行ガイド

2. **TypeScript Blog**
   https://devblogs.microsoft.com/typescript/
   TypeScript チームによる公式ブログ、新機能の背景と設計思想

3. **TC39 Proposals**
   https://github.com/tc39/proposals
   ECMAScript の提案一覧、TypeScript が実装する機能の元仕様

4. **Explicit Resource Management Proposal**
   https://github.com/tc39/proposal-explicit-resource-management
   using 宣言の ECMAScript 提案

5. **Decorator Metadata Proposal**
   https://github.com/tc39/proposal-decorator-metadata
   デコレータメタデータの ECMAScript 提案

6. **Iterator Helpers Proposal**
   https://github.com/tc39/proposal-iterator-helpers
   Iterator ヘルパーメソッドの ECMAScript 提案

---

**文字数**: 43,247文字

このガイドは TypeScript 5.0 から 5.7 までの主要な新機能を網羅し、実務で即座に活用できる実践的なコード例と、深い理解のための詳細な解説を提供します。