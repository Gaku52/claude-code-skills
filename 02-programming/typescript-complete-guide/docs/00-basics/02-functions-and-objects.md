# 関数とオブジェクト型

> TypeScriptにおける関数シグネチャ、オーバーロード、interface、type aliasの使い分けを網羅する。

## この章で学ぶこと

1. **関数の型付け** -- パラメータ型、戻り値型、オプショナル引数、デフォルト値、rest引数、オーバーロード
2. **interface** -- オブジェクトの構造を定義し、クラスやモジュール間の契約として使う
3. **type alias** -- 型エイリアスによる柔軟な型定義とinterfaceとの使い分け
4. **構造的型付け** -- TypeScript独自の型互換性判定メカニズム
5. **高度な関数パターン** -- ジェネリック関数、this型、コンストラクタ型、コールバックパターン
6. **オブジェクト型の高度なパターン** -- インデックスシグネチャ、Record、Mapped Types

---

## 1. 関数の型付け

### コード例1: 基本的な関数型

```typescript
// 関数宣言
function add(a: number, b: number): number {
  return a + b;
}

// アロー関数
const multiply = (a: number, b: number): number => a * b;

// 関数型の変数
const divide: (a: number, b: number) => number = (a, b) => a / b;

// 型エイリアスで関数型を定義
type MathOp = (a: number, b: number) => number;
const subtract: MathOp = (a, b) => a - b;
```

### 関数型の記法詳細

```typescript
// ===== 関数宣言のバリエーション =====

// 1. function宣言（ホイスティングされる）
function greet(name: string): string {
  return `Hello, ${name}!`;
}

// 2. 関数式
const greetExpr = function (name: string): string {
  return `Hello, ${name}!`;
};

// 3. アロー関数（thisをバインドしない）
const greetArrow = (name: string): string => `Hello, ${name}!`;

// 4. ジェネリック関数
function identity<T>(value: T): T {
  return value;
}

// 5. ジェネリックアロー関数（TSXとの衝突を避けるため extends を使う）
const identityArrow = <T extends unknown>(value: T): T => value;

// ===== 関数型の定義方法 =====

// 方法1: 型エイリアス
type Formatter = (input: string) => string;

// 方法2: interface（call signature）
interface FormatterInterface {
  (input: string): string;
}

// 方法3: interface（メソッドシグネチャ）
interface StringUtils {
  format(input: string): string;
  trim(input: string): string;
}

// 方法4: オブジェクトリテラル内のメソッド
type Logger = {
  log(message: string): void;
  error(message: string, error?: Error): void;
  warn(message: string): void;
};

// ===== 関数型とプロパティの複合型 =====
// 関数でありながらプロパティも持つ型
interface CreateElement {
  (tag: string): HTMLElement;
  defaultNamespace: string;
  supportedTags: string[];
}

// 実装例
const createElement: CreateElement = Object.assign(
  (tag: string) => document.createElement(tag),
  {
    defaultNamespace: "http://www.w3.org/1999/xhtml",
    supportedTags: ["div", "span", "p", "a"],
  }
);

createElement("div"); // HTMLElement
createElement.defaultNamespace; // string
```

### コード例2: オプショナル引数とデフォルト値

```typescript
// オプショナル引数（?）
function greet(name: string, greeting?: string): string {
  return `${greeting ?? "Hello"}, ${name}!`;
}
greet("Alice");           // "Hello, Alice!"
greet("Alice", "Hi");     // "Hi, Alice!"

// デフォルト値
function createUser(name: string, role: string = "viewer"): { name: string; role: string } {
  return { name, role };
}
createUser("Bob");              // { name: "Bob", role: "viewer" }
createUser("Bob", "admin");     // { name: "Bob", role: "admin" }

// rest引数
function sum(...numbers: number[]): number {
  return numbers.reduce((total, n) => total + n, 0);
}
sum(1, 2, 3, 4, 5); // 15
```

### オプショナル引数の詳細パターン

```typescript
// オプショナル引数 vs デフォルト値 の違い
function example1(x: number, y?: number): number {
  // y の型は number | undefined
  return x + (y ?? 0);
}

function example2(x: number, y: number = 0): number {
  // y の型は number（デフォルト値があるため）
  return x + y;
}

// 呼び出し方の違い
example1(1);           // OK: y は undefined
example1(1, undefined); // OK: y は undefined
example1(1, 2);        // OK: y は 2

example2(1);           // OK: y は 0
example2(1, undefined); // OK: y は 0（undefinedの場合もデフォルト値が使われる）
example2(1, 2);        // OK: y は 2

// オプショナル引数は最後に配置する必要がある
// function bad(x?: number, y: number) {} // エラー
function good(y: number, x?: number) {} // OK

// デフォルト値は途中の引数にも使える
function createRange(start: number = 0, end: number, step: number = 1): number[] {
  const result: number[] = [];
  for (let i = start; i < end; i += step) {
    result.push(i);
  }
  return result;
}
createRange(undefined, 5);     // [0, 1, 2, 3, 4] (start = 0)
createRange(2, 10, 3);         // [2, 5, 8]

// rest引数の型付け詳細
function createLogger(prefix: string, ...tags: string[]): void {
  console.log(`[${prefix}]`, ...tags.map(t => `#${t}`));
}
createLogger("APP", "info", "startup"); // [APP] #info #startup

// rest引数のタプル型
function query(sql: string, ...params: [string, ...number[]]): void {
  console.log(sql, params);
}
query("SELECT * FROM users WHERE name = ? AND age > ?", "Alice", 30);

// スプレッド引数の型安全性
function add3(a: number, b: number, c: number): number {
  return a + b + c;
}
const args = [1, 2, 3] as const; // readonly [1, 2, 3]
add3(...args); // OK（as const が必要。なければ number[] と推論され、3引数に合わない）
```

### コード例3: 関数オーバーロード

```typescript
// オーバーロードシグネチャ
function createElement(tag: "div"): HTMLDivElement;
function createElement(tag: "span"): HTMLSpanElement;
function createElement(tag: "input"): HTMLInputElement;
// 実装シグネチャ
function createElement(tag: string): HTMLElement {
  return document.createElement(tag);
}

const div = createElement("div");     // 型: HTMLDivElement
const span = createElement("span");   // 型: HTMLSpanElement
const input = createElement("input"); // 型: HTMLInputElement
```

### オーバーロードの詳細パターン

```typescript
// パターン1: 引数の数によるオーバーロード
function padding(all: number): string;
function padding(vertical: number, horizontal: number): string;
function padding(top: number, right: number, bottom: number, left: number): string;
function padding(a: number, b?: number, c?: number, d?: number): string {
  if (b === undefined) {
    return `${a}px`;
  }
  if (c === undefined) {
    return `${a}px ${b}px`;
  }
  return `${a}px ${b}px ${c}px ${d}px`;
}

padding(10);           // "10px"
padding(10, 20);       // "10px 20px"
padding(10, 20, 30, 40); // "10px 20px 30px 40px"

// パターン2: 引数の型によるオーバーロード
function parseInput(input: string): string[];
function parseInput(input: number): number[];
function parseInput(input: string | number): (string | number)[] {
  if (typeof input === "string") {
    return input.split(",");
  }
  return [input];
}

const strResult = parseInput("a,b,c"); // string[]
const numResult = parseInput(42);       // number[]

// パターン3: 戻り値型のオーバーロード
function fetchData(url: string, format: "json"): Promise<object>;
function fetchData(url: string, format: "text"): Promise<string>;
function fetchData(url: string, format: "blob"): Promise<Blob>;
function fetchData(url: string, format: string): Promise<unknown> {
  return fetch(url).then(response => {
    switch (format) {
      case "json": return response.json();
      case "text": return response.text();
      case "blob": return response.blob();
      default: return response.text();
    }
  });
}

// パターン4: ジェネリクスを使ったオーバーロードの代替
// オーバーロードの代わりに条件型を使う方法
type ParseResult<T> = T extends string ? string[] : T extends number ? number[] : never;

function parseInputGeneric<T extends string | number>(input: T): ParseResult<T> {
  if (typeof input === "string") {
    return input.split(",") as ParseResult<T>;
  }
  return [input] as ParseResult<T>;
}

// パターン5: メソッドオーバーロード
class EventEmitter {
  on(event: "click", handler: (x: number, y: number) => void): void;
  on(event: "keypress", handler: (key: string) => void): void;
  on(event: "scroll", handler: (position: number) => void): void;
  on(event: string, handler: (...args: unknown[]) => void): void {
    // 実装
  }
}

const emitter = new EventEmitter();
emitter.on("click", (x, y) => {
  // x: number, y: number と推論される
  console.log(x, y);
});
emitter.on("keypress", (key) => {
  // key: string と推論される
  console.log(key);
});
```

### this型の制御

```typescript
// this パラメータ（実際の引数ではなく、this の型を指定する）
interface User {
  name: string;
  greet(this: User): string;
}

const user: User = {
  name: "Alice",
  greet() {
    return `Hello, I'm ${this.name}`;
  },
};

user.greet(); // OK
// const greetFn = user.greet;
// greetFn(); // エラー: this の型が User ではない

// クラスでの this 型
class Builder {
  private items: string[] = [];

  add(item: string): this {
    this.items.push(item);
    return this; // this を返すことでメソッドチェーンを可能に
  }

  build(): string[] {
    return [...this.items];
  }
}

class EnhancedBuilder extends Builder {
  private prefix: string = "";

  setPrefix(prefix: string): this {
    this.prefix = prefix;
    return this;
  }
}

// メソッドチェーンが型安全に動作
const result = new EnhancedBuilder()
  .setPrefix("item-")  // EnhancedBuilder を返す
  .add("one")           // EnhancedBuilder を返す（Builder ではない）
  .add("two")
  .build();

// thisの型ガード
class FileReader {
  private content: string | null = null;
  private loaded: boolean = false;

  isLoaded(): this is FileReader & { content: string } {
    return this.loaded && this.content !== null;
  }

  load(path: string): void {
    this.content = "file content";
    this.loaded = true;
  }

  getContent(): string {
    if (this.isLoaded()) {
      return this.content; // string として安全にアクセス
    }
    throw new Error("File not loaded");
  }
}
```

### 関数型の記法比較

```
  関数宣言          アロー関数型         call signature
+---------------+  +------------------+  +---------------------+
| function      |  | (a: T, b: U)     |  | interface Fn {      |
|   fn(a: T):U  |  |   => R           |  |   (a: T, b: U): R  |
+---------------+  +------------------+  +---------------------+

  メソッドシグネチャ     コンストラクタシグネチャ
+---------------------+  +-----------------------+
| interface Obj {     |  | interface Ctor {      |
|   method(a: T): R   |  |   new (a: T): Obj     |
| }                   |  | }                     |
+---------------------+  +-----------------------+
```

### コールバック関数の型パターン

```typescript
// ===== コールバック関数の型定義 =====

// シンプルなコールバック
type SimpleCallback = () => void;
type ErrorCallback = (error: Error | null) => void;
type DataCallback<T> = (error: Error | null, data: T) => void;

// Node.jsスタイルのコールバック
type NodeCallback<T> = (error: NodeJS.ErrnoException | null, result: T) => void;

function readFile(
  path: string,
  callback: DataCallback<string>
): void {
  try {
    const content = "file content";
    callback(null, content);
  } catch (err) {
    callback(err instanceof Error ? err : new Error(String(err)), "" as never);
  }
}

// Promise型の関数
type AsyncFunction<T, R> = (input: T) => Promise<R>;

// ミドルウェアパターン
type Middleware<T> = (
  context: T,
  next: () => Promise<void>
) => Promise<void>;

// Express風のミドルウェア
interface Request {
  path: string;
  method: string;
  body: unknown;
}
interface Response {
  status(code: number): Response;
  json(data: unknown): void;
}
type NextFunction = () => void;

type ExpressMiddleware = (
  req: Request,
  res: Response,
  next: NextFunction
) => void | Promise<void>;

// イベントリスナーの型
type EventListener<T = Event> = (event: T) => void;

// 高階関数の型
type Predicate<T> = (value: T) => boolean;
type Mapper<T, U> = (value: T, index: number) => U;
type Reducer<T, U> = (accumulator: U, value: T, index: number) => U;
type Comparator<T> = (a: T, b: T) => number;

// 高階関数の実装例
function pipe<T>(...fns: ((value: T) => T)[]): (value: T) => T {
  return (value: T) => fns.reduce((acc, fn) => fn(acc), value);
}

const processString = pipe<string>(
  (s) => s.trim(),
  (s) => s.toLowerCase(),
  (s) => s.replace(/\s+/g, "-"),
);

processString("  Hello World  "); // "hello-world"
```

### 非同期関数の型

```typescript
// async/await の型付け
async function fetchUser(id: number): Promise<User> {
  const response = await fetch(`/api/users/${id}`);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

// Promise ユーティリティの型
type PromiseType<T> = T extends Promise<infer U> ? U : T;
type Unwrapped = PromiseType<Promise<string>>; // string

// Awaited型（TypeScript 4.5+）
type AwaitedResult = Awaited<Promise<Promise<string>>>; // string（深いPromiseも解決）

// 非同期ジェネレーター
async function* generateNumbers(count: number): AsyncGenerator<number> {
  for (let i = 0; i < count; i++) {
    await new Promise(resolve => setTimeout(resolve, 100));
    yield i;
  }
}

// 非同期イテレータの消費
async function processNumbers(): Promise<void> {
  for await (const num of generateNumbers(5)) {
    console.log(num); // 0, 1, 2, 3, 4
  }
}

// 型安全なリトライ関数
async function withRetry<T>(
  fn: () => Promise<T>,
  options: {
    maxRetries?: number;
    delay?: number;
    backoff?: number;
    shouldRetry?: (error: unknown) => boolean;
  } = {}
): Promise<T> {
  const {
    maxRetries = 3,
    delay = 1000,
    backoff = 2,
    shouldRetry = () => true,
  } = options;

  let lastError: unknown;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      if (attempt < maxRetries && shouldRetry(error)) {
        await new Promise(resolve =>
          setTimeout(resolve, delay * Math.pow(backoff, attempt))
        );
      }
    }
  }

  throw lastError;
}

// 使用例
const user = await withRetry(
  () => fetchUser(1),
  {
    maxRetries: 3,
    delay: 500,
    shouldRetry: (err) => {
      if (err instanceof Error && err.message.includes("HTTP 404")) {
        return false; // 404はリトライしない
      }
      return true;
    },
  }
);
```

---

## 2. interface

### コード例4: interface の定義と使用

```typescript
// 基本的なinterface
interface User {
  readonly id: number;     // 読み取り専用
  name: string;            // 必須プロパティ
  email: string;           // 必須プロパティ
  age?: number;            // オプショナルプロパティ
}

// インデックスシグネチャ
interface Dictionary {
  [key: string]: string;
}

// 関数を持つinterface
interface Formatter {
  format(value: unknown): string;
  readonly prefix: string;
}

// interface の継承
interface Employee extends User {
  department: string;
  salary: number;
}

// 複数の継承
interface Manager extends Employee {
  reports: Employee[];
}

const manager: Manager = {
  id: 1,
  name: "Alice",
  email: "alice@example.com",
  department: "Engineering",
  salary: 120000,
  reports: [],
};
```

### interface の高度なパターン

```typescript
// ===== 複数のinterfaceの継承 =====
interface Serializable {
  serialize(): string;
}

interface Printable {
  print(): void;
}

interface Loggable {
  log(level: "info" | "warn" | "error"): void;
}

// 複数のinterfaceを同時に継承
interface Document extends Serializable, Printable, Loggable {
  title: string;
  content: string;
}

// ===== ジェネリックinterface =====
interface Repository<T> {
  findById(id: string): Promise<T | null>;
  findAll(filter?: Partial<T>): Promise<T[]>;
  create(data: Omit<T, "id">): Promise<T>;
  update(id: string, data: Partial<T>): Promise<T>;
  delete(id: string): Promise<boolean>;
}

interface User {
  id: string;
  name: string;
  email: string;
}

// 具体的な型でRepository を使用
class UserRepository implements Repository<User> {
  async findById(id: string): Promise<User | null> {
    // データベースから取得
    return null;
  }

  async findAll(filter?: Partial<User>): Promise<User[]> {
    return [];
  }

  async create(data: Omit<User, "id">): Promise<User> {
    return { id: crypto.randomUUID(), ...data };
  }

  async update(id: string, data: Partial<User>): Promise<User> {
    return { id, name: "", email: "", ...data };
  }

  async delete(id: string): Promise<boolean> {
    return true;
  }
}

// ===== コンストラクタシグネチャ =====
interface Constructable<T> {
  new (...args: unknown[]): T;
}

function createInstance<T>(Ctor: Constructable<T>): T {
  return new Ctor();
}

class MyService {
  constructor() {
    console.log("Service created");
  }
}

const service = createInstance(MyService); // MyService

// ===== ハイブリッドinterface（関数 + プロパティ） =====
interface JQuery {
  (selector: string): JQuery;
  ajax(settings: object): Promise<unknown>;
  version: string;
}

// ===== Mapped Types風のinterface（限定的） =====
interface StringMap {
  [key: string]: string;
}

interface NumberMap {
  [key: string]: number;
}

// ===== readonly インデックスシグネチャ =====
interface ReadonlyStringMap {
  readonly [key: string]: string;
}
```

### コード例5: interface のマージ（Declaration Merging）

```typescript
// 同名のinterfaceは自動的にマージされる
interface Window {
  myCustomProperty: string;
}

// これでグローバルの Window に myCustomProperty が追加される
// ライブラリの型拡張に便利

interface Config {
  host: string;
  port: number;
}

interface Config {
  debug: boolean;      // マージされる
}

// 結果の型: { host: string; port: number; debug: boolean }
const config: Config = {
  host: "localhost",
  port: 3000,
  debug: true,
};
```

### Declaration Merging の実務パターン

```typescript
// パターン1: サードパーティライブラリの型拡張
// Express の Request にカスタムプロパティを追加
declare global {
  namespace Express {
    interface Request {
      user?: {
        id: string;
        name: string;
        role: string;
      };
      requestId: string;
    }
  }
}

// パターン2: 環境変数の型定義
declare global {
  namespace NodeJS {
    interface ProcessEnv {
      NODE_ENV: "development" | "staging" | "production";
      PORT: string;
      DATABASE_URL: string;
      JWT_SECRET: string;
    }
  }
}

// これにより process.env.PORT が string 型として認識される
const port = parseInt(process.env.PORT, 10);

// パターン3: Window オブジェクトの拡張
declare global {
  interface Window {
    __APP_CONFIG__: {
      apiBaseUrl: string;
      featureFlags: Record<string, boolean>;
    };
    analytics: {
      track(event: string, properties?: Record<string, unknown>): void;
    };
  }
}

// パターン4: モジュール拡張
// date-fns のような既存ライブラリに型を追加
declare module "express-session" {
  interface SessionData {
    userId: string;
    loginAt: Date;
  }
}

// パターン5: namespace とのマージ
interface Color {
  r: number;
  g: number;
  b: number;
}

namespace Color {
  export function fromHex(hex: string): Color {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    if (!result) throw new Error("Invalid hex color");
    return {
      r: parseInt(result[1], 16),
      g: parseInt(result[2], 16),
      b: parseInt(result[3], 16),
    };
  }

  export const Red: Color = { r: 255, g: 0, b: 0 };
  export const Green: Color = { r: 0, g: 255, b: 0 };
  export const Blue: Color = { r: 0, g: 0, b: 255 };
}

// interface としても namespace としても使える
const color: Color = Color.fromHex("#ff0000");
const red: Color = Color.Red;
```

---

## 3. type alias

### コード例6: type alias の柔軟性

```typescript
// オブジェクト型
type Point = {
  x: number;
  y: number;
};

// Union型
type Result<T> = { success: true; data: T } | { success: false; error: Error };

// 関数型
type EventHandler = (event: Event) => void;

// タプル型
type Coordinate = [number, number];

// マップ型
type Readonly<T> = { readonly [K in keyof T]: T[K] };

// 条件型
type NonNullable<T> = T extends null | undefined ? never : T;

// テンプレートリテラル型
type HttpMethod = `${"GET" | "POST" | "PUT" | "DELETE"}`;
type Endpoint = `/${string}`;
type ApiRoute = `${HttpMethod} ${Endpoint}`;
```

### type alias の高度なパターン

```typescript
// ===== 条件型の活用 =====
type IsString<T> = T extends string ? true : false;
type A = IsString<"hello">; // true
type B = IsString<42>;      // false

// 条件型による型の抽出
type ExtractArrayType<T> = T extends (infer U)[] ? U : never;
type Elem = ExtractArrayType<string[]>; // string

// Promiseの中身を取得
type UnwrapPromise<T> = T extends Promise<infer U> ? U : T;
type Result = UnwrapPromise<Promise<string>>; // string

// 関数の引数型と戻り値型の抽出
type ParamTypes<T> = T extends (...args: infer P) => unknown ? P : never;
type ReturnType<T> = T extends (...args: unknown[]) => infer R ? R : never;

type Params = ParamTypes<(a: string, b: number) => void>; // [a: string, b: number]
type Ret = ReturnType<(a: string) => boolean>; // boolean

// ===== Mapped Types =====
type Optional<T> = { [K in keyof T]?: T[K] };
type Required<T> = { [K in keyof T]-?: T[K] };
type Mutable<T> = { -readonly [K in keyof T]: T[K] };

// Key Remapping（TypeScript 4.1+）
type Getters<T> = {
  [K in keyof T as `get${Capitalize<string & K>}`]: () => T[K];
};

type Setters<T> = {
  [K in keyof T as `set${Capitalize<string & K>}`]: (value: T[K]) => void;
};

interface Person {
  name: string;
  age: number;
}

type PersonGetters = Getters<Person>;
// { getName: () => string; getAge: () => number }

type PersonSetters = Setters<Person>;
// { setName: (value: string) => void; setAge: (value: number) => void }

// ===== ユーティリティ型の組み合わせ =====
// APIレスポンスのCRUD型を自動生成
type CreateInput<T> = Omit<T, "id" | "createdAt" | "updatedAt">;
type UpdateInput<T> = Partial<Omit<T, "id" | "createdAt" | "updatedAt">>;
type ListResponse<T> = {
  data: T[];
  pagination: {
    page: number;
    perPage: number;
    total: number;
    totalPages: number;
  };
};

interface Article {
  id: string;
  title: string;
  body: string;
  author: string;
  tags: string[];
  createdAt: Date;
  updatedAt: Date;
}

type CreateArticleInput = CreateInput<Article>;
// { title: string; body: string; author: string; tags: string[] }

type UpdateArticleInput = UpdateInput<Article>;
// { title?: string; body?: string; author?: string; tags?: string[] }

type ArticleListResponse = ListResponse<Article>;

// ===== 再帰型 =====
type JSON =
  | string
  | number
  | boolean
  | null
  | JSON[]
  | { [key: string]: JSON };

// 深い読み取り専用
type DeepReadonly<T> = T extends (infer U)[]
  ? ReadonlyArray<DeepReadonly<U>>
  : T extends object
  ? { readonly [K in keyof T]: DeepReadonly<T[K]> }
  : T;

// 深いPartial
type DeepPartial<T> = T extends object
  ? { [K in keyof T]?: DeepPartial<T[K]> }
  : T;

// パスの型安全なアクセス
type Path<T, K extends keyof T> = K extends string
  ? T[K] extends Record<string, unknown>
    ? `${K}.${Path<T[K], keyof T[K]>}` | K
    : K
  : never;
```

### interface vs type alias 比較

| 特性 | interface | type alias |
|------|-----------|------------|
| オブジェクト型 | OK | OK |
| Union型 | 不可 | OK |
| Intersection | extends で継承 | `&` で合成 |
| Declaration Merging | OK（同名で自動マージ） | 不可（重複エラー） |
| implements | OK | OK（一部制限あり） |
| 条件型・マップ型 | 不可 | OK |
| パフォーマンス | やや高速（キャッシュ） | 複雑な型は遅くなる場合あり |
| 推奨シーン | オブジェクト構造、公開API | Union、複雑な型変換 |

### 使い分けの判断フロー

```
  型を定義したい
      |
      v
  Union型が必要？ ----Yes----> type alias
      |
      No
      |
      v
  条件型/マップ型が必要？ ----Yes----> type alias
      |
      No
      |
      v
  Declaration Mergingが必要？ ----Yes----> interface
      |
      No
      |
      v
  オブジェクトの構造定義？ ----Yes----> interface（推奨）
      |                                  または type（好み）
      No
      |
      v
  type alias を使用
```

### 実務でのスタイルガイド

```typescript
// ===== Google TypeScript Style Guide の方針 =====
// interface と type のどちらでも表現できる場合は interface を使う

// interface が適切な場合
interface UserService {
  getUser(id: string): Promise<User>;
  createUser(data: CreateUserInput): Promise<User>;
  updateUser(id: string, data: UpdateUserInput): Promise<User>;
  deleteUser(id: string): Promise<void>;
}

interface ApiResponse<T> {
  data: T;
  status: number;
  message: string;
}

// type が必要な場合
type UserId = string;
type UserRole = "admin" | "editor" | "viewer";
type Result<T, E = Error> = { ok: true; value: T } | { ok: false; error: E };
type Handler = (req: Request, res: Response) => Promise<void>;

// ===== 別のスタイル: 常に type を使う =====
// 一部のチームではシンプルさのため全て type を使う方針もある
type User = {
  id: string;
  name: string;
  email: string;
};

type UserService = {
  getUser(id: string): Promise<User>;
  createUser(data: CreateUserInput): Promise<User>;
};

// どちらのスタイルでも、チーム内で統一することが重要
```

---

## 4. 構造的型付け（Structural Typing）

### コード例7: ダックタイピング

```typescript
interface Point {
  x: number;
  y: number;
}

// Point インターフェースを明示的にimplementsしていなくてもOK
const point = { x: 10, y: 20, z: 30 };

function printPoint(p: Point): void {
  console.log(`(${p.x}, ${p.y})`);
}

// point は x, y を持っているので Point として受け入れられる
printPoint(point); // OK: 構造が一致していれば良い

// 過剰プロパティチェック（直接オブジェクトリテラルの場合のみ）
// printPoint({ x: 10, y: 20, z: 30 }); // エラー: z は Point に存在しない
```

### 構造的型付けの詳細

```typescript
// ===== 構造的型付けの基本原則 =====
// 「必要なプロパティを全て持っていれば、その型として扱える」

interface HasName {
  name: string;
}

interface HasAge {
  age: number;
}

interface Person extends HasName, HasAge {}

// 全く関係のないオブジェクトでも、構造が一致すればOK
const dog = {
  name: "Buddy",
  age: 5,
  breed: "Labrador", // 余分なプロパティ
};

function greetPerson(person: Person): string {
  return `Hello, ${person.name}! You are ${person.age} years old.`;
}

greetPerson(dog); // OK: dog は name と age を持っている

// ===== Excess Property Check（余剰プロパティチェック） =====
// オブジェクトリテラルを直接代入する場合のみ発動

// エラー: オブジェクトリテラルを直接渡す場合
// greetPerson({ name: "Alice", age: 30, extra: true }); // エラー

// OK: 一度変数に代入してから渡す場合
const alice = { name: "Alice", age: 30, extra: true };
greetPerson(alice); // OK

// OK: スプレッド構文で渡す場合
greetPerson({ ...alice }); // OK（これはオブジェクトリテラルだがスプレッドなので...）
// 実際にはこれもエラーになる。正確にはスプレッドもExcess Property Checkの対象

// Excess Property Checkを回避する方法
// 方法1: インデックスシグネチャを追加
interface FlexiblePerson {
  name: string;
  age: number;
  [key: string]: unknown; // 任意のプロパティを許容
}

// 方法2: 変数経由で渡す（上述）
// 方法3: 型アサーション
greetPerson({ name: "Alice", age: 30, extra: true } as Person);

// ===== 関数の構造的互換性 =====
type Handler = (event: MouseEvent) => void;

// パラメータが少ない関数は互換性がある
const simpleHandler: Handler = () => {}; // OK: パラメータを無視
const eventHandler: Handler = (event) => {
  console.log(event.clientX);
}; // OK

// パラメータが多い関数は互換性がない
// const badHandler: Handler = (event: MouseEvent, extra: string) => {}; // エラー

// ===== クラスの構造的互換性 =====
class Cat {
  name: string;
  constructor(name: string) { this.name = name; }
  meow(): void { console.log("Meow!"); }
}

class FakeCat {
  name: string;
  constructor(name: string) { this.name = name; }
  meow(): void { console.log("Fake meow!"); }
}

// FakeCat は Cat と構造が同じなので互換性がある
const cat: Cat = new FakeCat("Kitty"); // OK

// private/protected メンバーがある場合は別
class RealCat {
  private id: number = 0;
  name: string;
  constructor(name: string) { this.name = name; }
}

class AnotherCat {
  private id: number = 0;
  name: string;
  constructor(name: string) { this.name = name; }
}

// private メンバーの出所が異なるため互換性がない
// const realCat: RealCat = new AnotherCat("Kitty"); // エラー
```

### 構造的型付けの図解

```
  名前的型付け (Java, C# など)           構造的型付け (TypeScript)
+----------------------------+    +----------------------------+
| class Dog implements       |    | interface HasName {        |
|   Animal { ... }           |    |   name: string;            |
|                            |    | }                          |
| → Dog は Animal の名前で   |    |                            |
|   型チェック               |    | // { name: string } を持つ |
+----------------------------+    | // 全てのオブジェクトが     |
                                  | // HasName として使える     |
                                  +----------------------------+
```

---

## 5. オブジェクト型の高度なパターン

### インデックスシグネチャの詳細

```typescript
// 基本的なインデックスシグネチャ
interface StringMap {
  [key: string]: string;
}

// 明示的なプロパティとインデックスシグネチャの共存
interface Config {
  name: string;                  // 明示的なプロパティ
  version: number;               // 明示的なプロパティ
  [key: string]: string | number; // インデックスシグネチャ（上のプロパティの型を含む必要がある）
}

// number インデックスシグネチャ
interface StringArray {
  [index: number]: string;
  length: number;
}

// string と number のインデックスシグネチャの共存
interface MixedIndex {
  [key: string]: string | number;
  [index: number]: string; // number インデックスは string インデックスのサブタイプでなければならない
}

// Record型（インデックスシグネチャの代替として推奨）
type UserRoles = Record<string, "admin" | "editor" | "viewer">;

const roles: UserRoles = {
  alice: "admin",
  bob: "editor",
  charlie: "viewer",
};

// Record の応用
type HttpHeaders = Record<string, string | string[]>;
type QueryParams = Record<string, string | number | boolean>;
type Translations = Record<string, Record<string, string>>;

const translations: Translations = {
  en: { greeting: "Hello", farewell: "Goodbye" },
  ja: { greeting: "こんにちは", farewell: "さようなら" },
};
```

### readonly の詳細

```typescript
// ===== readonly プロパティ =====
interface ImmutableUser {
  readonly id: string;
  readonly name: string;
  readonly email: string;
  readonly createdAt: Date;
}

const user: ImmutableUser = {
  id: "1",
  name: "Alice",
  email: "alice@example.com",
  createdAt: new Date(),
};

// user.name = "Bob"; // エラー: readonly プロパティは変更できない

// ===== Readonly<T> ユーティリティ型 =====
interface MutableConfig {
  host: string;
  port: number;
  debug: boolean;
}

type FrozenConfig = Readonly<MutableConfig>;
// { readonly host: string; readonly port: number; readonly debug: boolean }

// ===== readonly の限界 =====
// readonly は浅い（shallow）: ネストしたオブジェクトは変更可能
interface Settings {
  readonly theme: {
    primary: string;
    secondary: string;
  };
}

const settings: Settings = {
  theme: { primary: "#007bff", secondary: "#6c757d" },
};

// settings.theme = { primary: "#000", secondary: "#fff" }; // エラー
settings.theme.primary = "#000"; // OK！（ネストした中身は変更可能）

// 深い readonly を実現する DeepReadonly
type DeepReadonly<T> = T extends (infer U)[]
  ? ReadonlyArray<DeepReadonly<U>>
  : T extends object
  ? { readonly [K in keyof T]: DeepReadonly<T[K]> }
  : T;

type DeepFrozenSettings = DeepReadonly<Settings>;
// 全階層が readonly になる

// ===== const assertion との組み合わせ =====
const CONFIG = {
  api: {
    baseUrl: "https://api.example.com",
    timeout: 5000,
    retries: 3,
  },
  features: {
    darkMode: true,
    notifications: false,
  },
} as const;
// 全プロパティが readonly かつリテラル型
```

### ユーティリティ型の網羅的解説

```typescript
// TypeScript 組み込みのユーティリティ型

// ===== オブジェクト操作 =====

// Partial<T>: 全プロパティをオプショナルに
type PartialUser = Partial<User>;
// { id?: string; name?: string; email?: string }

// Required<T>: 全プロパティを必須に
interface OptionalUser {
  id: string;
  name?: string;
  email?: string;
}
type RequiredUser = Required<OptionalUser>;
// { id: string; name: string; email: string }

// Pick<T, K>: 指定したプロパティのみ取得
type UserName = Pick<User, "name" | "email">;
// { name: string; email: string }

// Omit<T, K>: 指定したプロパティを除外
type UserWithoutId = Omit<User, "id">;
// { name: string; email: string }

// Record<K, V>: キーと値の型を指定したオブジェクト
type StatusMessages = Record<"success" | "error" | "warning", string>;
// { success: string; error: string; warning: string }

// Readonly<T>: 全プロパティを readonly に
type ImmutableUser = Readonly<User>;

// ===== Union操作 =====

// Exclude<T, U>: T から U を除外
type NonString = Exclude<string | number | boolean, string>;
// number | boolean

// Extract<T, U>: T から U に代入可能な型を抽出
type StringOrNumber = Extract<string | number | boolean, string | number>;
// string | number

// NonNullable<T>: null と undefined を除外
type Defined = NonNullable<string | null | undefined>;
// string

// ===== 関数操作 =====

// Parameters<T>: 関数のパラメータ型をタプルで取得
type AddParams = Parameters<typeof add>;
// [a: number, b: number]

// ReturnType<T>: 関数の戻り値型を取得
type AddReturn = ReturnType<typeof add>;
// number

// ConstructorParameters<T>: コンストラクタのパラメータ型
class MyClass {
  constructor(name: string, age: number) {}
}
type CtorParams = ConstructorParameters<typeof MyClass>;
// [name: string, age: number]

// InstanceType<T>: コンストラクタのインスタンス型
type Instance = InstanceType<typeof MyClass>;
// MyClass

// ===== 文字列操作 =====
type Upper = Uppercase<"hello">;        // "HELLO"
type Lower = Lowercase<"HELLO">;        // "hello"
type Cap = Capitalize<"hello">;         // "Hello"
type Uncap = Uncapitalize<"Hello">;     // "hello"

// ===== Promise操作 =====
type AwaitedType = Awaited<Promise<Promise<string>>>;
// string

// ===== その他 =====
// ThisParameterType<T>: this パラメータの型を取得
// OmitThisParameter<T>: this パラメータを除外した関数型
// ThisType<T>: this の型を指定するマーカー型

// NoInfer<T>（TypeScript 5.4+）: 型推論を抑制
function createPair<T>(a: T, b: NoInfer<T>): [T, T] {
  return [a, b];
}
createPair("hello", "world"); // OK
// createPair("hello", 42);   // エラー: T は string と推論され、42 は string に代入不可
```

---

## アンチパターン

### アンチパターン1: 過度にネストした型定義

```typescript
// BAD: インライン型定義が深くネストして読めない
function processOrder(
  order: {
    items: {
      product: { id: number; name: string; price: number };
      quantity: number;
      options?: { gift: boolean; message?: string };
    }[];
    customer: { name: string; address: { street: string; city: string } };
  }
): void { /* ... */ }

// GOOD: 型を分割して名前をつける
interface Address {
  street: string;
  city: string;
}
interface Customer {
  name: string;
  address: Address;
}
interface Product {
  id: number;
  name: string;
  price: number;
}
interface OrderItem {
  product: Product;
  quantity: number;
  options?: { gift: boolean; message?: string };
}
interface Order {
  items: OrderItem[];
  customer: Customer;
}
function processOrder(order: Order): void { /* ... */ }
```

### アンチパターン2: interfaceとtypeの無秩序な混在

```typescript
// BAD: 同じプロジェクト内でinterfaceとtypeを一貫性なく使う
interface User { name: string; }
type Product = { name: string; };    // なぜここだけtype？
interface Order { items: string[]; }
type Invoice = { total: number; };   // 一貫性がない

// GOOD: チームで方針を決めて統一する
// 方針例: オブジェクト構造はinterface、Unionや複雑な型はtype
interface User { name: string; }
interface Product { name: string; }
interface Order { items: string[]; }
type PaymentMethod = "credit" | "debit" | "cash"; // Union はtype
type Result<T> = Success<T> | Failure;             // Union はtype
```

### アンチパターン3: 巨大なinterfaceを作る

```typescript
// BAD: 1つのinterfaceに全てを詰め込む
interface User {
  id: string;
  name: string;
  email: string;
  password: string;
  avatar: string;
  bio: string;
  settings: {
    theme: string;
    language: string;
    notifications: boolean;
  };
  billing: {
    plan: string;
    card: string;
    expiry: string;
  };
  social: {
    twitter: string;
    github: string;
    linkedin: string;
  };
  stats: {
    posts: number;
    followers: number;
    following: number;
  };
  // ... さらに続く
}

// GOOD: 責務ごとにinterfaceを分割する
interface UserIdentity {
  id: string;
  name: string;
  email: string;
}

interface UserCredentials {
  password: string;
}

interface UserProfile {
  avatar: string;
  bio: string;
}

interface UserSettings {
  theme: "light" | "dark" | "system";
  language: string;
  notifications: boolean;
}

interface UserBilling {
  plan: "free" | "pro" | "enterprise";
  card: string;
  expiry: string;
}

interface UserSocial {
  twitter?: string;
  github?: string;
  linkedin?: string;
}

interface UserStats {
  posts: number;
  followers: number;
  following: number;
}

// 必要に応じて組み合わせる
interface User extends UserIdentity, UserProfile, UserSettings {
  billing: UserBilling;
  social: UserSocial;
  stats: UserStats;
}

// 用途に応じて必要な型だけ使う
type PublicUser = UserIdentity & UserProfile & { stats: UserStats };
type AdminUser = User & UserCredentials;
```

### アンチパターン4: 関数のパラメータが多すぎる

```typescript
// BAD: パラメータが多すぎて順序を間違えやすい
function createUser(
  name: string,
  email: string,
  age: number,
  role: string,
  department: string,
  isActive: boolean,
  createdBy: string
): User {
  // ...
}
// 呼び出し時にどの引数がどれか分からない
createUser("Alice", "alice@example.com", 30, "admin", "Engineering", true, "system");

// GOOD: オブジェクト引数を使う
interface CreateUserOptions {
  name: string;
  email: string;
  age: number;
  role: "admin" | "editor" | "viewer";
  department: string;
  isActive?: boolean;  // デフォルト true
  createdBy?: string;  // デフォルト "system"
}

function createUser(options: CreateUserOptions): User {
  const { isActive = true, createdBy = "system", ...rest } = options;
  // ...
}

// 呼び出し時に各フィールドが明確
createUser({
  name: "Alice",
  email: "alice@example.com",
  age: 30,
  role: "admin",
  department: "Engineering",
});
```

### アンチパターン5: 型のコピペ

```typescript
// BAD: 同じ型定義を複数ファイルで重複
// file: user-service.ts
interface User {
  id: string;
  name: string;
  email: string;
}

// file: user-controller.ts（同じ定義をコピペ）
interface User {
  id: string;
  name: string;
  email: string;
}

// GOOD: 共通の型ファイルからインポート
// file: types/user.ts
export interface User {
  id: string;
  name: string;
  email: string;
}

// file: user-service.ts
import type { User } from "../types/user";

// file: user-controller.ts
import type { User } from "../types/user";

// import type を使うことで、ランタイムには含まれないことを明示
```

---

## FAQ

### Q1: 関数のオーバーロードとUnion型パラメータ、どちらを使うべきですか？

**A:** 入力の型に応じて戻り値の型が変わる場合はオーバーロードが適切です。戻り値が同じなら Union型パラメータの方がシンプルです。

```typescript
// Union型で十分なケース
function len(x: string | any[]): number { return x.length; }

// オーバーロードが必要なケース（戻り値の型が変わる）
function parse(input: string): string[];
function parse(input: string[]): string[][];
function parse(input: string | string[]) { /* ... */ }

// ジェネリクスで解決できるケース（オーバーロードより推奨）
function firstElement<T>(arr: T[]): T | undefined {
  return arr[0];
}
// string[] → string | undefined
// number[] → number | undefined
```

### Q2: `readonly` と `Readonly<T>` の違いは？

**A:** `readonly` はプロパティ単位の修飾子で、`Readonly<T>` はオブジェクト全体の全プロパティを一括で readonly にするユーティリティ型です。ネストしたオブジェクトの深い部分までは `Readonly<T>` でも readonly にはなりません。深い immutability が必要な場合はカスタムの `DeepReadonly` 型を定義します。

### Q3: `{}` 型は何を表しますか？

**A:** `{}` は「null と undefined 以外の全ての値」を表します。空オブジェクト型ではありません。空オブジェクトを表したい場合は `Record<string, never>` を使うのが正確です。`{}` は意図せず広い型になるため、避けるべきです。

```typescript
// {} は null/undefined 以外の全てを受け入れる
const a: {} = "hello";     // OK
const b: {} = 42;          // OK
const c: {} = true;        // OK
const d: {} = { foo: 1 };  // OK
// const e: {} = null;     // エラー
// const f: {} = undefined; // エラー

// 空オブジェクトを表す正しい方法
type EmptyObject = Record<string, never>;
const empty: EmptyObject = {};
// const notEmpty: EmptyObject = { key: "value" }; // エラー

// object 型は非プリミティブを表す
const g: object = { foo: 1 }; // OK
const h: object = [1, 2, 3];  // OK
// const i: object = "hello";  // エラー（プリミティブ）
// const j: object = 42;       // エラー（プリミティブ）
```

### Q4: interfaceの継承(extends)とIntersection(&)の違いは？

**A:** 機能的にはほぼ同じですが、重要な違いがあります。

```typescript
// extends: プロパティが衝突するとコンパイルエラー
interface A { x: number; }
// interface B extends A { x: string; } // エラー: x の型が互換性なし

// Intersection(&): プロパティが衝突すると never になる
type A = { x: number; };
type B = { x: string; };
type C = A & B;
// C.x の型は number & string = never（使い物にならない）

// extends の方がエラーを早期に検出できるため、推奨
```

### Q5: コンストラクタ型とは何ですか？

**A:** `new` キーワードで呼び出せる関数の型です。クラスをファクトリ関数に渡す場合などに使います。

```typescript
interface Constructor<T> {
  new (...args: any[]): T;
}

function createInstance<T>(ctor: Constructor<T>): T {
  return new ctor();
}

class MyService {
  name = "service";
}

const instance = createInstance(MyService); // MyService
console.log(instance.name); // "service"

// abstract クラスは Constructor に代入できない
// abstract class を含める場合は Function を使う
type AbstractConstructor<T> = abstract new (...args: any[]) => T;
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| 関数型 | パラメータ型と戻り値型を明示。推論にも頼れる |
| オプショナル引数 | `?` で省略可能に。デフォルト値も指定可 |
| オーバーロード | 入力に応じて戻り値型を変えたいときに使う |
| this型 | メソッドチェーンやコンテキストの型安全性を確保 |
| interface | オブジェクト構造の定義。継承・マージが可能 |
| type alias | 柔軟な型定義。Union、条件型、マップ型に必須 |
| 構造的型付け | 名前ではなく構造で型の互換性を判定 |
| 使い分け | オブジェクト→interface、Union/複雑な型→type |
| readonly | 浅い不変性。深い不変性にはDeepReadonlyが必要 |
| ユーティリティ型 | Partial, Pick, Omit, Record等で型を効率的に操作 |

---

## 次に読むべきガイド

- [03-union-intersection.md](./03-union-intersection.md) -- Union型とIntersection型
- [04-generics.md](./04-generics.md) -- ジェネリクス

---

## 参考文献

1. **TypeScript Handbook: More on Functions** -- https://www.typescriptlang.org/docs/handbook/2/functions.html
2. **TypeScript Handbook: Object Types** -- https://www.typescriptlang.org/docs/handbook/2/objects.html
3. **TypeScript Handbook: Type Manipulation** -- https://www.typescriptlang.org/docs/handbook/2/types-from-types.html
4. **Effective TypeScript, Item 13: Know the Differences Between type and interface** -- Dan Vanderkam著, O'Reilly
5. **TypeScript Deep Dive: Functions** -- https://basarat.gitbook.io/typescript/type-system/functions
6. **Google TypeScript Style Guide** -- https://google.github.io/styleguide/tsguide.html
