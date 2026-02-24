# ジェネリクス

> 型パラメータを使い、再利用可能かつ型安全なコードを書く。制約、条件型、型推論の仕組みを理解する。

## この章で学ぶこと

1. **ジェネリクスの基本** -- 型パラメータ、型引数の推論、ジェネリック関数・クラス・インターフェース
2. **型制約（constraints）** -- extends による型パラメータの制約、複数制約
3. **ジェネリクスの応用** -- 条件型との組み合わせ、デフォルト型パラメータ、型推論（infer）
4. **実践パターン** -- Repository, Result型, Builder, 型安全なイベント, ファクトリパターン
5. **高度なジェネリクス** -- 再帰型、分配条件型、可変長タプル

---

## 1. ジェネリクスの基本

ジェネリクスは、TypeScriptにおける「型の変数化」の仕組みである。関数やクラスを定義する際に具体的な型を指定する代わりに型パラメータを使い、呼び出し時に実際の型が決まるようにする。これにより、同じロジックを異なる型に対して再利用でき、かつ型安全性を維持できる。

### なぜジェネリクスが必要なのか

ジェネリクスがないと、以下の2つの選択肢しかない：

```typescript
// 選択肢1: 型ごとに関数を定義（コード重複）
function identityString(value: string): string { return value; }
function identityNumber(value: number): number { return value; }
function identityBoolean(value: boolean): boolean { return value; }
// ... 型が増えるたびに関数が増える

// 選択肢2: any を使う（型安全性の喪失）
function identityAny(value: any): any { return value; }
const result = identityAny("hello"); // result の型は any → 型情報が失われる

// ジェネリクスによる解決: 再利用性と型安全性の両立
function identity<T>(value: T): T { return value; }
const result = identity("hello"); // result の型は string
```

### コード例1: ジェネリック関数

```typescript
// 型パラメータ T を使った汎用関数
function identity<T>(value: T): T {
  return value;
}

// 型引数の明示的指定
const str = identity<string>("hello");  // 型: string
const num = identity<number>(42);       // 型: number

// 型引数の推論（多くの場合、明示不要）
const inferred = identity("hello");     // 型: string（推論される）
```

### 型推論エンジンの動作

```
  呼び出し: identity("hello")
                |
                v
  型推論エンジン:
    1. T は引数の型から推論される
    2. "hello" の型は string
    3. よって T = string
                |
                v
  インスタンス化: identity<string>(value: string): string
                |
                v
  結果の型: string

  ────────────────────────────────

  呼び出し: identity<number>(42)
                |
                v
  型パラメータが明示的に指定されている
    T = number（明示的）
                |
                v
  引数チェック: 42 は number → OK
                |
                v
  結果の型: number
```

### コード例1b: アロー関数でのジェネリクス

```typescript
// 通常のアロー関数
const identity = <T>(value: T): T => value;

// .tsx ファイルでの回避策（JSX タグとの混同を防ぐ）
const identity1 = <T,>(value: T): T => value;           // 末尾カンマ
const identity2 = <T extends unknown>(value: T): T => value; // extends

// 複数型パラメータ
const pair = <A, B>(a: A, b: B): [A, B] => [a, b];

// 制約付き
const getLength = <T extends { length: number }>(value: T): number => value.length;
```

### コード例2: 実用的なジェネリック関数

```typescript
// 配列の最初の要素を返す
function first<T>(arr: T[]): T | undefined {
  return arr[0];
}

first([1, 2, 3]);       // 型: number | undefined
first(["a", "b"]);      // 型: string | undefined

// ペアを作る
function pair<A, B>(a: A, b: B): [A, B] {
  return [a, b];
}

const p = pair("name", 42);  // 型: [string, number]

// オブジェクトからキーの値を取得
function getProperty<T, K extends keyof T>(obj: T, key: K): T[K] {
  return obj[key];
}

const user = { name: "Alice", age: 30 };
const name = getProperty(user, "name");  // 型: string
const age = getProperty(user, "age");    // 型: number
// getProperty(user, "email");            // エラー: "email" は keyof User にない
```

### コード例2b: さらに実用的なジェネリック関数

```typescript
// 配列をグループ化する
function groupBy<T, K extends string | number>(
  items: T[],
  getKey: (item: T) => K,
): Record<K, T[]> {
  const result = {} as Record<K, T[]>;
  for (const item of items) {
    const key = getKey(item);
    if (!result[key]) {
      result[key] = [];
    }
    result[key].push(item);
  }
  return result;
}

interface Product {
  name: string;
  category: string;
  price: number;
}

const products: Product[] = [
  { name: "Apple", category: "fruit", price: 100 },
  { name: "Banana", category: "fruit", price: 80 },
  { name: "Carrot", category: "vegetable", price: 120 },
];

const grouped = groupBy(products, (p) => p.category);
// 型: Record<string, Product[]>
// { fruit: [...], vegetable: [...] }

// 配列のユニーク要素を取得
function unique<T>(items: T[], getKey?: (item: T) => unknown): T[] {
  if (!getKey) {
    return [...new Set(items)];
  }
  const seen = new Set();
  return items.filter((item) => {
    const key = getKey(item);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

// プロミスのリトライ
async function retry<T>(
  fn: () => Promise<T>,
  options: { maxRetries: number; delayMs: number },
): Promise<T> {
  let lastError: unknown;
  for (let i = 0; i <= options.maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      if (i < options.maxRetries) {
        await new Promise((r) => setTimeout(r, options.delayMs));
      }
    }
  }
  throw lastError;
}

// 型安全なメモ化
function memoize<Args extends unknown[], R>(
  fn: (...args: Args) => R,
): (...args: Args) => R {
  const cache = new Map<string, R>();
  return (...args: Args): R => {
    const key = JSON.stringify(args);
    if (cache.has(key)) {
      return cache.get(key)!;
    }
    const result = fn(...args);
    cache.set(key, result);
    return result;
  };
}

const memoizedFetch = memoize(async (url: string) => {
  const res = await fetch(url);
  return res.json();
});
```

---

## 2. ジェネリクスの様々な形

### コード例3: ジェネリックインターフェースとクラス

```typescript
// ジェネリックインターフェース
interface Repository<T> {
  findById(id: string): Promise<T | null>;
  findAll(): Promise<T[]>;
  save(entity: T): Promise<T>;
  delete(id: string): Promise<void>;
}

// ジェネリッククラス
class InMemoryRepository<T extends { id: string }> implements Repository<T> {
  private items: Map<string, T> = new Map();

  async findById(id: string): Promise<T | null> {
    return this.items.get(id) ?? null;
  }

  async findAll(): Promise<T[]> {
    return Array.from(this.items.values());
  }

  async save(entity: T): Promise<T> {
    this.items.set(entity.id, entity);
    return entity;
  }

  async delete(id: string): Promise<void> {
    this.items.delete(id);
  }
}

// 使用例
interface User {
  id: string;
  name: string;
}

const userRepo = new InMemoryRepository<User>();
await userRepo.save({ id: "1", name: "Alice" });
const user = await userRepo.findById("1");  // 型: User | null
```

### コード例3b: ジェネリッククラスの高度なパターン

```typescript
// --- パターン1: 型安全なスタック ---
class Stack<T> {
  private items: T[] = [];

  push(item: T): void {
    this.items.push(item);
  }

  pop(): T | undefined {
    return this.items.pop();
  }

  peek(): T | undefined {
    return this.items[this.items.length - 1];
  }

  isEmpty(): boolean {
    return this.items.length === 0;
  }

  get size(): number {
    return this.items.length;
  }

  // イテレータサポート
  *[Symbol.iterator](): IterableIterator<T> {
    for (let i = this.items.length - 1; i >= 0; i--) {
      yield this.items[i];
    }
  }
}

const numberStack = new Stack<number>();
numberStack.push(1);
numberStack.push(2);
numberStack.pop(); // 型: number | undefined

// --- パターン2: 型安全なLinkedList ---
class LinkedListNode<T> {
  constructor(
    public value: T,
    public next: LinkedListNode<T> | null = null,
  ) {}
}

class LinkedList<T> {
  private head: LinkedListNode<T> | null = null;
  private _size = 0;

  prepend(value: T): void {
    this.head = new LinkedListNode(value, this.head);
    this._size++;
  }

  find(predicate: (value: T) => boolean): T | undefined {
    let current = this.head;
    while (current) {
      if (predicate(current.value)) return current.value;
      current = current.next;
    }
    return undefined;
  }

  toArray(): T[] {
    const result: T[] = [];
    let current = this.head;
    while (current) {
      result.push(current.value);
      current = current.next;
    }
    return result;
  }

  get size(): number {
    return this._size;
  }
}

// --- パターン3: Observable / EventEmitter ---
class TypedObservable<T> {
  private observers: ((value: T) => void)[] = [];

  subscribe(observer: (value: T) => void): () => void {
    this.observers.push(observer);
    // unsubscribe 関数を返す
    return () => {
      this.observers = this.observers.filter((o) => o !== observer);
    };
  }

  notify(value: T): void {
    for (const observer of this.observers) {
      observer(value);
    }
  }
}

const priceUpdates = new TypedObservable<{ symbol: string; price: number }>();
const unsub = priceUpdates.subscribe((data) => {
  // data: { symbol: string; price: number }
  console.log(`${data.symbol}: $${data.price}`);
});

priceUpdates.notify({ symbol: "AAPL", price: 150.5 });
unsub(); // 購読解除
```

### コード例4: ジェネリック型エイリアス

```typescript
// 非同期操作の結果型
type AsyncResult<T> = {
  data: T | null;
  loading: boolean;
  error: Error | null;
};

// APIレスポンス
type ApiResponse<T> = {
  status: number;
  data: T;
  timestamp: string;
};

// ページネーション付きレスポンス
type Paginated<T> = {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasNext: boolean;
};

// 使用例
type UserListResponse = ApiResponse<Paginated<User>>;
```

### コード例4b: 型エイリアスの高度なパターン

```typescript
// --- パターン1: Result型（Railway指向プログラミング） ---
type Result<T, E = Error> =
  | { ok: true; value: T }
  | { ok: false; error: E };

function ok<T>(value: T): Result<T, never> {
  return { ok: true, value };
}

function err<E>(error: E): Result<never, E> {
  return { ok: false, error };
}

function map<T, U, E>(
  result: Result<T, E>,
  fn: (value: T) => U,
): Result<U, E> {
  if (result.ok) {
    return ok(fn(result.value));
  }
  return result;
}

function flatMap<T, U, E>(
  result: Result<T, E>,
  fn: (value: T) => Result<U, E>,
): Result<U, E> {
  if (result.ok) {
    return fn(result.value);
  }
  return result;
}

// 使用例: バリデーションのチェーン
function validateAge(age: number): Result<number, string> {
  if (age < 0 || age > 150) return err("Invalid age");
  return ok(age);
}

function validateName(name: string): Result<string, string> {
  if (name.length === 0) return err("Name is required");
  if (name.length > 100) return err("Name is too long");
  return ok(name.trim());
}

// --- パターン2: DeepPartial ---
type DeepPartial<T> = T extends object
  ? { [K in keyof T]?: DeepPartial<T[K]> }
  : T;

interface AppConfig {
  database: {
    host: string;
    port: number;
    credentials: {
      username: string;
      password: string;
    };
  };
  cache: {
    enabled: boolean;
    ttlSeconds: number;
  };
}

// 一部のネストされたプロパティのみ上書き
function mergeConfig(
  base: AppConfig,
  overrides: DeepPartial<AppConfig>,
): AppConfig {
  // deep merge 実装
  return { ...base, ...overrides } as AppConfig;
}

// --- パターン3: DeepReadonly ---
type DeepReadonly<T> = T extends Function
  ? T
  : T extends object
    ? { readonly [K in keyof T]: DeepReadonly<T[K]> }
    : T;

const config: DeepReadonly<AppConfig> = {
  database: {
    host: "localhost",
    port: 5432,
    credentials: { username: "admin", password: "secret" },
  },
  cache: { enabled: true, ttlSeconds: 3600 },
};

// config.database.host = "other"; // エラー: readonly
// config.cache.enabled = false;   // エラー: readonly

// --- パターン4: Nullable<T> ---
type Nullable<T> = { [K in keyof T]: T[K] | null };

interface UserForm {
  name: string;
  email: string;
  bio: string;
}

type NullableUserForm = Nullable<UserForm>;
// { name: string | null; email: string | null; bio: string | null }
```

### ジェネリクスの適用箇所

```
+------------------+---------------------------+---------------------+
| 適用箇所         | 構文                      | 例                  |
+------------------+---------------------------+---------------------+
| 関数             | function fn<T>(x: T): T   | identity<T>         |
| アロー関数       | const fn = <T>(x: T): T   | <T>(x: T) => T     |
| インターフェース | interface I<T> { ... }    | Repository<T>       |
| クラス           | class C<T> { ... }        | Stack<T>            |
| 型エイリアス     | type T<U> = { ... }       | Result<T>           |
| メソッド         | method<T>(x: T): T        | Array#map           |
+------------------+---------------------------+---------------------+
```

---

## 3. 型制約（Constraints）

型制約は、ジェネリック型パラメータが満たすべき条件を指定する仕組みである。`extends` キーワードを使って「Tはこの型のサブタイプでなければならない」という制約を課す。

### コード例5: extends による制約

```typescript
// T は { length: number } を持つ型に制約される
function logLength<T extends { length: number }>(value: T): T {
  console.log(`Length: ${value.length}`);
  return value;
}

logLength("hello");       // OK: string は length を持つ
logLength([1, 2, 3]);     // OK: number[] は length を持つ
logLength({ length: 10 }); // OK
// logLength(42);          // エラー: number は length を持たない

// keyof による制約
function pick<T, K extends keyof T>(obj: T, keys: K[]): Pick<T, K> {
  const result = {} as Pick<T, K>;
  for (const key of keys) {
    result[key] = obj[key];
  }
  return result;
}

const user = { id: 1, name: "Alice", email: "alice@test.com", age: 30 };
const picked = pick(user, ["name", "email"]);
// 型: { name: string; email: string }
```

### コード例5b: 制約の実践パターン

```typescript
// --- パターン1: Comparable な型に制約 ---
interface Comparable<T> {
  compareTo(other: T): number;
}

function max<T extends Comparable<T>>(a: T, b: T): T {
  return a.compareTo(b) >= 0 ? a : b;
}

class Money implements Comparable<Money> {
  constructor(public readonly amount: number, public readonly currency: string) {}

  compareTo(other: Money): number {
    if (this.currency !== other.currency) {
      throw new Error("Cannot compare different currencies");
    }
    return this.amount - other.amount;
  }
}

const a = new Money(100, "USD");
const b = new Money(200, "USD");
const result = max(a, b); // 型: Money

// --- パターン2: コンストラクタ制約 ---
type Constructor<T = unknown> = new (...args: any[]) => T;

function createInstance<T>(ctor: Constructor<T>, ...args: any[]): T {
  return new ctor(...args);
}

class UserEntity {
  constructor(public name: string, public email: string) {}
}

const user2 = createInstance(UserEntity, "Alice", "alice@test.com");
// 型: UserEntity

// --- パターン3: Record制約 ---
function mergeObjects<
  T extends Record<string, unknown>,
  U extends Record<string, unknown>,
>(a: T, b: U): T & U {
  return { ...a, ...b };
}

const merged = mergeObjects(
  { name: "Alice", age: 30 },
  { email: "alice@test.com", active: true },
);
// 型: { name: string; age: number } & { email: string; active: boolean }

// --- パターン4: 再帰的制約 ---
interface TreeNode<T extends TreeNode<T>> {
  parent: T | null;
  children: T[];
}

class DOMElement implements TreeNode<DOMElement> {
  parent: DOMElement | null = null;
  children: DOMElement[] = [];
  tag: string;

  constructor(tag: string) {
    this.tag = tag;
  }

  appendChild(child: DOMElement): void {
    child.parent = this;
    this.children.push(child);
  }
}
```

### コード例6: 複数の型パラメータと制約

```typescript
// マージ関数
function merge<
  T extends Record<string, unknown>,
  U extends Record<string, unknown>
>(target: T, source: U): T & U {
  return { ...target, ...source };
}

const merged = merge(
  { name: "Alice" },
  { age: 30 }
);
// 型: { name: string } & { age: number }

// デフォルト型パラメータ
interface FetchOptions<T = unknown> {
  url: string;
  method?: "GET" | "POST";
  body?: T;
}

// T を指定しなければ unknown になる
const opts: FetchOptions = { url: "/api/users" };
const opts2: FetchOptions<{ name: string }> = {
  url: "/api/users",
  method: "POST",
  body: { name: "Alice" },
};
```

### コード例6b: 条件付き型パラメータのデフォルト

```typescript
// デフォルト型パラメータの活用
type Container<T = string> = {
  value: T;
  toString(): string;
};

// T を省略すると string
const c1: Container = { value: "hello", toString: () => "hello" };
// T を指定
const c2: Container<number> = { value: 42, toString: () => "42" };

// 複数のデフォルト型パラメータ
type ApiCall<
  TResponse = unknown,
  TError = Error,
  TParams = Record<string, string>,
> = {
  execute(params: TParams): Promise<TResponse>;
  onError(handler: (error: TError) => void): void;
};

// 一部だけ指定（前から順に）
type SimpleCall = ApiCall<{ data: string }>;
// TResponse = { data: string }, TError = Error, TParams = Record<string, string>

type FullCall = ApiCall<User[], string, { page: number }>;
// 全て指定
```

### 制約の階層図

```
  <T>                     制約なし（全ての型を受け入れる）
    |
    v
  <T extends object>     オブジェクト型に制限
    |
    v
  <T extends Record<string, unknown>>  文字列キーのオブジェクト
    |
    v
  <T extends { id: number }>   id プロパティを持つ型に制限
    |
    v
  <T extends User>        User 型を満たす型に制限
    |
    v
  <T extends Admin>       Admin（extends User）を満たす型に制限

  制約が強くなるほど:
  - 受け入れる型の範囲が狭まる
  - 型パラメータ内で使えるプロパティ/メソッドが増える
  - 型安全性が向上する
```

---

## 4. ジェネリクスの応用

### コード例7: 条件型と型推論（infer）

```typescript
// Promiseの中身を取り出す型
type Unwrap<T> = T extends Promise<infer U> ? U : T;

type A = Unwrap<Promise<string>>;  // string
type B = Unwrap<Promise<number>>;  // number
type C = Unwrap<string>;           // string（Promiseでなければそのまま）

// 深くネストしたPromiseも再帰的にアンラップ
type DeepUnwrap<T> = T extends Promise<infer U> ? DeepUnwrap<U> : T;

type D = DeepUnwrap<Promise<Promise<Promise<string>>>>; // string

// 関数の戻り値型を取り出す（ReturnType相当）
type MyReturnType<T> = T extends (...args: any[]) => infer R ? R : never;

type E = MyReturnType<() => string>;            // string
type F = MyReturnType<(x: number) => boolean>;  // boolean

// 関数の引数型を取り出す（Parameters相当）
type MyParameters<T> = T extends (...args: infer P) => any ? P : never;

type G = MyParameters<(a: string, b: number) => void>; // [string, number]

// 配列の要素型を取り出す
type ElementOf<T> = T extends (infer U)[] ? U : never;

type H = ElementOf<string[]>;     // string
type I = ElementOf<number[]>;     // number
```

### コード例7b: inferの高度な活用

```typescript
// --- コンストラクタの引数型を取り出す ---
type ConstructorArgs<T> = T extends new (...args: infer A) => any ? A : never;

class UserEntity {
  constructor(public name: string, public age: number) {}
}

type UserArgs = ConstructorArgs<typeof UserEntity>; // [string, number]

// --- オブジェクト型からメソッドのみ抽出 ---
type MethodsOf<T> = {
  [K in keyof T as T[K] extends Function ? K : never]: T[K];
};

interface UserService {
  name: string;
  getUser(id: string): Promise<User>;
  createUser(data: Omit<User, "id">): Promise<User>;
  maxRetries: number;
}

type UserServiceMethods = MethodsOf<UserService>;
// {
//   getUser: (id: string) => Promise<User>;
//   createUser: (data: Omit<User, "id">) => Promise<User>;
// }

// --- テンプレートリテラル型からの推論 ---
type ParseRoute<T extends string> =
  T extends `${string}:${infer Param}/${infer Rest}`
    ? Param | ParseRoute<Rest>
    : T extends `${string}:${infer Param}`
      ? Param
      : never;

type RouteParams = ParseRoute<"/users/:userId/posts/:postId">;
// "userId" | "postId"

// --- Promiseの型からasync関数の型を構築 ---
type AsyncFunction<T extends (...args: any[]) => any> =
  ReturnType<T> extends Promise<any>
    ? T
    : (...args: Parameters<T>) => Promise<ReturnType<T>>;

function toAsync<T extends (...args: any[]) => any>(
  fn: T,
): AsyncFunction<T> {
  return (async (...args: Parameters<T>) => {
    return fn(...args);
  }) as AsyncFunction<T>;
}
```

### コード例8: ジェネリクスとマップ型

```typescript
// 全プロパティをオプショナルかつnullableにする
type NullablePartial<T> = {
  [K in keyof T]?: T[K] | null;
};

interface User {
  id: number;
  name: string;
  email: string;
}

type UpdateUserInput = NullablePartial<User>;
// { id?: number | null; name?: string | null; email?: string | null }

// イベントマップからハンドラ型を生成
type EventMap = {
  click: { x: number; y: number };
  keypress: { key: string; code: number };
  scroll: { offset: number };
};

type EventHandlers<T> = {
  [K in keyof T as `on${Capitalize<string & K>}`]: (event: T[K]) => void;
};

type Handlers = EventHandlers<EventMap>;
// {
//   onClick: (event: { x: number; y: number }) => void;
//   onKeypress: (event: { key: string; code: number }) => void;
//   onScroll: (event: { offset: number }) => void;
// }
```

### コード例8b: マップ型の高度なパターン

```typescript
// --- パターン1: Getter/Setter の自動生成 ---
type Getters<T> = {
  [K in keyof T as `get${Capitalize<string & K>}`]: () => T[K];
};

type Setters<T> = {
  [K in keyof T as `set${Capitalize<string & K>}`]: (value: T[K]) => void;
};

interface UserProps {
  name: string;
  age: number;
  active: boolean;
}

type UserGetters = Getters<UserProps>;
// {
//   getName: () => string;
//   getAge: () => number;
//   getActive: () => boolean;
// }

type UserSetters = Setters<UserProps>;
// {
//   setName: (value: string) => void;
//   setAge: (value: number) => void;
//   setActive: (value: boolean) => void;
// }

// --- パターン2: Readonly を特定のキーだけに適用 ---
type ReadonlyPick<T, K extends keyof T> = {
  readonly [P in K]: T[P];
} & {
  [P in Exclude<keyof T, K>]: T[P];
};

type UserWithReadonlyId = ReadonlyPick<User, "id">;
// { readonly id: number; name: string; email: string }

// --- パターン3: 条件付きプロパティの変換 ---
type StringToNumber<T> = {
  [K in keyof T]: T[K] extends string ? number : T[K];
};

interface RawData {
  name: string;
  count: string;   // 文字列として受信
  active: boolean;
}

type ParsedData = StringToNumber<RawData>;
// { name: number; count: number; active: boolean }
// ↑ name も number になってしまう。より精密な制御が必要:

type ParseNumericStrings<T, K extends keyof T> = {
  [P in keyof T]: P extends K
    ? T[P] extends string ? number : T[P]
    : T[P];
};

type BetterParsed = ParseNumericStrings<RawData, "count">;
// { name: string; count: number; active: boolean }

// --- パターン4: フィルタリング ---
type FilterByType<T, ValueType> = {
  [K in keyof T as T[K] extends ValueType ? K : never]: T[K];
};

interface Config {
  host: string;
  port: number;
  debug: boolean;
  maxConnections: number;
  name: string;
}

type StringConfigs = FilterByType<Config, string>;
// { host: string; name: string }

type NumberConfigs = FilterByType<Config, number>;
// { port: number; maxConnections: number }
```

---

## 5. 可変長タプルと高度なジェネリクス

### コード例9: 可変長タプル型（Variadic Tuple Types）

TypeScript 4.0 で導入された可変長タプル型により、タプルの型パラメータにスプレッド構文を使えるようになった。

```typescript
// 基本的な可変長タプル
type Prepend<T, Tuple extends unknown[]> = [T, ...Tuple];
type Append<Tuple extends unknown[], T> = [...Tuple, T];

type A = Prepend<string, [number, boolean]>; // [string, number, boolean]
type B = Append<[number, boolean], string>;  // [number, boolean, string]

// 複数の配列を連結する型
type Concat<A extends unknown[], B extends unknown[]> = [...A, ...B];
type C = Concat<[1, 2], [3, 4]>; // [1, 2, 3, 4]

// 実用例: 型安全な関数合成
function compose<A extends unknown[], B, C>(
  f: (arg: B) => C,
  g: (...args: A) => B,
): (...args: A) => C {
  return (...args: A) => f(g(...args));
}

const toNumber = (s: string): number => parseInt(s, 10);
const add = (a: number, b: number): number => a + b;

const addStrings = compose(String, add);
// 型: (a: number, b: number) => string
addStrings(1, 2); // "3"

// pipe 関数
function pipe<A, B>(value: A, fn1: (a: A) => B): B;
function pipe<A, B, C>(value: A, fn1: (a: A) => B, fn2: (b: B) => C): C;
function pipe<A, B, C, D>(
  value: A,
  fn1: (a: A) => B,
  fn2: (b: B) => C,
  fn3: (c: C) => D,
): D;
function pipe(value: unknown, ...fns: Function[]): unknown {
  return fns.reduce((acc, fn) => fn(acc), value);
}

const result = pipe(
  " Hello, World! ",
  (s: string) => s.trim(),
  (s: string) => s.toLowerCase(),
  (s: string) => s.split(", "),
);
// 型: string[]
// 値: ["hello", "world!"]
```

### コード例10: 再帰型

```typescript
// --- JSON型の定義 ---
type JsonPrimitive = string | number | boolean | null;
type JsonArray = JsonValue[];
type JsonObject = { [key: string]: JsonValue };
type JsonValue = JsonPrimitive | JsonArray | JsonObject;

// --- パスによるネストアクセスの型 ---
type PathOf<T, Prefix extends string = ""> = T extends object
  ? {
      [K in keyof T & string]: T[K] extends object
        ? `${Prefix}${K}` | PathOf<T[K], `${Prefix}${K}.`>
        : `${Prefix}${K}`;
    }[keyof T & string]
  : never;

interface AppState {
  user: {
    name: string;
    settings: {
      theme: "light" | "dark";
      notifications: boolean;
    };
  };
  cart: {
    items: string[];
    total: number;
  };
}

type StatePaths = PathOf<AppState>;
// "user" | "user.name" | "user.settings" | "user.settings.theme"
// | "user.settings.notifications" | "cart" | "cart.items" | "cart.total"

// --- 型レベルの数値演算（型安全なカウンタ） ---
type BuildTuple<N extends number, T extends unknown[] = []> =
  T["length"] extends N ? T : BuildTuple<N, [...T, unknown]>;

type Add<A extends number, B extends number> =
  [...BuildTuple<A>, ...BuildTuple<B>]["length"];

type Sum = Add<3, 4>; // 7
```

---

## 型制約パターン比較

| パターン | 構文 | 用途 |
|----------|------|------|
| 上界制約 | `<T extends U>` | T を U のサブタイプに制限 |
| keyof制約 | `<K extends keyof T>` | K を T のキーに制限 |
| 複数制約 | `<T extends A & B>` | T を A かつ B を満たす型に制限 |
| デフォルト型 | `<T = DefaultType>` | 型引数省略時のデフォルト |
| 条件型 | `T extends U ? X : Y` | 型レベルの条件分岐 |
| 推論 | `T extends X<infer U>` | 構造から型を抽出 |
| コンストラクタ | `<T extends new (...) => any>` | クラスコンストラクタに制限 |
| 関数 | `<T extends (...) => any>` | 関数型に制限 |

---

## 6. 実践パターン集

### パターン1: 型安全なAPIクライアント

```typescript
// エンドポイント定義
interface ApiEndpoints {
  "GET /users": {
    params: { page?: number; limit?: number };
    response: { users: User[]; total: number };
  };
  "GET /users/:id": {
    params: { id: string };
    response: User;
  };
  "POST /users": {
    body: { name: string; email: string };
    response: User;
  };
  "PUT /users/:id": {
    params: { id: string };
    body: Partial<User>;
    response: User;
  };
  "DELETE /users/:id": {
    params: { id: string };
    response: { success: boolean };
  };
}

// 型安全なfetch関数
type ExtractMethod<T extends string> = T extends `${infer M} ${string}` ? M : never;
type ExtractPath<T extends string> = T extends `${string} ${infer P}` ? P : never;

type HasBody<T> = T extends { body: infer B } ? B : never;
type HasParams<T> = T extends { params: infer P } ? P : Record<string, never>;
type GetResponse<T> = T extends { response: infer R } ? R : never;

async function apiCall<K extends keyof ApiEndpoints>(
  endpoint: K,
  options: Omit<ApiEndpoints[K], "response">,
): Promise<GetResponse<ApiEndpoints[K]>> {
  // 実装は省略
  return {} as any;
}

// 使用例: 型安全なAPI呼び出し
const users = await apiCall("GET /users", { params: { page: 1, limit: 20 } });
// 型: { users: User[]; total: number }

const user = await apiCall("POST /users", { body: { name: "Alice", email: "a@test.com" } });
// 型: User
```

### パターン2: 型安全なDIコンテナ

```typescript
class Token<T> {
  // ブランド型として機能
  private readonly _brand: T = undefined!;
  constructor(public readonly name: string) {}
}

class DIContainer {
  private bindings = new Map<Token<any>, () => any>();

  bind<T>(token: Token<T>, factory: () => T): void {
    this.bindings.set(token, factory);
  }

  resolve<T>(token: Token<T>): T {
    const factory = this.bindings.get(token);
    if (!factory) {
      throw new Error(`No binding for token: ${token.name}`);
    }
    return factory() as T;
  }
}

// トークン定義
const LoggerToken = new Token<ILogger>("Logger");
const UserRepoToken = new Token<IUserRepository>("UserRepo");

const container = new DIContainer();
container.bind(LoggerToken, () => new ConsoleLogger());
container.bind(UserRepoToken, () => new PostgresUserRepo());

const logger = container.resolve(LoggerToken); // 型: ILogger
const repo = container.resolve(UserRepoToken); // 型: IUserRepository
```

### パターン3: 型安全なフォームバリデーション

```typescript
type ValidationRule<T> = {
  validate: (value: T) => boolean;
  message: string;
};

type FormSchema<T extends Record<string, unknown>> = {
  [K in keyof T]: ValidationRule<T[K]>[];
};

type FormErrors<T extends Record<string, unknown>> = {
  [K in keyof T]?: string[];
};

function validateForm<T extends Record<string, unknown>>(
  data: T,
  schema: FormSchema<T>,
): FormErrors<T> {
  const errors: FormErrors<T> = {};

  for (const key in schema) {
    const rules = schema[key];
    const value = data[key];
    const fieldErrors: string[] = [];

    for (const rule of rules) {
      if (!rule.validate(value)) {
        fieldErrors.push(rule.message);
      }
    }

    if (fieldErrors.length > 0) {
      errors[key] = fieldErrors;
    }
  }

  return errors;
}

// 使用例
interface LoginForm {
  email: string;
  password: string;
}

const loginSchema: FormSchema<LoginForm> = {
  email: [
    { validate: (v) => v.length > 0, message: "メールアドレスは必須です" },
    { validate: (v) => v.includes("@"), message: "有効なメールアドレスを入力してください" },
  ],
  password: [
    { validate: (v) => v.length >= 8, message: "パスワードは8文字以上です" },
    { validate: (v) => /[A-Z]/.test(v), message: "大文字を含めてください" },
  ],
};

const errors = validateForm(
  { email: "test", password: "abc" },
  loginSchema,
);
// errors.email?: string[] | undefined
// errors.password?: string[] | undefined
```

---

## アンチパターン

### アンチパターン1: 不要なジェネリクス

```typescript
// BAD: T を使っていないのにジェネリクスにしている
function greet<T>(name: string): string {
  return `Hello, ${name}`;
}

// BAD: ジェネリクスが過剰（T を返さないので不要）
function getLength<T extends { length: number }>(x: T): number {
  return x.length;
}
// GOOD: ジェネリクス不要
function getLength(x: { length: number }): number {
  return x.length;
}

// ジェネリクスが必要なケースの判断基準:
// 1. 入力型と出力型を関連付ける場合 → 必要
//    function first<T>(arr: T[]): T | undefined
// 2. 複数の引数の型を関連付ける場合 → 必要
//    function merge<T>(a: T, b: Partial<T>): T
// 3. 入力型の情報を出力に伝播させない場合 → 不要
//    function getLength(x: { length: number }): number
```

### アンチパターン2: any で制約を回避

```typescript
// BAD: 制約エラーを any で黙らせる
function merge<T>(a: T, b: T): T {
  return { ...(a as any), ...(b as any) } as T;
}

// GOOD: 適切な制約をつける
function merge<T extends Record<string, unknown>>(a: T, b: Partial<T>): T {
  return { ...a, ...b };
}
```

### アンチパターン3: 型パラメータの名前が不明瞭

```typescript
// BAD: 何を表しているか分からない
function process<A, B, C, D>(a: A, b: B): C {
  // ...
}

// GOOD: 意味のある名前を使う
function transform<TInput, TOutput>(
  input: TInput,
  transformer: (item: TInput) => TOutput,
): TOutput {
  return transformer(input);
}

// 型パラメータの命名慣例:
// T, U, V        - 汎用的な型（1〜3個の場合）
// TInput, TOutput - 入出力の関係
// TKey, TValue    - キーと値の関係
// TEntity, TDto   - ドメインオブジェクト
// K extends keyof T - オブジェクトのキー
// E               - エラー型 or 要素型
// R               - 戻り値型
```

### アンチパターン4: ジェネリクスの過度なネスト

```typescript
// BAD: 読みにくいネスト
type Complex<T> = Promise<Result<Array<Partial<Readonly<T>>>>>;

// GOOD: 中間型を定義して段階的に構築
type ReadonlyPartial<T> = Readonly<Partial<T>>;
type ResultList<T> = Result<ReadonlyPartial<T>[]>;
type AsyncResultList<T> = Promise<ResultList<T>>;
```

---

## FAQ

### Q1: `<T>` と `<T extends unknown>` は同じですか？

**A:** はい、実質同じです。全ての型は `unknown` のサブタイプなので、`<T>` と `<T extends unknown>` は等価です。ただし、`<T extends object>` とすると null, undefined, プリミティブは除外されます。

### Q2: ジェネリクスの型パラメータ名の慣例は？

**A:** 一般的な慣例:
- `T` (Type): 汎用的な型
- `K` (Key): キーの型
- `V` (Value): 値の型
- `E` (Element / Error): 要素型やエラー型
- `R` (Return / Result): 戻り値型

複雑な場合は `TInput`, `TOutput` のように説明的な名前を使うことも推奨されます。

### Q3: アロー関数でジェネリクスを使うとJSXと衝突しませんか？

**A:** `.tsx` ファイルでは `<T>` が JSX タグと誤認される場合があります。回避策として `<T,>` や `<T extends unknown>` を使います。
```typescript
// .tsx ファイルでの回避策
const identity = <T,>(value: T): T => value;
const identity = <T extends unknown>(value: T): T => value;
```

### Q4: ジェネリクスはランタイムに影響しますか？

**A:** いいえ。ジェネリクスはコンパイル時にのみ存在し、JavaScript に変換された後は完全に消去されます（type erasure）。ランタイムのパフォーマンスへの影響はゼロです。

### Q5: ジェネリクスと共変性・反変性の関係は？

**A:** TypeScriptの型パラメータの変性（variance）は使用位置で決まります：
- **共変（covariant）**: 出力位置（戻り値型）→ サブタイプの方向に互換
- **反変（contravariant）**: 入力位置（引数型）→ スーパータイプの方向に互換
- **不変（invariant）**: 入出力の両方で使用 → 完全一致が必要

```typescript
// TypeScript 4.7+ では in/out 修飾子で変性を明示できる
interface Producer<out T> {  // T は共変
  produce(): T;
}

interface Consumer<in T> {  // T は反変
  consume(value: T): void;
}

interface Processor<in out T> {  // T は不変
  process(value: T): T;
}
```

### Q6: 型パラメータにデフォルト値を指定する場合の注意点は？

**A:** デフォルト型パラメータは末尾に配置する必要があります（関数のデフォルト引数と同様）。また、デフォルト値は制約を満たす必要があります。

```typescript
// OK: デフォルト型は制約を満たしている
type Container<T extends object = Record<string, unknown>> = { data: T };

// エラー: デフォルト型 string が object 制約を満たさない
// type Bad<T extends object = string> = { data: T };

// OK: デフォルト型パラメータは末尾
type Result<T, E = Error> = { value: T } | { error: E };

// エラー: デフォルト型パラメータの後に必須パラメータは置けない
// type Bad<T = string, U> = { a: T; b: U };
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| ジェネリクスとは | 型をパラメータ化し、再利用可能な型安全コードを書く仕組み |
| 型推論 | 多くの場合、TypeScriptが型引数を自動推論する |
| 制約 (extends) | 型パラメータが満たすべき条件を指定する |
| デフォルト型 | 型引数を省略した場合の既定値を設定 |
| infer | 条件型の中で型を抽出するキーワード |
| 適用箇所 | 関数、クラス、インターフェース、型エイリアス |
| 設計指針 | 不要なジェネリクスは避け、型を返す場合に使う |
| 可変長タプル | スプレッド構文でタプルの型を合成 |
| 再帰型 | 自己参照する型定義で木構造やパス型を表現 |
| 変性 | in/out 修飾子で型パラメータの変性を明示 |

---

## 演習問題

### 問題1: 汎用的なキャッシュクラス

以下の仕様を満たすジェネリックな `Cache<K, V>` クラスを実装してください。

- `get(key: K): V | undefined` -- キーに対応する値を取得
- `set(key: K, value: V, ttlMs?: number)` -- 値をセット（オプションでTTL指定）
- `has(key: K): boolean` -- キーが存在するか確認
- `delete(key: K): boolean` -- キーを削除
- TTLが切れたエントリは自動的に無効になること

```typescript
class Cache<K, V> {
  // ここに実装を書いてください
}
```

### 問題2: 型安全なEventEmitter

以下のイベント定義から型安全なEventEmitterを実装してください。`on` で登録したイベント名と `emit` で発火するイベント名の型チェック、およびペイロードの型チェックが行われること。

```typescript
type Events = {
  "user:login": { userId: string; timestamp: Date };
  "user:logout": { userId: string };
  "error": { code: number; message: string };
};

class TypedEmitter<E extends Record<string, unknown>> {
  // on, emit, off, once を実装
}
```

### 問題3: DeepPick の実装

ネストされたオブジェクト型から、ドット区切りのパスで指定したプロパティのみを抽出する `DeepPick` 型を実装してください。

```typescript
interface User {
  id: string;
  name: string;
  address: {
    city: string;
    zip: string;
    country: {
      code: string;
      name: string;
    };
  };
}

type Result = DeepPick<User, "name" | "address.city" | "address.country.code">;
// {
//   name: string;
//   address: {
//     city: string;
//     country: {
//       code: string;
//     };
//   };
// }
```

### 問題4: パイプライン関数

任意の数の変換関数を受け取り、左から右に適用する `pipe` 関数を実装してください。各関数の入力型と前の関数の出力型が一致することを型レベルで保証すること。

```typescript
function pipe<A>(value: A): A;
function pipe<A, B>(value: A, fn1: (a: A) => B): B;
function pipe<A, B, C>(value: A, fn1: (a: A) => B, fn2: (b: B) => C): C;
// ... オーバーロードを追加

// 使用例
const result = pipe(
  "  Hello, World!  ",
  (s) => s.trim(),
  (s) => s.toLowerCase(),
  (s) => s.split(" "),
  (arr) => arr.length,
);
// result: number (= 2)
```

### 問題5: 型安全な状態管理

以下の仕様を満たすジェネリックな状態管理クラスを実装してください。

- 初期状態を受け取りストアを作成する
- `getState()` で現在の状態を取得
- `setState(updater: (state: T) => T)` で状態を更新
- `subscribe(listener: (state: T) => void)` で変更を監視
- `select<U>(selector: (state: T) => U)` で部分的な状態を取得

```typescript
class Store<T extends Record<string, unknown>> {
  // ここに実装を書いてください
}
```

---

## 次に読むべきガイド

- [../01-advanced-types/00-conditional-types.md](../01-advanced-types/00-conditional-types.md) -- 条件型（詳細）
- [../01-advanced-types/01-mapped-types.md](../01-advanced-types/01-mapped-types.md) -- マップ型（詳細）
- [../02-patterns/00-error-handling.md](../02-patterns/00-error-handling.md) -- Result型の実践

---

## 参考文献

1. **TypeScript Handbook: Generics** -- https://www.typescriptlang.org/docs/handbook/2/generics.html
2. **TypeScript Deep Dive: Generics** -- https://basarat.gitbook.io/typescript/type-system/generics
3. **Effective TypeScript, Item 26: Understand How Context Is Used in Type Inference** -- Dan Vanderkam著, O'Reilly
4. **Programming TypeScript** -- Boris Cherny著, O'Reilly. Chapter 4: Functions (Generics section)
5. **TypeScript 4.0: Variadic Tuple Types** -- https://www.typescriptlang.org/docs/handbook/release-notes/typescript-4-0.html
