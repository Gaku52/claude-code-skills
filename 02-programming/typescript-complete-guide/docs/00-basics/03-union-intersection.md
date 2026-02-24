# Union型とIntersection型

> 型を「または」「かつ」で組み合わせる強力な仕組み。判別共用体と型ガードによる安全な型の絞り込みを習得する。

## この章で学ぶこと

1. **Union型** -- `|` 演算子による型の合成、判別共用体、型の絞り込み
2. **Intersection型** -- `&` 演算子による型の合成、ミックスインパターン
3. **型ガード** -- typeof, instanceof, in, ユーザー定義型ガードによるナローイング
4. **網羅性チェック** -- never型を活用した安全な分岐処理
5. **実践パターン** -- 実務で頻出するUnion/Intersection型の設計手法

---

## 1. Union型

Union型は、ある値が「複数の型のいずれか」であることを表現する型システムの基本機能である。集合論的に言えば「和集合（union）」に相当し、型 A または型 B のどちらかの値を受け入れる。

### なぜUnion型が重要なのか

実務のプログラミングでは、関数が複数の型の引数を受け取ったり、APIレスポンスが成功時と失敗時で異なる構造を返したりする場面が頻繁に発生する。Union型を使うことで、これらの「複数の可能性」を型レベルで正確に表現でき、型ガードと組み合わせることで各分岐先でのプロパティアクセスを安全に行える。

### コード例1: 基本的なUnion型

```typescript
// 文字列または数値を受け取る
function formatId(id: string | number): string {
  if (typeof id === "string") {
    return id.toUpperCase();
  }
  return id.toString().padStart(6, "0");
}

formatId("abc");  // "ABC"
formatId(42);     // "000042"

// Union型の変数
let value: string | number | boolean;
value = "hello";  // OK
value = 42;       // OK
value = true;     // OK
// value = [];    // エラー: Type 'never[]' is not assignable to type 'string | number | boolean'
```

### コード例1b: Union型でのメソッドアクセス制限

Union型の変数に対しては、**すべての構成型に共通するメンバー**にのみアクセスできる。これは型安全性を保つための重要な制約である。

```typescript
function describe(value: string | number): string {
  // toString() は string と number の両方に存在するため OK
  return value.toString();

  // value.toUpperCase() はエラー
  // → number には toUpperCase が存在しない

  // value.toFixed(2) もエラー
  // → string には toFixed が存在しない
}

// Union型の配列
type StringOrNumber = string | number;
const mixed: StringOrNumber[] = [1, "two", 3, "four"];

// 配列メソッドは使えるが、要素の型は string | number
mixed.forEach((item) => {
  console.log(item.toString()); // OK: 共通メソッド
  // console.log(item.toUpperCase()); // エラー
});
```

### コード例1c: リテラル型のUnion

```typescript
// 文字列リテラル型のUnion（列挙的な使い方）
type Direction = "north" | "south" | "east" | "west";
type HttpMethod = "GET" | "POST" | "PUT" | "DELETE" | "PATCH";
type LogLevel = "debug" | "info" | "warn" | "error";

function move(direction: Direction): void {
  // direction は4つの文字列リテラルのいずれか
  console.log(`Moving ${direction}`);
}

move("north"); // OK
// move("up"); // エラー: Argument of type '"up"' is not assignable

// 数値リテラル型のUnion
type DiceValue = 1 | 2 | 3 | 4 | 5 | 6;
type HttpStatusCode = 200 | 201 | 400 | 401 | 403 | 404 | 500;

// テンプレートリテラル型とUnionの組み合わせ
type EventName = `on${Capitalize<"click" | "hover" | "focus">}`;
// "onClick" | "onHover" | "onFocus"

type CSSUnit = `${number}${"px" | "em" | "rem" | "%"}`;
// "10px", "1.5em", "100%" などを許容
```

### Union型と型推論

```typescript
// TypeScriptはコンテキストからUnion型を推論する
const arr = [1, "hello", true]; // (string | number | boolean)[]

// 条件式でのUnion型推論
function getValue(flag: boolean) {
  return flag ? "text" : 42;
}
// 戻り値型: string | number

// as const でリテラル型のUnionを得る
const ROLES = ["admin", "user", "guest"] as const;
type Role = (typeof ROLES)[number]; // "admin" | "user" | "guest"
```

---

### コード例2: 判別共用体（Discriminated Unions）

判別共用体は、Union型の中で最も重要かつ実践的なパターンである。共通の「判別子（discriminant）」プロパティを持つオブジェクト型のUnionで、switch文やif文で安全に型を絞り込める。

```typescript
// 共通のリテラル型プロパティ（判別子）を持つUnion
interface Circle {
  kind: "circle";
  radius: number;
}

interface Rectangle {
  kind: "rectangle";
  width: number;
  height: number;
}

interface Triangle {
  kind: "triangle";
  base: number;
  height: number;
}

type Shape = Circle | Rectangle | Triangle;

function area(shape: Shape): number {
  switch (shape.kind) {
    case "circle":
      return Math.PI * shape.radius ** 2;
    case "rectangle":
      return shape.width * shape.height;
    case "triangle":
      return (shape.base * shape.height) / 2;
  }
}
```

### 判別共用体の構造

```
  Shape (Union型)
  +-----------+--------------+--------------+
  |  Circle   |  Rectangle   |  Triangle    |
  +-----------+--------------+--------------+
  | kind:     | kind:        | kind:        |
  | "circle"  | "rectangle"  | "triangle"   |
  | radius    | width        | base         |
  |           | height       | height       |
  +-----------+--------------+--------------+
       ^             ^              ^
       |             |              |
   kind = "circle"  kind = "rect"  kind = "tri"
   → radius が      → width,      → base,
     利用可能         height が      height が
                      利用可能       利用可能
```

### コード例2b: 実務における判別共用体

APIレスポンス、状態管理、イベントハンドリングなど、判別共用体は多岐にわたって活用される。

```typescript
// --- パターン1: API レスポンス ---
type ApiResponse<T> =
  | { status: "success"; data: T; timestamp: string }
  | { status: "error"; error: { code: string; message: string }; timestamp: string }
  | { status: "loading" };

function handleResponse<T>(response: ApiResponse<T>): void {
  switch (response.status) {
    case "success":
      console.log("Data:", response.data);
      console.log("At:", response.timestamp);
      break;
    case "error":
      console.error(`Error ${response.error.code}: ${response.error.message}`);
      break;
    case "loading":
      console.log("Loading...");
      break;
  }
}

// --- パターン2: Redux アクション ---
type UserAction =
  | { type: "USER_FETCH_REQUEST" }
  | { type: "USER_FETCH_SUCCESS"; payload: User[] }
  | { type: "USER_FETCH_FAILURE"; error: string }
  | { type: "USER_CREATE"; payload: Omit<User, "id"> }
  | { type: "USER_UPDATE"; payload: { id: string; changes: Partial<User> } }
  | { type: "USER_DELETE"; payload: { id: string } };

function userReducer(state: UserState, action: UserAction): UserState {
  switch (action.type) {
    case "USER_FETCH_REQUEST":
      return { ...state, loading: true, error: null };
    case "USER_FETCH_SUCCESS":
      return { ...state, loading: false, users: action.payload };
    case "USER_FETCH_FAILURE":
      return { ...state, loading: false, error: action.error };
    case "USER_CREATE":
      // action.payload は Omit<User, "id"> 型
      return state;
    case "USER_UPDATE":
      // action.payload.id と action.payload.changes にアクセス可能
      return state;
    case "USER_DELETE":
      return {
        ...state,
        users: state.users.filter((u) => u.id !== action.payload.id),
      };
  }
}

// --- パターン3: フォームフィールドバリデーション結果 ---
type ValidationResult =
  | { valid: true }
  | { valid: false; errors: string[] };

function validateEmail(email: string): ValidationResult {
  const errors: string[] = [];

  if (!email.includes("@")) {
    errors.push("メールアドレスには @ が必要です");
  }
  if (email.length > 255) {
    errors.push("255文字以内で入力してください");
  }

  return errors.length > 0
    ? { valid: false, errors }
    : { valid: true };
}

const result = validateEmail("test");
if (!result.valid) {
  // result.errors にアクセス可能（型安全）
  result.errors.forEach((err) => console.log(err));
}
```

### コード例2c: 複数の判別子を持つパターン

```typescript
// 2つの判別子を組み合わせる例
type Notification =
  | { channel: "email"; priority: "high"; to: string; subject: string; body: string }
  | { channel: "email"; priority: "low"; to: string; body: string }
  | { channel: "sms"; priority: "high"; phoneNumber: string; message: string }
  | { channel: "push"; priority: "high" | "low"; userId: string; title: string };

function sendNotification(notification: Notification): void {
  switch (notification.channel) {
    case "email":
      if (notification.priority === "high") {
        // subject にアクセス可能
        console.log(`[URGENT] ${notification.subject}: ${notification.body}`);
      } else {
        console.log(notification.body);
      }
      break;
    case "sms":
      console.log(`SMS to ${notification.phoneNumber}: ${notification.message}`);
      break;
    case "push":
      console.log(`Push to ${notification.userId}: ${notification.title}`);
      break;
  }
}
```

### 判別共用体のベストプラクティス

```
判別共用体の設計チェックリスト:

  [1] 判別子はリテラル型であること
      ✓ kind: "circle"        （文字列リテラル）
      ✓ type: 1               （数値リテラル）
      ✓ success: true          （booleanリテラル）
      ✗ kind: string           （広すぎる）

  [2] 判別子のプロパティ名はUnion全体で統一する
      ✓ { kind: "a", ... } | { kind: "b", ... }
      ✗ { kind: "a", ... } | { type: "b", ... }

  [3] 判別子の値はUnion内で一意であること
      ✓ { kind: "circle" } | { kind: "rect" }
      ✗ { kind: "shape" }  | { kind: "shape" }

  [4] 網羅性チェック（exhaustiveness check）を必ず入れる
```

---

## 2. Intersection型

Intersection型は、複数の型を「すべて満たす」型を作成する。集合論的には「積集合（intersection）」に相当する。Union型が「AまたはB」なのに対し、Intersection型は「AかつB」を意味する。

### コード例3: Intersection型の基本

```typescript
// 型の合成（全てのプロパティを持つ）
type HasId = { id: number };
type HasName = { name: string };
type HasEmail = { email: string };

type User = HasId & HasName & HasEmail;
// { id: number; name: string; email: string }

const user: User = {
  id: 1,
  name: "Alice",
  email: "alice@example.com",
};

// ミックスインパターン
type Timestamped = {
  createdAt: Date;
  updatedAt: Date;
};

type SoftDeletable = {
  deletedAt: Date | null;
};

type BaseEntity = HasId & Timestamped & SoftDeletable;
// { id: number; createdAt: Date; updatedAt: Date; deletedAt: Date | null }
```

### コード例3b: Intersection型の実用パターン

```typescript
// --- パターン1: 関心事の分離と合成 ---
type WithPagination = {
  page: number;
  pageSize: number;
  totalPages: number;
  totalItems: number;
};

type WithSorting = {
  sortBy: string;
  sortOrder: "asc" | "desc";
};

type WithFiltering = {
  filters: Record<string, string | number | boolean>;
};

// 必要な機能を組み合わせてリスト取得の型を構築
type PaginatedSortedList<T> = {
  items: T[];
} & WithPagination & WithSorting;

type FullFeaturedList<T> = {
  items: T[];
} & WithPagination & WithSorting & WithFiltering;

const userList: PaginatedSortedList<User> = {
  items: [{ id: 1, name: "Alice", email: "a@example.com" }],
  page: 1,
  pageSize: 20,
  totalPages: 5,
  totalItems: 100,
  sortBy: "createdAt",
  sortOrder: "desc",
};

// --- パターン2: ロール別の権限拡張 ---
type BasePermissions = {
  canRead: boolean;
  canWrite: boolean;
};

type AdminPermissions = BasePermissions & {
  canDelete: boolean;
  canManageUsers: boolean;
  canAccessLogs: boolean;
};

type SuperAdminPermissions = AdminPermissions & {
  canModifySettings: boolean;
  canDeployApp: boolean;
};

// --- パターン3: React コンポーネントのProps合成 ---
type WithClassName = {
  className?: string;
};

type WithTestId = {
  "data-testid"?: string;
};

type WithDisabled = {
  disabled?: boolean;
};

type ButtonProps = {
  label: string;
  onClick: () => void;
  variant: "primary" | "secondary" | "danger";
} & WithClassName & WithTestId & WithDisabled;

// --- パターン4: イベントハンドラのメタデータ付与 ---
type EventMetadata = {
  timestamp: number;
  source: string;
  correlationId: string;
};

type UserCreatedEvent = EventMetadata & {
  type: "user.created";
  data: { userId: string; email: string };
};

type OrderPlacedEvent = EventMetadata & {
  type: "order.placed";
  data: { orderId: string; amount: number };
};

type AppEvent = UserCreatedEvent | OrderPlacedEvent;
```

### Union vs Intersection 比較

| 特性 | Union (`A \| B`) | Intersection (`A & B`) |
|------|-------------------|------------------------|
| 意味 | A **または** B | A **かつ** B |
| プロパティ | 共通のプロパティのみアクセス可 | 全てのプロパティにアクセス可 |
| 集合論 | 和集合 | 積集合 |
| 値の範囲 | 広がる（受け入れが緩い） | 狭まる（要件が厳しい） |
| 型の範囲 | 広い（どちらかを満たせばOK） | 狭い（全てを満たす必要） |
| 使用場面 | 複数の可能性を表す | 型の合成・拡張 |
| 代入互換性 | 各メンバー型はUnion型に代入可能 | Intersection型は各メンバー型に代入可能 |

```
  Union (A | B)               Intersection (A & B)
+-------+-------+           +-------+
|       |  A&B  |           |  A&B  |
|   A   |       |   B      +-------+
|       +-------+           A の全プロパティ
+-------+       |           かつ
        |       |           B の全プロパティ
        +-------+           を持つ
A または B の値
```

### Intersection型で起きる型の矛盾

```typescript
// プリミティブ型同士のIntersectionは never になる
type Impossible1 = string & number;      // never
type Impossible2 = "hello" & "world";    // never
type Impossible3 = true & false;         // never

// オブジェクト型で同名プロパティの型が矛盾する場合
type A = { x: string; shared: number };
type B = { y: boolean; shared: string };
type C = A & B;
// C = { x: string; y: boolean; shared: never }
// shared は string & number = never
// → C型の値を作ることは実質不可能

// これを避けるにはOmitで矛盾するプロパティを除外する
type SafeMerge = A & Omit<B, "shared">;
// { x: string; shared: number; y: boolean }
```

### Intersection型と関数型の合成

```typescript
// 関数型のIntersection = オーバーロード
type NumberToString = (x: number) => string;
type StringToNumber = (x: string) => number;

type Combined = NumberToString & StringToNumber;
// オーバーロードのように振る舞う
// (x: number) => string
// (x: string) => number

declare const fn: Combined;
fn(42);      // string を返す
fn("hello"); // number を返す
```

---

## 3. 型ガードとナローイング

型ガードは、Union型の変数を特定の型に「絞り込む（narrow）」ための仕組みである。TypeScriptのコントロールフロー解析がif文やswitch文の条件を追跡し、各ブロック内での変数の型を自動的に狭める。

### ナローイングの概念図

```
  function handle(x: string | number | null) {

  x の型: string | number | null
      |
      v
  if (x === null) return;
      |
      v
  x の型: string | number    ← null が除外された
      |
      v
  if (typeof x === "string") {
      |
      v
    x の型: string             ← number が除外された
  } else {
      |
      v
    x の型: number             ← string が除外された
  }
```

### コード例4: 組み込み型ガード

```typescript
function process(value: string | number | boolean | Date) {
  // typeof ガード
  if (typeof value === "string") {
    console.log(value.toUpperCase()); // string
    return;
  }

  if (typeof value === "number") {
    console.log(value.toFixed(2));     // number
    return;
  }

  if (typeof value === "boolean") {
    console.log(value ? "yes" : "no"); // boolean
    return;
  }

  // この時点で value は Date 型に絞り込まれている
  console.log(value.toISOString());    // Date
}

// instanceof ガード
function formatError(error: Error | string): string {
  if (error instanceof Error) {
    return error.message;   // Error
  }
  return error;             // string
}

// in ガード
interface Dog { bark(): void; breed: string; }
interface Cat { meow(): void; indoor: boolean; }

function speak(pet: Dog | Cat): void {
  if ("bark" in pet) {
    pet.bark();   // Dog
  } else {
    pet.meow();   // Cat
  }
}
```

### コード例4b: typeofガードの完全リファレンス

```typescript
// typeof で判定できる型は7種類
function typeofDemo(value: unknown): string {
  switch (typeof value) {
    case "string":
      return `String: ${value.toUpperCase()}`;
    case "number":
      return `Number: ${value.toFixed(2)}`;
    case "boolean":
      return `Boolean: ${value ? "true" : "false"}`;
    case "bigint":
      return `BigInt: ${value.toString()}`;
    case "symbol":
      return `Symbol: ${value.toString()}`;
    case "undefined":
      return "Undefined";
    case "function":
      return `Function: ${value.name}`;
    case "object":
      if (value === null) return "Null";
      if (Array.isArray(value)) return `Array[${value.length}]`;
      return `Object: ${JSON.stringify(value)}`;
    default:
      return "Unknown";
  }
}

// typeof の注意点
typeof null === "object";        // JavaScript の歴史的バグ
typeof [] === "object";          // 配列もobject
typeof new Date() === "object";  // Dateもobject
// → これらの判別には instanceof や Array.isArray を使う
```

### コード例4c: 真偽値チェックによるナローイング

```typescript
// TypeScript は truthy/falsy チェックでも型を絞り込む
function processOptional(value: string | null | undefined): string {
  // falsy チェックで null と undefined を除外
  if (!value) {
    return "default";
  }
  // value は string 型（null, undefined, "" が除外される）
  return value.toUpperCase();
}

// !! による真偽値変換
function hasValue(x: string | null | undefined): x is string {
  return !!x; // null, undefined, "" は false
}

// nullish coalescing と optional chaining
type Config = {
  database?: {
    host?: string;
    port?: number;
  };
};

function getDbHost(config: Config): string {
  return config.database?.host ?? "localhost";
}
```

### コード例5: ユーザー定義型ガード

```typescript
// 型述語 (Type Predicate): `value is Type`
interface Fish { swim(): void; kind: "fish"; }
interface Bird { fly(): void; kind: "bird"; }

function isFish(pet: Fish | Bird): pet is Fish {
  return "swim" in pet;
}

function move(pet: Fish | Bird): void {
  if (isFish(pet)) {
    pet.swim();  // Fish として使える
  } else {
    pet.fly();   // Bird として使える
  }
}

// アサーション関数: asserts
function assertIsString(value: unknown): asserts value is string {
  if (typeof value !== "string") {
    throw new Error(`Expected string, got ${typeof value}`);
  }
}

function processInput(input: unknown): void {
  assertIsString(input);
  // この時点で input は string 型
  console.log(input.toUpperCase());
}
```

### コード例5b: 実践的なユーザー定義型ガード

```typescript
// --- パターン1: APIレスポンスの型ガード ---
interface SuccessResponse<T> {
  success: true;
  data: T;
}

interface ErrorResponse {
  success: false;
  error: { code: string; message: string };
}

type ApiResult<T> = SuccessResponse<T> | ErrorResponse;

function isSuccess<T>(result: ApiResult<T>): result is SuccessResponse<T> {
  return result.success === true;
}

function isError<T>(result: ApiResult<T>): result is ErrorResponse {
  return result.success === false;
}

async function fetchUser(id: string): Promise<User> {
  const result: ApiResult<User> = await fetch(`/api/users/${id}`).then(
    (r) => r.json()
  );

  if (isError(result)) {
    throw new Error(result.error.message); // ErrorResponse として型安全
  }

  return result.data; // SuccessResponse<User> として型安全
}

// --- パターン2: 配列フィルタリングでの型ガード ---
type MaybeUser = User | null | undefined;

function isUser(value: MaybeUser): value is User {
  return value != null;
}

const mixedResults: MaybeUser[] = [
  { id: "1", name: "Alice", email: "a@test.com" },
  null,
  { id: "2", name: "Bob", email: "b@test.com" },
  undefined,
];

// filter + 型ガードで型安全に null/undefined を除去
const validUsers: User[] = mixedResults.filter(isUser);
// validUsers の型は User[]（null | undefined が除外されている）

// --- パターン3: unknown型の安全な処理 ---
interface JsonObject {
  [key: string]: unknown;
}

function isJsonObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function parseConfig(raw: unknown): Record<string, string> {
  if (!isJsonObject(raw)) {
    throw new Error("Config must be an object");
  }

  const result: Record<string, string> = {};
  for (const [key, value] of Object.entries(raw)) {
    if (typeof value === "string") {
      result[key] = value;
    }
  }
  return result;
}

// --- パターン4: asserts でのバリデーション ---
function assertPositive(value: number): asserts value is number {
  if (value <= 0) {
    throw new RangeError(`Expected positive number, got ${value}`);
  }
}

function assertNonEmpty(arr: unknown[]): asserts arr is [unknown, ...unknown[]] {
  if (arr.length === 0) {
    throw new Error("Array must not be empty");
  }
}

function processOrder(quantity: number, items: string[]): void {
  assertPositive(quantity);
  assertNonEmpty(items);
  // quantity > 0 が保証されている
  // items は少なくとも1要素ある
  console.log(`Processing ${quantity} of ${items[0]}`);
}
```

### 型ガードの一覧と使い分け

| 型ガード | 構文 | 適用対象 | 用途 |
|---------|------|---------|------|
| typeof | `typeof x === "string"` | プリミティブ型 | string, number, boolean, symbol, bigint, undefined, function |
| instanceof | `x instanceof Error` | クラスインスタンス | Error, Date, RegExp, カスタムクラス |
| in | `"key" in x` | オブジェクトのプロパティ | プロパティ有無で型を判別 |
| Array.isArray | `Array.isArray(x)` | 配列 | 配列かオブジェクトかの判別 |
| 等値チェック | `x === null` | リテラル型 | null, undefined, 特定の文字列 |
| 型述語 | `x is Type` | カスタム判定 | 複雑な条件での型絞り込み |
| asserts | `asserts x is Type` | アサーション | 条件を満たさない場合に例外送出 |

```
型ガードの選択フローチャート:

  絞り込みたい型は？
      |
      +-- プリミティブ → typeof
      |
      +-- クラスインスタンス → instanceof
      |
      +-- 配列 → Array.isArray
      |
      +-- null / undefined → 等値チェック (=== null)
      |
      +-- オブジェクトの構造 → in演算子 or ユーザー定義型ガード
      |
      +-- 複雑な条件 → ユーザー定義型ガード (is / asserts)
```

---

## 4. 網羅性チェック

### コード例6: never を使った網羅性チェック

判別共用体のswitch文で、全ケースを処理したことをコンパイラに保証させるパターン。新しいメンバーが追加された際にコンパイルエラーとなり、処理漏れを防止できる。

```typescript
type Status = "pending" | "approved" | "rejected";

function handleStatus(status: Status): string {
  switch (status) {
    case "pending":
      return "審査中です";
    case "approved":
      return "承認されました";
    case "rejected":
      return "却下されました";
    default:
      // 全てのケースを処理した場合、ここに到達しない
      // 新しいStatusが追加された場合、コンパイルエラーになる
      const _exhaustive: never = status;
      throw new Error(`Unknown status: ${_exhaustive}`);
  }
}

// assertNever ヘルパー関数
function assertNever(value: never): never {
  throw new Error(`Unexpected value: ${value}`);
}
```

### コード例6b: 網羅性チェックの応用

```typescript
// --- パターン1: 複雑な判別共用体での網羅性チェック ---
type PaymentMethod =
  | { type: "credit_card"; cardNumber: string; expiry: string }
  | { type: "bank_transfer"; bankCode: string; accountNumber: string }
  | { type: "crypto"; walletAddress: string; network: "ethereum" | "bitcoin" }
  | { type: "paypal"; email: string };

function processPayment(payment: PaymentMethod): string {
  switch (payment.type) {
    case "credit_card":
      return `Credit Card ending in ${payment.cardNumber.slice(-4)}`;
    case "bank_transfer":
      return `Bank Transfer to ${payment.bankCode}`;
    case "crypto":
      return `Crypto payment to ${payment.walletAddress} on ${payment.network}`;
    case "paypal":
      return `PayPal payment to ${payment.email}`;
    default:
      return assertNever(payment);
      // もし新しい payment type を追加したら、
      // ここでコンパイルエラーが発生する
  }
}

// --- パターン2: if-else チェーンでの網羅性チェック ---
type Animal = { kind: "dog"; breed: string } | { kind: "cat"; indoor: boolean } | { kind: "bird"; canFly: boolean };

function describeAnimal(animal: Animal): string {
  if (animal.kind === "dog") {
    return `Dog (${animal.breed})`;
  }
  if (animal.kind === "cat") {
    return `Cat (${animal.indoor ? "indoor" : "outdoor"})`;
  }
  if (animal.kind === "bird") {
    return `Bird (${animal.canFly ? "can fly" : "cannot fly"})`;
  }
  // TypeScript は animal が never であることを認識
  return assertNever(animal);
}

// --- パターン3: マップオブジェクトによる網羅性チェック ---
type Fruit = "apple" | "banana" | "cherry";

// Record<Fruit, T> は全ての Fruit をキーに持つことを要求する
const fruitEmoji: Record<Fruit, string> = {
  apple: "apple",
  banana: "banana",
  cherry: "cherry",
  // grape: "grape" ← Fruit に含まれないのでエラー
  // cherry を削除するとエラー（全キーが必要）
};

// satisfies を使ったさらに柔軟な網羅性チェック（TypeScript 4.9+）
const fruitColors = {
  apple: "red",
  banana: "yellow",
  cherry: "dark red",
} satisfies Record<Fruit, string>;
// fruitColors.apple は "red" リテラル型（Record では string に広がる）
```

---

## 5. Intersection型の高度な利用

### コード例7: Intersection型の高度な利用

```typescript
// 条件付きプロパティの合成
type BaseConfig = {
  host: string;
  port: number;
};

type WithAuth = {
  auth: {
    username: string;
    password: string;
  };
};

type WithSSL = {
  ssl: {
    cert: string;
    key: string;
  };
};

type WithRetry = {
  retry: {
    maxAttempts: number;
    backoffMs: number;
  };
};

// 組み合わせて様々な構成を作る
type SecureConfig = BaseConfig & WithAuth & WithSSL;
type BasicConfig = BaseConfig & WithAuth;
type PublicConfig = BaseConfig;
type ResilientConfig = BaseConfig & WithRetry;
type FullConfig = BaseConfig & WithAuth & WithSSL & WithRetry;

const secureConfig: SecureConfig = {
  host: "db.example.com",
  port: 5432,
  auth: { username: "admin", password: "secret" },
  ssl: { cert: "...", key: "..." },
};
```

### コード例7b: ジェネリクスとIntersection/Unionの組み合わせ

```typescript
// --- パターン1: 汎用的なResult型 ---
type Result<T, E = Error> =
  | { ok: true; value: T }
  | { ok: false; error: E };

function divide(a: number, b: number): Result<number, string> {
  if (b === 0) {
    return { ok: false, error: "Division by zero" };
  }
  return { ok: true, value: a / b };
}

const result = divide(10, 3);
if (result.ok) {
  console.log(result.value.toFixed(2)); // "3.33"
} else {
  console.log(result.error.toUpperCase()); // エラーメッセージ
}

// --- パターン2: Branded Types とUnion ---
type Brand<T, B extends string> = T & { readonly __brand: B };

type UserId = Brand<string, "UserId">;
type OrderId = Brand<string, "OrderId">;
type ProductId = Brand<string, "ProductId">;

function createUserId(id: string): UserId {
  return id as UserId;
}

function createOrderId(id: string): OrderId {
  return id as OrderId;
}

function getUserById(id: UserId): Promise<User> {
  // UserId 型のみ受け入れ、OrderId を渡すとコンパイルエラー
  return fetch(`/api/users/${id}`).then((r) => r.json());
}

const userId = createUserId("user-123");
const orderId = createOrderId("order-456");

getUserById(userId);  // OK
// getUserById(orderId); // エラー: OrderId は UserId に代入できない

// --- パターン3: Union型の分配条件型 ---
type ToArray<T> = T extends unknown ? T[] : never;

type StringOrNumberArray = ToArray<string | number>;
// string[] | number[]（分配される）

type NonDistributed<T> = [T] extends [unknown] ? T[] : never;
type Mixed = NonDistributed<string | number>;
// (string | number)[]（分配されない）

// --- パターン4: Mapped TypesとUnion ---
type EventMap = {
  click: { x: number; y: number };
  keypress: { key: string };
  scroll: { offset: number };
};

type EventHandler<T> = (event: T) => void;

type EventHandlers = {
  [K in keyof EventMap]: EventHandler<EventMap[K]>;
};
// {
//   click: (event: { x: number; y: number }) => void;
//   keypress: (event: { key: string }) => void;
//   scroll: (event: { offset: number }) => void;
// }

// Union からイベント名を取得
type EventName = keyof EventMap; // "click" | "keypress" | "scroll"
```

### コード例7c: Union型のユーティリティ型

```typescript
// Union型から特定の型を抽出する
type Extract<T, U> = T extends U ? T : never;
type Exclude<T, U> = T extends U ? never : T;

type AllTypes = string | number | boolean | null | undefined;

type OnlyStrings = Extract<AllTypes, string>;       // string
type NoNullish = Exclude<AllTypes, null | undefined>; // string | number | boolean

// Union型のメンバー数をカウントする（型レベル）
type UnionToIntersection<U> =
  (U extends unknown ? (k: U) => void : never) extends
  (k: infer I) => void ? I : never;

// 条件型でUnionをフィルタリング
type FilterByKind<T, K extends string> = T extends { kind: K } ? T : never;

type Shape =
  | { kind: "circle"; radius: number }
  | { kind: "rectangle"; width: number; height: number }
  | { kind: "triangle"; base: number; height: number };

type CircleOnly = FilterByKind<Shape, "circle">;
// { kind: "circle"; radius: number }

type RectOrTriangle = FilterByKind<Shape, "rectangle" | "triangle">;
// { kind: "rectangle"; ... } | { kind: "triangle"; ... }
```

---

## 6. 実践的なパターン集

### パターン1: 状態マシン（State Machine）

```typescript
// 注文の状態遷移を判別共用体で表現
type OrderState =
  | { status: "draft"; items: CartItem[] }
  | { status: "submitted"; items: CartItem[]; submittedAt: Date }
  | { status: "paid"; items: CartItem[]; submittedAt: Date; paidAt: Date; paymentId: string }
  | { status: "shipped"; items: CartItem[]; submittedAt: Date; paidAt: Date; paymentId: string; trackingNumber: string; shippedAt: Date }
  | { status: "delivered"; items: CartItem[]; submittedAt: Date; paidAt: Date; paymentId: string; trackingNumber: string; shippedAt: Date; deliveredAt: Date }
  | { status: "cancelled"; items: CartItem[]; cancelledAt: Date; reason: string };

interface CartItem {
  productId: string;
  quantity: number;
  price: number;
}

// 状態遷移関数（型安全）
function submitOrder(order: Extract<OrderState, { status: "draft" }>): Extract<OrderState, { status: "submitted" }> {
  return {
    ...order,
    status: "submitted",
    submittedAt: new Date(),
  };
}

function payOrder(
  order: Extract<OrderState, { status: "submitted" }>,
  paymentId: string,
): Extract<OrderState, { status: "paid" }> {
  return {
    ...order,
    status: "paid",
    paidAt: new Date(),
    paymentId,
  };
}

// 不正な遷移はコンパイルエラー
// payOrder(draftOrder, "pay-123"); // エラー: draft → paid は不可

// 状態に応じた表示
function renderOrderStatus(order: OrderState): string {
  switch (order.status) {
    case "draft":
      return `下書き（${order.items.length}件の商品）`;
    case "submitted":
      return `注文済み（${order.submittedAt.toLocaleDateString()}）`;
    case "paid":
      return `決済完了（決済ID: ${order.paymentId}）`;
    case "shipped":
      return `発送済み（追跡番号: ${order.trackingNumber}）`;
    case "delivered":
      return `配達完了（${order.deliveredAt.toLocaleDateString()}）`;
    case "cancelled":
      return `キャンセル（理由: ${order.reason}）`;
    default:
      return assertNever(order);
  }
}
```

### パターン2: ビルダーパターンとIntersection型

```typescript
// Intersection型を活用した型安全なビルダー
type QueryBuilder<T extends Record<string, unknown>> = {
  select<K extends keyof T>(
    ...keys: K[]
  ): QueryBuilder<Pick<T, K>>;
  where(
    condition: Partial<T>
  ): QueryBuilder<T>;
  orderBy(
    key: keyof T,
    direction: "asc" | "desc"
  ): QueryBuilder<T>;
  limit(n: number): QueryBuilder<T>;
  execute(): Promise<T[]>;
};

// 使用イメージ
declare const userQuery: QueryBuilder<User>;
// 型安全なチェーン
const result = await userQuery
  .select("name", "email")    // Pick<User, "name" | "email">
  .where({ role: "admin" })
  .orderBy("name", "asc")     // "name" | "email" のみ指定可
  .limit(10)
  .execute();
// result: Pick<User, "name" | "email">[]
```

### パターン3: 型安全なイベントエミッター

```typescript
type EventDefinitions = {
  "user:login": { userId: string; timestamp: Date };
  "user:logout": { userId: string; timestamp: Date };
  "order:created": { orderId: string; total: number };
  "order:shipped": { orderId: string; trackingNumber: string };
  "error": { code: string; message: string; stack?: string };
};

class TypedEventEmitter<Events extends Record<string, unknown>> {
  private listeners = new Map<string, Set<Function>>();

  on<K extends keyof Events & string>(
    event: K,
    handler: (data: Events[K]) => void,
  ): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(handler);
  }

  emit<K extends keyof Events & string>(
    event: K,
    data: Events[K],
  ): void {
    this.listeners.get(event)?.forEach((handler) => handler(data));
  }

  off<K extends keyof Events & string>(
    event: K,
    handler: (data: Events[K]) => void,
  ): void {
    this.listeners.get(event)?.delete(handler);
  }
}

const emitter = new TypedEventEmitter<EventDefinitions>();

// 型安全なイベントリスナー
emitter.on("user:login", (data) => {
  // data: { userId: string; timestamp: Date }
  console.log(`User ${data.userId} logged in at ${data.timestamp}`);
});

emitter.on("order:created", (data) => {
  // data: { orderId: string; total: number }
  console.log(`Order ${data.orderId}: $${data.total}`);
});

// 型安全な emit
emitter.emit("user:login", {
  userId: "user-123",
  timestamp: new Date(),
});

// エラー: 型が一致しない
// emitter.emit("user:login", { orderId: "xxx" }); // コンパイルエラー
// emitter.emit("unknown:event", {}); // コンパイルエラー
```

---

## アンチパターン

### アンチパターン1: 型ガードなしでUnion型を使う

```typescript
// BAD: 型ガードなしでプロパティアクセス
function getLength(value: string | string[]): number {
  // return value.split("").length; // エラー: string[] に split はない
  return (value as string).length;  // as で逃げる → 配列の場合にバグ
}

// GOOD: 型ガードで安全に処理
function getLength(value: string | string[]): number {
  if (typeof value === "string") {
    return value.length;
  }
  return value.length;
}

// さらに良い: Array.isArray を使う
function getLength(value: string | string[]): number {
  if (Array.isArray(value)) {
    return value.length;
  }
  return value.length;
}
```

### アンチパターン2: 判別子なしのUnion型オブジェクト

```typescript
// BAD: 判別するプロパティがない
type Response = { data: string } | { error: string };

function handle(res: Response) {
  // res.data にアクセスできない（error側の可能性があるため）
  // res.error にもアクセスできない
  if ("data" in res) {  // in ガードで対処可能だが不安定
    console.log(res.data);
  }
}

// GOOD: 判別子を設ける
type Response =
  | { success: true; data: string }
  | { success: false; error: string };

function handle(res: Response) {
  if (res.success) {
    console.log(res.data);   // 安全
  } else {
    console.log(res.error);  // 安全
  }
}
```

### アンチパターン3: Union型が膨大になる

```typescript
// BAD: Union型のメンバーが多すぎてメンテナンス不能
type Event =
  | { type: "a"; data: A }
  | { type: "b"; data: B }
  | { type: "c"; data: C }
  // ... 50個以上のメンバー
  | { type: "zz"; data: ZZ };

// GOOD: ジェネリクスでパターンを抽出し、サブグループに分割
type CrudEvent<Entity extends string, T> =
  | { type: `${Entity}:created`; data: T }
  | { type: `${Entity}:updated`; data: T; changes: Partial<T> }
  | { type: `${Entity}:deleted`; id: string };

type UserEvent = CrudEvent<"user", User>;
type OrderEvent = CrudEvent<"order", Order>;
type ProductEvent = CrudEvent<"product", Product>;

type AppEvent = UserEvent | OrderEvent | ProductEvent;
```

### アンチパターン4: as による不安全なキャスト

```typescript
// BAD: Union型を as で無理やりキャスト
function processShape(shape: Shape) {
  const circle = shape as Circle; // kind が "circle" でなくてもキャストされる
  console.log(circle.radius); // ランタイムエラーの可能性
}

// GOOD: 型ガードで安全に絞り込む
function processShape(shape: Shape) {
  if (shape.kind === "circle") {
    console.log(shape.radius); // Circle 型として安全にアクセス
  }
}
```

### アンチパターン5: Intersection型の意図しない never

```typescript
// BAD: 矛盾するIntersectionに気づかない
type Config = { mode: "development" } & { mode: "production" };
// mode: "development" & "production" = never
// → Config型の値は作れない

// GOOD: Union型を使う
type Config = { mode: "development" } | { mode: "production" };

// BAD: 関数型のIntersectionの誤解
type F = ((x: string) => void) & ((x: number) => void);
// これはオーバーロードとして動作する（エラーではない）
// ただし意図しないオーバーロードに注意
```

---

## FAQ

### Q1: Union型のメンバーが多くなりすぎた場合はどうしますか？

**A:** 判別共用体を使いつつ、関連するメンバーをグループ化してサブUnionに分割します。また、ジェネリクスを活用して共通パターンを抽出することも有効です。
```typescript
type CrudEvent<T> =
  | { type: "created"; entity: T }
  | { type: "updated"; entity: T; changes: Partial<T> }
  | { type: "deleted"; id: string };
```

### Q2: `A & B` で A と B のプロパティ型が矛盾する場合はどうなりますか？

**A:** 矛盾するプロパティの型は `never` になります。
```typescript
type A = { x: string };
type B = { x: number };
type C = A & B;
// C の x は string & number = never
// → C 型の値を作ることは実質不可能
```

Omitを使って矛盾するプロパティを除外するか、型設計を見直すことを推奨します。
```typescript
type SafeMerge = Omit<A, keyof B> & B;
// B のプロパティが優先される
```

### Q3: 型ガードの `is` 構文は必ず必要ですか？

**A:** `typeof`, `instanceof`, `in` などの組み込みガードではTypeScriptが自動的にナローイングします。ユーザー定義の関数で型を絞り込みたい場合のみ `is` 構文（型述語）が必要です。カスタムのバリデーション関数を作る際に特に有用です。

### Q4: Union型とenumはどう使い分けますか？

**A:** TypeScriptでは文字列リテラルのUnion型が一般的に推奨されます。enumはツリーシェイキングされにくく、JavaScriptに変換されるとオブジェクトとして残ります。一方、Union型はコンパイル後に消えるため、バンドルサイズに影響しません。

```typescript
// enumの場合
enum Status {
  Active = "active",
  Inactive = "inactive",
}

// Union型の場合（推奨）
type Status = "active" | "inactive";

// const enum は消えるがバレル再エクスポートで問題が起きる場合がある
const enum Color {
  Red = "red",
  Blue = "blue",
}
```

### Q5: 判別共用体の判別子にはどんな型が使えますか？

**A:** 文字列リテラル、数値リテラル、boolean リテラル（true/false）が使えます。最も一般的なのは文字列リテラルです。

```typescript
// 文字列リテラル（最も一般的）
type A = { kind: "a"; ... } | { kind: "b"; ... };

// 数値リテラル
type B = { code: 200; data: T } | { code: 404; message: string };

// boolean リテラル
type C = { success: true; data: T } | { success: false; error: E };
```

### Q6: Union型のメンバーの順序は重要ですか？

**A:** 型レベルでは順序は関係ありません。`string | number` と `number | string` は同じ型です。ただし、コードの可読性のためにメンバーを論理的にグループ化することを推奨します。

### Q7: Intersection型はいつ使うべきですか？

**A:** 主に以下の場面で使用します：
1. **型の合成**: 小さな型を組み合わせて大きな型を作る
2. **ミックスイン**: 既存の型に機能を追加する
3. **ジェネリクスの制約**: `T extends A & B` で複数の制約を課す
4. **関数のオーバーロード**: 関数型のIntersectionはオーバーロードとして動作する

---

## まとめ

| 項目 | 内容 |
|------|------|
| Union型 (`\|`) | 「いずれかの型」を表す。型ガードで絞り込んで使う |
| Intersection型 (`&`) | 「全ての型の組み合わせ」を表す。型の合成に使う |
| 判別共用体 | 共通のリテラル型プロパティで型を区別する安全なパターン |
| typeof | プリミティブ型の判定。string, number, boolean 等 |
| instanceof | クラスインスタンスの判定 |
| in | プロパティの存在チェック |
| ユーザー定義型ガード | `value is Type` で独自の型判定関数を定義 |
| asserts | 条件を満たさない場合に例外を投げるアサーション関数 |
| 網羅性チェック | never + default で全ケースの処理漏れを検出 |
| 分配条件型 | Union型に条件型を適用するとメンバーごとに分配される |

---

## 演習問題

### 問題1: 判別共用体の設計

以下の要件を満たす判別共用体 `Shape` を定義し、面積を計算する `calculateArea` 関数を実装してください。

- Circle（半径）
- Rectangle（幅、高さ）
- Triangle（底辺、高さ）
- Ellipse（長径、短径）

網羅性チェックを含めること。

```typescript
// ここに実装を書いてください
type Shape = /* ... */;

function calculateArea(shape: Shape): number {
  // ...
}
```

### 問題2: 型ガード関数の実装

以下の `unknown` 型のデータが特定のインターフェースを満たすかどうかを判定する型ガード関数を実装してください。

```typescript
interface UserProfile {
  id: string;
  name: string;
  email: string;
  age: number;
}

function isUserProfile(value: unknown): value is UserProfile {
  // ここに実装を書いてください
}
```

### 問題3: Result型の実装

以下の仕様に従って `Result<T, E>` 型を定義し、ユーティリティ関数を実装してください。

- `Result<T, E>` は `Ok<T>` または `Err<E>` のUnion型
- `map` 関数: 成功値を変換する
- `flatMap` 関数: 成功値から新しいResultを生成する
- `unwrapOr` 関数: 成功値を取得するか、デフォルト値を返す

```typescript
// ここに実装を書いてください
type Result<T, E> = /* ... */;

function map<T, U, E>(result: Result<T, E>, fn: (value: T) => U): Result<U, E> {
  // ...
}

function flatMap<T, U, E>(result: Result<T, E>, fn: (value: T) => Result<U, E>): Result<U, E> {
  // ...
}

function unwrapOr<T, E>(result: Result<T, E>, defaultValue: T): T {
  // ...
}
```

### 問題4: 型安全なイベントシステム

以下のイベント定義に対応する型安全なイベントバスを実装してください。`emit` 時にイベント名と一致しないデータを渡すとコンパイルエラーになること。

```typescript
type Events = {
  "user:created": { id: string; name: string };
  "user:deleted": { id: string };
  "order:placed": { orderId: string; items: string[] };
};

// EventBus クラスを実装してください
class EventBus<E extends Record<string, unknown>> {
  // on, emit, off メソッドを実装
}
```

### 問題5: 状態マシンの型安全な遷移

以下の状態遷移図に基づいて、不正な遷移をコンパイルエラーにする関数群を実装してください。

```
  draft → submitted → approved → published
                  ↘ rejected
  (いずれの状態からも cancelled に遷移可能)
```

```typescript
// 各状態の型と遷移関数を実装してください
type ArticleState = /* ... */;

function submit(article: /* draft */): /* submitted */ { ... }
function approve(article: /* submitted */): /* approved */ { ... }
function reject(article: /* submitted */): /* rejected */ { ... }
function publish(article: /* approved */): /* published */ { ... }
function cancel(article: ArticleState): /* cancelled */ { ... }
```

### 問題6: Intersection型によるプラグインシステム

基本機能を持つ `BaseApp` に対して、Intersection型で機能を拡張するプラグインシステムを設計してください。各プラグインは独自のメソッドとプロパティを追加します。

```typescript
type BaseApp = {
  name: string;
  version: string;
  start(): void;
};

type WithAuth = { /* 認証機能 */ };
type WithLogging = { /* ログ機能 */ };
type WithCache = { /* キャッシュ機能 */ };

// 任意のプラグインの組み合わせでアプリを構成する型を定義してください
type MyApp = BaseApp & WithAuth & WithLogging;

function createApp<T extends BaseApp>(config: T): T {
  // ...
}
```

---

## 次に読むべきガイド

- [04-generics.md](./04-generics.md) -- ジェネリクス
- [../02-patterns/02-discriminated-unions.md](../02-patterns/02-discriminated-unions.md) -- 判別共用体パターン（実践編）
- [../01-advanced-types/00-conditional-types.md](../01-advanced-types/00-conditional-types.md) -- 条件型

---

## 参考文献

1. **TypeScript Handbook: Narrowing** -- https://www.typescriptlang.org/docs/handbook/2/narrowing.html
2. **TypeScript Handbook: Unions and Intersection Types** -- https://www.typescriptlang.org/docs/handbook/2/everyday-types.html#union-types
3. **Discriminated Unions in TypeScript** -- https://www.typescriptlang.org/docs/handbook/typescript-in-5-minutes-func.html#discriminated-unions
4. **Effective TypeScript** -- Dan Vanderkam著, O'Reilly. Item 22: Understand Type Narrowing
5. **Programming TypeScript** -- Boris Cherny著, O'Reilly. Chapter 6: Advanced Types
