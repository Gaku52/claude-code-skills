# 型の基礎

> TypeScriptの根幹をなすプリミティブ型、リテラル型、配列、タプル、enum、特殊型（any/unknown/never）を体系的に学ぶ。

## この章で学ぶこと

1. **プリミティブ型とリテラル型** -- string, number, boolean, symbol, bigint およびリテラル型による値の制限
2. **コレクション型** -- 配列型、タプル型、readonly修飾子による不変性の表現
3. **特殊型** -- any, unknown, never, void, null, undefined の正しい使い分け
4. **型推論** -- TypeScriptが自動的に型を推論する仕組み
5. **型の拡張と絞り込み** -- widening, narrowing, const assertion
6. **実務パターン** -- 各型を実務でどのように活用するか

---

## 1. プリミティブ型

### コード例1: 基本的なプリミティブ型

```typescript
// 文字列
const name: string = "TypeScript";

// 数値（整数・浮動小数点の区別なし）
const age: number = 12;
const pi: number = 3.14159;

// 真偽値
const isReady: boolean = true;

// シンボル
const uniqueKey: symbol = Symbol("key");

// BigInt（大きな整数）
const huge: bigint = 9007199254740991n;

// null と undefined
const nothing: null = null;
const notDefined: undefined = undefined;
```

### プリミティブ型の一覧

```
+------------------+-------------------+------------------------+
| 型               | 値の例            | 用途                   |
+------------------+-------------------+------------------------+
| string           | "hello"           | テキストデータ         |
| number           | 42, 3.14          | 数値全般               |
| boolean          | true, false       | 論理値                 |
| symbol           | Symbol("id")      | ユニークキー           |
| bigint           | 100n              | 巨大整数               |
| null             | null              | 値の不在（意図的）     |
| undefined        | undefined         | 値の未定義             |
+------------------+-------------------+------------------------+
```

### string型の詳細

```typescript
// 文字列の基本操作と型
const greeting: string = "Hello, TypeScript!";
const templateStr: string = `Count: ${42}`; // テンプレートリテラル
const multiLine: string = `
  複数行の
  文字列も
  string型
`;

// string と String の違い（重要）
const primitive: string = "hello";       // プリミティブ型（推奨）
// const wrapped: String = new String("hello"); // ラッパーオブジェクト型（非推奨）

// プリミティブ型を使うべき理由
// 1. String オブジェクトは typeof で "object" を返す
// 2. String オブジェクトは == 比較で予期しない結果になる
// 3. TypeScript のほぼ全てのAPIはプリミティブ string を期待する

// 文字列操作の型推論
const upper = "hello".toUpperCase(); // string と推論
const includes = "hello".includes("ell"); // boolean と推論
const split = "a,b,c".split(","); // string[] と推論
const charAt = "hello".charAt(0); // string と推論（"h" ではない）

// テンプレートリテラル型（型レベルの文字列操作）
type Greeting = `Hello, ${string}!`;
const greet1: Greeting = "Hello, World!"; // OK
const greet2: Greeting = "Hello, TypeScript!"; // OK
// const greet3: Greeting = "Hi, World!"; // エラー: "Hi, World!" は "Hello, ${string}!" に代入不可
```

### number型の詳細

```typescript
// 数値の基本
const integer: number = 42;
const float: number = 3.14;
const negative: number = -100;
const hex: number = 0xff;         // 16進数 = 255
const octal: number = 0o77;      // 8進数 = 63
const binary: number = 0b1010;   // 2進数 = 10
const scientific: number = 1e10; // 指数表記 = 10000000000

// 特殊な数値
const inf: number = Infinity;
const negInf: number = -Infinity;
const notANumber: number = NaN;
// NaN は number 型に含まれることに注意

// NaN のチェック
function isValidNumber(value: number): boolean {
  return !Number.isNaN(value) && Number.isFinite(value);
}

// number と bigint は互換性がない
const num: number = 42;
const big: bigint = 42n;
// const mixed: number = big; // エラー: bigint を number に代入できない
// const result = num + big;  // エラー: number と bigint は演算できない

// 安全な整数範囲
const maxSafe: number = Number.MAX_SAFE_INTEGER;  // 9007199254740991
const minSafe: number = Number.MIN_SAFE_INTEGER;  // -9007199254740991
// この範囲を超える場合は bigint を使用する

// 数値の型ガード
function processNumber(value: unknown): number {
  if (typeof value === "number" && !Number.isNaN(value)) {
    return value;
  }
  throw new Error(`Invalid number: ${value}`);
}
```

### boolean型の詳細

```typescript
// 真偽値の基本
const isActive: boolean = true;
const isDisabled: boolean = false;

// 型推論と boolean
const result = 10 > 5; // boolean と推論
const comparison = "a" === "b"; // boolean と推論

// boolean のリテラル型
type True = true;
type False = false;

// 条件分岐での活用
interface Feature {
  enabled: boolean;
  name: string;
}

function isFeatureEnabled(feature: Feature): feature is Feature & { enabled: true } {
  return feature.enabled;
}

// truthy/falsy とTypeScriptの型システム
// JavaScriptの falsy 値: false, 0, "", null, undefined, NaN
// TypeScriptの boolean 型は true と false のみ
// 他の falsy 値は boolean 型ではない

// strictNullChecks 有効時の安全な boolean 変換
function toBooleanSafe(value: unknown): boolean {
  return Boolean(value); // 明示的な変換
  // return !!value; // ダブルバング（こちらも一般的）
}
```

### symbol型の詳細

```typescript
// symbol の基本
const sym1: symbol = Symbol("description");
const sym2: symbol = Symbol("description");
console.log(sym1 === sym2); // false（常にユニーク）

// unique symbol: より厳密なシンボル型
const UNIQUE_KEY: unique symbol = Symbol("uniqueKey");
// unique symbol は const 宣言でのみ使用可能

// シンボルの実務での用途

// 1. オブジェクトのプライベートプロパティ（WeakMapの代替）
const _privateData = Symbol("privateData");

class MyClass {
  [_privateData]: string;

  constructor(data: string) {
    this[_privateData] = data;
  }

  getData(): string {
    return this[_privateData];
  }
}

// 2. Well-known Symbols（組み込みシンボル）
class CustomCollection {
  private items: number[] = [];

  // Symbol.iterator を実装してイテラブルにする
  [Symbol.iterator](): Iterator<number> {
    let index = 0;
    const items = this.items;
    return {
      next(): IteratorResult<number> {
        if (index < items.length) {
          return { value: items[index++], done: false };
        }
        return { value: undefined, done: true };
      },
    };
  }

  add(item: number): void {
    this.items.push(item);
  }
}

// 3. Symbol.dispose（TypeScript 5.2+）
class Resource {
  [Symbol.dispose](): void {
    console.log("Resource disposed");
  }
}

// using 宣言でスコープ終了時に自動的に dispose される
// function useResource() {
//   using resource = new Resource();
//   // ... resource を使用
// } // ← ここで自動的に [Symbol.dispose]() が呼ばれる
```

### bigint型の詳細

```typescript
// bigint の基本
const big1: bigint = 100n;
const big2: bigint = BigInt(100);
const big3: bigint = BigInt("9007199254740991");

// bigint の演算
const sum: bigint = 100n + 200n;       // 300n
const product: bigint = 10n * 20n;     // 200n
const division: bigint = 10n / 3n;     // 3n（切り捨て）
const remainder: bigint = 10n % 3n;    // 1n
const power: bigint = 2n ** 64n;       // 18446744073709551616n

// bigint と number は混在演算できない
// const mixed = 1n + 1; // エラー

// bigint の比較は number と可能
console.log(1n === 1);  // false（型が異なる）
console.log(1n == 1);   // true（値が同じ）
console.log(1n < 2);    // true

// bigint の実務での用途
// 1. データベースの大きなID（snowflake ID等）
type SnowflakeId = bigint;
const discordId: SnowflakeId = 1234567890123456789n;

// 2. 暗号学的な計算
// 3. 高精度の金融計算（整数での扱い）
// 4. タイムスタンプのナノ秒精度

// bigint の制約
// - JSON.stringify() で直接シリアライズできない
// - Math オブジェクトの関数は使えない
// - number への暗黙変換はない

// JSON シリアライズの解決策
function bigintReplacer(_key: string, value: unknown): unknown {
  if (typeof value === "bigint") {
    return value.toString();
  }
  return value;
}

const data = { id: 123456789012345n };
JSON.stringify(data, bigintReplacer); // '{"id":"123456789012345"}'
```

---

## 2. リテラル型

### コード例2: リテラル型で値を制限する

```typescript
// 文字列リテラル型
type Direction = "north" | "south" | "east" | "west";
let dir: Direction = "north"; // OK
// dir = "up"; // コンパイルエラー

// 数値リテラル型
type DiceRoll = 1 | 2 | 3 | 4 | 5 | 6;
let roll: DiceRoll = 3; // OK
// roll = 7; // コンパイルエラー

// 真偽値リテラル型
type Success = true;
const result: Success = true;
```

### リテラル型の実務パターン

```typescript
// パターン1: HTTPメソッドの制限
type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE" | "HEAD" | "OPTIONS";

interface RequestConfig {
  method: HttpMethod;
  url: string;
  body?: unknown;
  headers?: Record<string, string>;
}

function makeRequest(config: RequestConfig): Promise<Response> {
  return fetch(config.url, {
    method: config.method,
    body: config.body ? JSON.stringify(config.body) : undefined,
    headers: config.headers,
  });
}

// makeRequest({ method: "GETTT", url: "/" }); // エラー: "GETTT" は HttpMethod に代入不可
makeRequest({ method: "GET", url: "/api/users" }); // OK

// パターン2: ステータスの状態遷移
type OrderStatus = "pending" | "confirmed" | "shipped" | "delivered" | "cancelled";

interface Order {
  id: string;
  status: OrderStatus;
  amount: number;
}

// 状態遷移の制約を型で表現
function transitionOrder(
  order: Order,
  newStatus: OrderStatus
): Order {
  const validTransitions: Record<OrderStatus, OrderStatus[]> = {
    pending: ["confirmed", "cancelled"],
    confirmed: ["shipped", "cancelled"],
    shipped: ["delivered"],
    delivered: [],
    cancelled: [],
  };

  if (!validTransitions[order.status].includes(newStatus)) {
    throw new Error(
      `Invalid transition: ${order.status} → ${newStatus}`
    );
  }

  return { ...order, status: newStatus };
}

// パターン3: 数値リテラル型の活用
type Bit = 0 | 1;
type Nibble = [Bit, Bit, Bit, Bit]; // 4ビット

type LogLevel = 0 | 1 | 2 | 3 | 4;
const LOG_LEVELS = {
  TRACE: 0 as const,
  DEBUG: 1 as const,
  INFO: 2 as const,
  WARN: 3 as const,
  ERROR: 4 as const,
};

// パターン4: テンプレートリテラル型との組み合わせ
type CSSUnit = "px" | "em" | "rem" | "%" | "vh" | "vw";
type CSSValue = `${number}${CSSUnit}`;

function setWidth(element: HTMLElement, width: CSSValue): void {
  element.style.width = width;
}

// setWidth(element, "100px");  // OK
// setWidth(element, "2.5rem"); // OK
// setWidth(element, "100");    // エラー: 単位がない

// パターン5: リテラル型を使ったオーバーロードの代替
type EventType = "click" | "hover" | "focus";

interface EventPayload {
  click: { x: number; y: number; button: number };
  hover: { x: number; y: number };
  focus: { target: string };
}

function handleEvent<T extends EventType>(
  type: T,
  payload: EventPayload[T]
): void {
  console.log(`Event: ${type}`, payload);
}

handleEvent("click", { x: 100, y: 200, button: 0 }); // OK
handleEvent("hover", { x: 100, y: 200 }); // OK
// handleEvent("click", { x: 100, y: 200 }); // エラー: button が不足
```

### コード例3: const アサーション

```typescript
// as const でリテラル型に狭める
const config = {
  host: "localhost",
  port: 3000,
} as const;
// 型: { readonly host: "localhost"; readonly port: 3000 }

// as const なしの場合
const config2 = {
  host: "localhost",
  port: 3000,
};
// 型: { host: string; port: number } -- 広い型になる
```

### const アサーションの詳細パターン

```typescript
// 配列への as const
const colors = ["red", "green", "blue"] as const;
// 型: readonly ["red", "green", "blue"]
// colors[0] は "red" 型（string ではなく）

// as const から Union 型を生成
type Color = (typeof colors)[number]; // "red" | "green" | "blue"

// ネストしたオブジェクトへの as const
const theme = {
  colors: {
    primary: "#007bff",
    secondary: "#6c757d",
    danger: "#dc3545",
  },
  spacing: {
    small: 4,
    medium: 8,
    large: 16,
  },
  breakpoints: {
    mobile: 320,
    tablet: 768,
    desktop: 1024,
  },
} as const;

// 深いネストの値もリテラル型として認識される
type PrimaryColor = typeof theme.colors.primary; // "#007bff"
type AllColors = (typeof theme.colors)[keyof typeof theme.colors];
// "#007bff" | "#6c757d" | "#dc3545"

// 関数の引数で as const を活用
function createConfig<const T extends Record<string, unknown>>(config: T): T {
  return config;
}

const appConfig = createConfig({
  apiUrl: "https://api.example.com",
  timeout: 5000,
  retry: 3,
});
// appConfig.apiUrl は "https://api.example.com" 型

// satisfies と as const の組み合わせ
interface ThemeConfig {
  colors: Record<string, string>;
  spacing: Record<string, number>;
}

const validatedTheme = {
  colors: {
    primary: "#007bff",
    secondary: "#6c757d",
  },
  spacing: {
    small: 4,
    large: 16,
  },
} as const satisfies ThemeConfig;
// 型チェックされつつ、リテラル型が維持される
// validatedTheme.colors.primary は "#007bff" 型
```

---

## 3. 配列とタプル

### コード例4: 配列型

```typescript
// 配列の2つの記法
const numbers: number[] = [1, 2, 3];
const strings: Array<string> = ["a", "b", "c"];

// 読み取り専用配列
const frozen: readonly number[] = [1, 2, 3];
// frozen.push(4); // コンパイルエラー: push は readonly 配列に存在しない

const frozenAlt: ReadonlyArray<number> = [1, 2, 3];
```

### 配列型の詳細パターン

```typescript
// 多次元配列
const matrix: number[][] = [
  [1, 2, 3],
  [4, 5, 6],
  [7, 8, 9],
];

// 3次元配列
const cube: number[][][] = [
  [[1, 2], [3, 4]],
  [[5, 6], [7, 8]],
];

// Union型の配列
const mixed: (string | number)[] = [1, "hello", 2, "world"];

// オブジェクト配列
interface User {
  id: number;
  name: string;
  email: string;
}

const users: User[] = [
  { id: 1, name: "Alice", email: "alice@example.com" },
  { id: 2, name: "Bob", email: "bob@example.com" },
];

// 配列メソッドの型推論
const doubled = numbers.map(n => n * 2); // number[]
const filtered = numbers.filter(n => n > 1); // number[]
const found = numbers.find(n => n === 2); // number | undefined
const sum = numbers.reduce((acc, n) => acc + n, 0); // number
const names = users.map(u => u.name); // string[]

// filter で型を絞り込む
const mixedArray: (string | number | null)[] = [1, "hello", null, 2, "world", null];

// 型述語を使った filter
const nonNull = mixedArray.filter(
  (item): item is string | number => item !== null
); // (string | number)[]

const onlyStrings = mixedArray.filter(
  (item): item is string => typeof item === "string"
); // string[]

// Array.isArray の型ガード
function processInput(input: string | string[]): string[] {
  if (Array.isArray(input)) {
    return input; // string[]
  }
  return [input]; // string[]
}

// readonly配列の実務パターン
function processItems(items: readonly string[]): string {
  // items.push("new"); // エラー: readonly 配列は変更できない
  // items.sort();       // エラー: sort は配列を変更する

  // 新しい配列を作成するメソッドは使える
  const sorted = [...items].sort(); // OK: スプレッドでコピー
  return sorted.join(", ");
}

// ReadonlyArray と readonly の違い
const arr1: ReadonlyArray<number> = [1, 2, 3]; // ジェネリック記法
const arr2: readonly number[] = [1, 2, 3];     // 省略記法
// 両者は完全に等価
```

### コード例5: タプル型

```typescript
// タプル: 要素数と各位置の型が固定された配列
type Point2D = [number, number];
type Point3D = [number, number, number];

const origin: Point2D = [0, 0];
const point: Point3D = [1, 2, 3];

// ラベル付きタプル（可読性向上）
type UserEntry = [id: number, name: string, active: boolean];
const user: UserEntry = [1, "Alice", true];

// 可変長タプル（rest要素）
type StringNumberBooleans = [string, number, ...boolean[]];
const data: StringNumberBooleans = ["hello", 42, true, false, true];
```

### タプル型の詳細パターン

```typescript
// タプルの分割代入
type NameAge = [name: string, age: number];
const [userName, userAge]: NameAge = ["Alice", 30];
// userName は string 型、userAge は number 型

// 関数の戻り値としてのタプル
function useState<T>(initial: T): [T, (newValue: T) => void] {
  let value = initial;
  const setValue = (newValue: T) => {
    value = newValue;
  };
  return [value, setValue];
}

const [count, setCount] = useState(0);
// count は number 型
// setCount は (newValue: number) => void 型

// 複数の値を返す関数
function divmod(a: number, b: number): [quotient: number, remainder: number] {
  return [Math.floor(a / b), a % b];
}
const [quotient, remainder] = divmod(17, 5); // [3, 2]

// オプショナル要素を持つタプル
type PartialPoint = [number, number, number?];
const point2d: PartialPoint = [1, 2];     // OK
const point3d: PartialPoint = [1, 2, 3];  // OK

// 先頭にrest要素
type TailString = [...number[], string]; // 最後が string、それ以前は number[]
const data1: TailString = ["end"];            // OK
const data2: TailString = [1, 2, 3, "end"];  // OK

// 中間にrest要素
type Sandwich = [string, ...number[], string]; // 先頭と末尾が string
const s1: Sandwich = ["start", "end"];               // OK
const s2: Sandwich = ["start", 1, 2, 3, "end"];     // OK

// readonly タプル
type ReadonlyPoint = readonly [number, number];
const p: ReadonlyPoint = [1, 2];
// p[0] = 3; // エラー: readonly タプルは変更できない

// タプルの型推論とas const
const pair = [1, "hello"] as const; // readonly [1, "hello"]
// as const なしの場合: (string | number)[] と推論される（タプルではなく配列）

// Variadic Tuple Types（TypeScript 4.0+）
type Concat<A extends readonly unknown[], B extends readonly unknown[]> = [...A, ...B];
type Result = Concat<[1, 2], [3, 4]>; // [1, 2, 3, 4]

// タプルを使ったイベントエミッタ
type EventMap = {
  click: [x: number, y: number];
  keypress: [key: string, modifiers: string[]];
  resize: [width: number, height: number];
};

class TypedEventEmitter<T extends Record<string, unknown[]>> {
  private listeners = new Map<keyof T, Set<(...args: any[]) => void>>();

  on<K extends keyof T>(event: K, listener: (...args: T[K]) => void): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(listener);
  }

  emit<K extends keyof T>(event: K, ...args: T[K]): void {
    this.listeners.get(event)?.forEach(listener => listener(...args));
  }
}

const emitter = new TypedEventEmitter<EventMap>();
emitter.on("click", (x, y) => {
  // x は number, y は number と推論
  console.log(`Clicked at (${x}, ${y})`);
});
emitter.emit("click", 100, 200); // OK
// emitter.emit("click", "100", 200); // エラー: string は number に代入不可
```

### 配列 vs タプル 比較

| 特性 | 配列 (Array) | タプル (Tuple) |
|------|-------------|----------------|
| 要素数 | 可変 | 固定（rest要素で可変も可） |
| 要素の型 | 全要素同一型 | 位置ごとに異なる型が可能 |
| 用途 | 同種データの集合 | 異種データの組み合わせ |
| アクセス | インデックスで同一型 | インデックスで位置に応じた型 |
| 例 | `number[]` | `[string, number]` |
| 分割代入 | 型は同一 | 各変数が対応する型を持つ |

```
  配列 (number[])           タプル ([string, number, boolean])
+---+---+---+---+...      +--------+--------+---------+
| n | n | n | n |          | string | number | boolean |
+---+---+---+---+...      +--------+--------+---------+
 全て同じ型                  位置ごとに異なる型
 長さ不定                    長さ固定
```

---

## 4. enum（列挙型）

### コード例6: enum の種類

```typescript
// 数値enum（デフォルト: 0から自動インクリメント）
enum Status {
  Pending,   // 0
  Active,    // 1
  Inactive,  // 2
}

// 文字列enum（推奨: 値が明示的）
enum Color {
  Red = "RED",
  Green = "GREEN",
  Blue = "BLUE",
}

// const enum（コンパイル時にインライン化、パフォーマンス向上）
const enum HttpMethod {
  GET = "GET",
  POST = "POST",
  PUT = "PUT",
  DELETE = "DELETE",
}

// 使用例
const status: Status = Status.Active;
const method: HttpMethod = HttpMethod.GET;
```

### enum の詳細パターン

```typescript
// 数値enumの開始値指定
enum Priority {
  Low = 1,
  Medium = 5,
  High = 10,
  Critical = 100,
}

// 計算されたメンバー
enum FileAccess {
  None = 0,
  Read = 1 << 0,      // 1
  Write = 1 << 1,     // 2
  Execute = 1 << 2,   // 4
  ReadWrite = Read | Write,      // 3
  ReadExecute = Read | Execute,  // 5
  All = Read | Write | Execute,  // 7
}

// ビットフラグとして使う
function hasPermission(userPermissions: FileAccess, required: FileAccess): boolean {
  return (userPermissions & required) === required;
}

const myPerms = FileAccess.ReadWrite;
console.log(hasPermission(myPerms, FileAccess.Read));    // true
console.log(hasPermission(myPerms, FileAccess.Execute)); // false

// 数値enumの逆引き
enum Direction {
  Up = 0,
  Down = 1,
  Left = 2,
  Right = 3,
}
console.log(Direction[0]); // "Up"（逆引き）
console.log(Direction.Up); // 0（正引き）
// 注意: 文字列enumには逆引きは存在しない

// ヘテロジニアスenum（数値と文字列の混在、非推奨）
enum Mixed {
  No = 0,
  Yes = "YES",
}

// enum をイテレートする
enum Fruit {
  Apple = "APPLE",
  Banana = "BANANA",
  Cherry = "CHERRY",
}

// Object.values で文字列enumの値を取得
const fruitValues = Object.values(Fruit); // ["APPLE", "BANANA", "CHERRY"]

// 数値enumの場合は逆引きのキーも含まれるため注意
const dirValues = Object.keys(Direction);
// ["0", "1", "2", "3", "Up", "Down", "Left", "Right"]
// → 数値enumのイテレートは避けるか、フィルタリングが必要

// const enum の注意点
const enum Speeds {
  Slow = 10,
  Medium = 50,
  Fast = 100,
}
const speed = Speeds.Fast;
// コンパイル後: const speed = 100; // インライン化される
// 利点: バンドルサイズ削減
// 注意: --isolatedModules 使用時は const enum が使えない場合がある
```

### enum vs Union型 比較

| 特性 | enum | Union型 |
|------|------|---------|
| ランタイムコード | 生成される（const enum除く） | なし（型情報のみ） |
| バンドルサイズ | 増加する | 影響なし |
| 逆引き | 数値enumは可能 | 不可 |
| Tree-shaking | 困難な場合がある | 問題なし |
| 型の拡張性 | 不可 | Union で柔軟に拡張可能 |
| 推奨度 | const enum または非推奨傾向 | 多くの場面で推奨 |

```typescript
// モダンなTypeScriptではUnion型が推奨されるケースが多い
// enum の代替
type Color = "RED" | "GREEN" | "BLUE";

// 定数オブジェクト + as const パターン
const Color = {
  Red: "RED",
  Green: "GREEN",
  Blue: "BLUE",
} as const;
type Color = (typeof Color)[keyof typeof Color];
// "RED" | "GREEN" | "BLUE"
```

### as const パターンの実務的活用

```typescript
// 定数オブジェクトパターンの完全な例
const HTTP_STATUS = {
  OK: 200,
  Created: 201,
  BadRequest: 400,
  Unauthorized: 401,
  Forbidden: 403,
  NotFound: 404,
  InternalServerError: 500,
} as const;

// 値のUnion型
type HttpStatusCode = (typeof HTTP_STATUS)[keyof typeof HTTP_STATUS];
// 200 | 201 | 400 | 401 | 403 | 404 | 500

// キーのUnion型
type HttpStatusName = keyof typeof HTTP_STATUS;
// "OK" | "Created" | "BadRequest" | "Unauthorized" | "Forbidden" | "NotFound" | "InternalServerError"

// 逆引きマップの型安全な実装
type ReverseMap<T extends Record<string, string | number>> = {
  [V in T[keyof T]]: {
    [K in keyof T]: T[K] extends V ? K : never;
  }[keyof T];
};

// ラベルの定義
const LABELS = {
  OK: "成功",
  Created: "作成完了",
  BadRequest: "不正なリクエスト",
  Unauthorized: "認証が必要",
  Forbidden: "アクセス拒否",
  NotFound: "見つかりません",
  InternalServerError: "サーバーエラー",
} as const satisfies Record<HttpStatusName, string>;

function getStatusLabel(code: HttpStatusCode): string {
  const entry = Object.entries(HTTP_STATUS).find(([_, v]) => v === code);
  if (!entry) return "不明なステータス";
  return LABELS[entry[0] as HttpStatusName];
}
```

---

## 5. 特殊型: any, unknown, never

### 型の階層図

```
           any（全ての型のスーパータイプ）
          / | \
   string number boolean ... object
          \ | /
         unknown（安全なany）
            |
          never（全ての型のサブタイプ、値を持たない）
```

正確な型の階層はこのようになっている。

```
         any（特殊：全ての型に代入可能 & 全ての型から代入可能）
          |
        unknown（トップ型：全ての型から代入可能）
       / | | \
string number boolean object ... void null undefined
       \ | | /
        never（ボトム型：全ての型に代入可能、値を持たない）
```

### コード例7: any vs unknown

```typescript
// any: 型チェックを完全に無効化する（危険）
let dangerous: any = "hello";
dangerous.foo.bar.baz(); // コンパイルエラーなし → 実行時エラー

// unknown: 型安全な「何でも受け取れる型」
let safe: unknown = "hello";
// safe.foo; // コンパイルエラー！ まず型を確認する必要がある

// unknownの正しい使い方: 型ガードで絞り込む
if (typeof safe === "string") {
  console.log(safe.toUpperCase()); // OK: string として安全に使える
}
```

### any の実務的な使いどころ（限定的）

```typescript
// any が許容されるケース（極めて限定的）

// 1. 型定義が存在しないサードパーティライブラリの一時的な利用
// @ts-expect-error: 型定義が不完全なライブラリ
declare const legacyLib: any;

// 2. JSON.parse の戻り値（ただし unknown が望ましい）
// JSON.parse は any を返すが、即座にバリデーションすべき
const parsed: unknown = JSON.parse(jsonString) as unknown;

// 3. テストコードでの意図的な型違反
// テストでエッジケースを検証する場合
// expect(() => processUser(null as any)).toThrow();

// 4. 型の複雑さを一時的に回避（TODO付きで）
// TODO: #1234 で適切な型を定義する
function temporaryHandler(event: any): void {
  // ...
}

// any を段階的に排除するための tsconfig 設定
// {
//   "compilerOptions": {
//     "noImplicitAny": true,       // 暗黙のanyを禁止
//     "noExplicitAny": false       // 明示的なanyは許可（将来的に禁止）
//   }
// }

// ESLint ルールで any を制限
// {
//   "rules": {
//     "@typescript-eslint/no-explicit-any": "warn",
//     "@typescript-eslint/no-unsafe-assignment": "error",
//     "@typescript-eslint/no-unsafe-member-access": "error",
//     "@typescript-eslint/no-unsafe-call": "error",
//     "@typescript-eslint/no-unsafe-return": "error"
//   }
// }
```

### unknown の実務パターン

```typescript
// unknown の使いどころ

// 1. 外部データの受け取り
async function fetchData(url: string): Promise<unknown> {
  const response = await fetch(url);
  return response.json(); // unknown として返す
}

// 2. 型安全なパーサー
function parseJSON(json: string): unknown {
  try {
    return JSON.parse(json);
  } catch {
    return undefined;
  }
}

// 3. unknown からの型安全な変換
// 方法A: typeof による絞り込み
function processUnknown(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number") return String(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === null) return "null";
  if (value === undefined) return "undefined";
  if (Array.isArray(value)) return JSON.stringify(value);
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

// 方法B: ユーザー定義型ガード
interface User {
  id: number;
  name: string;
  email: string;
}

function isUser(value: unknown): value is User {
  return (
    typeof value === "object" &&
    value !== null &&
    "id" in value &&
    "name" in value &&
    "email" in value &&
    typeof (value as User).id === "number" &&
    typeof (value as User).name === "string" &&
    typeof (value as User).email === "string"
  );
}

const data: unknown = await fetchData("/api/user/1");
if (isUser(data)) {
  console.log(data.name); // 安全にアクセス可能
}

// 方法C: Zodによるバリデーション（推奨）
import { z } from "zod";

const UserSchema = z.object({
  id: z.number(),
  name: z.string(),
  email: z.string().email(),
});

function parseUser(data: unknown): User {
  return UserSchema.parse(data); // 不正なデータはZodError
}

// 4. catch ブロックでの unknown
try {
  await riskyOperation();
} catch (error: unknown) {
  // TypeScript 4.4+ で catch の変数は unknown 型（strict モード時）
  if (error instanceof Error) {
    console.error(error.message);
  } else if (typeof error === "string") {
    console.error(error);
  } else {
    console.error("Unknown error:", error);
  }
}
```

### コード例8: never型

```typescript
// never: 決して値を返さない（到達不能）
function throwError(message: string): never {
  throw new Error(message);
}

function infiniteLoop(): never {
  while (true) {
    // 永遠に終わらない
  }
}

// 網羅性チェック（exhaustive check）に使う
type Shape = "circle" | "square" | "triangle";

function getArea(shape: Shape): number {
  switch (shape) {
    case "circle":
      return Math.PI * 10 * 10;
    case "square":
      return 10 * 10;
    case "triangle":
      return (10 * 10) / 2;
    default:
      // shape が never 型になる = 全てのケースを処理済み
      const _exhaustive: never = shape;
      return _exhaustive;
  }
}
```

### never型の詳細パターン

```typescript
// never型の高度な活用

// 1. 網羅性チェックのヘルパー関数
function assertNever(value: never, message?: string): never {
  throw new Error(message ?? `Unexpected value: ${value}`);
}

type Action =
  | { type: "INCREMENT"; amount: number }
  | { type: "DECREMENT"; amount: number }
  | { type: "RESET" };

function reducer(state: number, action: Action): number {
  switch (action.type) {
    case "INCREMENT":
      return state + action.amount;
    case "DECREMENT":
      return state - action.amount;
    case "RESET":
      return 0;
    default:
      return assertNever(action); // 新しい Action 型を追加するとここでエラー
  }
}

// 2. 条件型での never の活用
// never はUnion型から除外される
type RemoveString<T> = T extends string ? never : T;
type Result = RemoveString<string | number | boolean>;
// number | boolean（string が除外された）

// 3. Exclude ユーティリティ型の内部実装
// type Exclude<T, U> = T extends U ? never : T;
type WithoutNull = Exclude<string | null | undefined, null | undefined>;
// string

// 4. never を使った型レベルのアサーション
type Assert<T extends true> = T;
type IsString<T> = T extends string ? true : false;

// コンパイル時のテスト
type _test1 = Assert<IsString<"hello">>; // OK
// type _test2 = Assert<IsString<42>>;   // エラー: false は true に代入不可

// 5. never と条件分岐
type IsNever<T> = [T] extends [never] ? true : false;
type Test1 = IsNever<never>;  // true
type Test2 = IsNever<string>; // false

// 6. プロパティの禁止（特定のキーを使えなくする）
type Without<T, K extends keyof T> = {
  [P in keyof T as P extends K ? never : P]: T[P];
};

interface FullUser {
  id: string;
  name: string;
  email: string;
  password: string;
}

type PublicUser = Without<FullUser, "password">;
// { id: string; name: string; email: string }
```

### コード例9: void と undefined

```typescript
// void: 戻り値がないことを示す
function logMessage(msg: string): void {
  console.log(msg);
  // return undefined; は暗黙的に行われる
}

// undefined: 値としてのundefined
let u: undefined = undefined;

// void と undefined の違い
// void はコールバックの戻り値を「無視する」意味を持つ
type Callback = () => void;

// void コールバックは実際には値を返しても良い（値が無視される）
const cb: Callback = () => 42; // OK（42は無視される）
```

### void と undefined の詳細

```typescript
// void の正確な意味
// void は「戻り値を使わない」という意図を表す

// undefined は「値が undefined である」という具体的な型
function getUndefined(): undefined {
  return undefined; // 明示的に undefined を返す必要がある
}

function getVoid(): void {
  // return; も return undefined; も OK
  // 何も return しなくても OK
}

// 重要な違い: void を返す関数型の代入互換性
type VoidFunc = () => void;
type UndefinedFunc = () => undefined;

// void型の関数は実際には何でも返せる（値は無視される）
const f1: VoidFunc = () => 42;         // OK
const f2: VoidFunc = () => "hello";    // OK
const f3: VoidFunc = () => true;       // OK

// undefined型の関数は undefined のみ返せる
// const f4: UndefinedFunc = () => 42; // エラー
const f5: UndefinedFunc = () => undefined; // OK

// この仕様の理由: Array.prototype.forEach の型
// forEach のコールバックは void を返す
// void でないと、map のように値を返すコールバックも使えなくなる
const arr = [1, 2, 3];
arr.forEach(n => n * 2); // OK: n * 2 の結果は無視される

// null と undefined の使い分け
// TypeScript では strictNullChecks: true が推奨
// null: 意図的に「値が存在しない」ことを示す
// undefined: 「値が設定されていない」ことを示す

interface UserProfile {
  name: string;
  bio: string | null;        // 意図的に空: ユーザーが「未設定」を選択
  middleName?: string;       // 省略可能: 設定されていない可能性
  // middleName: string | undefined と同等
}

// 実務での慣例
// - API レスポンス: null を使う（JSONに undefined は存在しない）
// - オプショナルプロパティ: undefined（省略可能という意味）
// - 関数の戻り値（見つからない場合）: undefined（Array.find 等の慣例）
//   ただし null を返すパターン（document.getElementById 等）も一般的
```

---

## 6. 型推論（Type Inference）

### 基本的な型推論

```typescript
// TypeScriptは多くの場面で型を自動的に推論する

// 変数の初期化時
const name = "TypeScript";  // string と推論
const age = 12;             // 12 と推論（const のため リテラル型）
let count = 0;              // number と推論（let のため ワイド型）

// const vs let の推論の違い
const x = "hello";  // 型: "hello"（リテラル型）
let y = "hello";    // 型: string（ワイド型）

// const でもオブジェクトの場合は中身がワイドになる
const config = { host: "localhost", port: 3000 };
// 型: { host: string; port: number }
// host は "localhost" ではなく string と推論される

// as const でリテラル型を維持
const configConst = { host: "localhost", port: 3000 } as const;
// 型: { readonly host: "localhost"; readonly port: 3000 }

// 関数の戻り値推論
function add(a: number, b: number) {
  return a + b; // 戻り値は number と推論
}

function createPair<T>(value: T) {
  return [value, value] as const; // readonly [T, T] と推論
}

// 条件式の推論
const result = Math.random() > 0.5 ? "yes" : "no";
// 型: "yes" | "no"

const value = Math.random() > 0.5 ? 42 : "hello";
// 型: 42 | "hello" (const の場合)
// 型: number | string (let の場合)
```

### 文脈的型推論（Contextual Typing）

```typescript
// 関数の引数の型から、コールバックのパラメータ型が推論される

// 例1: 配列メソッド
const numbers = [1, 2, 3, 4, 5];
const doubled = numbers.map(n => n * 2);
// n は number と推論される（numbers が number[] なので）

// 例2: イベントハンドラ
document.addEventListener("click", event => {
  // event は MouseEvent と推論される（"click" イベントの型定義から）
  console.log(event.clientX, event.clientY);
});

document.addEventListener("keydown", event => {
  // event は KeyboardEvent と推論される
  console.log(event.key);
});

// 例3: コールバック関数
type Comparator<T> = (a: T, b: T) => number;

function sort<T>(arr: T[], comparator: Comparator<T>): T[] {
  return [...arr].sort(comparator);
}

// a と b は number と推論される
const sorted = sort([3, 1, 2], (a, b) => a - b);

// 例4: Promise のコールバック
const promise = new Promise<string>((resolve, reject) => {
  // resolve は (value: string) => void と推論
  // reject は (reason?: any) => void と推論
  resolve("done");
});

// 例5: satisfies を使った文脈的型推論
type Routes = Record<string, { method: "GET" | "POST"; handler: () => void }>;

const routes = {
  "/users": { method: "GET", handler: () => {} },
  "/users/create": { method: "POST", handler: () => {} },
} satisfies Routes;
// routes の型は推論された具体的な型を維持しつつ、Routes との互換性が保証される
// routes["/users"].method は "GET" 型（"GET" | "POST" ではない）
```

### 型の拡張（Widening）と絞り込み（Narrowing）

```typescript
// ===== Widening（型の拡張） =====
// let で宣言した変数は、リテラル型ではなくワイド型に推論される

let x = "hello";  // string（widenされた）
const y = "hello"; // "hello"（widenされない）

// Wideningが起こる場面
let a = 42;        // number
let b = true;      // boolean
let c = null;      // any（strictNullChecks無効時）/ null（有効時）
let d = undefined; // any（strictNullChecks無効時）/ undefined（有効時）

// Wideningを防ぐ方法
let e: "hello" = "hello"; // 明示的な型アノテーション
let f = "hello" as const; // as const アサーション（ただしletでは代入できなくなる）

// ===== Narrowing（型の絞り込み） =====
// TypeScriptの制御フロー分析により、型が自動的に絞り込まれる

function processValue(value: string | number | null) {
  // この時点: string | number | null

  if (value === null) {
    // ここ: null
    return;
  }
  // ここ: string | number

  if (typeof value === "string") {
    // ここ: string
    console.log(value.toUpperCase());
  } else {
    // ここ: number
    console.log(value.toFixed(2));
  }
}

// Narrowing のパターン一覧

// 1. typeof ガード
function typeofGuard(x: unknown) {
  if (typeof x === "string") { /* x: string */ }
  if (typeof x === "number") { /* x: number */ }
  if (typeof x === "boolean") { /* x: boolean */ }
  if (typeof x === "bigint") { /* x: bigint */ }
  if (typeof x === "symbol") { /* x: symbol */ }
  if (typeof x === "function") { /* x: Function */ }
  if (typeof x === "object" && x !== null) { /* x: object */ }
}

// 2. instanceof ガード
function instanceofGuard(x: Date | RegExp) {
  if (x instanceof Date) {
    x.getFullYear(); // x: Date
  } else {
    x.test("hello"); // x: RegExp
  }
}

// 3. in ガード
interface Fish { swim(): void; }
interface Bird { fly(): void; }

function inGuard(animal: Fish | Bird) {
  if ("swim" in animal) {
    animal.swim(); // animal: Fish
  } else {
    animal.fly(); // animal: Bird
  }
}

// 4. 等価性チェック
function equalityGuard(x: string | number, y: string | boolean) {
  if (x === y) {
    // x と y の共通型: string
    x.toUpperCase(); // x: string
    y.toUpperCase(); // y: string
  }
}

// 5. truthiness チェック
function truthinessGuard(x: string | null | undefined) {
  if (x) {
    x.toUpperCase(); // x: string（null, undefined, "" は除外）
  }
}

// 6. Discriminated Union（タグ付きユニオン）
type Shape =
  | { kind: "circle"; radius: number }
  | { kind: "square"; side: number }
  | { kind: "rectangle"; width: number; height: number };

function area(shape: Shape): number {
  switch (shape.kind) {
    case "circle":
      return Math.PI * shape.radius ** 2; // shape: { kind: "circle"; radius: number }
    case "square":
      return shape.side ** 2; // shape: { kind: "square"; side: number }
    case "rectangle":
      return shape.width * shape.height; // shape: { kind: "rectangle"; ... }
  }
}

// 7. ユーザー定義型ガード（Type Predicate）
function isString(value: unknown): value is string {
  return typeof value === "string";
}

function isNonNull<T>(value: T | null | undefined): value is T {
  return value !== null && value !== undefined;
}

const values: (string | null)[] = ["hello", null, "world", null];
const nonNullValues = values.filter(isNonNull); // string[]

// 8. assertion 関数
function assertIsString(value: unknown): asserts value is string {
  if (typeof value !== "string") {
    throw new Error(`Expected string, got ${typeof value}`);
  }
}

function processInput(input: unknown): string {
  assertIsString(input);
  // ここ以降、input は string 型として扱える
  return input.toUpperCase();
}
```

---

## 7. 型エイリアスとリテラル型の高度な活用

### テンプレートリテラル型

```typescript
// テンプレートリテラル型の基本
type Greeting = `Hello, ${string}`;
type EventName = `on${string}`;

// リテラル型の組み合わせ
type Vertical = "top" | "middle" | "bottom";
type Horizontal = "left" | "center" | "right";
type Position = `${Vertical}-${Horizontal}`;
// "top-left" | "top-center" | "top-right" | "middle-left" | ... (9通り)

// 組み込みの文字列操作型
type Upper = Uppercase<"hello">;     // "HELLO"
type Lower = Lowercase<"HELLO">;     // "hello"
type Capitalized = Capitalize<"hello">; // "Hello"
type Uncapitalized = Uncapitalize<"Hello">; // "hello"

// テンプレートリテラル型の実務パターン
// CSS プロパティ名の型安全な生成
type CSSProperty = "margin" | "padding" | "border";
type CSSDirection = "top" | "right" | "bottom" | "left";
type CSSDirectionalProperty = `${CSSProperty}-${CSSDirection}`;
// "margin-top" | "margin-right" | ... | "border-left"

// APIエンドポイントの型
type Entity = "user" | "post" | "comment";
type CrudEndpoint = `/${Entity}s` | `/${Entity}s/:id`;
// "/users" | "/users/:id" | "/posts" | "/posts/:id" | "/comments" | "/comments/:id"

// イベント名の自動生成
type ModelEvents<T extends string> = `${T}Created` | `${T}Updated` | `${T}Deleted`;
type UserEvents = ModelEvents<"user">;
// "userCreated" | "userUpdated" | "userDeleted"

// Getter/Setter の型
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
```

### 型のブランディング（Branded Types）

```typescript
// TypeScript は構造的型付けなので、同じ構造の型は互換性がある
// これが問題になる場合、ブランド型で名前的型付けを模倣する

// 問題: string 同士は全て互換性がある
type UserId = string;
type ProductId = string;

function getUser(id: UserId): void { /* ... */ }
function getProduct(id: ProductId): void { /* ... */ }

const userId: UserId = "user-123";
const productId: ProductId = "prod-456";

getUser(productId); // エラーにならない！（両方 string なので）

// 解決: ブランド型を使う
type BrandedUserId = string & { readonly __brand: unique symbol };
type BrandedProductId = string & { readonly __brand: unique symbol };

function createUserId(id: string): BrandedUserId {
  // バリデーションを行う
  if (!id.startsWith("user-")) {
    throw new Error("Invalid user ID format");
  }
  return id as BrandedUserId;
}

function createProductId(id: string): BrandedProductId {
  if (!id.startsWith("prod-")) {
    throw new Error("Invalid product ID format");
  }
  return id as BrandedProductId;
}

function getUserById(id: BrandedUserId): void { /* ... */ }
function getProductById(id: BrandedProductId): void { /* ... */ }

const safeUserId = createUserId("user-123");
const safeProductId = createProductId("prod-456");

getUserById(safeUserId);     // OK
// getUserById(safeProductId); // エラー！型が異なる

// 汎用的なブランド型のユーティリティ
type Brand<T, B extends string> = T & { readonly __brand: B };

type Email = Brand<string, "Email">;
type URL = Brand<string, "URL">;
type PositiveNumber = Brand<number, "PositiveNumber">;

function createEmail(value: string): Email {
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) {
    throw new Error("Invalid email");
  }
  return value as Email;
}

function createPositiveNumber(value: number): PositiveNumber {
  if (value <= 0) {
    throw new Error("Must be positive");
  }
  return value as PositiveNumber;
}
```

---

## アンチパターン

### アンチパターン1: any で逃げる

```typescript
// BAD: 型がわからないときに any を使う
function parseJSON(json: string): any {
  return JSON.parse(json);
}
const data = parseJSON('{"name":"Alice"}');
data.nonExistent.property; // 実行時エラー、コンパイラは警告しない

// GOOD: unknown を使い、型ガードで安全に処理
function parseJSON(json: string): unknown {
  return JSON.parse(json);
}
const data = parseJSON('{"name":"Alice"}');
if (data !== null && typeof data === "object" && "name" in data) {
  console.log((data as { name: string }).name);
}
// さらに良い: Zodなどでバリデーション
```

### アンチパターン2: 不必要な型アサーション

```typescript
// BAD: 根拠なく型アサーション（as）を使う
const input = document.getElementById("name") as HTMLInputElement;
input.value; // 実際にはnullかもしれない → 実行時エラー

// GOOD: 型ガードで安全に確認
const input = document.getElementById("name");
if (input instanceof HTMLInputElement) {
  input.value; // 安全
}
```

### アンチパターン3: 過度にワイドな型

```typescript
// BAD: string や number を使いすぎる
interface Config {
  mode: string;      // 何でも入る
  retries: number;   // 負の数も入る
  level: string;     // 意味のない値も入る
}

// GOOD: リテラル型やブランド型で制約する
interface Config {
  mode: "development" | "staging" | "production";
  retries: 0 | 1 | 2 | 3 | 5 | 10;
  level: "debug" | "info" | "warn" | "error";
}

// さらに良い: 値の制約をランタイムでも保証
import { z } from "zod";

const ConfigSchema = z.object({
  mode: z.enum(["development", "staging", "production"]),
  retries: z.number().int().min(0).max(10),
  level: z.enum(["debug", "info", "warn", "error"]),
});

type Config = z.infer<typeof ConfigSchema>;
```

### アンチパターン4: null チェックの漏れ

```typescript
// BAD: strictNullChecks を無効にする
// tsconfig: "strictNullChecks": false

// GOOD: strictNullChecks を有効にし、null を明示的に扱う
function findUser(id: string): User | null {
  const user = database.get(id);
  return user ?? null;
}

const user = findUser("123");
// user.name; // エラー: Object is possibly 'null'
if (user !== null) {
  user.name; // OK
}

// Optional Chaining と Nullish Coalescing の活用
const userName = user?.name ?? "Anonymous";
const userAge = user?.profile?.age ?? 0;
```

### アンチパターン5: 配列の安全でないインデックスアクセス

```typescript
// BAD: 配列の要素が必ず存在すると仮定する
const items: string[] = ["a", "b", "c"];
const item: string = items[10]; // undefined が返るが、型は string
item.toUpperCase(); // 実行時エラー！

// GOOD: noUncheckedIndexedAccess を有効にする
// tsconfig: "noUncheckedIndexedAccess": true
// items[10] の型は string | undefined になる

const safeItem = items[10];
// safeItem.toUpperCase(); // エラー: Object is possibly 'undefined'
if (safeItem !== undefined) {
  safeItem.toUpperCase(); // OK
}

// at() メソッドの活用（ES2022+）
const lastItem = items.at(-1); // string | undefined（常にundefinedを含む）
```

---

## 8. 型アサーションと型ガードのベストプラクティス

### 型アサーションの正しい使い方

```typescript
// 型アサーション（as）は「コンパイラより詳しいことを知っている」場合のみ使う

// 許容されるケース1: DOM要素の型
const canvas = document.getElementById("myCanvas");
if (canvas instanceof HTMLCanvasElement) {
  const ctx = canvas.getContext("2d"); // OK: canvas は HTMLCanvasElement
}

// 許容されるケース2: 外部ライブラリの型が不正確
// ライブラリの型定義が間違っている場合の一時的な回避策
const result = externalLib.getData() as unknown as CorrectType;
// TODO: @types/external-lib にPRを送る

// 許容されるケース3: テストコード
// テストでは意図的に不正な値を渡すことがある
it("should handle invalid input", () => {
  expect(() => processUser(null as unknown as User)).toThrow();
});

// 非推奨: 根拠のない型アサーション
// const data = fetchData() as UserData; // データの形状が保証されない
```

### 型ガードの設計パターン

```typescript
// 汎用的な型ガードの実装

// 1. プリミティブ型のガード集
const is = {
  string: (value: unknown): value is string => typeof value === "string",
  number: (value: unknown): value is number =>
    typeof value === "number" && !Number.isNaN(value),
  boolean: (value: unknown): value is boolean => typeof value === "boolean",
  null: (value: unknown): value is null => value === null,
  undefined: (value: unknown): value is undefined => value === undefined,
  array: (value: unknown): value is unknown[] => Array.isArray(value),
  object: (value: unknown): value is Record<string, unknown> =>
    typeof value === "object" && value !== null && !Array.isArray(value),
  function: (value: unknown): value is Function => typeof value === "function",
  date: (value: unknown): value is Date => value instanceof Date,
  regExp: (value: unknown): value is RegExp => value instanceof RegExp,
  error: (value: unknown): value is Error => value instanceof Error,
};

// 使用例
function processValue(value: unknown): string {
  if (is.string(value)) return value.toUpperCase();
  if (is.number(value)) return value.toFixed(2);
  if (is.boolean(value)) return value ? "true" : "false";
  if (is.null(value)) return "null";
  if (is.undefined(value)) return "undefined";
  if (is.array(value)) return `[${value.length} items]`;
  if (is.object(value)) return JSON.stringify(value);
  return String(value);
}

// 2. オブジェクトのプロパティチェック
function hasProperty<K extends string>(
  obj: unknown,
  key: K
): obj is Record<K, unknown> {
  return typeof obj === "object" && obj !== null && key in obj;
}

function hasProperties<K extends string>(
  obj: unknown,
  keys: K[]
): obj is Record<K, unknown> {
  return (
    typeof obj === "object" &&
    obj !== null &&
    keys.every(key => key in obj)
  );
}

// 3. 配列要素の型ガード
function isArrayOf<T>(
  value: unknown,
  guard: (item: unknown) => item is T
): value is T[] {
  return Array.isArray(value) && value.every(guard);
}

const data: unknown = [1, 2, 3];
if (isArrayOf(data, is.number)) {
  // data は number[] として扱える
  const sum = data.reduce((a, b) => a + b, 0);
}
```

---

## FAQ

### Q1: `string` と `String` の違いは？

**A:** 小文字の `string` はTypeScriptのプリミティブ型です。大文字の `String` はJavaScriptのラッパーオブジェクト型です。常に小文字の `string` を使ってください。`number` / `Number`、`boolean` / `Boolean` も同様です。

```typescript
// プリミティブ型（推奨）
const name: string = "Alice";
const age: number = 30;
const active: boolean = true;

// ラッパーオブジェクト型（非推奨）
// const name: String = new String("Alice"); // 使わない
// const age: Number = new Number(30);       // 使わない
// const active: Boolean = new Boolean(true); // 使わない

// ラッパーオブジェクトはプリミティブに代入できるが、逆はできない
const s: String = "hello"; // OK（暗黙変換）
// const p: string = new String("hello"); // エラー
```

### Q2: タプルの要素数チェックは実行時にも行われますか？

**A:** いいえ。タプルの型チェックはコンパイル時のみです。実行時にはただの配列になります。実行時の要素数チェックが必要な場合は、手動でバリデーションを書くか、Zodなどを使います。

```typescript
// コンパイル時のチェック
type Point = [number, number];
// const p: Point = [1]; // エラー: 要素が足りない
// const p: Point = [1, 2, 3]; // エラー: 要素が多い
const p: Point = [1, 2]; // OK

// 実行時はただの配列
console.log(Array.isArray(p)); // true
console.log(p.length); // 2（ただし型レベルでは length: 2）

// 実行時にタプルの形状を検証する関数
function isPoint(value: unknown): value is Point {
  return (
    Array.isArray(value) &&
    value.length === 2 &&
    typeof value[0] === "number" &&
    typeof value[1] === "number"
  );
}
```

### Q3: enum は使うべきですか？

**A:** 2025年現在、多くのTypeScriptスタイルガイドでは `const enum` か、Union型 + `as const` オブジェクトパターンが推奨されています。通常の数値 enum は逆引きマッピングのためにランタイムコードを生成し、バンドルサイズに影響するためです。

### Q4: noUncheckedIndexedAccess は有効にすべきですか？

**A:** はい、推奨します。このオプションを有効にすると、配列やオブジェクトのインデックスアクセスの結果に `undefined` が追加され、安全なコードを書くことを促します。

```typescript
// noUncheckedIndexedAccess: true の場合
const arr: string[] = ["a", "b", "c"];
const item = arr[0]; // string | undefined

// Record のアクセスも影響を受ける
const map: Record<string, number> = { a: 1, b: 2 };
const value = map["c"]; // number | undefined

// 非破壊的な回避方法
// 1. 存在チェック
if (item !== undefined) {
  console.log(item.toUpperCase());
}

// 2. Non-null assertion（確実に存在する場合のみ）
const first = arr[0]!; // string（undefined を除外）

// 3. at() メソッド（返り値は常に T | undefined）
const last = arr.at(-1);
```

### Q5: symbol はどういう場面で使うべきですか？

**A:** symbolは以下の場面で活用されます。
1. **オブジェクトのメタデータプロパティ**: 通常のプロパティと衝突しないキーとして
2. **Well-known Symbols**: `Symbol.iterator`, `Symbol.dispose` などの組み込みプロトコルの実装
3. **プライベートに近いプロパティ**: 外部からアクセスされにくいプロパティキーとして（ただし真のプライベートではない）
4. **ライブラリのプラグインシステム**: 識別子の衝突を避けるため

一般的なアプリケーション開発では直接使う機会は少ないですが、ライブラリやフレームワークの内部実装では頻繁に使われます。

---

## まとめ

| 型カテゴリ | 型 | 用途 |
|-----------|-----|------|
| プリミティブ | string, number, boolean, symbol, bigint | 基本的な値 |
| リテラル | "hello", 42, true | 特定の値に限定 |
| テンプレートリテラル | `Hello, ${string}` | 文字列パターンの制約 |
| コレクション | T[], [T1, T2] | 同種/異種データの集合 |
| 列挙 | enum, const enum | 名前付き定数の集合 |
| 特殊（危険） | any | 型チェック無効化（非推奨） |
| 特殊（安全） | unknown | 型安全な任意値 |
| 特殊（不在） | never | 到達不能、網羅性チェック |
| 特殊（空） | void, null, undefined | 戻り値なし、値の不在 |
| ブランド型 | Brand<T, B> | 構造的型付けに名前的型付けを追加 |

---

## 次に読むべきガイド

- [02-functions-and-objects.md](./02-functions-and-objects.md) -- 関数とオブジェクト型
- [03-union-intersection.md](./03-union-intersection.md) -- Union型とIntersection型

---

## 参考文献

1. **TypeScript Handbook: Everyday Types** -- https://www.typescriptlang.org/docs/handbook/2/everyday-types.html
2. **TypeScript Handbook: Narrowing** -- https://www.typescriptlang.org/docs/handbook/2/narrowing.html
3. **TypeScript Handbook: Template Literal Types** -- https://www.typescriptlang.org/docs/handbook/2/template-literal-types.html
4. **TypeScript Deep Dive: TypeScript's Type System** -- https://basarat.gitbook.io/typescript/type-system
5. **Effective TypeScript (Dan Vanderkam著, O'Reilly)** -- 特に Item 7: Think of Types as Sets of Values
6. **TypeScript Playground** -- https://www.typescriptlang.org/play -- 型の動作を即座に確認できる
