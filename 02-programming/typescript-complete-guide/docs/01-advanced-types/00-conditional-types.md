# 条件型（Conditional Types）

> 型レベルの条件分岐を実現する強力な型機能。extends、infer、分配条件型、再帰的条件型によって高度な型変換を行う。

## この章で学ぶこと

1. **条件型の基本** -- `T extends U ? X : Y` の構文、型レベルの if/else
2. **infer キーワード** -- 型パターンマッチングによる型の抽出
3. **分配条件型** -- Union型の分配処理とその制御
4. **再帰的条件型** -- 深いネスト構造に対する再帰的な型変換
5. **実践的なユーティリティ型の構築** -- 条件型を組み合わせた実用的な型の設計
6. **パフォーマンスとデバッグ** -- 条件型のコンパイル性能と問題解決手法

---

## 1. 条件型の基本

### 1.1 基本構文と評価ルール

条件型は TypeScript の型システムにおける「if/else」に相当する。型パラメータが特定の条件を満たすかどうかに基づいて、異なる型を返す。

```typescript
// 基本構文: T extends U ? TrueType : FalseType
// 「T が U に代入可能であれば TrueType、そうでなければ FalseType」

// 最も単純な例: 型が string かどうかを判定
type IsString<T> = T extends string ? true : false;

type A = IsString<string>;   // true
type B = IsString<number>;   // false
type C = IsString<"hello">;  // true（リテラル型もstringのサブタイプ）
type D = IsString<any>;      // boolean（any は特殊: true | false に展開される）
type E = IsString<never>;    // never（never は空のユニオンとして扱われる）

// extends の意味: 「サブタイプである」
// string extends string → true
// "hello" extends string → true（リテラル型はstringのサブタイプ）
// number extends string → false
// string extends "hello" → false（stringは"hello"のスーパータイプ）
```

### 1.2 ネストした条件型

```typescript
// 型に応じて異なる変換を適用
type TypeName<T> =
  T extends string ? "string" :
  T extends number ? "number" :
  T extends boolean ? "boolean" :
  T extends undefined ? "undefined" :
  T extends null ? "null" :
  T extends symbol ? "symbol" :
  T extends bigint ? "bigint" :
  T extends Function ? "function" :
  T extends any[] ? "array" :
  "object";

type T1 = TypeName<string>;       // "string"
type T2 = TypeName<() => void>;   // "function"
type T3 = TypeName<string[]>;     // "array"
type T4 = TypeName<null>;         // "null"
type T5 = TypeName<{ x: 1 }>;    // "object"
type T6 = TypeName<42>;           // "number"
type T7 = TypeName<true>;         // "boolean"
type T8 = TypeName<symbol>;       // "symbol"
type T9 = TypeName<bigint>;       // "bigint"
```

### 1.3 実用例: null安全なユーティリティ型

```typescript
// null と undefined を除外する型（標準ライブラリの NonNullable）
type NonNullable<T> = T extends null | undefined ? never : T;

type R1 = NonNullable<string | null>;             // string
type R2 = NonNullable<number | undefined>;         // number
type R3 = NonNullable<string | null | undefined>;  // string
type R4 = NonNullable<null | undefined>;           // never

// 実務例: API レスポンスの型からnullを除外
interface ApiResponse {
  user: {
    name: string;
    email: string | null;
    phone: string | null | undefined;
  };
}

// Nullableなフィールドの型を安全に取得
type UserEmail = NonNullable<ApiResponse["user"]["email"]>;  // string
type UserPhone = NonNullable<ApiResponse["user"]["phone"]>;  // string
```

### 1.4 条件型とジェネリック制約の組み合わせ

```typescript
// ジェネリック制約と条件型を組み合わせた実用パターン
type MessageFor<T extends { type: string }> =
  T extends { type: "error" } ? { message: string; code: number } :
  T extends { type: "warning" } ? { message: string } :
  T extends { type: "info" } ? { text: string } :
  never;

// 使用例
interface ErrorEvent { type: "error"; }
interface WarningEvent { type: "warning"; }
interface InfoEvent { type: "info"; }

type ErrorMsg = MessageFor<ErrorEvent>;    // { message: string; code: number }
type WarningMsg = MessageFor<WarningEvent>; // { message: string }
type InfoMsg = MessageFor<InfoEvent>;       // { text: string }

// 関数での使用: 型安全なイベントハンドラ
function handleEvent<T extends { type: string }>(
  event: T,
  handler: (msg: MessageFor<T>) => void
): void {
  // 実装...
}
```

### 条件型の評価フロー

```
  条件型: T extends U ? X : Y

  評価プロセス:
  +----------+
  | T は U の |---Yes---> X を返す
  | サブタイプ|
  | か？      |---No----> Y を返す
  +----------+

  例: IsString<number>
  +----------+
  | number   |
  | extends  |---No----> false
  | string?  |
  +----------+

  複数段のネスト:
  TypeName<string[]>
  +----------+
  | string[] |           +----------+
  | extends  |---No----> | string[] |           +----------+
  | string?  |           | extends  |---No----> | ...      |
  +----------+           | number?  |           |          |---...
                         +----------+           +----------+
                         ...最終的に "array" に到達
```

### 1.5 条件型と関数オーバーロードの比較

```typescript
// 関数オーバーロードによるアプローチ
function processValue(value: string): string[];
function processValue(value: number): number;
function processValue(value: string | number): string[] | number {
  if (typeof value === "string") {
    return value.split(",");
  }
  return value * 2;
}

// 条件型によるアプローチ（よりスケーラブル）
type ProcessResult<T> =
  T extends string ? string[] :
  T extends number ? number :
  never;

function processValueGeneric<T extends string | number>(
  value: T
): ProcessResult<T> {
  if (typeof value === "string") {
    return value.split(",") as ProcessResult<T>;
  }
  return (value as number) * 2 as ProcessResult<T>;
}

// 両方とも型安全に動作
const strResult = processValueGeneric("a,b,c");  // string[]
const numResult = processValueGeneric(42);         // number
```

### 1.6 extends の意味を深く理解する

```typescript
// extends は「サブタイプ関係」を判定する
// TypeScript のサブタイプ関係は構造的型付け（structural typing）に基づく

// 基本的なサブタイプ関係
type Test1 = "hello" extends string ? true : false;     // true（リテラル→一般）
type Test2 = string extends "hello" ? true : false;     // false（一般→リテラルは不可）
type Test3 = 42 extends number ? true : false;          // true
type Test4 = true extends boolean ? true : false;       // true

// オブジェクト型のサブタイプ: プロパティが多いほうがサブタイプ
type Test5 = { a: string; b: number } extends { a: string } ? true : false;  // true
type Test6 = { a: string } extends { a: string; b: number } ? true : false;  // false

// 関数型のサブタイプ: 反変性（引数は逆方向）
type Test7 = ((x: string) => void) extends ((x: string | number) => void) ? true : false;
// false（引数は反変）

type Test8 = ((x: string | number) => void) extends ((x: string) => void) ? true : false;
// true（より広い引数型を受け取れる関数はサブタイプ）

// any と unknown の特殊性
type Test9 = any extends string ? true : false;      // boolean（true | false）
type Test10 = unknown extends string ? true : false;  // false
type Test11 = string extends any ? true : false;      // true
type Test12 = string extends unknown ? true : false;  // true

// never の特殊性
type Test13 = never extends string ? true : false;    // never（分配で空のユニオン）
type TestNeverWrapped = [never] extends [string] ? true : false;  // true
```

---

## 2. infer キーワード

### 2.1 基本的な型の抽出

`infer` キーワードは条件型の `extends` 節内で使用し、型のパターンマッチングにより部分的な型を抽出する。

```typescript
// 関数の戻り値型を取得（標準ライブラリの ReturnType）
type ReturnType<T> = T extends (...args: any[]) => infer R ? R : never;

type A = ReturnType<() => string>;             // string
type B = ReturnType<(x: number) => boolean>;   // boolean
type C = ReturnType<typeof Math.max>;          // number
type D = ReturnType<typeof JSON.parse>;        // any

// Promiseの中身を取得（標準ライブラリの Awaited）
type Awaited<T> = T extends Promise<infer U> ? Awaited<U> : T;

type E = Awaited<Promise<string>>;            // string
type F = Awaited<Promise<Promise<number>>>;   // number（再帰的に展開）
type G = Awaited<string>;                     // string（Promiseでなければそのまま）

// 関数のパラメータ型を取得（標準ライブラリの Parameters）
type Parameters<T> = T extends (...args: infer P) => any ? P : never;

type H = Parameters<(a: string, b: number) => void>;  // [string, number]
type I = Parameters<() => void>;                        // []

// 配列の要素型を取得
type ElementType<T> = T extends (infer U)[] ? U : never;

type J = ElementType<string[]>;   // string
type K = ElementType<number[]>;   // number
type L = ElementType<string>;     // never（配列でないため）
```

### 2.2 infer の高度な使い方

```typescript
// オブジェクトのプロパティ型を抽出
type PropType<T, K extends string> =
  T extends { [key in K]: infer V } ? V : never;

type UserName = PropType<{ name: string; age: number }, "name">;  // string
type UserAge = PropType<{ name: string; age: number }, "age">;    // number

// テンプレートリテラル型でのinfer
type ExtractRouteParams<T extends string> =
  T extends `${string}:${infer Param}/${infer Rest}`
    ? Param | ExtractRouteParams<`/${Rest}`>
    : T extends `${string}:${infer Param}`
      ? Param
      : never;

type Params = ExtractRouteParams<"/users/:userId/posts/:postId">;
// "userId" | "postId"

// コンストラクタのインスタンス型を取得（標準ライブラリの InstanceType）
type InstanceType<T> = T extends new (...args: any[]) => infer R ? R : never;

class MyClass {
  constructor(public name: string) {}
}

type MyInstance = InstanceType<typeof MyClass>;  // MyClass

// コンストラクタのパラメータ型を取得（標準ライブラリの ConstructorParameters）
type ConstructorParameters<T> =
  T extends new (...args: infer P) => any ? P : never;

type MyCtorParams = ConstructorParameters<typeof MyClass>;  // [string]
```

### 2.3 複数の infer を使ったパターンマッチング

```typescript
// 関数型から引数と戻り値を同時に抽出
type FuncParts<T> =
  T extends (...args: infer A) => infer R
    ? { args: A; returnType: R }
    : never;

type Parts = FuncParts<(x: string, y: number) => boolean>;
// { args: [x: string, y: number]; returnType: boolean }

// Map型からキーと値を同時に抽出
type MapParts<T> =
  T extends Map<infer K, infer V>
    ? { key: K; value: V }
    : never;

type MP = MapParts<Map<string, number>>;
// { key: string; value: number }

// タプル型の先頭と残りを分割
type Head<T extends any[]> =
  T extends [infer H, ...any[]] ? H : never;

type Tail<T extends any[]> =
  T extends [any, ...infer R] ? R : never;

type H = Head<[1, 2, 3]>;  // 1
type TT = Tail<[1, 2, 3]>; // [2, 3]

// タプル型の最後の要素を取得
type Last<T extends any[]> =
  T extends [...any[], infer L] ? L : never;

type LL = Last<[1, 2, 3]>;  // 3

// タプル型から最後の要素を除いた残りを取得
type Init<T extends any[]> =
  T extends [...infer I, any] ? I : never;

type II = Init<[1, 2, 3]>;  // [1, 2]
```

### 2.4 infer と共変位置・反変位置

```typescript
// infer の位置によって推論結果が異なる

// 共変位置（covariant position）での infer: ユニオン型が推論される
type Foo<T> =
  T extends { a: infer U; b: infer U } ? U : never;

type FooResult = Foo<{ a: string; b: number }>;
// string | number（共変位置なのでユニオン）

// 反変位置（contravariant position）での infer: インターセクション型が推論される
type Bar<T> =
  T extends {
    a: (x: infer U) => void;
    b: (x: infer U) => void;
  } ? U : never;

type BarResult = Bar<{
  a: (x: string) => void;
  b: (x: number) => void;
}>;
// string & number → never（stringとnumberのインターセクションは存在しない）

// 実用例: 関数のオーバーロードからすべてのパラメータ型を取得
type UnionOfParams<T> =
  T extends {
    (x: infer A): any;
    (x: infer B): any;
  } ? A | B : never;
```

### 2.5 infer を使った文字列操作

```typescript
// 文字列の先頭文字を取得
type FirstChar<T extends string> =
  T extends `${infer F}${string}` ? F : never;

type FC = FirstChar<"hello">;  // "h"

// 文字列の残り部分を取得
type RestOfString<T extends string> =
  T extends `${string}${infer R}` ? R : never;

type RS = RestOfString<"hello">;  // "ello"

// 文字列を指定文字で分割
type Split<S extends string, D extends string> =
  S extends `${infer Head}${D}${infer Tail}`
    ? [Head, ...Split<Tail, D>]
    : [S];

type Splitted = Split<"a.b.c", ".">;  // ["a", "b", "c"]

// 文字列の置換
type Replace<
  S extends string,
  From extends string,
  To extends string
> = S extends `${infer Before}${From}${infer After}`
  ? `${Before}${To}${After}`
  : S;

type Replaced = Replace<"hello world", "world", "TypeScript">;
// "hello TypeScript"

// すべてを置換
type ReplaceAll<
  S extends string,
  From extends string,
  To extends string
> = S extends `${infer Before}${From}${infer After}`
  ? ReplaceAll<`${Before}${To}${After}`, From, To>
  : S;

type ReplacedAll = ReplaceAll<"a-b-c-d", "-", ".">;
// "a.b.c.d"
```

### infer の位置と抽出される型

```
  パターン                          infer の位置       抽出される型
+-------------------------------------+----------------+------------------+
| T extends (infer U)[]               | 配列の要素     | 要素型           |
| T extends (...args: infer P) => any | 関数の引数     | パラメータタプル |
| T extends (...args: any) => infer R | 関数の戻り値   | 戻り値型         |
| T extends Promise<infer U>          | Promiseの中身  | 解決値の型       |
| T extends Map<infer K, infer V>     | Mapの型引数    | キーと値の型     |
| T extends Set<infer U>              | Setの型引数    | 要素型           |
| T extends { a: infer U }            | プロパティ値   | プロパティの型   |
| T extends [infer H, ...infer T]     | タプルの分割   | 先頭と残り       |
| T extends `${infer A}.${infer B}`   | 文字列の分割   | 部分文字列       |
| T extends new (...args: infer P)... | コンストラクタ | コンストラクタ引数|
+-------------------------------------+----------------+------------------+
```

---

## 3. 分配条件型

### 3.1 分配（Distribution）の仕組み

条件型にユニオン型が「裸の（naked）」型パラメータとして渡されると、ユニオンの各メンバーに対して個別に条件型が評価される。これを「分配（distribution）」と呼ぶ。

```typescript
// Union型が条件型に渡されると「分配」される
type ToArray<T> = T extends any ? T[] : never;

// string | number が渡されると:
// ToArray<string> | ToArray<number>
// = string[] | number[]
type A = ToArray<string | number>;  // string[] | number[]

// 分配が起こる条件:
// 1. 条件型の被チェック型が「裸の型パラメータ」であること
// 2. 型パラメータにユニオン型が渡されること

// 分配を防ぎたい場合は [] で囲む
type ToArrayNonDist<T> = [T] extends [any] ? T[] : never;

type B = ToArrayNonDist<string | number>;  // (string | number)[]
```

### 3.2 分配の詳細な挙動

```typescript
// 分配が起こるパターンの詳細

// パターン1: 裸の型パラメータ → 分配する
type Dist1<T> = T extends string ? T : never;
type R1 = Dist1<string | number | boolean>;  // string

// パターン2: 型パラメータが別の型で包まれている → 分配しない
type NoDist1<T> = [T] extends [string] ? T : never;
type R2 = NoDist1<string | number>;  // never（ユニオン全体がstringではない）

// パターン3: 型パラメータがプロパティとして使われている → 分配しない
type NoDist2<T> = { value: T } extends { value: string } ? T : never;
type R3 = NoDist2<string | number>;  // never

// パターン4: 型パラメータが配列の要素として使われている → 分配しない
type NoDist3<T> = T[] extends string[] ? T : never;
type R4 = NoDist3<string | number>;  // never

// 重要: 分配はextends句の左辺の型パラメータのみに適用
type Example<T, U> = T extends U ? T : never;
// T が分配される（左辺）、U は分配されない（右辺）
type R5 = Example<"a" | "b" | "c", "a" | "c">;  // "a" | "c"
```

### 3.3 分配条件型の実用例

```typescript
// Union型からnullとundefinedを除外（NonNullable の実装）
type MyNonNullable<T> = T extends null | undefined ? never : T;

type A = MyNonNullable<string | null | undefined>;  // string

// Union型から特定の型を抽出（Extract の実装）
type MyExtract<T, U> = T extends U ? T : never;

type B = MyExtract<"a" | "b" | "c", "a" | "c">;  // "a" | "c"

// Union型から特定の型を除外（Exclude の実装）
type MyExclude<T, U> = T extends U ? never : T;

type C = MyExclude<"a" | "b" | "c", "a">;  // "b" | "c"

// 判別共用体から特定のメンバーを抽出
type Action =
  | { type: "fetch"; url: string }
  | { type: "save"; data: unknown }
  | { type: "delete"; id: string };

type FetchAction = Extract<Action, { type: "fetch" }>;
// { type: "fetch"; url: string }

type NonFetchAction = Exclude<Action, { type: "fetch" }>;
// { type: "save"; data: unknown } | { type: "delete"; id: string }

// オブジェクト型のフィルタリング: 特定の型のプロパティだけを残す
type FilterByValueType<T, ValueType> = {
  [K in keyof T as T[K] extends ValueType ? K : never]: T[K];
};

interface User {
  name: string;
  age: number;
  email: string;
  isActive: boolean;
}

type StringProps = FilterByValueType<User, string>;
// { name: string; email: string }

type NumberProps = FilterByValueType<User, number>;
// { age: number }
```

### 3.4 分配条件型と never

```typescript
// never はユニオン型の「空」として扱われる
// 分配条件型に never を渡すと、処理すべきメンバーがないため結果も never

type Test<T> = T extends string ? "yes" : "no";
type Result1 = Test<never>;  // never（"yes" でも "no" でもない）

// never を検出したい場合は分配を防ぐ
type IsNever<T> = [T] extends [never] ? true : false;

type Result2 = IsNever<never>;       // true
type Result3 = IsNever<string>;      // false
type Result4 = IsNever<undefined>;   // false

// 実用例: 型がneverかどうかで処理を分岐
type SafeReturn<T> =
  [T] extends [never]
    ? undefined
    : T;

type R1 = SafeReturn<string>;  // string
type R2 = SafeReturn<never>;   // undefined

// Union型が空かどうかの判定
type IsEmptyUnion<T> = [T] extends [never] ? true : false;

// never と void の違い
type TestVoid = Test<void>;    // "no"（void はユニオンメンバーとして評価される）
type TestNever = Test<never>;  // never（空のユニオン）
```

### 3.5 分配条件型を使ったユニオン操作

```typescript
// ユニオン型のメンバー数を数える（概念的なアプローチ）
// 直接的にはできないが、タプルに変換して長さを取得する手法がある

// ユニオンをタプルに変換（順序は保証されない）
type UnionToIntersection<U> =
  (U extends any ? (k: U) => void : never) extends (k: infer I) => void
    ? I
    : never;

type LastOfUnion<T> =
  UnionToIntersection<
    T extends any ? () => T : never
  > extends () => infer R
    ? R
    : never;

type UnionToTuple<T, Last = LastOfUnion<T>> =
  [T] extends [never]
    ? []
    : [...UnionToTuple<Exclude<T, Last>>, Last];

type Tuple = UnionToTuple<"a" | "b" | "c">;  // ["a", "b", "c"]

// ユニオン型のすべてのペアを生成
type Pairs<T, U = T> =
  T extends any
    ? U extends any
      ? T extends U
        ? never
        : [T, U]
      : never
    : never;

type AllPairs = Pairs<"a" | "b" | "c">;
// ["a", "b"] | ["a", "c"] | ["b", "a"] | ["b", "c"] | ["c", "a"] | ["c", "b"]
```

---

## 4. 再帰的条件型

### 4.1 深いオブジェクトの型変換

```typescript
// ネストしたオブジェクトの深いReadonly
type DeepReadonly<T> =
  T extends (...args: any[]) => any
    ? T  // 関数はそのまま
    : T extends any[]
      ? DeepReadonlyArray<T>  // 配列は別処理
      : T extends object
        ? { readonly [K in keyof T]: DeepReadonly<T[K]> }
        : T;  // プリミティブはそのまま

type DeepReadonlyArray<T extends any[]> =
  readonly [...{ [K in keyof T]: DeepReadonly<T[K]> }];

interface Config {
  db: {
    host: string;
    port: number;
    credentials: {
      user: string;
      pass: string;
    };
  };
  features: string[];
  nested: {
    list: { id: number; name: string }[];
  };
}

type ReadonlyConfig = DeepReadonly<Config>;
// 全てのネストされたプロパティがreadonly
// features は readonly string[]
// nested.list は readonly { readonly id: number; readonly name: string }[]

// 深いPartial
type DeepPartial<T> =
  T extends (...args: any[]) => any
    ? T
    : T extends any[]
      ? T
      : T extends object
        ? { [K in keyof T]?: DeepPartial<T[K]> }
        : T;

type PartialConfig = DeepPartial<Config>;
// 全てのネストされたプロパティがオプショナル

// 深いRequired
type DeepRequired<T> =
  T extends (...args: any[]) => any
    ? T
    : T extends any[]
      ? T
      : T extends object
        ? { [K in keyof T]-?: DeepRequired<T[K]> }
        : T;
```

### 4.2 フラット化型

```typescript
// フラット化（ネストした配列を1段階平坦化）
type Flatten<T> = T extends (infer U)[] ? U : T;

type A = Flatten<string[][]>;  // string[]
type B = Flatten<string[]>;    // string

// 深いフラット化
type DeepFlatten<T> = T extends (infer U)[] ? DeepFlatten<U> : T;

type C = DeepFlatten<string[][][]>;  // string

// 指定した深さまでフラット化
type FlattenDepth<
  T extends any[],
  Depth extends number = 1,
  Counter extends any[] = []
> = Counter["length"] extends Depth
  ? T
  : T extends [infer First, ...infer Rest]
    ? First extends any[]
      ? [...FlattenDepth<First, Depth, [...Counter, 0]>, ...FlattenDepth<Rest, Depth, Counter>]
      : [First, ...FlattenDepth<Rest, Depth, Counter>]
    : T;

type FD1 = FlattenDepth<[1, [2, [3, [4]]]], 1>;  // [1, 2, [3, [4]]]
type FD2 = FlattenDepth<[1, [2, [3, [4]]]], 2>;  // [1, 2, 3, [4]]
```

### 4.3 型レベルの文字列操作（再帰的）

```typescript
// 文字列を大文字に変換
type UpperCase<S extends string> =
  S extends `${infer First}${infer Rest}`
    ? `${Uppercase<First>}${UpperCase<Rest>}`
    : S;

// キャメルケースをスネークケースに変換
type CamelToSnake<S extends string> =
  S extends `${infer First}${infer Rest}`
    ? First extends Uppercase<First>
      ? First extends Lowercase<First>
        ? `${First}${CamelToSnake<Rest>}`
        : `_${Lowercase<First>}${CamelToSnake<Rest>}`
      : `${First}${CamelToSnake<Rest>}`
    : S;

type Snake = CamelToSnake<"camelCaseString">;
// "camel_case_string"

// スネークケースをキャメルケースに変換
type SnakeToCamel<S extends string> =
  S extends `${infer Head}_${infer Tail}`
    ? `${Head}${Capitalize<SnakeToCamel<Tail>>}`
    : S;

type Camel = SnakeToCamel<"snake_case_string">;
// "snakeCaseString"

// ケバブケースをキャメルケースに変換
type KebabToCamel<S extends string> =
  S extends `${infer Head}-${infer Tail}`
    ? `${Head}${Capitalize<KebabToCamel<Tail>>}`
    : S;

type Kebab = KebabToCamel<"kebab-case-string">;
// "kebabCaseString"

// 文字列の長さを取得
type StringLength<
  S extends string,
  Counter extends any[] = []
> = S extends `${string}${infer Rest}`
  ? StringLength<Rest, [...Counter, 0]>
  : Counter["length"];

type Len = StringLength<"hello">;  // 5
```

### 4.4 型レベルの算術演算

```typescript
// 型レベルの加算
type Add<A extends number, B extends number> =
  [...BuildTuple<A>, ...BuildTuple<B>]["length"] & number;

type BuildTuple<
  N extends number,
  T extends any[] = []
> = T["length"] extends N ? T : BuildTuple<N, [...T, 0]>;

type Sum = Add<3, 4>;  // 7

// 型レベルの比較
type GreaterThan<
  A extends number,
  B extends number,
  Counter extends any[] = []
> = Counter["length"] extends A
  ? false
  : Counter["length"] extends B
    ? true
    : GreaterThan<A, B, [...Counter, 0]>;

type GT = GreaterThan<5, 3>;  // true
type LT = GreaterThan<2, 4>;  // false

// 型レベルの範囲生成
type Range<
  Start extends number,
  End extends number,
  Result extends number[] = [],
  Counter extends any[] = BuildTuple<Start>
> = Counter["length"] extends End
  ? [...Result, End]
  : Range<Start, End, [...Result, Counter["length"] & number], [...Counter, 0]>;

type R = Range<1, 5>;  // [1, 2, 3, 4, 5]
```

### 4.5 ドットパスによるネストプロパティアクセス

```typescript
// パスからドット区切りのキーを生成
type DotPath<T, Prefix extends string = ""> =
  T extends object
    ? {
        [K in keyof T & string]: T[K] extends object
          ? T[K] extends any[]
            ? `${Prefix}${K}`
            : DotPath<T[K], `${Prefix}${K}.`> | `${Prefix}${K}`
          : `${Prefix}${K}`;
      }[keyof T & string]
    : never;

interface Settings {
  theme: {
    color: string;
    fontSize: number;
    font: {
      family: string;
      weight: number;
    };
  };
  user: {
    name: string;
    preferences: {
      darkMode: boolean;
    };
  };
}

type SettingPaths = DotPath<Settings>;
// "theme" | "theme.color" | "theme.fontSize" | "theme.font"
// | "theme.font.family" | "theme.font.weight"
// | "user" | "user.name" | "user.preferences" | "user.preferences.darkMode"

// ドットパスで値の型を取得
type PathValue<T, P extends string> =
  P extends `${infer Key}.${infer Rest}`
    ? Key extends keyof T
      ? PathValue<T[Key], Rest>
      : never
    : P extends keyof T
      ? T[P]
      : never;

type ThemeColor = PathValue<Settings, "theme.color">;      // string
type FontWeight = PathValue<Settings, "theme.font.weight">; // number
type DarkMode = PathValue<Settings, "user.preferences.darkMode">; // boolean

// 型安全なgetByPath関数
function getByPath<T, P extends DotPath<T>>(
  obj: T,
  path: P
): PathValue<T, P & string> {
  const keys = (path as string).split(".");
  let result: any = obj;
  for (const key of keys) {
    result = result[key];
  }
  return result;
}

// 使用例
declare const settings: Settings;
const color = getByPath(settings, "theme.color");           // string型
const weight = getByPath(settings, "theme.font.weight");    // number型
// getByPath(settings, "theme.invalid");  // コンパイルエラー
```

---

## 5. 実践的なユーティリティ型の構築

### 5.1 APIレスポンスの型変換

```typescript
// APIレスポンスのフィールドを自動変換する型

// スネークケースのキーをキャメルケースに変換
type SnakeToCamelKeys<T> =
  T extends any[]
    ? { [K in keyof T]: SnakeToCamelKeys<T[K]> }
    : T extends object
      ? {
          [K in keyof T as K extends string
            ? SnakeToCamel<K>
            : K]: SnakeToCamelKeys<T[K]>
        }
      : T;

// APIレスポンス型（サーバーはスネークケースで返す）
interface ApiUserResponse {
  user_id: number;
  first_name: string;
  last_name: string;
  email_address: string;
  created_at: string;
  profile_data: {
    avatar_url: string;
    display_name: string;
    bio_text: string | null;
  };
}

// フロントエンド用にキャメルケースに変換
type UserData = SnakeToCamelKeys<ApiUserResponse>;
// {
//   userId: number;
//   firstName: string;
//   lastName: string;
//   emailAddress: string;
//   createdAt: string;
//   profileData: {
//     avatarUrl: string;
//     displayName: string;
//     bioText: string | null;
//   };
// }

// 日付文字列をDateに変換する型
type ConvertDates<T> =
  T extends object
    ? {
        [K in keyof T]: K extends `${string}_at` | `${string}At`
          ? Date
          : T[K] extends object
            ? ConvertDates<T[K]>
            : T[K];
      }
    : T;

type UserWithDates = ConvertDates<UserData>;
// createdAt が Date 型に変換される
```

### 5.2 フォームバリデーション型

```typescript
// フォームのバリデーションルールを型安全に定義

// バリデーションルール
interface ValidationRules {
  required: boolean;
  minLength: number;
  maxLength: number;
  pattern: RegExp;
  min: number;
  max: number;
  custom: (value: any) => boolean;
}

// フォームフィールドの型からバリデーションエラー型を自動生成
type ValidationErrors<T> = {
  [K in keyof T]?: T[K] extends object
    ? T[K] extends any[]
      ? string[]
      : ValidationErrors<T[K]>
    : string;
};

// フォームの状態型
type FormState<T> = {
  values: T;
  errors: ValidationErrors<T>;
  touched: { [K in keyof T]?: boolean };
  isValid: boolean;
  isSubmitting: boolean;
};

// 使用例
interface LoginForm {
  email: string;
  password: string;
  rememberMe: boolean;
}

type LoginFormState = FormState<LoginForm>;
// {
//   values: LoginForm;
//   errors: { email?: string; password?: string; rememberMe?: string };
//   touched: { email?: boolean; password?: boolean; rememberMe?: boolean };
//   isValid: boolean;
//   isSubmitting: boolean;
// }

// ネストしたフォーム
interface RegistrationForm {
  personal: {
    firstName: string;
    lastName: string;
  };
  account: {
    email: string;
    password: string;
  };
  address: {
    street: string;
    city: string;
    zipCode: string;
  };
}

type RegistrationErrors = ValidationErrors<RegistrationForm>;
// {
//   personal?: {
//     firstName?: string;
//     lastName?: string;
//   };
//   account?: {
//     email?: string;
//     password?: string;
//   };
//   address?: {
//     street?: string;
//     city?: string;
//     zipCode?: string;
//   };
// }
```

### 5.3 状態管理の型安全な設計

```typescript
// Redux風の型安全なアクション定義

// アクションクリエータの型を定義
type ActionCreator<T extends string, P = void> =
  P extends void
    ? { type: T }
    : { type: T; payload: P };

// アクションの型マップからアクション型を自動生成
type ActionMap = {
  INCREMENT: void;
  DECREMENT: void;
  SET_COUNT: number;
  SET_NAME: string;
  SET_USER: { name: string; age: number };
  ADD_ITEM: { id: string; value: unknown };
  REMOVE_ITEM: string;
  RESET: void;
};

// マップからすべてのアクション型を生成
type Actions = {
  [K in keyof ActionMap]: ActionCreator<K & string, ActionMap[K]>;
}[keyof ActionMap];

// Actions は以下のユニオン型:
// | { type: "INCREMENT" }
// | { type: "DECREMENT" }
// | { type: "SET_COUNT"; payload: number }
// | { type: "SET_NAME"; payload: string }
// | { type: "SET_USER"; payload: { name: string; age: number } }
// | { type: "ADD_ITEM"; payload: { id: string; value: unknown } }
// | { type: "REMOVE_ITEM"; payload: string }
// | { type: "RESET" }

// reducer の型安全な定義
type State = {
  count: number;
  name: string;
  user: { name: string; age: number } | null;
  items: { id: string; value: unknown }[];
};

function reducer(state: State, action: Actions): State {
  switch (action.type) {
    case "INCREMENT":
      return { ...state, count: state.count + 1 };
    case "SET_COUNT":
      return { ...state, count: action.payload }; // payload は number
    case "SET_USER":
      return { ...state, user: action.payload };   // payload は { name: string; age: number }
    case "ADD_ITEM":
      return { ...state, items: [...state.items, action.payload] };
    case "REMOVE_ITEM":
      return {
        ...state,
        items: state.items.filter(i => i.id !== action.payload),  // payload は string
      };
    case "RESET":
      return { count: 0, name: "", user: null, items: [] };
    default:
      return state;
  }
}
```

### 5.4 型安全なイベントエミッタ

```typescript
// イベントマップの型定義
interface EventMap {
  connect: { host: string; port: number };
  disconnect: { reason: string };
  message: { from: string; text: string; timestamp: number };
  error: { code: number; message: string };
  "user:join": { userId: string; username: string };
  "user:leave": { userId: string };
}

// 条件型を使って型安全なイベントエミッタを定義
type EventHandler<T> = (event: T) => void;

type EventHandlers<M> = {
  [K in keyof M]?: EventHandler<M[K]>[];
};

class TypedEventEmitter<M extends Record<string, any>> {
  private handlers: EventHandlers<M> = {} as EventHandlers<M>;

  on<K extends keyof M>(event: K, handler: EventHandler<M[K]>): void {
    if (!this.handlers[event]) {
      this.handlers[event] = [];
    }
    this.handlers[event]!.push(handler);
  }

  emit<K extends keyof M>(event: K, data: M[K]): void {
    const eventHandlers = this.handlers[event];
    if (eventHandlers) {
      eventHandlers.forEach(handler => handler(data));
    }
  }

  off<K extends keyof M>(event: K, handler: EventHandler<M[K]>): void {
    const eventHandlers = this.handlers[event];
    if (eventHandlers) {
      this.handlers[event] = eventHandlers.filter(h => h !== handler) as any;
    }
  }
}

// 使用例
const emitter = new TypedEventEmitter<EventMap>();

emitter.on("connect", (event) => {
  // event は { host: string; port: number } と推論される
  console.log(`Connected to ${event.host}:${event.port}`);
});

emitter.on("message", (event) => {
  // event は { from: string; text: string; timestamp: number } と推論される
  console.log(`${event.from}: ${event.text}`);
});

// emitter.on("connect", (event: { invalid: true }) => {}); // コンパイルエラー
// emitter.emit("message", { invalid: true }); // コンパイルエラー
```

### 5.5 ビルダーパターンの型安全な実装

```typescript
// 条件型を使った型安全なビルダー

// 必須フィールドの追跡
type RequiredFields = {
  name: string;
  age: number;
  email: string;
};

type OptionalFields = {
  phone?: string;
  address?: string;
  bio?: string;
};

// ビルダーの状態を型パラメータで追跡
type BuilderState<
  Required extends Record<string, any>,
  Set extends string = never
> = {
  // すべての必須フィールドが設定済みなら build() が使える
  build: [Exclude<keyof Required, Set>] extends [never]
    ? () => Required & OptionalFields
    : never;
} & {
  // まだ設定されていないフィールドのセッター
  [K in Exclude<keyof Required, Set> & string as `set${Capitalize<K>}`]:
    (value: Required[K]) => BuilderState<Required, Set | K>;
} & {
  // オプショナルフィールドのセッター
  [K in keyof OptionalFields & string as `set${Capitalize<K>}`]:
    (value: NonNullable<OptionalFields[K]>) => BuilderState<Required, Set>;
};

// 使用イメージ（概念的な型のみ）
// const user = createBuilder<RequiredFields>()
//   .setName("Alice")   // BuilderState<RequiredFields, "name">
//   .setAge(30)          // BuilderState<RequiredFields, "name" | "age">
//   .setEmail("a@b.c")  // BuilderState<RequiredFields, "name" | "age" | "email">
//   .build();            // Required & OptionalFields（build が使える）
```

---

## 6. パフォーマンスとデバッグ

### 6.1 条件型のパフォーマンス考慮事項

```typescript
// 条件型のコンパイル性能に関する注意点

// BAD: 深い再帰はコンパイルが遅くなる
type DeepNested1<T, Depth extends any[] = []> =
  Depth["length"] extends 100
    ? T
    : DeepNested1<T[], [...Depth, 0]>;

// TypeScript には再帰の深さ制限がある（通常50〜100程度）
// 過度な再帰はエラー "Type instantiation is excessively deep" を引き起こす

// GOOD: 再帰の深さを制限する
type SafeDeepType<T, Depth extends any[] = []> =
  Depth["length"] extends 10  // 妥当な深さに制限
    ? T
    : T extends object
      ? { [K in keyof T]: SafeDeepType<T[K], [...Depth, 0]> }
      : T;

// BAD: ユニオン型の爆発
// 分配条件型 + 大きなユニオン = 計算量の爆発
type HugeUnion = "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h"; // 8メンバー
type Combinations<T, U = T> =
  T extends any
    ? U extends any
      ? [T, U]
      : never
    : never;

type AllCombinations = Combinations<HugeUnion>;
// 8 × 8 = 64 のユニオンメンバーが生成される
// メンバーが増えると指数的にコンパイルが遅くなる

// GOOD: 必要な組み合わせだけを生成
type SpecificCombinations = ["a", "b"] | ["a", "c"] | ["b", "c"];
```

### 6.2 条件型のデバッグテクニック

```typescript
// テクニック1: 中間型を変数化して確認
type Debug1 = string extends any ? true : false;  // true を確認

// テクニック2: エラーメッセージを利用した型の確認
type ShowType<T> = T & { __debug: T };

// テクニック3: 型テスト用のユーティリティ
type Expect<T extends true> = T;
type Equal<X, Y> =
  (<T>() => T extends X ? 1 : 2) extends
  (<T>() => T extends Y ? 1 : 2) ? true : false;

// テストの記述
type TestCase1 = Expect<Equal<IsString<string>, true>>;  // OK
type TestCase2 = Expect<Equal<IsString<number>, false>>; // OK
// type TestCase3 = Expect<Equal<IsString<string>, false>>; // コンパイルエラー（期待通り）

// テクニック4: 条件型の展開を追跡
// TypeScript の Language Server の "Expand Type" 機能を使う
// VSCode で変数の上にホバーすると展開された型が表示される

// テクニック5: @ts-expect-error を使ったネガティブテスト
// @ts-expect-error -- number は string ではないのでエラーになるべき
type NegativeTest: IsString<number> = true;

// テクニック6: 型の等値性を確認するヘルパー
type Assert<T, Expected> = T extends Expected
  ? Expected extends T
    ? true
    : { error: "Types are not equal"; got: T; expected: Expected }
  : { error: "Types are not equal"; got: T; expected: Expected };

type Check1 = Assert<ReturnType<() => string>, string>;  // true
type Check2 = Assert<ReturnType<() => number>, string>;  // エラー情報付きオブジェクト
```

### 6.3 条件型の制限事項

```typescript
// 制限1: 条件型の中で型ガードは使えない
// BAD: ランタイムの typeof は型レベルでは使えない
// type RuntimeCheck<T> = typeof T === "string" ? true : false; // エラー

// 制限2: 条件型の遅延評価
// 条件型は型パラメータが未解決の場合、評価が遅延される
function example<T>(x: T) {
  // この時点では T が不明なので条件型は展開されない
  type Result = T extends string ? "string" : "other";
  // Result は T extends string ? "string" : "other" のまま
}

// 制限3: 再帰の深さ制限
// TypeScript 4.5+ では1000レベルまでの再帰が可能だが
// 実際にはもっと浅い段階でパフォーマンスが問題になる

// 制限4: 型推論の優先度
// 複数の infer が同じ型変数名を使う場合の挙動
type Ambiguous<T> =
  T extends { a: infer U; b: infer U }
    ? U
    : never;

// 共変位置なのでユニオンになる
type AmbiguousResult = Ambiguous<{ a: string; b: number }>;
// string | number

// 制限5: 条件型と関数引数の型推論の相互作用
// 条件型の結果を関数の引数として使う場合、型推論が期待通りに動かないことがある
function problematic<T extends string | number>(
  value: T,
  transform: (x: T extends string ? string : number) => void
): void {
  // T が未解決なので条件型も未解決
  // transform(value); // エラー: T は T extends string ? string : number に代入できない
}

// 回避策: オーバーロードや型アサーションを使う
function fixed(value: string, transform: (x: string) => void): void;
function fixed(value: number, transform: (x: number) => void): void;
function fixed(value: string | number, transform: (x: any) => void): void {
  transform(value);
}
```

---

## 7. 高度な条件型パターン

### 7.1 型レベルの JSON パーサー

```typescript
// JSON 文字列を型レベルでパースする（概念的な実装）

// 空白のスキップ
type SkipWhitespace<S extends string> =
  S extends ` ${infer Rest}` | `\n${infer Rest}` | `\t${infer Rest}`
    ? SkipWhitespace<Rest>
    : S;

// 文字列リテラルのパース
type ParseString<S extends string> =
  S extends `"${infer Content}"${infer Rest}`
    ? [Content, Rest]
    : never;

// 数値のパース（簡略版）
type ParseNumber<S extends string> =
  S extends `${infer N extends number}${infer Rest}`
    ? [N, Rest]
    : never;

// ブーリアンのパース
type ParseBool<S extends string> =
  S extends `true${infer Rest}` ? [true, Rest] :
  S extends `false${infer Rest}` ? [false, Rest] :
  never;

// null のパース
type ParseNull<S extends string> =
  S extends `null${infer Rest}` ? [null, Rest] : never;

// これらを組み合わせてJSONの各値型をパースできる
// 実際のフルパーサーは非常に複雑だが、型レベルプログラミングの可能性を示す例
```

### 7.2 型レベルのパス解析

```typescript
// URLパスのパラメータを型安全に抽出

// パスパラメータの抽出
type ExtractParams<Path extends string> =
  Path extends `${string}:${infer Param}/${infer Rest}`
    ? { [K in Param | keyof ExtractParamsObj<Rest>]: string }
    : Path extends `${string}:${infer Param}`
      ? { [K in Param]: string }
      : {};

type ExtractParamsObj<Path extends string> =
  Path extends `${string}:${infer Param}/${infer Rest}`
    ? { [K in Param]: string } & ExtractParamsObj<Rest>
    : Path extends `${string}:${infer Param}`
      ? { [K in Param]: string }
      : {};

// クエリパラメータの型安全な定義
type QueryParams<T extends string> =
  T extends `${string}?${infer Query}`
    ? ParseQuery<Query>
    : {};

type ParseQuery<Q extends string> =
  Q extends `${infer Key}=${infer Value}&${infer Rest}`
    ? { [K in Key]: Value } & ParseQuery<Rest>
    : Q extends `${infer Key}=${infer Value}`
      ? { [K in Key]: Value }
      : {};

// Express風のルーター型定義
type Route<
  Method extends "GET" | "POST" | "PUT" | "DELETE",
  Path extends string,
  Body = never,
  Response = unknown
> = {
  method: Method;
  path: Path;
  params: ExtractParamsObj<Path>;
  body: Body;
  response: Response;
};

// 使用例
type UserRoute = Route<"GET", "/users/:userId/posts/:postId">;
// params は { userId: string; postId: string }

// 型安全なルートハンドラ
type RouteHandler<R extends Route<any, any, any, any>> = (
  req: {
    params: R["params"];
    body: R["body"];
  }
) => Promise<R["response"]>;

// 実装例
type GetUserRoute = Route<"GET", "/users/:userId", never, { name: string; age: number }>;

const getUserHandler: RouteHandler<GetUserRoute> = async (req) => {
  const userId = req.params.userId;  // string 型
  return { name: "Alice", age: 30 };  // { name: string; age: number } を返す必要がある
};
```

### 7.3 型レベルの状態マシン

```typescript
// 条件型で状態遷移を型安全に表現

// 状態の定義
type OrderState = "draft" | "pending" | "confirmed" | "shipped" | "delivered" | "cancelled";

// 状態遷移の定義（条件型を使用）
type ValidTransition<From extends OrderState> =
  From extends "draft" ? "pending" | "cancelled" :
  From extends "pending" ? "confirmed" | "cancelled" :
  From extends "confirmed" ? "shipped" | "cancelled" :
  From extends "shipped" ? "delivered" :
  From extends "delivered" ? never :
  From extends "cancelled" ? never :
  never;

// 状態遷移が有効かどうかの判定
type CanTransition<From extends OrderState, To extends OrderState> =
  To extends ValidTransition<From> ? true : false;

// 型テスト
type Test1 = CanTransition<"draft", "pending">;      // true
type Test2 = CanTransition<"draft", "shipped">;       // false
type Test3 = CanTransition<"confirmed", "shipped">;   // true
type Test4 = CanTransition<"delivered", "draft">;      // false
type Test5 = CanTransition<"cancelled", "pending">;    // false

// 型安全な状態遷移関数
class Order<S extends OrderState> {
  constructor(
    public readonly state: S,
    public readonly id: string
  ) {}

  transition<Next extends ValidTransition<S>>(
    nextState: Next
  ): Order<Next> {
    return new Order(nextState, this.id);
  }
}

// 使用例
const order = new Order("draft", "order-001");
const pending = order.transition("pending");      // Order<"pending">
const confirmed = pending.transition("confirmed"); // Order<"confirmed">
const shipped = confirmed.transition("shipped");    // Order<"shipped">
// shipped.transition("draft");                    // コンパイルエラー！
// confirmed.transition("delivered");              // コンパイルエラー！

// 各状態で利用可能なアクションを型で表現
type ActionsForState<S extends OrderState> =
  S extends "draft" ? { edit: () => void; submit: () => void; cancel: () => void } :
  S extends "pending" ? { confirm: () => void; cancel: () => void } :
  S extends "confirmed" ? { ship: () => void; cancel: () => void } :
  S extends "shipped" ? { deliver: () => void; track: () => string } :
  S extends "delivered" ? { review: () => void; returnItem: () => void } :
  S extends "cancelled" ? { reorder: () => void } :
  never;
```

### 7.4 条件型を使ったバリデーションDSL

```typescript
// 型レベルのバリデーションルール定義

// バリデーションルールの型
type Rule<T, Message extends string = string> = {
  validate: (value: T) => boolean;
  message: Message;
};

// フィールドの型からバリデーションルールを条件型で決定
type DefaultRules<T> =
  T extends string
    ? Rule<string, "文字列は空にできません"> | Rule<string, "最大文字数を超えています">
    : T extends number
      ? Rule<number, "数値は0以上である必要があります"> | Rule<number, "数値が大きすぎます">
      : T extends boolean
        ? Rule<boolean, "必須項目です">
        : T extends any[]
          ? Rule<T, "少なくとも1つの要素が必要です">
          : Rule<T, "無効な値です">;

// フォームフィールド定義の型
type FieldConfig<T> = {
  type: T extends string ? "text" | "email" | "password" | "url"
    : T extends number ? "number" | "range" | "slider"
    : T extends boolean ? "checkbox" | "toggle"
    : T extends Date ? "date" | "datetime"
    : T extends any[] ? "multiselect" | "tags"
    : "custom";
  label: string;
  defaultValue: T;
  rules?: DefaultRules<T>[];
};

// 使用例
type UserFormConfig = {
  [K in keyof RequiredFields]: FieldConfig<RequiredFields[K]>;
};

// 型推論により、各フィールドに適切な設定型が割り当てられる
const formConfig: UserFormConfig = {
  name: {
    type: "text",        // "text" | "email" | "password" | "url" のみ許可
    label: "名前",
    defaultValue: "",
    rules: [
      { validate: (v) => v.length > 0, message: "文字列は空にできません" },
    ],
  },
  age: {
    type: "number",      // "number" | "range" | "slider" のみ許可
    label: "年齢",
    defaultValue: 0,
    rules: [
      { validate: (v) => v >= 0, message: "数値は0以上である必要があります" },
    ],
  },
  email: {
    type: "email",
    label: "メールアドレス",
    defaultValue: "",
  },
};
```

---

## 8. 実務での条件型のベストプラクティス

### 8.1 段階的な型設計

```typescript
// GOOD: 複雑な条件型を小さな部品に分割

// Step 1: 基本的な判定型
type IsArray<T> = T extends any[] ? true : false;
type IsFunction<T> = T extends (...args: any[]) => any ? true : false;
type IsObject<T> = T extends object
  ? T extends any[]
    ? false
    : T extends (...args: any[]) => any
      ? false
      : true
  : false;

// Step 2: 変換型
type Writable<T> = { -readonly [K in keyof T]: T[K] };
type DeepWritable<T> =
  IsFunction<T> extends true ? T :
  IsArray<T> extends true ? { [K in keyof T]: DeepWritable<T[K]> } :
  IsObject<T> extends true ? { -readonly [K in keyof T]: DeepWritable<T[K]> } :
  T;

// Step 3: 組み合わせ型
type Mutable<T, Deep extends boolean = false> =
  Deep extends true ? DeepWritable<T> : Writable<T>;

// 使用例
interface ReadonlyConfig {
  readonly host: string;
  readonly port: number;
  readonly db: {
    readonly name: string;
    readonly options: {
      readonly ssl: boolean;
    };
  };
}

type ShallowMutable = Mutable<ReadonlyConfig, false>;
// { host: string; port: number; db: { readonly name: string; ... } }

type DeepMutable = Mutable<ReadonlyConfig, true>;
// { host: string; port: number; db: { name: string; options: { ssl: boolean } } }
```

### 8.2 条件型の命名規則

```typescript
// 命名規則のベストプラクティス

// 1. Is- プレフィックス: boolean を返す型
type IsString<T> = T extends string ? true : false;
type IsNullable<T> = null extends T ? true : false;
type IsUnion<T> = [T] extends [UnionToIntersection<T>] ? false : true;

// 2. Extract/Get プレフィックス: 一部を取り出す型
type ExtractArrayElement<T> = T extends (infer U)[] ? U : never;
type GetReturnType<T> = T extends (...args: any[]) => infer R ? R : never;

// 3. To/As プレフィックス: 変換する型
type ToArray<T> = T extends any ? T[] : never;
type ToPromise<T> = T extends any ? Promise<T> : never;
type AsReadonly<T> = { readonly [K in keyof T]: T[K] };

// 4. Without/Omit プレフィックス: 除外する型
type WithoutNullable<T> = T extends null | undefined ? never : T;
type WithoutFunctions<T> = {
  [K in keyof T as T[K] extends Function ? never : K]: T[K];
};

// 5. Deep プレフィックス: 再帰的に適用する型
type DeepReadonly<T> = /* ... */;
type DeepPartial<T> = /* ... */;
type DeepRequired<T> = /* ... */;
```

### 8.3 エラーメッセージの改善

```typescript
// 条件型でカスタムエラーメッセージを提供する

// 基本的なアプローチ: never の代わりにエラーメッセージを含む型を返す
type MustBeString<T> =
  T extends string
    ? T
    : { __error: `Expected string but got ${T & string}` };

// より高度なアプローチ: ブランド型でエラーメッセージを表現
type TypeError<Message extends string> = { readonly __typeError: Message } & never;

type SafeDivide<A extends number, B extends number> =
  B extends 0
    ? TypeError<"Division by zero is not allowed">
    : number;

// 使用時
type Result1 = SafeDivide<10, 2>;  // number
type Result2 = SafeDivide<10, 0>;  // TypeError<"Division by zero is not allowed">

// 型制約の違反をわかりやすく報告
type ValidateConfig<T> =
  T extends { host: string }
    ? T extends { port: number }
      ? T
      : TypeError<"Config must include 'port' as number">
    : TypeError<"Config must include 'host' as string">;
```

---

## 組み込み条件型比較

| ユーティリティ型 | 定義 | 用途 |
|-----------------|------|------|
| `NonNullable<T>` | `T extends null \| undefined ? never : T` | null/undefinedを除外 |
| `Extract<T, U>` | `T extends U ? T : never` | Union型から特定の型を抽出 |
| `Exclude<T, U>` | `T extends U ? never : T` | Union型から特定の型を除外 |
| `ReturnType<T>` | `T extends (...) => infer R ? R : any` | 関数の戻り値型を取得 |
| `Parameters<T>` | `T extends (...args: infer P) => any ? P : never` | 関数のパラメータ型を取得 |
| `InstanceType<T>` | `T extends new (...) => infer R ? R : any` | コンストラクタのインスタンス型 |
| `ConstructorParameters<T>` | `T extends new (...args: infer P) => any ? P : never` | コンストラクタのパラメータ型 |
| `Awaited<T>` | 再帰的にPromiseを展開 | Promise解決後の型を取得 |
| `ThisParameterType<T>` | `T extends (this: infer U, ...) => any ? U : unknown` | this パラメータの型 |
| `OmitThisParameter<T>` | this パラメータを除外 | this なしの関数型 |

---

## 分配 vs 非分配 比較

| 特性 | 分配条件型 | 非分配条件型 |
|------|-----------|-------------|
| 構文 | `T extends U ? X : Y` | `[T] extends [U] ? X : Y` |
| Union入力 | 各メンバーに個別適用 | Union全体として評価 |
| `string \| number` | `F<string> \| F<number>` | `F<string \| number>` |
| never の扱い | never を返す | 正常に評価 |
| 主な用途 | フィルタリング、型変換 | Union全体の判定、never検出 |

```
  分配 (Distributive)
  ToArray<string | number>
    → ToArray<string> | ToArray<number>
    → string[] | number[]

  非分配 (Non-distributive)
  ToArrayNonDist<string | number>
    → (string | number)[]

  never の扱いの違い
  分配: Test<never> → never
  非分配: [never] extends [string] ? true : false → true
```

---

## アンチパターン

### アンチパターン1: 過度に複雑な条件型

```typescript
// BAD: 読めない条件型のネスト
type ComplexType<T> =
  T extends string
    ? T extends `${infer A}.${infer B}`
      ? A extends `${infer C}-${infer D}`
        ? [C, D, B]
        : [A, B]
      : [T]
    : never;

// GOOD: ステップごとに型を分割して名前をつける
type SplitDot<T extends string> =
  T extends `${infer A}.${infer B}` ? [A, B] : [T];

type SplitDash<T extends string> =
  T extends `${infer A}-${infer B}` ? [A, B] : [T];

// 組み合わせて使う（各ステップの意味が明確）
type ParseSegment<T extends string> =
  SplitDot<T> extends [infer First extends string, infer Second]
    ? [...SplitDash<First>, Second]
    : SplitDot<T>;
```

### アンチパターン2: 再帰の深さ制限を無視

```typescript
// BAD: 無限再帰に近い型（コンパイルが非常に遅くなる）
type InfiniteNest<T> = {
  value: T;
  children: InfiniteNest<T>[];
};

// GOOD: 再帰の深さを制限する
type Nest<T, Depth extends number[] = []> =
  Depth["length"] extends 5
    ? T
    : {
        value: T;
        children: Nest<T, [...Depth, 0]>[];
      };
```

### アンチパターン3: 不必要な条件型の使用

```typescript
// BAD: 条件型が不要なケース
type Unnecessary<T> = T extends any ? T : never;
// T をそのまま返しているだけ（分配によるユニオン展開以外に意味がない）

// BAD: 単純なマッピングに条件型を使う
type BadMapping<T> =
  T extends "a" ? 1 :
  T extends "b" ? 2 :
  T extends "c" ? 3 :
  never;

// GOOD: レコード型を使う
type GoodMapping = {
  a: 1;
  b: 2;
  c: 3;
};
type Mapped<T extends keyof GoodMapping> = GoodMapping[T];
```

### アンチパターン4: 型アサーションに頼りすぎる

```typescript
// BAD: 条件型の結果を常にアサーションで上書き
function process<T extends string | number>(value: T) {
  // 条件型の結果を活用できていない
  const result = transform(value) as any;
  return result;
}

// GOOD: 型の絞り込みを使って条件型と連携
function processGood<T extends string | number>(
  value: T
): T extends string ? string[] : number {
  if (typeof value === "string") {
    return value.split(",") as T extends string ? string[] : number;
  }
  return (value as number) * 2 as T extends string ? string[] : number;
}
// ※ 条件型の戻り値ではアサーションが必要だが、最小限に留める
```

### アンチパターン5: 分配の予期しない動作

```typescript
// BAD: 分配を意識せずにユニオンを渡す
type Wrapper<T> = T extends any ? { value: T } : never;
type Result = Wrapper<string | number>;
// { value: string } | { value: number }（意図と異なる場合がある）

// GOOD: 意図を明確にする
// ユニオン全体を包みたい場合
type WrapperUnion<T> = [T] extends [any] ? { value: T } : never;
type ResultUnion = WrapperUnion<string | number>;
// { value: string | number }

// 個別に包みたい場合（分配を意図的に使用）
type WrapperDist<T> = T extends any ? { value: T } : never;
type ResultDist = WrapperDist<string | number>;
// { value: string } | { value: number }
```

---

## FAQ

### Q1: 条件型はどこで使うのが最も効果的ですか？

**A:** ライブラリの型定義や、APIの型変換（レスポンス型の自動導出など）で最も効果的です。アプリケーションコードでは、組み込みの `ReturnType`, `Parameters`, `Awaited` などのユーティリティ型を使うだけで十分なことが多いです。具体的な使いどころ:

- ライブラリの公開API型定義
- APIレスポンスの型変換（スネークケース→キャメルケースなど）
- 状態管理のアクション型の自動生成
- フォームバリデーションの型安全な定義
- ルーティングのパラメータ型の抽出

### Q2: infer は条件型の中でしか使えませんか？

**A:** はい、`infer` は `extends` 節の中でのみ使用できます。`T extends ... infer U ... ? X : Y` の形式でのみ有効です。条件型の外で型を「抽出」したい場合は、別途条件型を定義する必要があります。

```typescript
// infer は必ず extends 節の中で使う
type GetFirst<T> = T extends [infer F, ...any[]] ? F : never;

// 以下はエラー
// type Invalid<T> = infer U;  // エラー: infer は条件型でのみ使用可能
```

### Q3: 分配条件型で never はどう扱われますか？

**A:** `never` は空の Union として扱われるため、分配条件型に `never` が渡されると結果も `never` になります。

```typescript
type Test<T> = T extends string ? "yes" : "no";
type Result = Test<never>; // never（"yes" でも "no" でもない）

// never を正しく検出するには非分配にする
type IsNever<T> = [T] extends [never] ? true : false;
type Check = IsNever<never>; // true
```

### Q4: 条件型のコンパイル時間が遅い場合の対処法は？

**A:** 以下の対策が有効です:

1. **再帰の深さを制限する** -- カウンター用タプルで深さを管理
2. **ユニオン型の爆発を避ける** -- 分配で大きなユニオンが生成されないよう注意
3. **型をキャッシュする** -- 中間結果を型エイリアスに保存
4. **条件型を分割する** -- 1つの条件型でやりすぎない
5. **`declare` でテスト用の型インスタンスを作成** -- 全体ビルドせずに確認

### Q5: 条件型と型ガード（type predicate）の違いは？

**A:** 条件型はコンパイル時に型を変換するための仕組みで、型ガードはランタイムの値チェックに基づいて型を絞り込む仕組みです。

```typescript
// 条件型: コンパイル時
type IsString<T> = T extends string ? true : false;

// 型ガード: ランタイム
function isString(value: unknown): value is string {
  return typeof value === "string";
}

// 両者を組み合わせるパターン
function processValue<T>(value: T): T extends string ? string[] : T {
  if (isString(value)) {
    return value.split(",") as any;
  }
  return value as any;
}
```

### Q6: 条件型の中で複数の infer を使う場合の注意点は？

**A:** 同じ名前の `infer` 型変数を複数箇所で使う場合、それらが共変位置にあればユニオン型、反変位置にあればインターセクション型として推論されます。意図しない結果を避けるため、異なる名前を使うか、位置を意識して設計してください。

```typescript
// 同名の infer が共変位置に複数ある場合 → ユニオン
type CovariantInfer<T> =
  T extends { a: infer U; b: infer U } ? U : never;

type R1 = CovariantInfer<{ a: string; b: number }>;  // string | number

// 異なる名前を使えば個別に取得可能
type SeparateInfer<T> =
  T extends { a: infer A; b: infer B } ? [A, B] : never;

type R2 = SeparateInfer<{ a: string; b: number }>;  // [string, number]
```

### Q7: TypeScript のバージョンによる条件型の違いは？

**A:** 主な進化の経緯:

- **TypeScript 2.8**: 条件型の導入（`T extends U ? X : Y`）
- **TypeScript 4.1**: テンプレートリテラル型の導入で文字列操作が可能に
- **TypeScript 4.5**: 再帰制限の緩和（1000レベルまで）、Tail-call optimization
- **TypeScript 4.7**: `infer` に `extends` 制約を追加可能に（`infer U extends string`）
- **TypeScript 4.9**: `satisfies` 演算子の導入（条件型と併用で型安全性向上）
- **TypeScript 5.0**: const型パラメータの導入

```typescript
// TypeScript 4.7+: infer に制約を追加
type FirstIfString<T> =
  T extends [infer F extends string, ...any[]]
    ? F
    : never;

type R1 = FirstIfString<["hello", 42]>;  // "hello"
type R2 = FirstIfString<[42, "hello"]>;  // never
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| 条件型 | `T extends U ? X : Y` で型レベルの条件分岐 |
| infer | パターンマッチで型を抽出するキーワード |
| 分配条件型 | Union型の各メンバーに個別に条件型を適用 |
| 非分配 | `[T] extends [U]` で分配を抑制 |
| 再帰的条件型 | 深いネスト構造の型変換に使用 |
| 組み込み型 | ReturnType, Parameters, Awaited, Extract, Exclude 等 |
| 共変/反変 | infer の位置によりユニオン/インターセクションが推論される |
| パフォーマンス | 再帰深度とユニオンサイズに注意 |
| デバッグ | Equal, Expect 等のテストユーティリティを活用 |
| ベストプラクティス | 小さな部品に分割、命名規則を統一、エラーメッセージを改善 |

---

## 次に読むべきガイド

- [01-mapped-types.md](./01-mapped-types.md) -- マップ型
- [02-template-literal-types.md](./02-template-literal-types.md) -- テンプレートリテラル型
- [03-type-challenges.md](./03-type-challenges.md) -- 型パズル・チャレンジ

---

## 参考文献

1. **TypeScript Handbook: Conditional Types** -- https://www.typescriptlang.org/docs/handbook/2/conditional-types.html
2. **TypeScript Deep Dive: Conditional Types** -- https://basarat.gitbook.io/typescript/type-system/conditional-types
3. **Matt Pocock: Conditional Types in TypeScript** -- https://www.totaltypescript.com/books/total-typescript-essentials/conditional-types-and-infer
4. **TypeScript Release Notes** -- https://www.typescriptlang.org/docs/handbook/release-notes/overview.html
5. **Type-Level TypeScript** -- https://type-level-typescript.com/
6. **TypeScript Type Challenges** -- https://github.com/type-challenges/type-challenges
