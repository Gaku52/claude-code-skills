# 条件型（Conditional Types）

> 型レベルの条件分岐を実現する強力な型機能。extends、infer、分配条件型、再帰的条件型によって高度な型変換を行う。

## この章で学ぶこと

1. **条件型の基本** -- `T extends U ? X : Y` の構文、型レベルの if/else
2. **infer キーワード** -- 型パターンマッチングによる型の抽出
3. **分配条件型と再帰** -- Union型の分配処理、再帰的な型変換

---

## 1. 条件型の基本

### コード例1: 基本構文

```typescript
// 型レベルの if/else
type IsString<T> = T extends string ? true : false;

type A = IsString<string>;   // true
type B = IsString<number>;   // false
type C = IsString<"hello">;  // true（リテラル型もstringのサブタイプ）

// 実用例: null除外
type NonNullable<T> = T extends null | undefined ? never : T;

type D = NonNullable<string | null>;       // string
type E = NonNullable<number | undefined>;  // number
```

### コード例2: ネストした条件型

```typescript
// 型に応じて異なる変換を適用
type TypeName<T> =
  T extends string ? "string" :
  T extends number ? "number" :
  T extends boolean ? "boolean" :
  T extends undefined ? "undefined" :
  T extends Function ? "function" :
  "object";

type T1 = TypeName<string>;     // "string"
type T2 = TypeName<() => void>; // "function"
type T3 = TypeName<string[]>;   // "object"
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
```

---

## 2. infer キーワード

### コード例3: 型の抽出

```typescript
// 関数の戻り値型を取得
type ReturnType<T> = T extends (...args: any[]) => infer R ? R : never;

type A = ReturnType<() => string>;            // string
type B = ReturnType<(x: number) => boolean>;  // boolean

// Promiseの中身を取得
type Awaited<T> = T extends Promise<infer U> ? Awaited<U> : T;

type C = Awaited<Promise<string>>;            // string
type D = Awaited<Promise<Promise<number>>>;   // number（再帰的に展開）

// 関数のパラメータ型を取得
type Parameters<T> = T extends (...args: infer P) => any ? P : never;

type E = Parameters<(a: string, b: number) => void>;  // [string, number]

// 配列の要素型を取得
type ElementType<T> = T extends (infer U)[] ? U : never;

type F = ElementType<string[]>;  // string
type G = ElementType<number[]>;  // number
```

### コード例4: infer の高度な使い方

```typescript
// オブジェクトのプロパティ型を抽出
type PropType<T, K extends string> =
  T extends { [key in K]: infer V } ? V : never;

type UserName = PropType<{ name: string; age: number }, "name">;  // string

// テンプレートリテラル型でのinfer
type ExtractRouteParams<T extends string> =
  T extends `${string}:${infer Param}/${infer Rest}`
    ? Param | ExtractRouteParams<`/${Rest}`>
    : T extends `${string}:${infer Param}`
      ? Param
      : never;

type Params = ExtractRouteParams<"/users/:userId/posts/:postId">;
// "userId" | "postId"

// コンストラクタのインスタンス型を取得
type InstanceType<T> = T extends new (...args: any[]) => infer R ? R : never;
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
| T extends { a: infer U }            | プロパティ値   | プロパティの型   |
+-------------------------------------+----------------+------------------+
```

---

## 3. 分配条件型

### コード例5: 分配（Distribution）の仕組み

```typescript
// Union型が条件型に渡されると「分配」される
type ToArray<T> = T extends any ? T[] : never;

// string | number が渡されると:
// ToArray<string> | ToArray<number>
// = string[] | number[]
type A = ToArray<string | number>;  // string[] | number[]

// 分配を防ぎたい場合は [] で囲む
type ToArrayNonDist<T> = [T] extends [any] ? T[] : never;

type B = ToArrayNonDist<string | number>;  // (string | number)[]
```

### コード例6: 分配条件型の実用例

```typescript
// Union型からnullとundefinedを除外（NonNullable の実装）
type MyNonNullable<T> = T extends null | undefined ? never : T;

type A = MyNonNullable<string | null | undefined>;  // string

// Union型から特定の型を抽出
type Extract<T, U> = T extends U ? T : never;

type B = Extract<"a" | "b" | "c", "a" | "c">;  // "a" | "c"

// Union型から特定の型を除外
type Exclude<T, U> = T extends U ? never : T;

type C = Exclude<"a" | "b" | "c", "a">;  // "b" | "c"

// 判別共用体から特定のメンバーを抽出
type Action =
  | { type: "fetch"; url: string }
  | { type: "save"; data: unknown }
  | { type: "delete"; id: string };

type FetchAction = Extract<Action, { type: "fetch" }>;
// { type: "fetch"; url: string }
```

---

## 4. 再帰的条件型

### コード例7: 再帰型

```typescript
// ネストしたオブジェクトの深いReadonly
type DeepReadonly<T> =
  T extends (...args: any[]) => any
    ? T
    : T extends object
      ? { readonly [K in keyof T]: DeepReadonly<T[K]> }
      : T;

interface Config {
  db: {
    host: string;
    port: number;
    credentials: {
      user: string;
      pass: string;
    };
  };
}

type ReadonlyConfig = DeepReadonly<Config>;
// 全てのネストされたプロパティがreadonly

// フラット化（ネストした配列を1段階平坦化）
type Flatten<T> = T extends (infer U)[] ? U : T;

type A = Flatten<string[][]>;  // string[]
type B = Flatten<string[]>;    // string

// 深いフラット化
type DeepFlatten<T> = T extends (infer U)[] ? DeepFlatten<U> : T;

type C = DeepFlatten<string[][][]>;  // string
```

### コード例8: 型レベルのユーティリティ

```typescript
// パスからドット区切りのキーを生成
type DotPath<T, Prefix extends string = ""> =
  T extends object
    ? {
        [K in keyof T & string]: T[K] extends object
          ? DotPath<T[K], `${Prefix}${K}.`> | `${Prefix}${K}`
          : `${Prefix}${K}`;
      }[keyof T & string]
    : never;

interface Settings {
  theme: {
    color: string;
    fontSize: number;
  };
  user: {
    name: string;
  };
}

type SettingPaths = DotPath<Settings>;
// "theme" | "theme.color" | "theme.fontSize" | "user" | "user.name"
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
| `Awaited<T>` | 再帰的にPromiseを展開 | Promise解決後の型を取得 |

---

## 分配 vs 非分配 比較

| 特性 | 分配条件型 | 非分配条件型 |
|------|-----------|-------------|
| 構文 | `T extends U ? X : Y` | `[T] extends [U] ? X : Y` |
| Union入力 | 各メンバーに個別適用 | Union全体として評価 |
| `string \| number` | `F<string> \| F<number>` | `F<string \| number>` |
| 主な用途 | フィルタリング、型変換 | Union全体の判定 |

```
  分配 (Distributive)
  ToArray<string | number>
    → ToArray<string> | ToArray<number>
    → string[] | number[]

  非分配 (Non-distributive)
  ToArrayNonDist<string | number>
    → (string | number)[]
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

// 組み合わせて使う
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

---

## FAQ

### Q1: 条件型はどこで使うのが最も効果的ですか？

**A:** ライブラリの型定義や、APIの型変換（レスポンス型の自動導出など）で最も効果的です。アプリケーションコードでは、組み込みの `ReturnType`, `Parameters`, `Awaited` などのユーティリティ型を使うだけで十分なことが多いです。

### Q2: infer は条件型の中でしか使えませんか？

**A:** はい、`infer` は `extends` 節の中でのみ使用できます。`T extends ... infer U ... ? X : Y` の形式でのみ有効です。条件型の外で型を「抽出」したい場合は、別途条件型を定義する必要があります。

### Q3: 分配条件型で never はどう扱われますか？

**A:** `never` は空の Union として扱われるため、分配条件型に `never` が渡されると結果も `never` になります。
```typescript
type Test<T> = T extends string ? "yes" : "no";
type Result = Test<never>; // never（"yes" でも "no" でもない）
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
| 組み込み型 | ReturnType, Parameters, Awaited 等 |
| 注意点 | 過度な複雑さと再帰深度に注意 |

---

## 次に読むべきガイド

- [01-mapped-types.md](./01-mapped-types.md) -- マップ型
- [02-template-literal-types.md](./02-template-literal-types.md) -- テンプレートリテラル型

---

## 参考文献

1. **TypeScript Handbook: Conditional Types** -- https://www.typescriptlang.org/docs/handbook/2/conditional-types.html
2. **TypeScript Deep Dive: Conditional Types** -- https://basarat.gitbook.io/typescript/type-system/conditional-types
3. **Matt Pocock: Conditional Types in TypeScript** -- https://www.totaltypescript.com/books/total-typescript-essentials/conditional-types-and-infer
