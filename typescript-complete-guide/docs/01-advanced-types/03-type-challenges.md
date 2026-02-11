# 型チャレンジ

> Type Challenges の代表的な問題を解説し、型レベルプログラミングの実践力を養う。実用的な型パズルの解法パターンを習得する。

## この章で学ぶこと

1. **型レベルプログラミングの基礎テクニック** -- 再帰、パターンマッチ、アキュムレータ
2. **初級〜中級チャレンジ** -- Pick, Readonly, TupleToUnion, Last, Includes 等
3. **上級チャレンジ** -- DeepReadonly, Flatten, StringToUnion, CamelCase 等

---

## 1. 型レベルプログラミングの基礎テクニック

### 主要テクニック一覧

```
+------------------+-------------------------------+------------------------+
| テクニック       | 用途                          | 使用する構文           |
+------------------+-------------------------------+------------------------+
| 条件型           | 型の分岐                      | T extends U ? X : Y    |
| infer            | 型の抽出                      | T extends F<infer U>   |
| 再帰             | 繰り返し処理                  | Type<T> = ... Type<U>  |
| タプル操作       | 長さカウント、ループ          | [...T, U], T["length"] |
| 文字列操作       | パターンマッチ                | `${infer H}${infer T}` |
| マップ型         | プロパティ変換                | { [K in keyof T]: ... }|
| Key Remapping    | キー名の変換                  | [K in ... as ...]      |
+------------------+-------------------------------+------------------------+
```

### コード例1: タプルを使ったカウンター

```typescript
// 型レベルの数値演算にはタプルの長さを使う
type Length<T extends readonly unknown[]> = T["length"];

type A = Length<[1, 2, 3]>;  // 3
type B = Length<[]>;          // 0

// 型レベルの加算
type BuildTuple<N extends number, T extends unknown[] = []> =
  T["length"] extends N ? T : BuildTuple<N, [...T, unknown]>;

type Add<A extends number, B extends number> =
  [...BuildTuple<A>, ...BuildTuple<B>]["length"];

type Sum = Add<3, 4>;  // 7
```

---

## 2. 初級チャレンジ

### コード例2: MyPick（Pick の自作）

```typescript
// 課題: Pick<T, K> を自作せよ
type MyPick<T, K extends keyof T> = {
  [P in K]: T[P];
};

// テスト
interface Todo {
  title: string;
  description: string;
  completed: boolean;
}

type TodoPreview = MyPick<Todo, "title" | "completed">;
// { title: string; completed: boolean }
```

### コード例3: MyReadonly と TupleToUnion

```typescript
// 課題: Readonly<T> を自作せよ
type MyReadonly<T> = {
  readonly [K in keyof T]: T[K];
};

// 課題: タプル型をUnion型に変換せよ
type TupleToUnion<T extends readonly unknown[]> = T[number];

type A = TupleToUnion<[1, 2, 3]>;        // 1 | 2 | 3
type B = TupleToUnion<["a", "b", "c"]>;   // "a" | "b" | "c"

// 課題: First<T> - 配列の最初の要素の型を取得
type First<T extends readonly unknown[]> =
  T extends [infer F, ...unknown[]] ? F : never;

type C = First<[3, 2, 1]>;  // 3
type D = First<[]>;          // never
```

### コード例4: Last と Includes

```typescript
// 課題: Last<T> - 配列の最後の要素の型を取得
type Last<T extends readonly unknown[]> =
  T extends [...unknown[], infer L] ? L : never;

type A = Last<[1, 2, 3]>;  // 3
type B = Last<["a"]>;       // "a"

// 課題: Includes<T, U> - 配列にUが含まれるか
type IsEqual<A, B> =
  (<T>() => T extends A ? 1 : 2) extends (<T>() => T extends B ? 1 : 2)
    ? true
    : false;

type Includes<T extends readonly unknown[], U> =
  T extends [infer First, ...infer Rest]
    ? IsEqual<First, U> extends true
      ? true
      : Includes<Rest, U>
    : false;

type C = Includes<[1, 2, 3], 2>;     // true
type D = Includes<[1, 2, 3], 4>;     // false
type E = Includes<["a", "b"], "a">;   // true
```

### 初級チャレンジ解法パターン

```
  問題の種類            使うテクニック           典型例
+-------------------+----------------------+------------------+
| プロパティ変換     | マップ型              | Pick, Readonly   |
| 配列の先頭/末尾   | infer + rest          | First, Last      |
| 配列 → Union      | インデックスアクセス  | TupleToUnion     |
| 要素の検索         | 再帰 + IsEqual       | Includes         |
| 長さ取得           | T["length"]           | Length           |
+-------------------+----------------------+------------------+
```

---

## 3. 中級チャレンジ

### コード例5: DeepReadonly

```typescript
// 課題: 全てのネストしたプロパティをreadonlyにせよ
type DeepReadonly<T> =
  T extends (...args: any[]) => any
    ? T
    : T extends object
      ? { readonly [K in keyof T]: DeepReadonly<T[K]> }
      : T;

// テスト
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
```

### コード例6: Flatten と Chainable

```typescript
// 課題: ネストした配列をフラットにする型
type Flatten<T extends unknown[]> =
  T extends [infer First, ...infer Rest]
    ? First extends unknown[]
      ? [...Flatten<First>, ...Flatten<Rest>]
      : [First, ...Flatten<Rest>]
    : [];

type A = Flatten<[1, [2, [3]], 4]>;  // [1, 2, 3, 4]
type B = Flatten<[[1, 2], [3, 4]]>;  // [1, 2, 3, 4]

// 課題: チェイン可能なオプション設定
type Chainable<T = {}> = {
  option<K extends string, V>(
    key: K extends keyof T ? never : K,
    value: V
  ): Chainable<Omit<T, K> & Record<K, V>>;
  get(): T;
};

// 使用例
declare const config: Chainable;
const result = config
  .option("name", "TypeScript")
  .option("version", 5)
  .option("strict", true)
  .get();
// 型: { name: string; version: number; strict: boolean }
```

### コード例7: StringToUnion と Trim

```typescript
// 課題: 文字列をUnion型に変換
type StringToUnion<S extends string> =
  S extends `${infer C}${infer Rest}`
    ? C | StringToUnion<Rest>
    : never;

type A = StringToUnion<"hello">;  // "h" | "e" | "l" | "o"

// 課題: 文字列の前後の空白を除去
type TrimLeft<S extends string> =
  S extends `${" " | "\n" | "\t"}${infer Rest}`
    ? TrimLeft<Rest>
    : S;

type TrimRight<S extends string> =
  S extends `${infer Rest}${" " | "\n" | "\t"}`
    ? TrimRight<Rest>
    : S;

type Trim<S extends string> = TrimLeft<TrimRight<S>>;

type B = Trim<"  hello  ">;  // "hello"
type C = Trim<"\nhello\n">;  // "hello"
```

---

## 4. 上級チャレンジ

### コード例8: CamelCase

```typescript
// 課題: ケバブケースをキャメルケースに変換
type CamelCase<S extends string> =
  S extends `${infer Head}-${infer Tail}`
    ? `${Lowercase<Head>}${CamelCase<Capitalize<Tail>>}`
    : S;

type A = CamelCase<"foo-bar-baz">;      // "fooBarBaz"
type B = CamelCase<"hello-world">;       // "helloWorld"
type C = CamelCase<"no-dash">;           // "noDash"

// オブジェクトのキーをCamelCaseに変換
type CamelCaseKeys<T> = {
  [K in keyof T as K extends string ? CamelCase<K> : K]: T[K];
};

interface ApiResponse {
  "user-name": string;
  "created-at": string;
  "is-active": boolean;
}

type CamelResponse = CamelCaseKeys<ApiResponse>;
// { userName: string; createdAt: string; isActive: boolean }
```

### コード例9: 型レベルの算術

```typescript
// 型レベルの比較
type GreaterThan<
  A extends number,
  B extends number,
  Count extends unknown[] = []
> =
  Count["length"] extends A
    ? false
    : Count["length"] extends B
      ? true
      : GreaterThan<A, B, [...Count, unknown]>;

type A = GreaterThan<5, 3>;  // true
type B = GreaterThan<2, 7>;  // false

// 型レベルのFizzBuzz（概念的な例）
type FizzBuzz<N extends number, Acc extends string[] = [], Count extends unknown[] = [unknown]> =
  Count["length"] extends N
    ? [...Acc, FB<Count["length"]>]
    : FizzBuzz<N, [...Acc, FB<Count["length"]>], [...Count, unknown]>;

type Mod3<N extends number> = /* 省略: 3の倍数判定 */ any;
type FB<N extends number> = /* 省略: FizzBuzz判定 */ any;
```

---

## 難易度別チャレンジ分類

| 難易度 | チャレンジ | 必要テクニック |
|--------|-----------|---------------|
| Easy | Pick, Readonly, First, Length | マップ型、infer |
| Easy | TupleToUnion, Includes, If | インデックスアクセス、再帰 |
| Medium | DeepReadonly, Flatten, Trim | 再帰、文字列操作 |
| Medium | Chainable, CamelCase | Key Remapping、テンプレートリテラル |
| Hard | StringToNumber, CurryFn | 高度な再帰、複合テクニック |
| Extreme | JSON Parser, SQL Parser | 全テクニック総合 |

---

## 解法パターンの選び方

| パターン | 判断基準 | 例 |
|----------|---------|-----|
| マップ型 | オブジェクトのプロパティを変換 | Pick, Readonly, Partial |
| infer + 条件型 | 構造から型を抽出 | ReturnType, First, Last |
| 再帰 + タプル | 配列要素を順に処理 | Flatten, Includes, Reverse |
| 再帰 + 文字列 | 文字列を1文字ずつ処理 | Trim, CamelCase, StringToUnion |
| タプル長 | 数値の演算 | Add, Subtract, GreaterThan |
| Key Remapping | キー名の変換 | CamelCaseKeys, Getters |

---

## アンチパターン

### アンチパターン1: プロダクションコードで複雑な型パズルを使う

```typescript
// BAD: 型レベルのSQL パーサーをプロダクションで使う
type ParseSQL<T extends string> = /* 数百行の型定義 */;
// → コンパイル時間が爆発、エラーメッセージが意味不明

// GOOD: ライブラリの型を使う、またはシンプルな型で十分
import { sql } from "drizzle-orm";
// ライブラリが適切な型安全性を提供
```

### アンチパターン2: 再帰深度の限界を考慮しない

```typescript
// BAD: 深い再帰
type Repeat<S extends string, N extends number, Acc extends string = ""> =
  /* ... */;
type Long = Repeat<"a", 1000>;  // エラー: 再帰が深すぎる

// TypeScriptの再帰制限は約1000回程度
// GOOD: 再帰深度を意識して設計する
// 大きな数値の演算は型レベルでは避ける
```

---

## FAQ

### Q1: Type Challenges はどのように取り組むべきですか？

**A:** Easy から順に取り組みましょう。各問題について (1) 問題を理解する (2) 使えそうなテクニックを考える (3) 実装する (4) エッジケースを確認する、の手順で進めます。https://tsch.js.org/ でブラウザ上で挑戦できます。

### Q2: 型レベルプログラミングは実務で役立ちますか？

**A:** ライブラリ作者にとっては非常に重要です。アプリケーション開発者にとっても、型エラーの理解やユーティリティ型の活用において役立ちます。ただし、過度に複雑な型を書く必要はなく、基本的な型パズルの解法を知っていれば十分です。

### Q3: TypeScriptの型システムはチューリング完全ですか？

**A:** はい、再帰制限を除けばチューリング完全です。理論的にはあらゆる計算を型レベルで表現できますが、実用上は再帰深度制限（約1000回）やUnionメンバー数制限（約100,000）があるため、計算に限界があります。

---

## まとめ

| 項目 | 内容 |
|------|------|
| 型レベルプログラミング | 条件型、infer、再帰を使った型の計算 |
| 基本テクニック | マップ型、タプル操作、文字列パターンマッチ |
| Easy | Pick, Readonly, First, TupleToUnion |
| Medium | DeepReadonly, Flatten, CamelCase, Chainable |
| Hard | 型レベル算術、パーサー |
| 実務での活用 | ユーティリティ型の理解、ライブラリ型の設計 |
| 注意点 | 再帰深度制限、コンパイル速度への影響 |

---

## 次に読むべきガイド

- [04-declaration-files.md](./04-declaration-files.md) -- 宣言ファイル
- [../02-patterns/03-branded-types.md](../02-patterns/03-branded-types.md) -- ブランド型

---

## 参考文献

1. **Type Challenges** -- https://tsch.js.org/
2. **TypeScript Type-Level Programming** -- https://type-level-typescript.com/
3. **Matt Pocock: Total TypeScript** -- https://www.totaltypescript.com/
