# 型チャレンジ

> Type Challenges の代表的な問題を解説し、型レベルプログラミングの実践力を養う。実用的な型パズルの解法パターンを習得する。

## この章で学ぶこと

1. **型レベルプログラミングの基礎テクニック** -- 再帰、パターンマッチ、アキュムレータ
2. **初級チャレンジ** -- Pick, Readonly, TupleToUnion, Last, Includes 等
3. **中級チャレンジ** -- DeepReadonly, Flatten, StringToUnion, CamelCase, Chainable 等
4. **上級チャレンジ** -- 型レベル算術、パーサー、Union操作 等
5. **実務への応用** -- 型パズルのテクニックを実務コードに活かす方法

---

## 1. 型レベルプログラミングの基礎テクニック

### 1.1 主要テクニック一覧

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
| 分配条件型       | Union の各メンバーに適用      | T extends any ? F<T>   |
| タプル→ユニオン  | 配列型からユニオン型へ        | T[number]              |
| ユニオン→交差    | ユニオン型からインターセクション | 反変位置での infer    |
+------------------+-------------------------------+------------------------+
```

### 1.2 テスト用ユーティリティ型

型チャレンジの解答を検証するために、以下のユーティリティ型を使用する。

```typescript
// 2つの型が等しいかの判定（最も正確な実装）
type Equal<X, Y> =
  (<T>() => T extends X ? 1 : 2) extends
  (<T>() => T extends Y ? 1 : 2) ? true : false;

// Equal が true であることを期待する型（falseだとコンパイルエラー）
type Expect<T extends true> = T;

// Equal が false であることを期待する型
type ExpectFalse<T extends false> = T;

// テスト例
type Test1 = Expect<Equal<1, 1>>;           // OK
type Test2 = Expect<Equal<"a", "a">>;       // OK
// type Test3 = Expect<Equal<1, 2>>;         // コンパイルエラー（期待通り）
// type Test4 = Expect<Equal<string, number>>; // コンパイルエラー（期待通り）

// Not型: boolean を反転
type Not<T extends boolean> = T extends true ? false : true;

// テスト
type Test5 = Expect<Not<false>>;            // OK
type Test6 = ExpectFalse<Not<true>>;        // OK
```

### 1.3 タプルを使ったカウンター

```typescript
// 型レベルの数値演算にはタプルの長さを使う
type Length<T extends readonly unknown[]> = T["length"];

type A = Length<[1, 2, 3]>;  // 3
type B = Length<[]>;          // 0

// 指定した長さのタプルを構築
type BuildTuple<N extends number, T extends unknown[] = []> =
  T["length"] extends N ? T : BuildTuple<N, [...T, unknown]>;

type Tuple3 = BuildTuple<3>;   // [unknown, unknown, unknown]
type Tuple5 = BuildTuple<5>;   // [unknown, unknown, unknown, unknown, unknown]

// 型レベルの加算
type Add<A extends number, B extends number> =
  [...BuildTuple<A>, ...BuildTuple<B>]["length"] & number;

type Sum = Add<3, 4>;  // 7

// 型レベルの減算
type Subtract<A extends number, B extends number> =
  BuildTuple<A> extends [...BuildTuple<B>, ...infer Rest]
    ? Rest["length"]
    : never;  // B > A の場合は never

type Diff = Subtract<7, 3>;  // 4

// 型レベルの乗算
type Multiply<
  A extends number,
  B extends number,
  Acc extends unknown[] = []
> = B extends 0
  ? 0
  : BuildTuple<B> extends [unknown, ...infer Rest]
    ? Multiply<A, Rest["length"] & number, [...Acc, ...BuildTuple<A>]>
    : Acc["length"] & number;

type Product = Multiply<3, 4>;  // 12

// 型レベルの比較
type GreaterThanOrEqual<
  A extends number,
  B extends number,
  Count extends unknown[] = []
> = Count["length"] extends A
  ? Count["length"] extends B
    ? true  // A === B
    : false // Count reached A first, A < B
  : Count["length"] extends B
    ? true  // Count reached B first, A > B
    : GreaterThanOrEqual<A, B, [...Count, unknown]>;
```

### 1.4 再帰のパターン

```typescript
// パターン1: 配列の再帰処理（先頭から処理）
type ProcessArray<T extends unknown[]> =
  T extends [infer First, ...infer Rest]
    ? /* First を処理 */ [First, ...ProcessArray<Rest>]
    : [];

// パターン2: 配列の再帰処理（末尾から処理）
type ProcessFromEnd<T extends unknown[]> =
  T extends [...infer Init, infer Last]
    ? [...ProcessFromEnd<Init>, Last]
    : [];

// パターン3: アキュムレータパターン（結果を蓄積）
type Accumulate<
  T extends unknown[],
  Acc extends unknown[] = []  // 結果を蓄積
> = T extends [infer First, ...infer Rest]
  ? Accumulate<Rest, [...Acc, First]>
  : Acc;

// パターン4: 文字列の再帰処理
type ProcessString<S extends string> =
  S extends `${infer First}${infer Rest}`
    ? `${First}${ProcessString<Rest>}`
    : "";

// パターン5: 深さ制限付き再帰
type LimitedRecursion<
  T,
  Depth extends unknown[] = []
> = Depth["length"] extends 10
  ? T  // 深さ制限で停止
  : T extends object
    ? { [K in keyof T]: LimitedRecursion<T[K], [...Depth, unknown]> }
    : T;

// パターン6: 2分割による効率的な再帰
// 直線的な再帰より深い再帰が可能
type DeepBuild<
  N extends number,
  T extends unknown[] = [unknown]
> = T["length"] extends N
  ? T
  : DeepBuild<N, [...T, ...T]> extends infer R extends unknown[]
    ? R["length"] extends N
      ? R
      : /* trim excess */ R
    : never;
```

---

## 2. 初級チャレンジ

### 2.1 MyPick（Pick の自作）

```typescript
// 課題: Pick<T, K> を自作せよ
// 要件: T から K で指定されたプロパティだけを選択する型を実装する

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

// 検証
type TestPick1 = Expect<Equal<
  MyPick<Todo, "title">,
  { title: string }
>>;
type TestPick2 = Expect<Equal<
  MyPick<Todo, "title" | "completed">,
  { title: string; completed: boolean }
>>;

// 解説:
// - K extends keyof T で K を T のキーに制限
// - [P in K] で K のメンバーをイテレーション
// - T[P] で元の型の値をそのまま使用
```

### 2.2 MyReadonly

```typescript
// 課題: Readonly<T> を自作せよ
// 要件: T の全プロパティを readonly にする

type MyReadonly<T> = {
  readonly [K in keyof T]: T[K];
};

// テスト
type ReadonlyTodo = MyReadonly<Todo>;
// { readonly title: string; readonly description: string; readonly completed: boolean }

type TestReadonly = Expect<Equal<
  MyReadonly<Todo>,
  { readonly title: string; readonly description: string; readonly completed: boolean }
>>;
```

### 2.3 TupleToUnion

```typescript
// 課題: タプル型をUnion型に変換せよ
// 要件: [1, 2, 3] → 1 | 2 | 3

type TupleToUnion<T extends readonly unknown[]> = T[number];

type A = TupleToUnion<[1, 2, 3]>;        // 1 | 2 | 3
type B = TupleToUnion<["a", "b", "c"]>;   // "a" | "b" | "c"
type C = TupleToUnion<[string, number]>;   // string | number

// 検証
type TestTTU = Expect<Equal<TupleToUnion<[1, 2, 3]>, 1 | 2 | 3>>;

// 解説:
// T[number] は配列型のインデックスシグネチャで、全要素の型のユニオンを返す
// [1, 2, 3] の場合: [1, 2, 3][number] = 1 | 2 | 3

// 再帰版（理解を深めるため）
type TupleToUnionRecursive<T extends readonly unknown[]> =
  T extends [infer First, ...infer Rest]
    ? First | TupleToUnionRecursive<Rest>
    : never;
```

### 2.4 First

```typescript
// 課題: 配列の最初の要素の型を取得せよ
// 要件: First<[3, 2, 1]> = 3、First<[]> = never

// 解法1: infer を使用
type First<T extends readonly unknown[]> =
  T extends [infer F, ...unknown[]] ? F : never;

// 解法2: 条件型で空配列チェック
type First2<T extends readonly unknown[]> =
  T extends [] ? never : T[0];

// 解法3: T["length"] を使用
type First3<T extends readonly unknown[]> =
  T["length"] extends 0 ? never : T[0];

type D = First<[3, 2, 1]>;  // 3
type E = First<[]>;          // never
type F = First<[undefined]>; // undefined

// 検証
type TestFirst1 = Expect<Equal<First<[3, 2, 1]>, 3>>;
type TestFirst2 = Expect<Equal<First<[]>, never>>;
type TestFirst3 = Expect<Equal<First<[undefined]>, undefined>>;
```

### 2.5 Last

```typescript
// 課題: 配列の最後の要素の型を取得せよ
// 要件: Last<[1, 2, 3]> = 3

type Last<T extends readonly unknown[]> =
  T extends [...unknown[], infer L] ? L : never;

type G = Last<[1, 2, 3]>;  // 3
type H = Last<["a"]>;       // "a"
type I = Last<[]>;           // never

// 検証
type TestLast1 = Expect<Equal<Last<[1, 2, 3]>, 3>>;
type TestLast2 = Expect<Equal<Last<["a"]>, "a">>;
type TestLast3 = Expect<Equal<Last<[]>, never>>;

// 解説:
// [...unknown[], infer L] は「最後の要素を L として抽出」するパターン
// TypeScript 4.0+ の Variadic Tuple Types を活用
```

### 2.6 Includes

```typescript
// 課題: 配列にUが含まれるか判定せよ
// 要件: Includes<[1, 2, 3], 2> = true

// 正確な等値判定（IsEqual）
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

type J = Includes<[1, 2, 3], 2>;       // true
type K = Includes<[1, 2, 3], 4>;       // false
type L = Includes<["a", "b"], "a">;     // true
type M = Includes<[true, false], true>; // true
type N = Includes<[boolean], true>;     // false（boolean !== true）

// 検証
type TestInc1 = Expect<Equal<Includes<[1, 2, 3], 2>, true>>;
type TestInc2 = Expect<Equal<Includes<[1, 2, 3], 4>, false>>;
type TestInc3 = Expect<Equal<Includes<[boolean], true>, false>>;

// 解説:
// IsEqual は (<T>() => T extends X ? 1 : 2) のパターンで厳密な型比較を行う
// extends による比較では boolean と true が一致してしまうため、
// この特殊なパターンが必要
```

### 2.7 Push と Unshift

```typescript
// 課題: Push<T, U> - 配列の末尾に要素を追加
type Push<T extends unknown[], U> = [...T, U];

type P1 = Push<[1, 2], 3>;       // [1, 2, 3]
type P2 = Push<[], 1>;            // [1]

// 課題: Unshift<T, U> - 配列の先頭に要素を追加
type Unshift<T extends unknown[], U> = [U, ...T];

type U1 = Unshift<[1, 2], 0>;    // [0, 1, 2]
type U2 = Unshift<[], 1>;         // [1]

// 課題: Pop<T> - 配列の末尾要素を除去
type Pop<T extends unknown[]> =
  T extends [...infer Init, unknown] ? Init : [];

type Pop1 = Pop<[1, 2, 3]>;  // [1, 2]
type Pop2 = Pop<[1]>;         // []
type Pop3 = Pop<[]>;           // []

// 課題: Shift<T> - 配列の先頭要素を除去
type Shift<T extends unknown[]> =
  T extends [unknown, ...infer Rest] ? Rest : [];

type Shift1 = Shift<[1, 2, 3]>;  // [2, 3]
type Shift2 = Shift<[1]>;         // []
```

### 2.8 Concat

```typescript
// 課題: Concat<T, U> - 2つの配列を結合
type Concat<T extends unknown[], U extends unknown[]> = [...T, ...U];

type C1 = Concat<[1, 2], [3, 4]>;     // [1, 2, 3, 4]
type C2 = Concat<[], [1]>;             // [1]
type C3 = Concat<[1], []>;             // [1]

// 検証
type TestConcat = Expect<Equal<Concat<[1, 2], [3, 4]>, [1, 2, 3, 4]>>;
```

### 2.9 If

```typescript
// 課題: If<C, T, F> - Cがtrueなら T、falseなら F
type If<C extends boolean, T, F> = C extends true ? T : F;

type If1 = If<true, "a", "b">;   // "a"
type If2 = If<false, "a", "b">;  // "b"

// 検証
type TestIf1 = Expect<Equal<If<true, "a", "b">, "a">>;
type TestIf2 = Expect<Equal<If<false, "a", "b">, "b">>;
```

### 2.10 Awaited

```typescript
// 課題: Promise の解決値の型を取得せよ
// 要件: ネストした Promise も再帰的に展開

type MyAwaited<T> =
  T extends Promise<infer U>
    ? U extends Promise<any>
      ? MyAwaited<U>  // ネストした Promise を再帰的に展開
      : U
    : never;

type Aw1 = MyAwaited<Promise<string>>;            // string
type Aw2 = MyAwaited<Promise<Promise<number>>>;    // number
type Aw3 = MyAwaited<Promise<Promise<Promise<boolean>>>>; // boolean

// もっとシンプルな実装
type MyAwaited2<T> =
  T extends Promise<infer U> ? MyAwaited2<U> : T;

// 検証
type TestAwaited = Expect<Equal<MyAwaited<Promise<string>>, string>>;
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
| 配列操作           | スプレッド構文        | Push, Concat     |
| 条件分岐           | 条件型                | If, Awaited      |
| Promise展開        | infer + 再帰          | Awaited          |
+-------------------+----------------------+------------------+
```

---

## 3. 中級チャレンジ

### 3.1 DeepReadonly

```typescript
// 課題: 全てのネストしたプロパティをreadonlyにせよ
type DeepReadonly<T> =
  T extends (...args: any[]) => any
    ? T  // 関数はそのまま
    : T extends object
      ? { readonly [K in keyof T]: DeepReadonly<T[K]> }
      : T;  // プリミティブはそのまま

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
  features: string[];
}

type ReadonlyConfig = DeepReadonly<Config>;
// 全てのネストされたプロパティがreadonly

// 検証
type TestDeepReadonly = Expect<Equal<
  DeepReadonly<{ a: { b: string } }>,
  { readonly a: { readonly b: string } }
>>;

// 解説:
// 1. 関数は再帰しない（無限ループ防止）
// 2. object の場合は再帰的に DeepReadonly を適用
// 3. プリミティブ（string, number等）はそのまま返す
```

### 3.2 Flatten

```typescript
// 課題: ネストした配列をフラットにする型
type Flatten<T extends unknown[]> =
  T extends [infer First, ...infer Rest]
    ? First extends unknown[]
      ? [...Flatten<First>, ...Flatten<Rest>]
      : [First, ...Flatten<Rest>]
    : [];

type F1 = Flatten<[1, [2, [3]], 4]>;    // [1, 2, 3, 4]
type F2 = Flatten<[[1, 2], [3, 4]]>;    // [1, 2, 3, 4]
type F3 = Flatten<[1, 2, 3]>;           // [1, 2, 3]（フラットなら変わらない）
type F4 = Flatten<[]>;                   // []

// 検証
type TestFlatten = Expect<Equal<Flatten<[1, [2, [3]], 4]>, [1, 2, 3, 4]>>;

// 深さ制限付きFlatten
type FlattenDepth<
  T extends unknown[],
  Depth extends number = 1,
  Counter extends unknown[] = []
> = Counter["length"] extends Depth
  ? T
  : T extends [infer First, ...infer Rest]
    ? First extends unknown[]
      ? [...FlattenDepth<First, Depth, [...Counter, unknown]>, ...FlattenDepth<Rest, Depth, Counter>]
      : [First, ...FlattenDepth<Rest, Depth, Counter>]
    : T;

type FD1 = FlattenDepth<[1, [2, [3, [4]]]], 1>;  // [1, 2, [3, [4]]]
type FD2 = FlattenDepth<[1, [2, [3, [4]]]], 2>;  // [1, 2, 3, [4]]
```

### 3.3 Chainable

```typescript
// 課題: チェイン可能なオプション設定
// 要件:
// - option(key, value) で設定を追加
// - 同じキーを2回設定するとエラー
// - get() で最終的な型を取得

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

// 解説:
// 1. Chainable<T> はこれまでに設定されたオプションの型 T を保持
// 2. option() は新しいキーと値を T に追加した Chainable を返す
// 3. K extends keyof T ? never : K で重複キーを防止
// 4. get() は蓄積された型 T を返す
```

### 3.4 StringToUnion

```typescript
// 課題: 文字列をUnion型に変換
type StringToUnion<S extends string> =
  S extends `${infer C}${infer Rest}`
    ? C | StringToUnion<Rest>
    : never;

type STU1 = StringToUnion<"hello">;  // "h" | "e" | "l" | "o"
type STU2 = StringToUnion<"abc">;    // "a" | "b" | "c"
type STU3 = StringToUnion<"">;       // never

// 検証
type TestSTU = Expect<Equal<StringToUnion<"abc">, "a" | "b" | "c">>;

// 注意: "hello" の結果に "l" は1つだけ含まれる（Unionは重複を除去）
```

### 3.5 Trim

```typescript
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

type T1 = Trim<"  hello  ">;   // "hello"
type T2 = Trim<"\nhello\n">;   // "hello"
type T3 = Trim<"\t hello \t">; // "hello"

// 検証
type TestTrim = Expect<Equal<Trim<"  hello  ">, "hello">>;
```

### 3.6 Replace と ReplaceAll

```typescript
// 課題: 文字列の最初の一致箇所を置換
type Replace<S extends string, From extends string, To extends string> =
  From extends ""
    ? S
    : S extends `${infer Before}${From}${infer After}`
      ? `${Before}${To}${After}`
      : S;

type R1 = Replace<"types are fun!", "fun", "awesome">;  // "types are awesome!"
type R2 = Replace<"foobar", "bar", "baz">;               // "foobaz"
type R3 = Replace<"foobar", "", "baz">;                   // "foobar"

// 課題: 文字列の全ての一致箇所を置換
type ReplaceAll<S extends string, From extends string, To extends string> =
  From extends ""
    ? S
    : S extends `${infer Before}${From}${infer After}`
      ? ReplaceAll<`${Before}${To}${After}`, From, To>
      : S;

type RA1 = ReplaceAll<"t y p e s", " ", "">;  // "types"
type RA2 = ReplaceAll<"aaa", "a", "b">;        // "bbb"
```

### 3.7 Reverse

```typescript
// 課題: タプルを逆順にせよ
type Reverse<T extends unknown[]> =
  T extends [infer First, ...infer Rest]
    ? [...Reverse<Rest>, First]
    : [];

type Rev1 = Reverse<[1, 2, 3]>;       // [3, 2, 1]
type Rev2 = Reverse<["a", "b", "c"]>; // ["c", "b", "a"]
type Rev3 = Reverse<[]>;               // []

// 検証
type TestReverse = Expect<Equal<Reverse<[1, 2, 3]>, [3, 2, 1]>>;

// 文字列のReverse
type ReverseString<S extends string> =
  S extends `${infer First}${infer Rest}`
    ? `${ReverseString<Rest>}${First}`
    : "";

type RS1 = ReverseString<"hello">;  // "olleh"
```

### 3.8 Omit の自作

```typescript
// 課題: Omit<T, K> を自作せよ
// 方法1: Pick + Exclude
type MyOmit<T, K extends keyof T> = Pick<T, Exclude<keyof T, K>>;

// 方法2: マップ型 + as句
type MyOmit2<T, K extends keyof T> = {
  [P in keyof T as P extends K ? never : P]: T[P];
};

// テスト
type OmitTest = MyOmit<Todo, "description">;
// { title: string; completed: boolean }

// 検証
type TestOmit = Expect<Equal<
  MyOmit<Todo, "description">,
  { title: string; completed: boolean }
>>;
```

### 3.9 ReturnType の自作

```typescript
// 課題: ReturnType<T> を自作せよ
type MyReturnType<T extends (...args: any) => any> =
  T extends (...args: any) => infer R ? R : never;

type RT1 = MyReturnType<() => string>;            // string
type RT2 = MyReturnType<(x: number) => boolean>;  // boolean
type RT3 = MyReturnType<() => void>;               // void

// 検証
type TestRT = Expect<Equal<MyReturnType<() => string>, string>>;
```

### 3.10 Parameters の自作

```typescript
// 課題: Parameters<T> を自作せよ
type MyParameters<T extends (...args: any) => any> =
  T extends (...args: infer P) => any ? P : never;

type Params1 = MyParameters<(a: string, b: number) => void>;  // [a: string, b: number]
type Params2 = MyParameters<() => void>;                        // []

// 検証
type TestParams = Expect<Equal<
  MyParameters<(a: string, b: number) => void>,
  [a: string, b: number]
>>;
```

---

## 4. 上級チャレンジ

### 4.1 CamelCase

```typescript
// 課題: ケバブケースをキャメルケースに変換
type CamelCase<S extends string> =
  S extends `${infer Head}-${infer Tail}`
    ? Tail extends Capitalize<Tail>
      ? `${Head}-${CamelCase<Tail>}`  // 既に大文字始まりなら - を保持
      : `${Head}${CamelCase<Capitalize<Tail>>}`
    : S;

type CC1 = CamelCase<"foo-bar-baz">;       // "fooBarBaz"
type CC2 = CamelCase<"hello-world">;        // "helloWorld"
type CC3 = CamelCase<"no-dash">;            // "noDash"
type CC4 = CamelCase<"already">;            // "already"（ダッシュなし）

// より厳密な実装
type CamelCase2<S extends string> =
  S extends `${infer Head}-${infer Tail}`
    ? `${Lowercase<Head>}${CamelCase2<Capitalize<Tail>>}`
    : S;

// オブジェクトのキーをCamelCaseに変換
type CamelCaseKeys<T> = {
  [K in keyof T as K extends string ? CamelCase2<K> : K]:
    T[K] extends object
      ? T[K] extends any[]
        ? T[K]
        : CamelCaseKeys<T[K]>
      : T[K];
};

interface ApiResponse {
  "user-name": string;
  "created-at": string;
  "is-active": boolean;
  "profile-data": {
    "avatar-url": string;
    "display-name": string;
  };
}

type CamelResponse = CamelCaseKeys<ApiResponse>;
// {
//   userName: string;
//   createdAt: string;
//   isActive: boolean;
//   profileData: {
//     avatarUrl: string;
//     displayName: string;
//   };
// }
```

### 4.2 Unique

```typescript
// 課題: タプルから重複を除去
type Unique<T extends unknown[], Seen extends unknown[] = []> =
  T extends [infer First, ...infer Rest]
    ? Includes<Seen, First> extends true
      ? Unique<Rest, Seen>
      : Unique<Rest, [...Seen, First]>
    : Seen;

type U1 = Unique<[1, 1, 2, 2, 3, 3]>;           // [1, 2, 3]
type U2 = Unique<[1, "a", 2, "b", 2, "a"]>;      // [1, "a", 2, "b"]
type U3 = Unique<[string, string, number]>;        // [string, number]

// 検証
type TestUnique = Expect<Equal<Unique<[1, 1, 2, 2, 3, 3]>, [1, 2, 3]>>;
```

### 4.3 Zip

```typescript
// 課題: 2つの配列をペアにする
type Zip<
  T extends unknown[],
  U extends unknown[]
> = T extends [infer TFirst, ...infer TRest]
  ? U extends [infer UFirst, ...infer URest]
    ? [[TFirst, UFirst], ...Zip<TRest, URest>]
    : []
  : [];

type Z1 = Zip<[1, 2, 3], ["a", "b", "c"]>;
// [[1, "a"], [2, "b"], [3, "c"]]

type Z2 = Zip<[1, 2], ["a", "b", "c"]>;
// [[1, "a"], [2, "b"]]（短い方に合わせる）

// 検証
type TestZip = Expect<Equal<
  Zip<[1, 2, 3], ["a", "b", "c"]>,
  [[1, "a"], [2, "b"], [3, "c"]]
>>;
```

### 4.4 GroupBy

```typescript
// 課題: 配列を条件でグルーピング
type GroupBy<
  T extends Record<string, any>[],
  Key extends string
> = {
  [V in T[number][Key]]: Extract<T[number], Record<Key, V>>[];
};

type Items = [
  { type: "a"; value: 1 },
  { type: "b"; value: 2 },
  { type: "a"; value: 3 },
];

type Grouped = GroupBy<Items, "type">;
// {
//   a: ({ type: "a"; value: 1 } | { type: "a"; value: 3 })[];
//   b: { type: "b"; value: 2 }[];
// }
```

### 4.5 TupleToNestedObject

```typescript
// 課題: タプルからネストしたオブジェクト型を生成
type TupleToNestedObject<
  T extends string[],
  V
> = T extends [infer First extends string, ...infer Rest extends string[]]
  ? { [K in First]: TupleToNestedObject<Rest, V> }
  : V;

type Nested1 = TupleToNestedObject<["a", "b", "c"], string>;
// { a: { b: { c: string } } }

type Nested2 = TupleToNestedObject<["x"], number>;
// { x: number }

type Nested3 = TupleToNestedObject<[], boolean>;
// boolean

// 検証
type TestNested = Expect<Equal<
  TupleToNestedObject<["a", "b"], string>,
  { a: { b: string } }
>>;
```

### 4.6 UnionToIntersection

```typescript
// 課題: Union型をIntersection型に変換
// これは「型パズルの王道」と呼ばれる高度なテクニック

type UnionToIntersection<U> =
  (U extends any ? (k: U) => void : never) extends (k: infer I) => void
    ? I
    : never;

type UTI1 = UnionToIntersection<{ a: 1 } | { b: 2 }>;
// { a: 1 } & { b: 2 }

type UTI2 = UnionToIntersection<((x: string) => void) | ((x: number) => void)>;
// ((x: string) => void) & ((x: number) => void)
// = (x: string & number) => void = (x: never) => void

// 解説:
// 1. (U extends any ? (k: U) => void : never)
//    → 分配条件型により各ユニオンメンバーを関数引数にラップ
//    → ((k: { a: 1 }) => void) | ((k: { b: 2 }) => void)
// 2. extends (k: infer I) => void
//    → 関数引数は反変位置（contravariant position）
//    → 反変位置での infer はインターセクションを推論
//    → I = { a: 1 } & { b: 2 }
```

### 4.7 UnionToTuple

```typescript
// 課題: Union型をタプル型に変換
// 注意: ユニオンの順序は保証されない

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

type UT1 = UnionToTuple<"a" | "b" | "c">;
// ["a", "b", "c"]（順序は保証されない）

type UT2 = UnionToTuple<1 | 2 | 3>;
// [1, 2, 3]（順序は保証されない）
```

### 4.8 IsUnion

```typescript
// 課題: 型がUnion型かどうかを判定
type IsUnion<T, Copy = T> =
  T extends any  // 分配条件型で各メンバーに展開
    ? [Copy] extends [T]  // 元の型全体が現在のメンバーのサブタイプか
      ? false  // サブタイプなら単一型
      : true   // そうでなければUnion
    : never;

type IU1 = IsUnion<string>;           // false
type IU2 = IsUnion<string | number>;  // true
type IU3 = IsUnion<never>;            // false
type IU4 = IsUnion<boolean>;          // true（boolean = true | false）

// 検証
type TestIsUnion1 = Expect<Equal<IsUnion<string | number>, true>>;
type TestIsUnion2 = Expect<Equal<IsUnion<string>, false>>;
type TestIsUnion3 = Expect<Equal<IsUnion<boolean>, true>>;

// 解説:
// boolean は実は true | false のユニオン型
// 分配条件型で T extends any により各メンバーに展開される
// [Copy] extends [T] で「元の型全体」と「個別メンバー」を比較
// 単一型なら一致するが、ユニオンなら一致しない
```

### 4.9 PercentageParser

```typescript
// 課題: 文字列を符号、数値、パーセント記号に分解
type PercentageParser<S extends string> =
  S extends `${infer Sign extends "+" | "-"}${infer Num}%`
    ? [Sign, Num, "%"]
    : S extends `${infer Sign extends "+" | "-"}${infer Num}`
      ? [Sign, Num, ""]
      : S extends `${infer Num}%`
        ? ["", Num, "%"]
        : S extends ""
          ? ["", "", ""]
          : ["", S, ""];

type PP1 = PercentageParser<"+100%">;  // ["+", "100", "%"]
type PP2 = PercentageParser<"-50">;    // ["-", "50", ""]
type PP3 = PercentageParser<"100%">;   // ["", "100", "%"]
type PP4 = PercentageParser<"42">;     // ["", "42", ""]
type PP5 = PercentageParser<"">;       // ["", "", ""]
```

### 4.10 型レベルの FizzBuzz

```typescript
// 課題: 型レベルで FizzBuzz を実装

// 3で割り切れるか判定
type IsDivisibleBy3<
  N extends number,
  Count extends unknown[] = [],
  Triple extends unknown[] = []
> = Count["length"] extends N
  ? true
  : Triple["length"] extends 3
    ? IsDivisibleBy3<N, Count, []>
    : IsDivisibleBy3<N, [...Count, unknown], [...Triple, unknown]>;

// 5で割り切れるか判定
type IsDivisibleBy5<
  N extends number,
  Count extends unknown[] = [],
  Penta extends unknown[] = []
> = Count["length"] extends N
  ? true
  : Penta["length"] extends 5
    ? IsDivisibleBy5<N, Count, []>
    : IsDivisibleBy5<N, [...Count, unknown], [...Penta, unknown]>;

// FizzBuzz の1要素
type FBValue<N extends number> =
  IsDivisibleBy3<N> extends true
    ? IsDivisibleBy5<N> extends true
      ? "FizzBuzz"
      : "Fizz"
    : IsDivisibleBy5<N> extends true
      ? "Buzz"
      : N;

// FizzBuzz の実行
type FizzBuzz<
  N extends number,
  Acc extends (string | number)[] = [],
  Count extends unknown[] = [unknown]  // 1から開始
> = Count["length"] extends [...BuildTuple<N>, unknown]["length"]
  ? Acc
  : FizzBuzz<N, [...Acc, FBValue<Count["length"] & number>], [...Count, unknown]>;

// 小さな数でテスト
type FB15 = FizzBuzz<15>;
// [1, 2, "Fizz", 4, "Buzz", "Fizz", 7, 8, "Fizz", "Buzz", 11, "Fizz", 13, 14, "FizzBuzz"]
```

---

## 5. 実務への応用

### 5.1 型チャレンジのテクニックを実務で活かす

```typescript
// パターン1: APIレスポンスの型変換（CamelCase チャレンジの応用）
// スネークケースのAPIレスポンスをキャメルケースに自動変換

type SnakeToCamel<S extends string> =
  S extends `${infer Head}_${infer Tail}`
    ? `${Head}${Capitalize<SnakeToCamel<Tail>>}`
    : S;

type CamelizeKeys<T> = {
  [K in keyof T as K extends string ? SnakeToCamel<K> : K]:
    T[K] extends object
      ? T[K] extends any[]
        ? CamelizeKeys<T[K][number]>[]
        : CamelizeKeys<T[K]>
      : T[K];
};

// パターン2: 設定オブジェクトの型安全化（Chainable チャレンジの応用）
class ConfigBuilder<T extends Record<string, unknown> = {}> {
  private config: T;

  constructor(config: T = {} as T) {
    this.config = config;
  }

  set<K extends string, V>(
    key: K extends keyof T ? never : K,
    value: V
  ): ConfigBuilder<T & Record<K, V>> {
    return new ConfigBuilder({ ...this.config, [key]: value } as any);
  }

  build(): Readonly<T> {
    return Object.freeze({ ...this.config });
  }
}

const config = new ConfigBuilder()
  .set("host", "localhost")
  .set("port", 3000)
  .set("debug", true)
  .build();
// Readonly<{ host: string; port: number; debug: boolean }>

// パターン3: 型安全なパスアクセス（TupleToNestedObject チャレンジの応用）
type PathValue<T, P extends string> =
  P extends `${infer Key}.${infer Rest}`
    ? Key extends keyof T
      ? PathValue<T[Key], Rest>
      : never
    : P extends keyof T
      ? T[P]
      : never;

function get<T, P extends string>(obj: T, path: P): PathValue<T, P> {
  return path.split(".").reduce((acc: any, key) => acc?.[key], obj);
}

// パターン4: 型安全なイベントシステム（StringToUnion チャレンジの応用）
type EventCallback<T> = (data: T) => void;

class TypedEmitter<Events extends Record<string, any>> {
  private handlers = new Map<string, Function[]>();

  on<K extends keyof Events & string>(
    event: K,
    callback: EventCallback<Events[K]>
  ): this {
    const existing = this.handlers.get(event) || [];
    this.handlers.set(event, [...existing, callback]);
    return this;
  }

  emit<K extends keyof Events & string>(
    event: K,
    data: Events[K]
  ): void {
    const handlers = this.handlers.get(event) || [];
    handlers.forEach(h => h(data));
  }
}
```

### 5.2 型チャレンジの知識が必要になる場面

```typescript
// 場面1: ライブラリの型定義を読み解く
// Prisma、tRPC、Zod などの高度な型定義は
// 条件型、infer、再帰を駆使している

// 場面2: 自作ライブラリの型設計
// 型安全な API を提供するために
// マップ型 + テンプレートリテラル型の組み合わせが必要

// 場面3: 複雑なジェネリクスの問題解決
// 「なぜ型が合わない？」を理解するために
// 分配条件型やinferの挙動の知識が必要

// 場面4: カスタムユーティリティ型の作成
// 標準のPartial, Pickでは不十分な場合に
// DeepPartial, PickByType 等を自作する必要がある
```

---

## 難易度別チャレンジ分類

| 難易度 | チャレンジ | 必要テクニック |
|--------|-----------|---------------|
| Easy | Pick, Readonly, First, Length | マップ型、infer |
| Easy | TupleToUnion, Includes, If | インデックスアクセス、再帰 |
| Easy | Push, Unshift, Concat, Awaited | スプレッド、再帰 |
| Medium | DeepReadonly, Flatten, Trim | 再帰、文字列操作 |
| Medium | Chainable, CamelCase, Omit | Key Remapping、テンプレートリテラル |
| Medium | Replace, Reverse, StringToUnion | 文字列再帰 |
| Hard | UnionToIntersection, UnionToTuple | 反変位置、分配条件型 |
| Hard | IsUnion, FizzBuzz, CurryFn | 高度な再帰、複合テクニック |
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
| 分配条件型 | ユニオンの各メンバーに処理 | Extract, Exclude, NonNullable |
| 反変位置 infer | ユニオン→インターセクション | UnionToIntersection |
| アキュムレータ | 結果を蓄積しながら再帰 | Unique, FizzBuzz, GroupBy |

```
  問題を見たとき:

  1. 入力の型は何？
     +-- オブジェクト → マップ型を検討
     +-- 配列/タプル → infer + ... rest を検討
     +-- 文字列 → テンプレートリテラル + infer を検討
     +-- ユニオン → 分配条件型を検討
     +-- 数値 → タプル長カウンターを検討

  2. 処理の種類は？
     +-- 変換 → 元の構造を保持しつつ各要素を変換
     +-- フィルタ → never を返して除外
     +-- 抽出 → infer で一部を取り出す
     +-- 構築 → アキュムレータで結果を蓄積
     +-- 判定 → true / false を返す

  3. 再帰が必要？
     +-- ネストした構造 → 再帰
     +-- 可変長のデータ → 再帰 + 終了条件
     +-- 固定構造 → 非再帰
```

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

// GOOD: 必要十分な型を定義
type QueryResult<T extends string> = T extends `SELECT * FROM users`
  ? User[]
  : T extends `SELECT * FROM posts`
    ? Post[]
    : unknown[];
```

### アンチパターン2: 再帰深度の限界を考慮しない

```typescript
// BAD: 深い再帰
type Repeat<S extends string, N extends number, Acc extends string = ""> =
  BuildTuple<N> extends [unknown, ...infer Rest]
    ? Repeat<S, Rest["length"] & number, `${Acc}${S}`>
    : Acc;

// type Long = Repeat<"a", 1000>;  // エラー: 再帰が深すぎる

// TypeScriptの再帰制限は約1000回程度（TS 4.5+）
// GOOD: 再帰深度を意識して設計する
// 大きな数値の演算は型レベルでは避ける
```

### アンチパターン3: IsEqual の誤った実装

```typescript
// BAD: extends による比較
type BadEqual<A, B> = A extends B ? B extends A ? true : false : false;

// 問題: any, boolean, union 型で誤った結果になる
type Test1 = BadEqual<boolean, true>;   // boolean（true | false を返す）
type Test2 = BadEqual<any, string>;     // boolean

// GOOD: 正確な等値判定
type GoodEqual<X, Y> =
  (<T>() => T extends X ? 1 : 2) extends
  (<T>() => T extends Y ? 1 : 2) ? true : false;

type Test3 = GoodEqual<boolean, true>;  // false（正しい）
type Test4 = GoodEqual<any, string>;    // false（正しい）
```

### アンチパターン4: 分配条件型の意図しない動作

```typescript
// BAD: never や boolean で予期しない結果
type BadIsNever<T> = T extends never ? true : false;
type Test1 = BadIsNever<never>;  // never（true でも false でもない！）

// GOOD: 非分配で判定
type GoodIsNever<T> = [T] extends [never] ? true : false;
type Test2 = GoodIsNever<never>;  // true

// BAD: boolean の分配を忘れる
type BadCheck<T> = T extends true ? "yes" : "no";
type Test3 = BadCheck<boolean>;  // "yes" | "no"（boolean = true | false が分配される）

// GOOD: 意図を明確にする
type GoodCheck<T> = [T] extends [true] ? "yes" : "no";
type Test4 = GoodCheck<boolean>;  // "no"（boolean は [true] のサブタイプではない）
```

---

## FAQ

### Q1: Type Challenges はどのように取り組むべきですか？

**A:** Easy から順に取り組みましょう。各問題について以下の手順で進めます:
1. 問題の要件を理解する（入力と期待される出力を確認）
2. 使えそうなテクニックを考える（マップ型？再帰？infer？）
3. 小さなケースから実装する
4. エッジケースを確認する（空配列、never、boolean、any）

https://tsch.js.org/ でブラウザ上で挑戦できます。各問題にはテストケースが付属しているので、自分の解答が正しいか即座に確認できます。

### Q2: 型レベルプログラミングは実務で役立ちますか？

**A:** ライブラリ作者にとっては非常に重要です。アプリケーション開発者にとっても、以下の場面で役立ちます:
- 型エラーの理解（なぜこの型が通らないのかを理解）
- ユーティリティ型の活用（DeepPartial, PickByType 等）
- ライブラリの型定義の読解（Prisma, tRPC, Zod 等）
- カスタムユーティリティ型の作成

ただし、過度に複雑な型を書く必要はなく、基本的な型パズルの解法パターン（条件型、infer、マップ型、再帰）を理解していれば十分です。

### Q3: TypeScriptの型システムはチューリング完全ですか？

**A:** はい、再帰制限を除けばチューリング完全です。理論的にはあらゆる計算を型レベルで表現できますが、実用上は以下の制限があります:
- 再帰深度制限（約1000回、TS 4.5+）
- Unionメンバー数制限（約100,000）
- コンパイル時間の増大

型レベルで「できる」ことと「すべき」ことは異なります。ランタイムで簡単にできることを型レベルで複雑に実装する必要はありません。

### Q4: Equal 型の仕組みがわかりません

**A:** `Equal<X, Y>` の実装を段階的に説明します:

```typescript
type Equal<X, Y> =
  (<T>() => T extends X ? 1 : 2) extends
  (<T>() => T extends Y ? 1 : 2) ? true : false;
```

1. `<T>() => T extends X ? 1 : 2` は「T が X のサブタイプなら 1、そうでなければ 2 を返す関数型」
2. X と Y が同じ型なら、この2つの関数型も同じ → true
3. X と Y が異なる型なら、関数型も異なる → false

この方法が extends による比較より正確な理由:
- `any extends string` は条件分岐で特殊な挙動をする
- `boolean extends true` は分配されてしまう
- この関数型パターンはこれらの問題を回避する

### Q5: 再帰が深すぎるエラーが出た場合の対処法は？

**A:** 以下の対策があります:
1. **再帰深度カウンターを追加**: タプルの長さで深さを追跡し、上限で停止
2. **末尾再帰最適化**: TypeScript 4.5+ では一部の再帰が最適化される
3. **分割統治法**: 問題を小さく分割して再帰の深さを減らす
4. **ランタイムに移行**: 型レベルでやるべきでない処理かもしれない

```typescript
// 末尾再帰の最適化が効くパターン
type TailRecursive<T extends unknown[], Acc extends unknown[] = []> =
  T extends [infer First, ...infer Rest]
    ? TailRecursive<Rest, [...Acc, First]>  // 末尾位置で再帰
    : Acc;

// 末尾再帰でないパターン（最適化されない）
type NonTailRecursive<T extends unknown[]> =
  T extends [infer First, ...infer Rest]
    ? [First, ...NonTailRecursive<Rest>]  // スプレッドの後に再帰結果
    : [];
```

---

## まとめ

| 項目 | 内容 |
|------|------|
| 型レベルプログラミング | 条件型、infer、再帰を使った型の計算 |
| 基本テクニック | マップ型、タプル操作、文字列パターンマッチ |
| テスト | Equal + Expect で型の正しさを検証 |
| Easy | Pick, Readonly, First, TupleToUnion, Includes, Awaited |
| Medium | DeepReadonly, Flatten, CamelCase, Chainable, Trim |
| Hard | UnionToIntersection, UnionToTuple, IsUnion, FizzBuzz |
| 実務での活用 | API型変換、設定ビルダー、イベントシステム |
| 注意点 | 再帰深度制限、コンパイル速度、IsEqual の正確性 |

---

## 次に読むべきガイド

- [04-declaration-files.md](./04-declaration-files.md) -- 宣言ファイル
- [00-conditional-types.md](./00-conditional-types.md) -- 条件型
- [01-mapped-types.md](./01-mapped-types.md) -- マップ型
- [02-template-literal-types.md](./02-template-literal-types.md) -- テンプレートリテラル型

---

## 参考文献

1. **Type Challenges** -- https://tsch.js.org/
2. **TypeScript Type-Level Programming** -- https://type-level-typescript.com/
3. **Matt Pocock: Total TypeScript** -- https://www.totaltypescript.com/
4. **TypeScript Handbook: Conditional Types** -- https://www.typescriptlang.org/docs/handbook/2/conditional-types.html
5. **TypeScript Handbook: Mapped Types** -- https://www.typescriptlang.org/docs/handbook/2/mapped-types.html
6. **TypeScript Handbook: Template Literal Types** -- https://www.typescriptlang.org/docs/handbook/2/template-literal-types.html
