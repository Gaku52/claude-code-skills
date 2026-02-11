# テンプレートリテラル型

> 文字列パターンを型レベルで表現する。パス型、イベント名、CSSプロパティなどの型安全な文字列操作と型レベルパーサーの実装を学ぶ。

## この章で学ぶこと

1. **基本構文** -- テンプレートリテラル型の構文と文字列Unionの組み合わせ
2. **文字列操作型** -- Uppercase, Lowercase, Capitalize, Uncapitalize の活用
3. **高度なパターン** -- パス型、型レベルパーサー、文字列の分解と再構成

---

## 1. 基本構文

### コード例1: テンプレートリテラル型の基本

```typescript
// 文字列リテラル型の合成
type Greeting = `Hello, ${string}`;

const a: Greeting = "Hello, World";  // OK
const b: Greeting = "Hello, Alice";  // OK
// const c: Greeting = "Hi, World";  // エラー

// Union型との組み合わせ（直積）
type Color = "red" | "green" | "blue";
type Size = "small" | "medium" | "large";

type ColorSize = `${Color}-${Size}`;
// "red-small" | "red-medium" | "red-large" |
// "green-small" | "green-medium" | "green-large" |
// "blue-small" | "blue-medium" | "blue-large"
```

### Union型の直積展開

```
  Color = "red" | "green" | "blue"
  Size  = "small" | "medium" | "large"

  `${Color}-${Size}` の展開:

  "red"   × "small"  → "red-small"
  "red"   × "medium" → "red-medium"
  "red"   × "large"  → "red-large"
  "green" × "small"  → "green-small"
  "green" × "medium" → "green-medium"
  "green" × "large"  → "green-large"
  "blue"  × "small"  → "blue-small"
  "blue"  × "medium" → "blue-medium"
  "blue"  × "large"  → "blue-large"

  結果: 3 × 3 = 9 パターンのUnion
```

### コード例2: イベントハンドラ名の型

```typescript
type DomEvent = "click" | "focus" | "blur" | "change" | "submit";

// on + PascalCase のイベントハンドラ名
type EventHandler = `on${Capitalize<DomEvent>}`;
// "onClick" | "onFocus" | "onBlur" | "onChange" | "onSubmit"

// CSSカスタムプロパティ
type CSSCustomProperty = `--${string}`;

function setCustomProp(name: CSSCustomProperty, value: string) {
  document.documentElement.style.setProperty(name, value);
}

setCustomProp("--primary-color", "#333");   // OK
// setCustomProp("primary-color", "#333");  // エラー: -- で始まらない
```

---

## 2. 文字列操作型

### コード例3: 組み込み文字列操作型

```typescript
// Uppercase: 全て大文字に
type A = Uppercase<"hello">;  // "HELLO"

// Lowercase: 全て小文字に
type B = Lowercase<"HELLO">;  // "hello"

// Capitalize: 先頭を大文字に
type C = Capitalize<"hello">;  // "Hello"

// Uncapitalize: 先頭を小文字に
type D = Uncapitalize<"Hello">;  // "hello"

// 組み合わせ
type CamelToSnake<S extends string> =
  S extends `${infer Head}${infer Tail}`
    ? Tail extends Uncapitalize<Tail>
      ? `${Lowercase<Head>}${CamelToSnake<Tail>}`
      : `${Lowercase<Head>}_${CamelToSnake<Tail>}`
    : S;

type E = CamelToSnake<"camelCaseString">;  // "camel_case_string"
```

### 文字列操作型の一覧

| 型 | 入力 | 出力 | 用途 |
|----|------|------|------|
| `Uppercase<S>` | `"hello"` | `"HELLO"` | HTTP メソッド、定数名 |
| `Lowercase<S>` | `"HELLO"` | `"hello"` | 正規化 |
| `Capitalize<S>` | `"hello"` | `"Hello"` | PascalCase、イベント名 |
| `Uncapitalize<S>` | `"Hello"` | `"hello"` | camelCase 変換 |

---

## 3. パターンマッチングと infer

### コード例4: 文字列の分解

```typescript
// ドット区切りの分解
type Split<S extends string, D extends string> =
  S extends `${infer Head}${D}${infer Tail}`
    ? [Head, ...Split<Tail, D>]
    : [S];

type A = Split<"a.b.c", ".">;  // ["a", "b", "c"]
type B = Split<"hello", ".">;  // ["hello"]

// パスパラメータの抽出
type ExtractParams<T extends string> =
  T extends `${string}:${infer Param}/${infer Rest}`
    ? Param | ExtractParams<Rest>
    : T extends `${string}:${infer Param}`
      ? Param
      : never;

type Params = ExtractParams<"/users/:userId/posts/:postId">;
// "userId" | "postId"

// パラメータからオブジェクト型を生成
type RouteParams<T extends string> = {
  [K in ExtractParams<T>]: string;
};

type UserPostParams = RouteParams<"/users/:userId/posts/:postId">;
// { userId: string; postId: string }
```

### パス型のパターンマッチング

```
  入力: "/users/:userId/posts/:postId"

  ステップ1: `${string}:${infer P}/${infer R}`
    P = "userId"
    R = "posts/:postId"

  ステップ2: ExtractParams<"posts/:postId">
    `${string}:${infer P}`
    P = "postId"

  結果: "userId" | "postId"
```

### コード例5: 型安全なルーター

```typescript
// ルート定義
type Route = "/users" | "/users/:id" | "/posts/:postId/comments/:commentId";

// ルートからパラメータ型を自動推論
type ExtractRouteParams<T extends string> =
  T extends `${string}:${infer Param}/${infer Rest}`
    ? { [K in Param | keyof ExtractRouteParams<Rest>]: string }
    : T extends `${string}:${infer Param}`
      ? { [K in Param]: string }
      : {};

// 型安全なナビゲーション関数
function navigate<T extends Route>(
  route: T,
  ...args: keyof ExtractRouteParams<T> extends never
    ? []
    : [params: ExtractRouteParams<T>]
): void {
  // 実装省略
}

navigate("/users");                                           // OK: パラメータ不要
navigate("/users/:id", { id: "123" });                        // OK
navigate("/posts/:postId/comments/:commentId", {
  postId: "1",
  commentId: "42",
});                                                           // OK
```

---

## 4. 高度なテンプレートリテラル型

### コード例6: CSSプロパティの型安全化

```typescript
// CSS値の型
type CSSUnit = "px" | "em" | "rem" | "%" | "vh" | "vw";
type CSSValue = `${number}${CSSUnit}` | "auto" | "0";

function setWidth(width: CSSValue): void {
  // 実装
}

setWidth("100px");   // OK
setWidth("50%");     // OK
setWidth("auto");    // OK
// setWidth("100");  // エラー: 単位がない
// setWidth("abc");  // エラー

// CSS色の型
type HexDigit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
  | "a" | "b" | "c" | "d" | "e" | "f"
  | "A" | "B" | "C" | "D" | "E" | "F";

type HexColor = `#${string}`;  // 簡略版
```

### コード例7: 型レベルのJSON パーサー（概念）

```typescript
// 数字文字列を数値に変換（概念的な例）
type StringToNumber<S extends string> =
  S extends `${infer N extends number}` ? N : never;

type A = StringToNumber<"42">;  // 42
type B = StringToNumber<"0">;   // 0

// 文字列の長さを型レベルで計算
type StringLength<
  S extends string,
  Acc extends unknown[] = []
> = S extends `${string}${infer Rest}`
  ? StringLength<Rest, [...Acc, unknown]>
  : Acc["length"];

type L1 = StringLength<"hello">;  // 5
type L2 = StringLength<"">;       // 0

// 文字列の結合
type Join<T extends string[], D extends string> =
  T extends []
    ? ""
    : T extends [infer F extends string]
      ? F
      : T extends [infer F extends string, ...infer R extends string[]]
        ? `${F}${D}${Join<R, D>}`
        : never;

type Joined = Join<["a", "b", "c"], ".">;  // "a.b.c"
```

### テンプレートリテラル型の展開ルール

```
  `${A | B}${C | D}`
  展開:
    → `${A}${C}` | `${A}${D}` | `${B}${C}` | `${B}${D}`

  `${"get" | "set"}${"Name" | "Age"}`
  展開:
    → "getName" | "getAge" | "setName" | "setAge"

  注意: 組み合わせ数が爆発する場合あり
  |A| × |B| = 結果のUnionメンバー数
```

---

## テンプレートリテラル型 vs 通常の文字列型

| 特性 | string | リテラル型 | テンプレートリテラル型 |
|------|--------|-----------|---------------------|
| 範囲 | 任意の文字列 | 特定の値のみ | パターンに一致する文字列 |
| 例 | `string` | `"hello"` | `` `hello-${string}` `` |
| 型安全性 | 低 | 高（固定値） | 中〜高（パターン） |
| 用途 | 汎用 | 定数 | パターン化された文字列 |
| Union展開 | なし | なし | 自動的に直積展開 |

---

## アンチパターン

### アンチパターン1: Union爆発

```typescript
// BAD: 組み合わせが爆発する
type Letter = "a" | "b" | "c" | ... | "z";  // 26個
type TwoLetters = `${Letter}${Letter}`;       // 26 × 26 = 676個
type ThreeLetters = `${Letter}${Letter}${Letter}`; // 17,576個！
// → コンパイルが極端に遅くなる

// GOOD: パターンで妥協する
type ThreeLetters = `${string}${string}${string}`; // 広いが高速
// または正規表現ベースのランタイムバリデーションと併用
```

### アンチパターン2: 実行時検証の代わりにテンプレートリテラル型を使う

```typescript
// BAD: 型だけでメールアドレスを検証しようとする
type Email = `${string}@${string}.${string}`;
const email: Email = "not-an-email@"; // これは型エラーにならない場合がある

// GOOD: 型は大まかなパターンに留め、実行時バリデーションを併用
import { z } from "zod";
const emailSchema = z.string().email();
type Email = `${string}@${string}.${string}`; // 型レベルのヒント
```

---

## FAQ

### Q1: テンプレートリテラル型のパフォーマンスへの影響は？

**A:** Union型の直積展開により、組み合わせ数が急速に増加します。TypeScriptは内部的に最大100,000程度のUnionメンバーを処理できますが、数千を超えるとコンパイルが遅くなります。大きなUnionの直積は避けてください。

### Q2: テンプレートリテラル型でパスの型安全性を確保する実践的な方法は？

**A:** ルーティングライブラリ（React Router, tRPC など）の多くがテンプレートリテラル型を活用した型安全なルーティングを提供しています。自前で実装するよりも、既存ライブラリの型定義を活用するのが実用的です。

### Q3: `${number}` はどのような文字列にマッチしますか？

**A:** `"0"`, `"42"`, `"3.14"`, `"-1"` など、数値のリテラル表現にマッチします。ただし `"1e10"` のような科学表記にもマッチする場合があります。厳密な数値文字列の検証にはランタイムチェックを併用してください。

---

## まとめ

| 項目 | 内容 |
|------|------|
| 基本構文 | `` `prefix-${Type}` `` で文字列パターンを型に |
| Union展開 | `${A \| B}` は自動的に直積展開される |
| 文字列操作 | Uppercase, Lowercase, Capitalize, Uncapitalize |
| infer | テンプレートリテラル型の中でパターンマッチ可能 |
| パス型 | URLパラメータの型安全な抽出に有用 |
| 注意点 | Union爆発によるコンパイル速度低下に注意 |

---

## 次に読むべきガイド

- [03-type-challenges.md](./03-type-challenges.md) -- 型チャレンジ
- [04-declaration-files.md](./04-declaration-files.md) -- 宣言ファイル

---

## 参考文献

1. **TypeScript Handbook: Template Literal Types** -- https://www.typescriptlang.org/docs/handbook/2/template-literal-types.html
2. **TypeScript 4.1 Release Notes** -- https://www.typescriptlang.org/docs/handbook/release-notes/typescript-4-1.html
3. **Matt Pocock: Template Literal Types** -- https://www.totaltypescript.com/books/total-typescript-essentials/template-literal-types
