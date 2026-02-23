# テンプレートリテラル型

> 文字列パターンを型レベルで表現する。パス型、イベント名、CSSプロパティなどの型安全な文字列操作と型レベルパーサーの実装を学ぶ。

## この章で学ぶこと

1. **基本構文** -- テンプレートリテラル型の構文と文字列Unionの組み合わせ
2. **文字列操作型** -- Uppercase, Lowercase, Capitalize, Uncapitalize の活用
3. **パターンマッチングと infer** -- 文字列の分解と型の抽出
4. **高度なパターン** -- パス型、型レベルパーサー、文字列の分解と再構成
5. **実務パターン** -- CSSプロパティ、ルーティング、i18n、SQL型安全化
6. **パフォーマンスとベストプラクティス** -- Union爆発の回避と効率的な型設計

---

## 1. 基本構文

### 1.1 テンプレートリテラル型の基本

テンプレートリテラル型は、TypeScript 4.1 で導入された機能で、文字列リテラル型をテンプレートとして合成できる。JavaScript のテンプレートリテラル構文をそのまま型レベルに持ち込んだもの。

```typescript
// 文字列リテラル型の合成
type Greeting = `Hello, ${string}`;

const a: Greeting = "Hello, World";  // OK
const b: Greeting = "Hello, Alice";  // OK
// const c: Greeting = "Hi, World";  // エラー: "Hi, World" は `Hello, ${string}` に代入不可

// string 以外のプリミティブ型も埋め込み可能
type NumberString = `value-${number}`;
const d: NumberString = "value-42";    // OK
const e: NumberString = "value-3.14";  // OK
// const f: NumberString = "value-abc"; // エラー

type BooleanString = `is-${boolean}`;
const g: BooleanString = "is-true";   // OK
const h: BooleanString = "is-false";  // OK
// const i: BooleanString = "is-yes"; // エラー

// bigint も使用可能
type BigIntString = `big-${bigint}`;
const j: BigIntString = "big-12345678901234567890"; // OK

// null と undefined も使用可能（ただし実用性は低い）
type NullString = `value-${null}`;       // "value-null"
type UndefString = `value-${undefined}`; // "value-undefined"
```

### 1.2 Union型との組み合わせ（直積展開）

```typescript
// Union型との組み合わせ（直積）
type Color = "red" | "green" | "blue";
type Size = "small" | "medium" | "large";

type ColorSize = `${Color}-${Size}`;
// "red-small" | "red-medium" | "red-large" |
// "green-small" | "green-medium" | "green-large" |
// "blue-small" | "blue-medium" | "blue-large"

// 3つのUnionの直積
type Prefix = "btn" | "link";
type Variant = "primary" | "secondary";
type State = "active" | "disabled";

type ClassName = `${Prefix}-${Variant}-${State}`;
// 2 × 2 × 2 = 8 パターン
// "btn-primary-active" | "btn-primary-disabled" | "btn-secondary-active" | ...
// "link-primary-active" | "link-primary-disabled" | "link-secondary-active" | ...

// 数値リテラルとの組み合わせ
type Port = 80 | 443 | 8080 | 3000;
type Protocol = "http" | "https";
type URL = `${Protocol}://localhost:${Port}`;
// "http://localhost:80" | "http://localhost:443" | ... | "https://localhost:3000"
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

  3つのUnionの場合:
  |A| × |B| × |C| = 結果のUnionメンバー数
  例: 10 × 10 × 10 = 1,000 パターン → 注意が必要
```

### 1.3 イベントハンドラ名の型

```typescript
type DomEvent = "click" | "focus" | "blur" | "change" | "submit" |
  "mouseenter" | "mouseleave" | "keydown" | "keyup" | "scroll" |
  "resize" | "load" | "error" | "input" | "drag" | "drop";

// on + PascalCase のイベントハンドラ名
type EventHandler = `on${Capitalize<DomEvent>}`;
// "onClick" | "onFocus" | "onBlur" | "onChange" | "onSubmit" |
// "onMouseenter" | "onMouseleave" | "onKeydown" | "onKeyup" | ...

// data- 属性の型
type DataAttribute = `data-${string}`;

function setAttribute(element: HTMLElement, attr: DataAttribute, value: string): void {
  element.setAttribute(attr, value);
}

setAttribute(document.body, "data-theme", "dark");     // OK
setAttribute(document.body, "data-user-id", "123");    // OK
// setAttribute(document.body, "class", "main");        // エラー

// CSSカスタムプロパティ
type CSSCustomProperty = `--${string}`;

function setCustomProp(name: CSSCustomProperty, value: string): void {
  document.documentElement.style.setProperty(name, value);
}

setCustomProp("--primary-color", "#333");   // OK
setCustomProp("--font-size", "16px");       // OK
// setCustomProp("primary-color", "#333");  // エラー: -- で始まらない

// ARIA属性の型
type AriaAttribute = `aria-${string}`;
type AriaRole = "button" | "dialog" | "alert" | "navigation" | "main" | "complementary";

function setAria(element: HTMLElement, attr: AriaAttribute, value: string): void {
  element.setAttribute(attr, value);
}
```

### 1.4 名前空間付きの型定義

```typescript
// 名前空間付きイベント名（jQuery風）
type Namespace = "app" | "user" | "ui" | "data";
type BaseEvent = "init" | "update" | "destroy" | "error";

type NamespacedEvent = `${Namespace}:${BaseEvent}`;
// "app:init" | "app:update" | "app:destroy" | "app:error" |
// "user:init" | "user:update" | ... (4 × 4 = 16パターン)

// ワイルドカードを含むパターン
type EventPattern = NamespacedEvent | `${Namespace}:*` | "*";

// Redis のキーパターン
type CachePrefix = "user" | "post" | "session" | "config";
type CacheKey = `${CachePrefix}:${string}`;

function getCache(key: CacheKey): Promise<string | null> {
  // 実装...
  return Promise.resolve(null);
}

getCache("user:123");         // OK
getCache("session:abc-def");  // OK
// getCache("invalid");       // エラー

// 環境変数のプレフィックス
type EnvPrefix = "NEXT_PUBLIC" | "VITE" | "REACT_APP";
type PublicEnvVar = `${EnvPrefix}_${string}`;

function getPublicEnv(key: PublicEnvVar): string | undefined {
  return process.env[key];
}
```

---

## 2. 文字列操作型

### 2.1 組み込み文字列操作型

TypeScript には4つの組み込み文字列操作型がある。これらは条件型とは異なり、コンパイラに組み込まれた特殊な型。

```typescript
// Uppercase<S>: 全て大文字に
type A = Uppercase<"hello">;      // "HELLO"
type A2 = Uppercase<"Hello">;     // "HELLO"
type A3 = Uppercase<"HELLO">;     // "HELLO"（既に大文字）

// Lowercase<S>: 全て小文字に
type B = Lowercase<"HELLO">;      // "hello"
type B2 = Lowercase<"Hello">;     // "hello"
type B3 = Lowercase<"hello">;     // "hello"（既に小文字）

// Capitalize<S>: 先頭を大文字に
type C = Capitalize<"hello">;     // "Hello"
type C2 = Capitalize<"Hello">;    // "Hello"（既にCapitalize済み）
type C3 = Capitalize<"hELLO">;    // "HELLO"（先頭のみ変更、残りはそのまま）

// Uncapitalize<S>: 先頭を小文字に
type D = Uncapitalize<"Hello">;   // "hello"
type D2 = Uncapitalize<"hello">;  // "hello"（既にuncapitalize済み）
type D3 = Uncapitalize<"HELLO">;  // "hELLO"（先頭のみ変更）

// Union型との組み合わせ（各メンバーに個別適用）
type Events = "click" | "focus" | "blur";
type PascalEvents = Capitalize<Events>;  // "Click" | "Focus" | "Blur"
type UpperEvents = Uppercase<Events>;    // "CLICK" | "FOCUS" | "BLUR"
type LowerEvents = Lowercase<Uppercase<Events>>;  // "click" | "focus" | "blur"
```

### 2.2 ケース変換の実装

```typescript
// キャメルケース → スネークケース
type CamelToSnake<S extends string> =
  S extends `${infer Head}${infer Tail}`
    ? Tail extends Uncapitalize<Tail>
      ? `${Lowercase<Head>}${CamelToSnake<Tail>}`
      : `${Lowercase<Head>}_${CamelToSnake<Tail>}`
    : S;

type Snake1 = CamelToSnake<"camelCaseString">;    // "camel_case_string"
type Snake2 = CamelToSnake<"getElementById">;      // "get_element_by_id"
type Snake3 = CamelToSnake<"XMLParser">;           // "x_m_l_parser"（注意: 連続大文字は個別に変換）

// スネークケース → キャメルケース
type SnakeToCamel<S extends string> =
  S extends `${infer Head}_${infer Tail}`
    ? `${Lowercase<Head>}${Capitalize<SnakeToCamel<Tail>>}`
    : Lowercase<S>;

type Camel1 = SnakeToCamel<"snake_case_string">;  // "snakeCaseString"
type Camel2 = SnakeToCamel<"user_id">;             // "userId"
type Camel3 = SnakeToCamel<"created_at">;           // "createdAt"

// スネークケース → パスカルケース
type SnakeToPascal<S extends string> =
  S extends `${infer Head}_${infer Tail}`
    ? `${Capitalize<Lowercase<Head>>}${SnakeToPascal<Tail>}`
    : Capitalize<Lowercase<S>>;

type Pascal1 = SnakeToPascal<"user_profile">;      // "UserProfile"
type Pascal2 = SnakeToPascal<"api_response_data">;  // "ApiResponseData"

// ケバブケース → キャメルケース
type KebabToCamel<S extends string> =
  S extends `${infer Head}-${infer Tail}`
    ? `${Lowercase<Head>}${Capitalize<KebabToCamel<Tail>>}`
    : Lowercase<S>;

type Kebab1 = KebabToCamel<"kebab-case-string">;  // "kebabCaseString"
type Kebab2 = KebabToCamel<"font-size">;            // "fontSize"
type Kebab3 = KebabToCamel<"border-top-width">;     // "borderTopWidth"

// キャメルケース → ケバブケース
type CamelToKebab<S extends string> =
  S extends `${infer Head}${infer Tail}`
    ? Tail extends Uncapitalize<Tail>
      ? `${Lowercase<Head>}${CamelToKebab<Tail>}`
      : `${Lowercase<Head>}-${CamelToKebab<Tail>}`
    : S;

type KebabR1 = CamelToKebab<"fontSize">;           // "font-size"
type KebabR2 = CamelToKebab<"borderTopWidth">;      // "border-top-width"
type KebabR3 = CamelToKebab<"backgroundColor">;     // "background-color"
```

### 2.3 文字列の結合と分割

```typescript
// 文字列の結合（Join）
type Join<T extends string[], D extends string> =
  T extends []
    ? ""
    : T extends [infer F extends string]
      ? F
      : T extends [infer F extends string, ...infer R extends string[]]
        ? `${F}${D}${Join<R, D>}`
        : never;

type Joined1 = Join<["a", "b", "c"], ".">;    // "a.b.c"
type Joined2 = Join<["hello", "world"], " ">;  // "hello world"
type Joined3 = Join<["one"], ",">;             // "one"
type Joined4 = Join<[], ",">;                  // ""

// 文字列の分割（Split）
type Split<S extends string, D extends string> =
  S extends `${infer Head}${D}${infer Tail}`
    ? [Head, ...Split<Tail, D>]
    : [S];

type Splitted1 = Split<"a.b.c", ".">;       // ["a", "b", "c"]
type Splitted2 = Split<"hello", ".">;        // ["hello"]
type Splitted3 = Split<"a-b-c-d", "-">;      // ["a", "b", "c", "d"]
type Splitted4 = Split<"hello world", " ">;  // ["hello", "world"]

// 文字列の置換（Replace）
type Replace<
  S extends string,
  From extends string,
  To extends string
> = S extends `${infer Before}${From}${infer After}`
  ? `${Before}${To}${After}`
  : S;

type Replaced1 = Replace<"hello world", "world", "TypeScript">;
// "hello TypeScript"

// 全置換（ReplaceAll）
type ReplaceAll<
  S extends string,
  From extends string,
  To extends string
> = S extends `${infer Before}${From}${infer After}`
  ? ReplaceAll<`${Before}${To}${After}`, From, To>
  : S;

type ReplacedAll1 = ReplaceAll<"a-b-c-d", "-", ".">;  // "a.b.c.d"
type ReplacedAll2 = ReplaceAll<"aaa", "a", "bb">;      // "bbbbbb"

// トリム（前後の空白を除去）
type TrimLeft<S extends string> =
  S extends ` ${infer Rest}` | `\n${infer Rest}` | `\t${infer Rest}`
    ? TrimLeft<Rest>
    : S;

type TrimRight<S extends string> =
  S extends `${infer Rest} ` | `${infer Rest}\n` | `${infer Rest}\t`
    ? TrimRight<Rest>
    : S;

type Trim<S extends string> = TrimLeft<TrimRight<S>>;

type Trimmed = Trim<"  hello  ">;  // "hello"
```

### 文字列操作型の一覧

| 型 | 入力 | 出力 | 用途 |
|----|------|------|------|
| `Uppercase<S>` | `"hello"` | `"HELLO"` | HTTP メソッド、定数名 |
| `Lowercase<S>` | `"HELLO"` | `"hello"` | 正規化 |
| `Capitalize<S>` | `"hello"` | `"Hello"` | PascalCase、イベント名 |
| `Uncapitalize<S>` | `"Hello"` | `"hello"` | camelCase 変換 |

### カスタム文字列操作型の一覧

| 型 | 入力 | 出力 | 用途 |
|----|------|------|------|
| `CamelToSnake<S>` | `"camelCase"` | `"camel_case"` | API通信 |
| `SnakeToCamel<S>` | `"snake_case"` | `"snakeCase"` | API通信 |
| `KebabToCamel<S>` | `"kebab-case"` | `"kebabCase"` | CSS→JS |
| `CamelToKebab<S>` | `"camelCase"` | `"camel-case"` | JS→CSS |
| `Split<S, D>` | `"a.b.c", "."` | `["a","b","c"]` | パス解析 |
| `Join<T, D>` | `["a","b"], "."` | `"a.b"` | パス構築 |
| `Replace<S, F, T>` | `"ab", "a", "x"` | `"xb"` | 文字列変換 |
| `Trim<S>` | `" abc "` | `"abc"` | 入力正規化 |

---

## 3. パターンマッチングと infer

### 3.1 文字列の分解

テンプレートリテラル型の中で `infer` を使うと、文字列をパターンマッチングして部分文字列を抽出できる。

```typescript
// 先頭文字の取得
type FirstChar<S extends string> =
  S extends `${infer F}${string}` ? F : never;

type FC1 = FirstChar<"hello">;  // "h"
type FC2 = FirstChar<"A">;      // "A"
type FC3 = FirstChar<"">;       // never

// 残りの文字列の取得
type RestChars<S extends string> =
  S extends `${string}${infer R}` ? R : never;

type RC1 = RestChars<"hello">;  // "ello"
type RC2 = RestChars<"A">;      // ""
type RC3 = RestChars<"">;       // never

// 最後の文字の取得
type LastChar<S extends string> =
  S extends `${infer Rest}${infer Last}`
    ? Last extends ""
      ? Rest
      : LastChar<`${Last}`> extends never
        ? Last
        : LastChar<`${Rest}${LastChar<Last>}`>
    : never;

// 文字列の反転
type Reverse<S extends string> =
  S extends `${infer First}${infer Rest}`
    ? `${Reverse<Rest>}${First}`
    : "";

type Rev = Reverse<"hello">;  // "olleh"

// 文字列にある文字が含まれるか
type Includes<S extends string, Search extends string> =
  S extends `${string}${Search}${string}` ? true : false;

type Inc1 = Includes<"hello world", "world">;  // true
type Inc2 = Includes<"hello world", "xyz">;    // false
type Inc3 = Includes<"typescript", "script">;  // true

// 文字列が特定の文字列で始まるか
type StartsWith<S extends string, Prefix extends string> =
  S extends `${Prefix}${string}` ? true : false;

type SW1 = StartsWith<"hello world", "hello">;  // true
type SW2 = StartsWith<"hello world", "world">;  // false

// 文字列が特定の文字列で終わるか
type EndsWith<S extends string, Suffix extends string> =
  S extends `${string}${Suffix}` ? true : false;

type EW1 = EndsWith<"hello world", "world">;  // true
type EW2 = EndsWith<"hello world", "hello">;  // false
```

### 3.2 パスパラメータの抽出

```typescript
// URLパスパラメータの抽出（基本版）
type ExtractParams<T extends string> =
  T extends `${string}:${infer Param}/${infer Rest}`
    ? Param | ExtractParams<Rest>
    : T extends `${string}:${infer Param}`
      ? Param
      : never;

type Params1 = ExtractParams<"/users/:userId/posts/:postId">;
// "userId" | "postId"

type Params2 = ExtractParams<"/api/v1/:version/users/:id/profile">;
// "version" | "id"

type Params3 = ExtractParams<"/static/index.html">;
// never（パラメータなし）

// パラメータからオブジェクト型を生成
type RouteParams<T extends string> = {
  [K in ExtractParams<T>]: string;
};

type UserPostParams = RouteParams<"/users/:userId/posts/:postId">;
// { userId: string; postId: string }

// 型付きパラメータ（数値IDの場合）
type TypedRouteParams<T extends string> = {
  [K in ExtractParams<T>]: K extends `${string}Id` ? number : string;
};

type TypedParams = TypedRouteParams<"/users/:userId/posts/:postId">;
// { userId: number; postId: number }（Idで終わるパラメータは number）
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

  RouteParams による変換:
    "userId" | "postId"
    → { userId: string; postId: string }
```

### 3.3 型安全なルーター

```typescript
// ルート定義
type Routes = {
  "/": {};
  "/users": {};
  "/users/:id": { id: string };
  "/users/:id/posts": { id: string };
  "/users/:userId/posts/:postId": { userId: string; postId: string };
  "/settings": {};
  "/settings/:section": { section: string };
};

// ルートパスからパラメータ型を自動推論する版
type ExtractRouteParams<T extends string> =
  T extends `${string}:${infer Param}/${infer Rest}`
    ? { [K in Param | keyof ExtractRouteParamsHelper<Rest>]: string }
    : T extends `${string}:${infer Param}`
      ? { [K in Param]: string }
      : {};

type ExtractRouteParamsHelper<T extends string> =
  T extends `${string}:${infer Param}/${infer Rest}`
    ? { [K in Param]: string } & ExtractRouteParamsHelper<Rest>
    : T extends `${string}:${infer Param}`
      ? { [K in Param]: string }
      : {};

// 型安全なナビゲーション関数
type Route = "/users" | "/users/:id" | "/posts/:postId/comments/:commentId" |
  "/settings" | "/settings/:section";

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
// navigate("/users/:id");                                    // エラー: パラメータが必要
// navigate("/users/:id", { id: "123", extra: "x" });         // エラー: 余分なパラメータ

// パスの構築（パラメータを実際の値に置換）
type BuildPath<
  Path extends string,
  Params extends Record<string, string>
> = Path extends `${infer Before}:${infer Param}/${infer After}`
  ? Param extends keyof Params
    ? BuildPath<`${Before}${Params[Param]}/${After}`, Params>
    : never
  : Path extends `${infer Before}:${infer Param}`
    ? Param extends keyof Params
      ? `${Before}${Params[Param]}`
      : never
    : Path;

type Built = BuildPath<"/users/:id/posts/:postId", { id: "123"; postId: "456" }>;
// "/users/123/posts/456"
```

### 3.4 クエリパラメータの型安全な解析

```typescript
// クエリ文字列の解析
type ParseQueryString<S extends string> =
  S extends `${infer Key}=${infer Value}&${infer Rest}`
    ? { [K in Key]: Value } & ParseQueryString<Rest>
    : S extends `${infer Key}=${infer Value}`
      ? { [K in Key]: Value }
      : {};

type Query1 = ParseQueryString<"page=1&limit=10&sort=name">;
// { page: "1" } & { limit: "10" } & { sort: "name" }

type Query2 = ParseQueryString<"q=typescript&lang=ja">;
// { q: "typescript" } & { lang: "ja" }

// URL全体の解析
type ParseURL<S extends string> =
  S extends `${infer Protocol}://${infer Host}/${infer Path}?${infer Query}`
    ? {
        protocol: Protocol;
        host: Host;
        path: `/${Path}`;
        query: ParseQueryString<Query>;
      }
    : S extends `${infer Protocol}://${infer Host}/${infer Path}`
      ? {
          protocol: Protocol;
          host: Host;
          path: `/${Path}`;
          query: {};
        }
      : S extends `${infer Protocol}://${infer Host}`
        ? {
            protocol: Protocol;
            host: Host;
            path: "/";
            query: {};
          }
        : never;

type URLParsed = ParseURL<"https://api.example.com/users?page=1&limit=10">;
// {
//   protocol: "https";
//   host: "api.example.com";
//   path: "/users";
//   query: { page: "1" } & { limit: "10" };
// }
```

---

## 4. 高度なテンプレートリテラル型

### 4.1 CSSプロパティの型安全化

```typescript
// CSS値の型
type CSSUnit = "px" | "em" | "rem" | "%" | "vh" | "vw" | "vmin" | "vmax" | "ch" | "ex";
type CSSValue = `${number}${CSSUnit}` | "auto" | "0" | "inherit" | "initial" | "unset";

function setWidth(width: CSSValue): void {
  // 実装
}

setWidth("100px");    // OK
setWidth("50%");      // OK
setWidth("auto");     // OK
setWidth("2.5rem");   // OK
// setWidth("100");   // エラー: 単位がない
// setWidth("abc");   // エラー

// CSS色の型（16進数カラー）
type HexDigit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
  | "a" | "b" | "c" | "d" | "e" | "f"
  | "A" | "B" | "C" | "D" | "E" | "F";

type HexColor = `#${string}`;  // 簡略版（厳密には6桁 or 8桁チェックが必要）

// CSS関数値の型
type CSSFunction =
  | `rgb(${number}, ${number}, ${number})`
  | `rgba(${number}, ${number}, ${number}, ${number})`
  | `hsl(${number}, ${number}%, ${number}%)`
  | `calc(${string})`
  | `var(${CSSCustomProperty})`;

// CSS変換の型
type CSSTransform =
  | `translateX(${CSSValue})`
  | `translateY(${CSSValue})`
  | `translate(${CSSValue}, ${CSSValue})`
  | `rotate(${number}deg)`
  | `scale(${number})`
  | `scale(${number}, ${number})`
  | `skewX(${number}deg)`
  | `skewY(${number}deg)`;

// CSS グリッド / フレックスボックスの型
type FlexDirection = "row" | "column" | "row-reverse" | "column-reverse";
type FlexWrap = "nowrap" | "wrap" | "wrap-reverse";
type FlexFlow = `${FlexDirection} ${FlexWrap}`;

type GridTemplate = `repeat(${number}, ${CSSValue | "1fr" | "auto"})` | string;

// CSSプロパティからReactスタイルオブジェクトへ
type CSSPropertyName =
  | "background-color" | "font-size" | "font-weight" | "font-family"
  | "border-radius" | "border-width" | "border-color" | "border-style"
  | "margin-top" | "margin-right" | "margin-bottom" | "margin-left"
  | "padding-top" | "padding-right" | "padding-bottom" | "padding-left"
  | "line-height" | "letter-spacing" | "text-align" | "text-decoration"
  | "box-shadow" | "text-shadow" | "z-index" | "opacity";

type CSSToReact<S extends string> =
  S extends `${infer Head}-${infer Tail}`
    ? `${Head}${Capitalize<CSSToReact<Tail>>}`
    : S;

type ReactProperty = CSSToReact<"background-color">;  // "backgroundColor"
type ReactProperty2 = CSSToReact<"border-top-width">; // "borderTopWidth"
```

### 4.2 SQL クエリの型安全化

```typescript
// SELECT 句からカラム名を抽出
type ExtractColumns<S extends string> =
  S extends `${infer Col}, ${infer Rest}`
    ? Trim<Col> | ExtractColumns<Rest>
    : Trim<S>;

// テーブル名を抽出
type ExtractTable<S extends string> =
  S extends `${string} FROM ${infer Table} ${string}`
    ? Trim<Table>
    : S extends `${string} FROM ${infer Table}`
      ? Trim<Table>
      : never;

// SELECT クエリの型解析
type ParseSelect<S extends string> =
  S extends `SELECT ${infer Columns} FROM ${infer Rest}`
    ? {
        columns: ExtractColumns<Columns>;
        table: Rest extends `${infer Table} WHERE ${string}`
          ? Trim<Table>
          : Trim<Rest>;
      }
    : never;

type Parsed = ParseSelect<"SELECT name, email, age FROM users WHERE id = 1">;
// {
//   columns: "name" | "email" | "age";
//   table: "users";
// }

// 型安全な SQL ビルダー（概念的な実装）
type TableName = "users" | "posts" | "comments";

type TableSchema = {
  users: {
    id: number;
    name: string;
    email: string;
    createdAt: Date;
  };
  posts: {
    id: number;
    title: string;
    content: string;
    authorId: number;
    published: boolean;
  };
  comments: {
    id: number;
    text: string;
    postId: number;
    authorId: number;
  };
};

// SELECT できるカラムを型で制約
type ValidColumn<T extends TableName> = keyof TableSchema[T] & string;

type SelectResult<T extends TableName, Cols extends ValidColumn<T>> =
  Pick<TableSchema[T], Cols>;

// 使用イメージ
function select<T extends TableName, C extends ValidColumn<T>>(
  table: T,
  columns: C[]
): Promise<SelectResult<T, C>[]> {
  // 実装...
  return Promise.resolve([]);
}

// 型安全に使用
const result = await select("users", ["name", "email"]);
// result は Pick<TableSchema["users"], "name" | "email">[]
// = { name: string; email: string }[]
```

### 4.3 型レベルのJSONパーサー

```typescript
// 数字文字列を数値に変換
type StringToNumber<S extends string> =
  S extends `${infer N extends number}` ? N : never;

type N1 = StringToNumber<"42">;    // 42
type N2 = StringToNumber<"0">;     // 0
type N3 = StringToNumber<"3.14">;  // 3.14

// ブーリアン文字列を boolean に変換
type StringToBoolean<S extends string> =
  S extends "true" ? true :
  S extends "false" ? false :
  never;

// 文字列の長さを型レベルで計算
type StringLength<
  S extends string,
  Acc extends unknown[] = []
> = S extends `${string}${infer Rest}`
  ? StringLength<Rest, [...Acc, unknown]>
  : Acc["length"];

type L1 = StringLength<"hello">;      // 5
type L2 = StringLength<"">;           // 0
type L3 = StringLength<"TypeScript">; // 10

// 文字列の繰り返し
type Repeat<
  S extends string,
  N extends number,
  Counter extends any[] = [],
  Result extends string = ""
> = Counter["length"] extends N
  ? Result
  : Repeat<S, N, [...Counter, 0], `${Result}${S}`>;

type Rep1 = Repeat<"ab", 3>;    // "ababab"
type Rep2 = Repeat<"-", 5>;     // "-----"
type Rep3 = Repeat<"ha", 2>;    // "haha"

// パディング
type PadStart<
  S extends string,
  Length extends number,
  Pad extends string = " "
> = StringLength<S> extends Length
  ? S
  : PadStart<`${Pad}${S}`, Length, Pad>;

type Padded = PadStart<"42", 5, "0">;  // "00042"
```

### 4.4 型安全なテンプレートエンジン

```typescript
// テンプレート文字列からプレースホルダーを抽出
type ExtractPlaceholders<T extends string> =
  T extends `${string}{{${infer Name}}}${infer Rest}`
    ? Name | ExtractPlaceholders<Rest>
    : never;

type Placeholders = ExtractPlaceholders<"Hello, {{name}}! You have {{count}} messages.">;
// "name" | "count"

// テンプレートのデータ型を自動生成
type TemplateData<T extends string> = {
  [K in ExtractPlaceholders<T>]: string | number;
};

type Data = TemplateData<"Hello, {{name}}! You have {{count}} messages.">;
// { name: string | number; count: string | number }

// 型安全なテンプレートレンダリング関数
function render<T extends string>(
  template: T,
  data: TemplateData<T>
): string {
  let result: string = template;
  for (const [key, value] of Object.entries(data)) {
    result = result.replace(new RegExp(`\\{\\{${key}\\}\\}`, "g"), String(value));
  }
  return result;
}

// 使用例
const message = render(
  "Hello, {{name}}! You have {{count}} new messages.",
  { name: "Alice", count: 5 }  // name と count が必須
);

// render(
//   "Hello, {{name}}!",
//   { name: "Alice", count: 5 }  // エラー: count は不要
// );

// render(
//   "Hello, {{name}}! {{greeting}}",
//   { name: "Alice" }  // エラー: greeting が不足
// );

// Mustache風のセクション解析
type ExtractSections<T extends string> =
  T extends `${string}{{#${infer Section}}}${infer Content}{{/${infer _End}}}${infer Rest}`
    ? { section: Section; content: Content } | ExtractSections<Rest>
    : never;

type Sections = ExtractSections<"{{#items}}{{name}}: {{price}}{{/items}}{{#footer}}{{text}}{{/footer}}">;
// { section: "items"; content: "{{name}}: {{price}}" } | { section: "footer"; content: "{{text}}" }
```

### 4.5 型安全な正規表現風パターン

```typescript
// 簡易的な正規表現パターンマッチング型

// メールアドレスのパターン（簡易版）
type EmailPattern = `${string}@${string}.${string}`;

// IPv4アドレスのパターン（簡易版）
type IPv4Pattern = `${number}.${number}.${number}.${number}`;

// セマンティックバージョニング
type SemVer = `${number}.${number}.${number}`;
type SemVerWithPre = `${number}.${number}.${number}-${string}`;
type SemVerFull = SemVer | SemVerWithPre;

// セマンティックバージョンの解析
type ParseSemVer<S extends string> =
  S extends `${infer Major extends number}.${infer Minor extends number}.${infer Patch extends number}-${infer Pre}`
    ? { major: Major; minor: Minor; patch: Patch; prerelease: Pre }
    : S extends `${infer Major extends number}.${infer Minor extends number}.${infer Patch extends number}`
      ? { major: Major; minor: Minor; patch: Patch; prerelease: never }
      : never;

type Version = ParseSemVer<"1.2.3">;
// { major: 1; minor: 2; patch: 3; prerelease: never }

type VersionPre = ParseSemVer<"2.0.0-beta.1">;
// { major: 2; minor: 0; patch: 0; prerelease: "beta.1" }

// 日付パターン
type DatePattern = `${number}-${number}-${number}`;
type TimePattern = `${number}:${number}:${number}`;
type DateTimePattern = `${DatePattern}T${TimePattern}`;

// ISO 8601 日付の解析
type ParseDate<S extends string> =
  S extends `${infer Y extends number}-${infer M extends number}-${infer D extends number}`
    ? { year: Y; month: M; day: D }
    : never;

type ParsedDate = ParseDate<"2024-12-25">;
// { year: 2024; month: 12; day: 25 }

// UUID パターン
type UUIDSegment = `${string}`;
type UUIDPattern = `${UUIDSegment}-${UUIDSegment}-${UUIDSegment}-${UUIDSegment}-${UUIDSegment}`;
```

---

## 5. 実務パターン

### 5.1 国際化（i18n）の型安全化

```typescript
// 翻訳キーの型安全な管理
interface TranslationSchema {
  common: {
    save: "保存";
    cancel: "キャンセル";
    delete: "削除";
    confirm: "確認";
    loading: "読み込み中...";
  };
  auth: {
    login: "ログイン";
    logout: "ログアウト";
    register: "新規登録";
    forgotPassword: "パスワードを忘れた場合";
  };
  errors: {
    notFound: "ページが見つかりません";
    unauthorized: "認証が必要です";
    validation: {
      required: "必須項目です";
      minLength: "{{min}}文字以上入力してください";
      maxLength: "{{max}}文字以下で入力してください";
      email: "有効なメールアドレスを入力してください";
      pattern: "{{pattern}}の形式で入力してください";
    };
  };
}

// ドットパスで翻訳キーを生成
type TranslationPath<T, Prefix extends string = ""> =
  T extends string
    ? Prefix
    : {
        [K in keyof T & string]:
          TranslationPath<T[K], Prefix extends "" ? K : `${Prefix}.${K}`>;
      }[keyof T & string];

type TranslationKey = TranslationPath<TranslationSchema>;
// "common.save" | "common.cancel" | ... | "errors.validation.required" | ...

// 翻訳値の型を取得
type TranslationValue<T, Path extends string> =
  Path extends `${infer Key}.${infer Rest}`
    ? Key extends keyof T
      ? TranslationValue<T[Key], Rest>
      : never
    : Path extends keyof T
      ? T[Path]
      : never;

// プレースホルダーの抽出
type ExtractI18nParams<S extends string> =
  S extends `${string}{{${infer Param}}}${infer Rest}`
    ? Param | ExtractI18nParams<Rest>
    : never;

// 翻訳関数の型
type TranslateParams<Key extends TranslationKey> =
  ExtractI18nParams<TranslationValue<TranslationSchema, Key> & string> extends never
    ? [params?: never]
    : [params: Record<ExtractI18nParams<TranslationValue<TranslationSchema, Key> & string>, string | number>];

function t<K extends TranslationKey>(
  key: K,
  ...args: TranslateParams<K>
): string {
  // 実装...
  return "";
}

// 使用例
t("common.save");                     // OK: パラメータなし
t("errors.validation.minLength", { min: 3 });    // OK: min パラメータ必須
t("errors.validation.maxLength", { max: 100 });  // OK: max パラメータ必須
// t("errors.validation.minLength");  // エラー: パラメータが必要
// t("invalid.key");                  // エラー: 無効なキー
```

### 5.2 GraphQL クエリの型安全化

```typescript
// GraphQL スキーマの型定義
type GraphQLSchema = {
  Query: {
    user: { args: { id: string }; return: User };
    users: { args: { limit?: number; offset?: number }; return: User[] };
    post: { args: { id: string }; return: Post };
    posts: { args: { authorId?: string }; return: Post[] };
  };
  User: {
    id: string;
    name: string;
    email: string;
    posts: Post[];
  };
  Post: {
    id: string;
    title: string;
    content: string;
    author: User;
  };
};

// クエリフィールドの抽出
type ExtractFields<S extends string> =
  S extends `${infer Field} ${infer Rest}`
    ? Trim<Field> | ExtractFields<Rest>
    : S extends `${infer Field},${infer Rest}`
      ? Trim<Field> | ExtractFields<Rest>
      : Trim<S>;

// 型安全な GraphQL クエリ結果
type QueryResult<
  Schema extends Record<string, any>,
  Fields extends keyof Schema
> = Pick<Schema, Fields>;

// 使用イメージ
type UserQueryResult = QueryResult<GraphQLSchema["User"], "id" | "name" | "email">;
// { id: string; name: string; email: string }
```

### 5.3 環境変数の型安全化

```typescript
// 環境変数の型定義
type EnvSchema = {
  NODE_ENV: "development" | "production" | "test";
  PORT: `${number}`;
  DATABASE_URL: `${"postgres" | "mysql"}://${string}`;
  API_KEY: string;
  LOG_LEVEL: "debug" | "info" | "warn" | "error";
  CACHE_TTL: `${number}`;
  CORS_ORIGIN: `${"http" | "https"}://${string}`;
  REDIS_URL: `redis://${string}`;
};

// 環境変数のアクセサー型
type EnvAccessor<T extends Record<string, string>> = {
  get<K extends keyof T>(key: K): T[K];
  getOrDefault<K extends keyof T>(key: K, defaultValue: T[K]): T[K];
  has(key: keyof T): boolean;
  require<K extends keyof T>(key: K): NonNullable<T[K]>;
};

// 型安全な環境変数パーサー
type ParseEnvValue<T extends string> =
  T extends `${number}` ? number :
  T extends "true" | "false" ? boolean :
  string;

type ParsedEnv<T extends Record<string, string>> = {
  [K in keyof T]: ParseEnvValue<T[K]>;
};

type ParsedEnvSchema = ParsedEnv<EnvSchema>;
// {
//   NODE_ENV: string;  (リテラル型のため)
//   PORT: number;
//   DATABASE_URL: string;
//   API_KEY: string;
//   LOG_LEVEL: string;
//   CACHE_TTL: number;
//   CORS_ORIGIN: string;
//   REDIS_URL: string;
// }
```

### 5.4 コマンドラインの型安全化

```typescript
// CLI コマンドの型定義
type Command = "init" | "build" | "deploy" | "test" | "lint";
type Flag = "--verbose" | "--quiet" | "--dry-run" | "--force" | "--watch";
type Option = "--config" | "--output" | "--env" | "--port";

type CLIInput = `${Command}${` ${Flag}`}${` ${Option}=${string}`}`;

// コマンド引数のパーサー
type ParseFlags<S extends string> =
  S extends `${string}--${infer Flag} ${infer Rest}`
    ? `--${Flag extends `${infer Name} ${string}` ? Name : Flag}` | ParseFlags<Rest>
    : S extends `${string}--${infer Flag}`
      ? `--${Flag}`
      : never;

type ParseOptions<S extends string> =
  S extends `${string}--${infer Key}=${infer Value} ${infer Rest}`
    ? { [K in Key]: Value } & ParseOptions<Rest>
    : S extends `${string}--${infer Key}=${infer Value}`
      ? { [K in Key]: Value }
      : {};

type ParsedCommand = ParseOptions<"build --config=webpack.config.js --output=dist --env=production">;
// { config: "webpack.config.js" } & { output: "dist" } & { env: "production" }
```

---

## 6. パフォーマンスとベストプラクティス

### 6.1 Union爆発の回避

```typescript
// BAD: 組み合わせが爆発する
// type Letter = "a" | "b" | "c" | ... | "z";  // 26個
// type TwoLetters = `${Letter}${Letter}`;       // 26 × 26 = 676個
// type ThreeLetters = `${Letter}${Letter}${Letter}`; // 17,576個！
// → コンパイルが極端に遅くなる

// GOOD: パターンで妥協する
type ThreeLetters = `${string}${string}${string}`; // 広いが高速

// GOOD: 必要な組み合わせだけを生成
type ValidCodes = "AAA" | "BBB" | "CCC";  // 手動で定義

// GOOD: テンプレートリテラル型のUnionサイズを制限
// 各位置のUnionメンバー数の目安:
// - 2 × 2 × 2 = 8 → 問題なし
// - 10 × 10 = 100 → 許容範囲
// - 20 × 20 = 400 → やや遅くなる可能性
// - 50 × 50 = 2,500 → 注意が必要
// - 100 × 100 = 10,000 → 避けるべき

// 対処法1: ブランド型で制約
type ThreeLetterCode = string & { readonly __brand: unique symbol };

function createCode(code: string): ThreeLetterCode {
  if (!/^[A-Z]{3}$/.test(code)) {
    throw new Error("Invalid code");
  }
  return code as ThreeLetterCode;
}

// 対処法2: テンプレートリテラル型 + ランタイムバリデーション
type DateString = `${number}-${number}-${number}`;

function isDateString(s: string): s is DateString {
  return /^\d{4}-\d{2}-\d{2}$/.test(s);
}
```

### 6.2 デバッグテクニック

```typescript
// テクニック1: 段階的な型の確認
type Step1 = ExtractParams<"/users/:userId/posts/:postId">;
// ホバーして確認: "userId" | "postId"

type Step2 = RouteParams<"/users/:userId/posts/:postId">;
// ホバーして確認: { userId: string; postId: string }

// テクニック2: テスト用の型チェック
type Expect<T extends true> = T;
type Equal<X, Y> =
  (<T>() => T extends X ? 1 : 2) extends
  (<T>() => T extends Y ? 1 : 2) ? true : false;

// テストケース
type TestSplit1 = Expect<Equal<Split<"a.b.c", ".">, ["a", "b", "c"]>>;
type TestJoin1 = Expect<Equal<Join<["a", "b", "c"], ".">, "a.b.c">>;
type TestReplace1 = Expect<Equal<Replace<"ab", "a", "x">, "xb">>;
type TestTrim1 = Expect<Equal<Trim<"  hello  ">, "hello">>;

// テクニック3: エラーメッセージの改善
type ValidatePath<T extends string> =
  T extends `/${string}`
    ? T
    : `パスは '/' で始まる必要があります。受け取った値: '${T}'`;

type ValidPath = ValidatePath<"/users">;    // "/users"
type InvalidPath = ValidatePath<"users">;   // "パスは '/' で始まる必要があります。受け取った値: 'users'"
```

### 6.3 ベストプラクティス

```typescript
// 1. テンプレートリテラル型は「パターンのヒント」として使う
// 厳密なバリデーションはランタイムで行う

// GOOD: 型でパターンを示し、ランタイムで検証
type Email = `${string}@${string}.${string}`;

function validateEmail(email: string): email is Email {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

// 2. 再利用可能な小さな部品に分割
type Protocol = "http" | "https";
type Domain = `${string}.${string}`;
type Path = `/${string}`;
type URL = `${Protocol}://${Domain}${Path}`;

// 3. ケース変換は必要な場合のみ使用
// 型の可読性を優先する

// BAD: 過度なケース変換チェーン
type OverlyComplex<T extends string> =
  Capitalize<Lowercase<CamelToSnake<SnakeToCamel<T>>>>;

// GOOD: 必要な変換のみ
type ApiToCamel<T extends string> = SnakeToCamel<T>;

// 4. 条件型と組み合わせる場合は段階的に
// BAD: 1つの型で全てを行う
type DoEverything<T extends string> =
  T extends `${infer A}:${infer B}`
    ? `${Capitalize<A>}: ${B extends `${infer C}.${infer D}` ? `${Uppercase<C>}.${D}` : B}`
    : never;

// GOOD: 分割して名前をつける
type ParsePrefix<T extends string> =
  T extends `${infer Prefix}:${infer Rest}` ? [Prefix, Rest] : never;

type FormatPrefix<T extends string> = Capitalize<T>;

type FormatValue<T extends string> =
  T extends `${infer Head}.${infer Tail}` ? `${Uppercase<Head>}.${Tail}` : T;

// 5. ドキュメントとしてのテスト型を残す
type _TestCamelToSnake = Expect<Equal<CamelToSnake<"camelCase">, "camel_case">>;
type _TestSnakeToCamel = Expect<Equal<SnakeToCamel<"snake_case">, "snakeCase">>;
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
| infer | 不可 | 不可 | パターンマッチ可能 |
| パフォーマンス | 最速 | 高速 | Union数に依存 |

### テンプレートリテラル型の展開ルール

```
  `${A | B}${C | D}`
  展開:
    → `${A}${C}` | `${A}${D}` | `${B}${C}` | `${B}${D}`

  `${"get" | "set"}${"Name" | "Age"}`
  展開:
    → "getName" | "getAge" | "setName" | "setAge"

  `${string}` → string（パターンを表す）
  `${number}` → `${number}`（数値文字列パターン）
  `${boolean}` → "true" | "false"
  `${bigint}` → `${bigint}`（bigint文字列パターン）

  注意: 組み合わせ数が爆発する場合あり
  |A| × |B| = 結果のUnionメンバー数
```

---

## アンチパターン

### アンチパターン1: Union爆発

```typescript
// BAD: 組み合わせが爆発する
// type Letter = "a" | "b" | "c" | ... | "z";  // 26個
// type TwoLetters = `${Letter}${Letter}`;       // 26 × 26 = 676個
// type ThreeLetters = `${Letter}${Letter}${Letter}`; // 17,576個！
// → コンパイルが極端に遅くなる

// GOOD: パターンで妥協する
type ThreeLetterPattern = `${string}${string}${string}`; // 広いが高速
// または正規表現ベースのランタイムバリデーションと併用
```

### アンチパターン2: 実行時検証の代わりにテンプレートリテラル型を使う

```typescript
// BAD: 型だけでメールアドレスを検証しようとする
type Email = `${string}@${string}.${string}`;
const email: Email = "not-valid@@"; // これが通ってしまう場合がある

// GOOD: 型は大まかなパターンに留め、実行時バリデーションを併用
import { z } from "zod";
const emailSchema = z.string().email();
type EmailBrand = string & { readonly __email: unique symbol };

function parseEmail(input: string): EmailBrand {
  const result = emailSchema.parse(input);
  return result as EmailBrand;
}
```

### アンチパターン3: 過度に深い再帰

```typescript
// BAD: 文字列の全文字を型レベルでイテレーション
type CountChar<S extends string, C extends string, Acc extends any[] = []> =
  S extends `${infer Head}${infer Tail}`
    ? Head extends C
      ? CountChar<Tail, C, [...Acc, 0]>
      : CountChar<Tail, C, Acc>
    : Acc["length"];

// 短い文字列なら OK だが、長い文字列ではスタックオーバーフロー
type Count = CountChar<"hello world", "l">;  // 3

// GOOD: 型レベルでやるべきことと、ランタイムでやるべきことを分ける
function countChar(s: string, c: string): number {
  return s.split(c).length - 1;
}
```

### アンチパターン4: テンプレートリテラル型の可読性の無視

```typescript
// BAD: 1行に詰め込みすぎ
type ParseComplexURL<S extends string> = S extends `${infer P}://${infer U}@${infer H}:${infer Port extends number}/${infer Path}?${infer Q}#${infer F}` ? { protocol: P; user: U; host: H; port: Port; path: Path; query: Q; fragment: F } : never;

// GOOD: 段階的に分割
type ParseProtocol<S extends string> =
  S extends `${infer Protocol}://${infer Rest}`
    ? { protocol: Protocol; rest: Rest }
    : never;

type ParseAuth<S extends string> =
  S extends `${infer User}@${infer Rest}`
    ? { user: User; rest: Rest }
    : { user: never; rest: S };

type ParseHost<S extends string> =
  S extends `${infer Host}:${infer Port extends number}/${infer Rest}`
    ? { host: Host; port: Port; rest: Rest }
    : S extends `${infer Host}/${infer Rest}`
      ? { host: Host; port: never; rest: Rest }
      : never;
// 各部品を組み合わせて最終的な型を構築
```

---

## FAQ

### Q1: テンプレートリテラル型のパフォーマンスへの影響は？

**A:** Union型の直積展開により、組み合わせ数が急速に増加します。TypeScriptは内部的に最大100,000程度のUnionメンバーを処理できますが、数千を超えるとコンパイルが遅くなります。大きなUnionの直積は避けてください。目安として、各位置のUnionサイズが10以下であれば問題ありません。

### Q2: テンプレートリテラル型でパスの型安全性を確保する実践的な方法は？

**A:** ルーティングライブラリ（React Router, tRPC, Hono など）の多くがテンプレートリテラル型を活用した型安全なルーティングを提供しています。自前で実装するよりも、既存ライブラリの型定義を活用するのが実用的です。自前で実装する場合は、パスパラメータの抽出パターンを小さなユーティリティ型に分割してください。

### Q3: `${number}` はどのような文字列にマッチしますか？

**A:** `"0"`, `"42"`, `"3.14"`, `"-1"` など、数値のリテラル表現にマッチします。ただし `"1e10"` のような科学表記にもマッチする場合があります。`"NaN"` や `"Infinity"` もマッチします。厳密な数値文字列の検証にはランタイムチェックを併用してください。

```typescript
type Test1 = "42" extends `${number}` ? true : false;      // true
type Test2 = "3.14" extends `${number}` ? true : false;    // true
type Test3 = "-1" extends `${number}` ? true : false;      // true
type Test4 = "NaN" extends `${number}` ? true : false;     // true
type Test5 = "abc" extends `${number}` ? true : false;     // false
type Test6 = "" extends `${number}` ? true : false;        // false
```

### Q4: テンプレートリテラル型は TypeScript のどのバージョンから使えますか？

**A:** TypeScript 4.1 で基本機能が導入されました。以降のバージョンで改善されています:

- **4.1**: テンプレートリテラル型の導入、Key Remapping
- **4.3**: テンプレートリテラル型の改善（infer との組み合わせ強化）
- **4.5**: Tail-call optimization の改善（深い再帰のサポート向上）
- **4.7**: `infer extends` 制約の追加
- **4.8**: テンプレートリテラル型での `${infer N extends number}` のサポート

### Q5: テンプレートリテラル型と正規表現の違いは？

**A:** テンプレートリテラル型はコンパイル時の型レベルのパターンマッチングで、正規表現はランタイムの文字列マッチングです。テンプレートリテラル型でできることは正規表現よりもかなり限定的です（量指定子やキャラクタクラスがない）。複雑なパターン検証には、テンプレートリテラル型で大まかなパターンを定義し、ランタイムで正規表現による厳密な検証を行うアプローチが推奨されます。

### Q6: テンプレートリテラル型で生成したユニオン型のメンバー数を知る方法は？

**A:** 型レベルで直接メンバー数を数えることは難しいですが、`UnionToTuple` のような型を使えば概念的にはできます。実務では、エディタのホバー表示で展開された型を確認するか、型テストで期待するメンバーが含まれているかを検証するのが実用的です。

---

## まとめ

| 項目 | 内容 |
|------|------|
| 基本構文 | `` `prefix-${Type}` `` で文字列パターンを型に |
| Union展開 | `${A \| B}` は自動的に直積展開される |
| 文字列操作 | Uppercase, Lowercase, Capitalize, Uncapitalize |
| infer | テンプレートリテラル型の中でパターンマッチ可能 |
| ケース変換 | CamelToSnake, SnakeToCamel 等のカスタム型 |
| パス型 | URLパラメータの型安全な抽出に有用 |
| 文字列操作 | Split, Join, Replace, Trim 等の型レベル文字列操作 |
| 実務パターン | CSS, SQL, i18n, テンプレートエンジン等 |
| 注意点 | Union爆発によるコンパイル速度低下に注意 |
| ベストプラクティス | 小さく分割、ランタイム検証と併用 |

---

## 次に読むべきガイド

- [03-type-challenges.md](./03-type-challenges.md) -- 型チャレンジ
- [04-declaration-files.md](./04-declaration-files.md) -- 宣言ファイル
- [00-conditional-types.md](./00-conditional-types.md) -- 条件型
- [01-mapped-types.md](./01-mapped-types.md) -- マップ型

---

## 参考文献

1. **TypeScript Handbook: Template Literal Types** -- https://www.typescriptlang.org/docs/handbook/2/template-literal-types.html
2. **TypeScript 4.1 Release Notes** -- https://www.typescriptlang.org/docs/handbook/release-notes/typescript-4-1.html
3. **Matt Pocock: Template Literal Types** -- https://www.totaltypescript.com/books/total-typescript-essentials/template-literal-types
4. **Type-Level TypeScript: Template Literal Types** -- https://type-level-typescript.com/template-literal-types
5. **TypeScript 4.8 Release Notes** -- https://www.typescriptlang.org/docs/handbook/release-notes/typescript-4-8.html
