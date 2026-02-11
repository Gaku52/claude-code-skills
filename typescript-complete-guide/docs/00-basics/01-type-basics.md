# 型の基礎

> TypeScriptの根幹をなすプリミティブ型、リテラル型、配列、タプル、enum、特殊型（any/unknown/never）を体系的に学ぶ。

## この章で学ぶこと

1. **プリミティブ型とリテラル型** -- string, number, boolean, symbol, bigint およびリテラル型による値の制限
2. **コレクション型** -- 配列型、タプル型、readonly修飾子による不変性の表現
3. **特殊型** -- any, unknown, never, void, null, undefined の正しい使い分け

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

---

## FAQ

### Q1: `string` と `String` の違いは？

**A:** 小文字の `string` はTypeScriptのプリミティブ型です。大文字の `String` はJavaScriptのラッパーオブジェクト型です。常に小文字の `string` を使ってください。`number` / `Number`、`boolean` / `Boolean` も同様です。

### Q2: タプルの要素数チェックは実行時にも行われますか？

**A:** いいえ。タプルの型チェックはコンパイル時のみです。実行時にはただの配列になります。実行時の要素数チェックが必要な場合は、手動でバリデーションを書くか、Zodなどを使います。

### Q3: enum は使うべきですか？

**A:** 2025年現在、多くのTypeScriptスタイルガイドでは `const enum` か、Union型 + `as const` オブジェクトパターンが推奨されています。通常の数値 enum は逆引きマッピングのためにランタイムコードを生成し、バンドルサイズに影響するためです。

---

## まとめ

| 型カテゴリ | 型 | 用途 |
|-----------|-----|------|
| プリミティブ | string, number, boolean, symbol, bigint | 基本的な値 |
| リテラル | "hello", 42, true | 特定の値に限定 |
| コレクション | T[], [T1, T2] | 同種/異種データの集合 |
| 列挙 | enum, const enum | 名前付き定数の集合 |
| 特殊（危険） | any | 型チェック無効化（非推奨） |
| 特殊（安全） | unknown | 型安全な任意値 |
| 特殊（不在） | never | 到達不能、網羅性チェック |
| 特殊（空） | void, null, undefined | 戻り値なし、値の不在 |

---

## 次に読むべきガイド

- [02-functions-and-objects.md](./02-functions-and-objects.md) -- 関数とオブジェクト型
- [03-union-intersection.md](./03-union-intersection.md) -- Union型とIntersection型

---

## 参考文献

1. **TypeScript Handbook: Everyday Types** -- https://www.typescriptlang.org/docs/handbook/2/everyday-types.html
2. **TypeScript Deep Dive: TypeScript's Type System** -- https://basarat.gitbook.io/typescript/type-system
3. **Effective TypeScript (Dan Vanderkam著, O'Reilly)** -- 特に Item 7: Think of Types as Sets of Values
