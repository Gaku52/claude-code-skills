# 関数とオブジェクト型

> TypeScriptにおける関数シグネチャ、オーバーロード、interface、type aliasの使い分けを網羅する。

## この章で学ぶこと

1. **関数の型付け** -- パラメータ型、戻り値型、オプショナル引数、デフォルト値、rest引数、オーバーロード
2. **interface** -- オブジェクトの構造を定義し、クラスやモジュール間の契約として使う
3. **type alias** -- 型エイリアスによる柔軟な型定義とinterfaceとの使い分け

---

## 1. 関数の型付け

### コード例1: 基本的な関数型

```typescript
// 関数宣言
function add(a: number, b: number): number {
  return a + b;
}

// アロー関数
const multiply = (a: number, b: number): number => a * b;

// 関数型の変数
const divide: (a: number, b: number) => number = (a, b) => a / b;

// 型エイリアスで関数型を定義
type MathOp = (a: number, b: number) => number;
const subtract: MathOp = (a, b) => a - b;
```

### コード例2: オプショナル引数とデフォルト値

```typescript
// オプショナル引数（?）
function greet(name: string, greeting?: string): string {
  return `${greeting ?? "Hello"}, ${name}!`;
}
greet("Alice");           // "Hello, Alice!"
greet("Alice", "Hi");     // "Hi, Alice!"

// デフォルト値
function createUser(name: string, role: string = "viewer"): { name: string; role: string } {
  return { name, role };
}
createUser("Bob");              // { name: "Bob", role: "viewer" }
createUser("Bob", "admin");     // { name: "Bob", role: "admin" }

// rest引数
function sum(...numbers: number[]): number {
  return numbers.reduce((total, n) => total + n, 0);
}
sum(1, 2, 3, 4, 5); // 15
```

### コード例3: 関数オーバーロード

```typescript
// オーバーロードシグネチャ
function createElement(tag: "div"): HTMLDivElement;
function createElement(tag: "span"): HTMLSpanElement;
function createElement(tag: "input"): HTMLInputElement;
// 実装シグネチャ
function createElement(tag: string): HTMLElement {
  return document.createElement(tag);
}

const div = createElement("div");     // 型: HTMLDivElement
const span = createElement("span");   // 型: HTMLSpanElement
const input = createElement("input"); // 型: HTMLInputElement
```

### 関数型の記法比較

```
  関数宣言          アロー関数型         call signature
+---------------+  +------------------+  +---------------------+
| function      |  | (a: T, b: U)     |  | interface Fn {      |
|   fn(a: T):U  |  |   => R           |  |   (a: T, b: U): R  |
+---------------+  +------------------+  +---------------------+

  メソッドシグネチャ     コンストラクタシグネチャ
+---------------------+  +-----------------------+
| interface Obj {     |  | interface Ctor {      |
|   method(a: T): R   |  |   new (a: T): Obj     |
| }                   |  | }                     |
+---------------------+  +-----------------------+
```

---

## 2. interface

### コード例4: interface の定義と使用

```typescript
// 基本的なinterface
interface User {
  readonly id: number;     // 読み取り専用
  name: string;            // 必須プロパティ
  email: string;           // 必須プロパティ
  age?: number;            // オプショナルプロパティ
}

// インデックスシグネチャ
interface Dictionary {
  [key: string]: string;
}

// 関数を持つinterface
interface Formatter {
  format(value: unknown): string;
  readonly prefix: string;
}

// interface の継承
interface Employee extends User {
  department: string;
  salary: number;
}

// 複数の継承
interface Manager extends Employee {
  reports: Employee[];
}

const manager: Manager = {
  id: 1,
  name: "Alice",
  email: "alice@example.com",
  department: "Engineering",
  salary: 120000,
  reports: [],
};
```

### コード例5: interface のマージ（Declaration Merging）

```typescript
// 同名のinterfaceは自動的にマージされる
interface Window {
  myCustomProperty: string;
}

// これでグローバルの Window に myCustomProperty が追加される
// ライブラリの型拡張に便利

interface Config {
  host: string;
  port: number;
}

interface Config {
  debug: boolean;      // マージされる
}

// 結果の型: { host: string; port: number; debug: boolean }
const config: Config = {
  host: "localhost",
  port: 3000,
  debug: true,
};
```

---

## 3. type alias

### コード例6: type alias の柔軟性

```typescript
// オブジェクト型
type Point = {
  x: number;
  y: number;
};

// Union型
type Result<T> = { success: true; data: T } | { success: false; error: Error };

// 関数型
type EventHandler = (event: Event) => void;

// タプル型
type Coordinate = [number, number];

// マップ型
type Readonly<T> = { readonly [K in keyof T]: T[K] };

// 条件型
type NonNullable<T> = T extends null | undefined ? never : T;

// テンプレートリテラル型
type HttpMethod = `${"GET" | "POST" | "PUT" | "DELETE"}`;
type Endpoint = `/${string}`;
type ApiRoute = `${HttpMethod} ${Endpoint}`;
```

### interface vs type alias 比較

| 特性 | interface | type alias |
|------|-----------|------------|
| オブジェクト型 | OK | OK |
| Union型 | 不可 | OK |
| Intersection | extends で継承 | `&` で合成 |
| Declaration Merging | OK（同名で自動マージ） | 不可（重複エラー） |
| implements | OK | OK（一部制限あり） |
| 条件型・マップ型 | 不可 | OK |
| パフォーマンス | やや高速（キャッシュ） | 複雑な型は遅くなる場合あり |
| 推奨シーン | オブジェクト構造、公開API | Union、複雑な型変換 |

### 使い分けの判断フロー

```
  型を定義したい
      |
      v
  Union型が必要？ ----Yes----> type alias
      |
      No
      |
      v
  条件型/マップ型が必要？ ----Yes----> type alias
      |
      No
      |
      v
  Declaration Mergingが必要？ ----Yes----> interface
      |
      No
      |
      v
  オブジェクトの構造定義？ ----Yes----> interface（推奨）
      |                                  または type（好み）
      No
      |
      v
  type alias を使用
```

---

## 4. 構造的型付け（Structural Typing）

### コード例7: ダックタイピング

```typescript
interface Point {
  x: number;
  y: number;
}

// Point インターフェースを明示的にimplementsしていなくてもOK
const point = { x: 10, y: 20, z: 30 };

function printPoint(p: Point): void {
  console.log(`(${p.x}, ${p.y})`);
}

// point は x, y を持っているので Point として受け入れられる
printPoint(point); // OK: 構造が一致していれば良い

// 過剰プロパティチェック（直接オブジェクトリテラルの場合のみ）
// printPoint({ x: 10, y: 20, z: 30 }); // エラー: z は Point に存在しない
```

### 構造的型付けの図解

```
  名前的型付け (Java, C# など)           構造的型付け (TypeScript)
+----------------------------+    +----------------------------+
| class Dog implements       |    | interface HasName {        |
|   Animal { ... }           |    |   name: string;            |
|                            |    | }                          |
| → Dog は Animal の名前で   |    |                            |
|   型チェック               |    | // { name: string } を持つ |
+----------------------------+    | // 全てのオブジェクトが     |
                                  | // HasName として使える     |
                                  +----------------------------+
```

---

## アンチパターン

### アンチパターン1: 過度にネストした型定義

```typescript
// BAD: インライン型定義が深くネストして読めない
function processOrder(
  order: {
    items: {
      product: { id: number; name: string; price: number };
      quantity: number;
      options?: { gift: boolean; message?: string };
    }[];
    customer: { name: string; address: { street: string; city: string } };
  }
): void { /* ... */ }

// GOOD: 型を分割して名前をつける
interface Address {
  street: string;
  city: string;
}
interface Customer {
  name: string;
  address: Address;
}
interface Product {
  id: number;
  name: string;
  price: number;
}
interface OrderItem {
  product: Product;
  quantity: number;
  options?: { gift: boolean; message?: string };
}
interface Order {
  items: OrderItem[];
  customer: Customer;
}
function processOrder(order: Order): void { /* ... */ }
```

### アンチパターン2: interfaceとtypeの無秩序な混在

```typescript
// BAD: 同じプロジェクト内でinterfaceとtypeを一貫性なく使う
interface User { name: string; }
type Product = { name: string; };    // なぜここだけtype？
interface Order { items: string[]; }
type Invoice = { total: number; };   // 一貫性がない

// GOOD: チームで方針を決めて統一する
// 方針例: オブジェクト構造はinterface、Unionや複雑な型はtype
interface User { name: string; }
interface Product { name: string; }
interface Order { items: string[]; }
type PaymentMethod = "credit" | "debit" | "cash"; // Union はtype
type Result<T> = Success<T> | Failure;             // Union はtype
```

---

## FAQ

### Q1: 関数のオーバーロードとUnion型パラメータ、どちらを使うべきですか？

**A:** 入力の型に応じて戻り値の型が変わる場合はオーバーロードが適切です。戻り値が同じなら Union型パラメータの方がシンプルです。
```typescript
// Union型で十分なケース
function len(x: string | any[]): number { return x.length; }

// オーバーロードが必要なケース（戻り値の型が変わる）
function parse(input: string): string[];
function parse(input: string[]): string[][];
function parse(input: string | string[]) { /* ... */ }
```

### Q2: `readonly` と `Readonly<T>` の違いは？

**A:** `readonly` はプロパティ単位の修飾子で、`Readonly<T>` はオブジェクト全体の全プロパティを一括で readonly にするユーティリティ型です。ネストしたオブジェクトの深い部分までは `Readonly<T>` でも readonly にはなりません。深い immutability が必要な場合はカスタムの `DeepReadonly` 型を定義します。

### Q3: `{}` 型は何を表しますか？

**A:** `{}` は「null と undefined 以外の全ての値」を表します。空オブジェクト型ではありません。空オブジェクトを表したい場合は `Record<string, never>` を使うのが正確です。`{}` は意図せず広い型になるため、避けるべきです。

---

## まとめ

| 項目 | 内容 |
|------|------|
| 関数型 | パラメータ型と戻り値型を明示。推論にも頼れる |
| オプショナル引数 | `?` で省略可能に。デフォルト値も指定可 |
| オーバーロード | 入力に応じて戻り値型を変えたいときに使う |
| interface | オブジェクト構造の定義。継承・マージが可能 |
| type alias | 柔軟な型定義。Union、条件型、マップ型に必須 |
| 構造的型付け | 名前ではなく構造で型の互換性を判定 |
| 使い分け | オブジェクト→interface、Union/複雑な型→type |

---

## 次に読むべきガイド

- [03-union-intersection.md](./03-union-intersection.md) -- Union型とIntersection型
- [04-generics.md](./04-generics.md) -- ジェネリクス

---

## 参考文献

1. **TypeScript Handbook: More on Functions** -- https://www.typescriptlang.org/docs/handbook/2/functions.html
2. **TypeScript Handbook: Object Types** -- https://www.typescriptlang.org/docs/handbook/2/objects.html
3. **Effective TypeScript, Item 13: Know the Differences Between type and interface** -- Dan Vanderkam著, O'Reilly
