# Union型とIntersection型

> 型を「または」「かつ」で組み合わせる強力な仕組み。判別共用体と型ガードによる安全な型の絞り込みを習得する。

## この章で学ぶこと

1. **Union型** -- `|` 演算子による型の合成、判別共用体、型の絞り込み
2. **Intersection型** -- `&` 演算子による型の合成、ミックスインパターン
3. **型ガード** -- typeof, instanceof, in, ユーザー定義型ガードによるナローイング

---

## 1. Union型

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
// value = [];    // エラー
```

### コード例2: 判別共用体（Discriminated Unions）

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

---

## 2. Intersection型

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

### Union vs Intersection 比較

| 特性 | Union (`A \| B`) | Intersection (`A & B`) |
|------|-------------------|------------------------|
| 意味 | A **または** B | A **かつ** B |
| プロパティ | 共通のプロパティのみアクセス可 | 全てのプロパティにアクセス可 |
| 集合論 | 和集合 | 積集合 |
| 値の範囲 | 広がる | 狭まる |
| 型の範囲 | 広い（どちらかを満たせばOK） | 狭い（全てを満たす必要） |
| 使用場面 | 複数の可能性を表す | 型の合成・拡張 |

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

---

## 3. 型ガードとナローイング

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
interface Dog { bark(): void; }
interface Cat { meow(): void; }

function speak(pet: Dog | Cat): void {
  if ("bark" in pet) {
    pet.bark();   // Dog
  } else {
    pet.meow();   // Cat
  }
}
```

### コード例5: ユーザー定義型ガード

```typescript
// 型述語 (Type Predicate): `value is Type`
interface Fish { swim(): void; }
interface Bird { fly(): void; }

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

### ナローイングの流れ

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

---

## 4. 網羅性チェック

### コード例6: never を使った網羅性チェック

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

// 組み合わせて様々な構成を作る
type SecureConfig = BaseConfig & WithAuth & WithSSL;
type BasicConfig = BaseConfig & WithAuth;
type PublicConfig = BaseConfig;

const secureConfig: SecureConfig = {
  host: "db.example.com",
  port: 5432,
  auth: { username: "admin", password: "secret" },
  ssl: { cert: "...", key: "..." },
};
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

### Q3: 型ガードの `is` 構文は必ず必要ですか？

**A:** `typeof`, `instanceof`, `in` などの組み込みガードではTypeScriptが自動的にナローイングします。ユーザー定義の関数で型を絞り込みたい場合のみ `is` 構文（型述語）が必要です。カスタムのバリデーション関数を作る際に特に有用です。

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
| 網羅性チェック | never + default で全ケースの処理漏れを検出 |

---

## 次に読むべきガイド

- [04-generics.md](./04-generics.md) -- ジェネリクス
- [../02-patterns/02-discriminated-unions.md](../02-patterns/02-discriminated-unions.md) -- 判別共用体パターン（実践編）

---

## 参考文献

1. **TypeScript Handbook: Narrowing** -- https://www.typescriptlang.org/docs/handbook/2/narrowing.html
2. **TypeScript Handbook: Unions and Intersection Types** -- https://www.typescriptlang.org/docs/handbook/2/everyday-types.html#union-types
3. **Discriminated Unions in TypeScript** -- https://www.typescriptlang.org/docs/handbook/typescript-in-5-minutes-func.html#discriminated-unions
