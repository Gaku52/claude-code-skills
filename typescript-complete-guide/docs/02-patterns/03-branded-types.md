# TypeScript ブランド型（Branded Types）

> ブランド型・公称型・opaque 型で、同じプリミティブ型の値を区別し、ID の取り違えや単位の混同をコンパイル時に防止する

## この章で学ぶこと

1. **ブランド型の原理** -- 構造的型付けの限界と、擬似的な公称型を TypeScript で実現する方法
2. **実装パターン** -- `__brand` フィールド、`unique symbol`、テンプレートリテラル型の 3 つのアプローチ
3. **実践的な適用** -- ID の取り違え防止、単位付き数値、バリデーション済み値の追跡

---

## 1. なぜブランド型が必要か

### 1-1. 構造的型付けの落とし穴

```
構造的型付け（TypeScript のデフォルト）:

  type UserId = string;
  type OrderId = string;

  UserId と OrderId は同じ構造（string）
       ↓
  相互に代入可能!

  function getUser(id: UserId): User { ... }
  const orderId: OrderId = "order-123";
  getUser(orderId);  // エラーにならない!

公称型（ブランド型で擬似的に実現）:

  type UserId = string & { __brand: "UserId" };
  type OrderId = string & { __brand: "OrderId" };

  UserId と OrderId は構造が異なる
       ↓
  相互に代入不可!

  getUser(orderId);  // コンパイルエラー!
```

```typescript
// 構造的型付けの問題
type UserId = string;
type OrderId = string;
type ProductId = string;

function getUser(id: UserId): void {}
function getOrder(id: OrderId): void {}

const userId: UserId = "user-1";
const orderId: OrderId = "order-1";

// 全てコンパイルが通ってしまう
getUser(orderId);    // バグ! OrderId を渡している
getOrder(userId);    // バグ! UserId を渡している
```

---

## 2. 実装パターン

### 2-1. `__brand` フィールドパターン

```typescript
// ブランド型の定義
type Brand<T, B extends string> = T & { readonly __brand: B };

type UserId = Brand<string, "UserId">;
type OrderId = Brand<string, "OrderId">;
type ProductId = Brand<string, "ProductId">;

// コンストラクタ関数
function userId(id: string): UserId {
  return id as UserId;
}

function orderId(id: string): OrderId {
  return id as OrderId;
}

// 使用例
function getUser(id: UserId): void {
  console.log(`Fetching user: ${id}`);
}

const uid = userId("user-123");
const oid = orderId("order-456");

getUser(uid);  // OK
getUser(oid);  // コンパイルエラー!
// Error: Type 'OrderId' is not assignable to type 'UserId'.
//   Type 'OrderId' is not assignable to type '{ __brand: "UserId" }'.
```

### 2-2. `unique symbol` パターン（より厳密）

```typescript
// unique symbol で完全にユニークなブランドを作成
declare const UserIdBrand: unique symbol;
declare const OrderIdBrand: unique symbol;

type UserId = string & { readonly [UserIdBrand]: typeof UserIdBrand };
type OrderId = string & { readonly [OrderIdBrand]: typeof OrderIdBrand };

function userId(id: string): UserId {
  return id as UserId;
}

function orderId(id: string): OrderId {
  return id as OrderId;
}

// unique symbol は declare のみで、ランタイムコストゼロ
```

### 2-3. バリデーション付きブランド型

```
バリデーション付きブランド型のフロー:

  string (未検証)
     |
     v
  +------------------+
  | validate & brand |
  +------------------+
     |           |
     v           v
  Ok(Email)   Err("invalid")
  (検証済み)   (拒否)
     |
     v
  sendEmail(email: Email)
  → 型が保証するので再検証不要
```

```typescript
// バリデーション付きブランド型
type Email = Brand<string, "Email">;
type NonEmptyString = Brand<string, "NonEmptyString">;
type PositiveNumber = Brand<number, "PositiveNumber">;
type Percentage = Brand<number, "Percentage">; // 0-100

// スマートコンストラクタ（Result 型と組み合わせ）
function email(value: string): Result<Email, string> {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(value)) {
    return Err("Invalid email format");
  }
  return Ok(value as Email);
}

function nonEmpty(value: string): Result<NonEmptyString, string> {
  if (value.trim().length === 0) {
    return Err("String must not be empty");
  }
  return Ok(value.trim() as NonEmptyString);
}

function positiveNumber(value: number): Result<PositiveNumber, string> {
  if (value <= 0 || !Number.isFinite(value)) {
    return Err("Must be a positive finite number");
  }
  return Ok(value as PositiveNumber);
}

function percentage(value: number): Result<Percentage, string> {
  if (value < 0 || value > 100) {
    return Err("Must be between 0 and 100");
  }
  return Ok(value as Percentage);
}

// 使用例
function sendWelcomeEmail(to: Email, name: NonEmptyString): void {
  // to は検証済みメール、name は空でない文字列が保証されている
  console.log(`Sending welcome email to ${to} for ${name}`);
}

const emailResult = email("alice@example.com");
const nameResult = nonEmpty("Alice");

if (isOk(emailResult) && isOk(nameResult)) {
  sendWelcomeEmail(emailResult.value, nameResult.value);
}
```

---

## 3. 単位付き数値

### 3-1. 物理量のブランド型

```typescript
// 単位ごとにブランド型を定義
type Meters = Brand<number, "Meters">;
type Kilometers = Brand<number, "Kilometers">;
type Miles = Brand<number, "Miles">;
type Seconds = Brand<number, "Seconds">;
type MetersPerSecond = Brand<number, "MetersPerSecond">;

// コンストラクタ
const meters = (v: number) => v as Meters;
const kilometers = (v: number) => v as Kilometers;
const miles = (v: number) => v as Miles;
const seconds = (v: number) => v as Seconds;

// 単位変換関数
function kmToMeters(km: Kilometers): Meters {
  return (km * 1000) as unknown as Meters;
}

function milesToKm(mi: Miles): Kilometers {
  return (mi * 1.60934) as unknown as Kilometers;
}

// 型安全な演算
function speed(distance: Meters, time: Seconds): MetersPerSecond {
  return (distance / time) as unknown as MetersPerSecond;
}

// 使用例
const d = meters(100);
const t = seconds(10);
const v = speed(d, t); // OK: MetersPerSecond

const km = kilometers(5);
// speed(km, t);  // コンパイルエラー! Kilometers は Meters ではない
speed(kmToMeters(km), t); // OK: 変換してから渡す
```

### 3-2. 通貨のブランド型

```
通貨演算の型安全性:

  JPY + JPY → JPY  ✓
  USD + USD → USD  ✓
  JPY + USD → ???  ✗ コンパイルエラー
  JPY * 2   → JPY  ✓ (スカラー倍は OK)
```

```typescript
type JPY = Brand<number, "JPY">;
type USD = Brand<number, "USD">;
type EUR = Brand<number, "EUR">;

// 同一通貨同士の加算
function addMoney<T extends Brand<number, string>>(a: T, b: T): T {
  return ((a as number) + (b as number)) as unknown as T;
}

// スカラー倍
function multiplyMoney<T extends Brand<number, string>>(
  amount: T,
  factor: number
): T {
  return ((amount as number) * factor) as unknown as T;
}

const price1 = 1000 as JPY;
const price2 = 2000 as JPY;
const total = addMoney(price1, price2); // OK: JPY

const usd = 10 as USD;
// addMoney(price1, usd);  // エラー! JPY と USD は加算不可
```

---

## 4. zod との統合

```typescript
import { z } from "zod";

// zod スキーマでブランド型を生成
const UserIdSchema = z.string().uuid().brand<"UserId">();
type UserId = z.infer<typeof UserIdSchema>;
// 型: string & { __brand: "UserId" }

const EmailSchema = z.string().email().brand<"Email">();
type Email = z.infer<typeof EmailSchema>;

const PositiveSchema = z.number().positive().brand<"Positive">();
type Positive = z.infer<typeof PositiveSchema>;

// 使用例
function createUser(id: UserId, email: Email): void {
  // ...
}

const id = UserIdSchema.parse("550e8400-e29b-41d4-a716-446655440000");
const mail = EmailSchema.parse("alice@example.com");
createUser(id, mail); // OK

// createUser("raw-string", "email"); // コンパイルエラー!
```

---

## 比較表

### ブランド型の実装方法比較

| 方法 | 型安全性 | ランタイムコスト | 実装量 | zod統合 |
|------|---------|--------------|--------|---------|
| `__brand` フィールド | 高 | ゼロ | 少 | 不要 |
| `unique symbol` | 最高 | ゼロ | 中 | 不要 |
| zod `.brand()` | 高 | 検証コスト | 最少 | 組込み |
| class ラッパー | 最高 | ラップコスト | 多 | 別途必要 |

### ブランド型 vs 他のアプローチ

| 比較軸 | ブランド型 | class ラッパー | enum | テンプレートリテラル |
|--------|-----------|-------------|------|------------------|
| ランタイムコスト | ゼロ | インスタンス生成 | 小 | ゼロ |
| 元のメソッド | 利用可能 | ラップ必要 | 不可 | 利用可能 |
| JSON 互換性 | そのまま | `toJSON` 必要 | そのまま | そのまま |
| パターンマッチ | 不可 | instanceof | switch | パターン |
| バリデーション | スマートCtor | コンストラクタ | 定義済み | 正規表現 |

---

## アンチパターン

### AP-1: as で無条件にブランド付与

```typescript
// NG: バリデーションなしでブランド付与
function unsafeBrand(input: string): Email {
  return input as Email; // 不正な値もブランドが付く
}

const bad = unsafeBrand("not-an-email");
sendEmail(bad); // 実行時エラー

// OK: スマートコンストラクタで検証
function safeEmail(input: string): Result<Email, string> {
  if (!isValidEmail(input)) {
    return Err("Invalid email");
  }
  return Ok(input as Email);
}
```

### AP-2: ブランド型を使いすぎる

```typescript
// NG: 全てのプリミティブにブランドを付ける（過剰）
type FirstName = Brand<string, "FirstName">;
type LastName = Brand<string, "LastName">;
type MiddleName = Brand<string, "MiddleName">;
type Street = Brand<string, "Street">;
type City = Brand<string, "City">;
// ... 数十個のブランド型

// OK: 取り違えが危険な場面のみブランド付与
// - ID 系（UserId, OrderId, ProductId）
// - 単位系（Meters, Seconds, JPY）
// - バリデーション済み値（Email, Url, NonEmpty）
```

---

## 実践パターン: ブランド型ユーティリティ

```
ブランド型ユーティリティの構成:

+-------------------+     +-------------------+
| Brand<T, B>       |     | Unbrand<T>        |
| string → UserId   |     | UserId → string   |
+-------------------+     +-------------------+

+-------------------+     +-------------------+
| validate & brand  |     | isBranded<T>()    |
| 検証 + ブランド付与|     | 型ガード          |
+-------------------+     +-------------------+
```

```typescript
// ユーティリティ型
type Brand<T, B extends string> = T & { readonly __brand: B };
type Unbrand<T> = T extends Brand<infer U, string> ? U : T;

// ブランド除去（シリアライズ時に使用）
function unbrand<T extends Brand<unknown, string>>(value: T): Unbrand<T> {
  return value as Unbrand<T>;
}

// ブランド型のセット
type BrandedId<B extends string> = Brand<string, B>;

type UserId = BrandedId<"UserId">;
type OrderId = BrandedId<"OrderId">;
type ProductId = BrandedId<"ProductId">;

// ジェネリックな ID 生成
function createId<B extends string>(brand: B): () => Brand<string, B> {
  return () => crypto.randomUUID() as Brand<string, B>;
}

const newUserId = createId("UserId");
const newOrderId = createId("OrderId");

const uid: UserId = newUserId();     // OK
const oid: OrderId = newOrderId();   // OK
// const bad: UserId = newOrderId(); // エラー!
```

---

## FAQ

### Q1: ブランド型はランタイムにオーバーヘッドがありますか？

ありません。`as` によるキャストは型アサーションであり、JavaScript 出力からは完全に消えます。`__brand` フィールドは型レベルにのみ存在し、実際のオブジェクトには追加されません。ただし、スマートコンストラクタのバリデーションロジックにはランタイムコストがあります。

### Q2: ブランド型の値を JSON にシリアライズできますか？

はい。ブランドは型レベルの概念なので、`JSON.stringify` は元のプリミティブ値をそのままシリアライズします。デシリアライズ時にはスマートコンストラクタで再検証してブランドを付与する必要があります。

### Q3: ライブラリの型とブランド型を組み合わせるには？

外部ライブラリの関数にブランド型を渡す場合、`as` で元の型に戻す必要があることがあります。境界層でブランドを付与し、ドメインロジック内ではブランド型を使い、外部 API 呼び出し時に `unbrand` するパターンが一般的です。

---

## まとめ表

| 概念 | 要点 |
|------|------|
| ブランド型 | `T & { __brand: B }` で構造的に異なる型を作る |
| 公称型 | TypeScript にはないが、ブランドで擬似的に実現 |
| スマートコンストラクタ | バリデーション + ブランド付与を一箇所で行う |
| 適用場面 | ID, 単位, 通貨, 検証済み文字列 |
| ランタイムコスト | ブランド自体はゼロ、検証ロジックは別途 |
| zod 統合 | `.brand()` メソッドで宣言的にブランド付与 |

---

## 次に読むべきガイド

- [判別共用体](./02-discriminated-unions.md) -- ブランド型と判別共用体の組み合わせ
- [Zod バリデーション](../04-ecosystem/00-zod-validation.md) -- zod の `.brand()` を使った実践パターン
- [エラーハンドリング](./00-error-handling.md) -- スマートコンストラクタと Result 型

---

## 参考文献

1. **Branding and Type-Tagging** -- TypeScript Deep Dive
   https://basarat.gitbook.io/typescript/main-1/nominaltyping

2. **Nominal Typing Techniques in TypeScript** -- Michal Zalecki
   https://michalzalecki.com/nominal-typing-in-typescript/

3. **Zod - Brand** -- Zod Documentation
   https://zod.dev/?id=brand
