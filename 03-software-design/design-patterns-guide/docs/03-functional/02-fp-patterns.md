# 関数型パターン

> カリー化、パイプライン、レンズなど、関数型プログラミングの実践的パターンを習得し、合成可能で保守性の高いコードを書く

## この章で学ぶこと

1. **関数合成の基本** — カリー化、部分適用、ポイントフリースタイルの仕組みと使い分け
2. **パイプラインパターン** — データ変換チェーン、ミドルウェア合成、Railway Oriented Programming
3. **不変データ操作** — レンズ、Prism、構造共有、Immer との比較
4. **メモ化と最適化** — 計算キャッシュ、トランスデューサー、遅延評価
5. **実務応用** — React/Redux での関数型パターン、テスト容易性の向上

---

## 前提知識

このガイドを読む前に、以下の知識を持っていることを推奨します。

| 前提知識 | 参照先 |
|---|---|
| TypeScript の基本（ジェネリクス、型推論） | [02-programming カテゴリ](../../../../02-programming/) |
| 高階関数の概念（map, filter, reduce） | JavaScript/TypeScript の基本 |
| 不変性（Immutability）の基本概念 | [クリーンコードの原則](../../clean-code-principles/) |
| ファンクタとモナドの基礎 | [ファンクタ・アプリカティブ](./01-functor-applicative.md)、[モナド](./00-monad.md) |

---

## 1. カリー化と部分適用

### 1.1 カリー化とは何か — WHY から理解する

```
カリー化と部分適用
====================

通常の関数:
  add(a, b) = a + b
  add(3, 5)  --> 8

カリー化:
  add = a => b => a + b
  add(3)     --> b => 3 + b   (部分適用された関数)
  add(3)(5)  --> 8

部分適用:
  add3 = add(3)    // 引数を1つ固定
  add3(5)  --> 8
  add3(10) --> 13

利点: 関数の再利用と合成が容易になる
```

**WHY**: なぜカリー化が必要なのか？

カリー化は一見すると不必要な複雑さに見えますが、以下の3つの重要な利点があります。

1. **関数の再利用**: 引数の一部を固定して特化した関数を作れる
2. **関数合成**: 1引数関数のチェーンとして合成しやすい
3. **遅延評価**: 全引数が揃うまで計算を遅延できる

```
┌────────────────────────────────────────────────────────┐
│  カリー化の動作イメージ                                 │
│                                                        │
│  通常の関数: f(a, b, c) → 結果                         │
│  全引数を一度に渡す                                     │
│                                                        │
│  カリー化: f(a) → g(b) → h(c) → 結果                  │
│  引数を1つずつ渡すと、次の関数が返る                    │
│                                                        │
│  multiply(2)(5)                                        │
│       │    │                                           │
│       │    └─ 最終引数 → 結果 10                       │
│       └─ 最初の引数 → (b => 2 * b) 関数が返る         │
│                                                        │
│  const double = multiply(2)  ← 部分適用で再利用可能    │
│  const triple = multiply(3)  ← 別の特化関数も作れる    │
│                                                        │
│  double(5)  → 10                                       │
│  double(7)  → 14                                       │
│  triple(5)  → 15                                       │
└────────────────────────────────────────────────────────┘
```

### コード例 1: カリー化の実装と活用

```typescript
// === 汎用カリー化関数 ===

// 2引数のカリー化
function curry<A, B, C>(fn: (a: A, b: B) => C): (a: A) => (b: B) => C {
  return (a: A) => (b: B) => fn(a, b);
}

// 3引数のカリー化
function curry3<A, B, C, D>(
  fn: (a: A, b: B, c: C) => D
): (a: A) => (b: B) => (c: C) => D {
  return (a: A) => (b: B) => (c: C) => fn(a, b, c);
}

// 任意引数のカリー化（JavaScript の動的な機能を活用）
function curryN(fn: (...args: any[]) => any): (...args: any[]) => any {
  const arity = fn.length;
  return function curried(...args: any[]): any {
    if (args.length >= arity) {
      return fn(...args);
    }
    return (...moreArgs: any[]) => curried(...args, ...moreArgs);
  };
}

// === 実用例1: 数値計算 ===
const multiply = curry((a: number, b: number) => a * b);
const double = multiply(2);
const triple = multiply(3);

console.log(double(5));  // 10
console.log(triple(5));  // 15
console.log(double(7));  // 14

// === 実用例2: 文字列処理 ===
const prefix = curry((pre: string, str: string) => `${pre}${str}`);
const addHttps = prefix("https://");
const addApiPrefix = prefix("/api/v1");

console.log(addHttps("example.com"));   // "https://example.com"
console.log(addApiPrefix("/users"));     // "/api/v1/users"

// === 実用例3: フィルタの部分適用 ===
const filterBy = curry3(
  <T>(key: keyof T, value: T[keyof T], items: T[]) =>
    items.filter(item => item[key] === value)
);

interface Order {
  id: number;
  status: string;
  amount: number;
}

const filterByStatus = filterBy<Order>("status");
const getActiveOrders = filterByStatus("active");
const getPendingOrders = filterByStatus("pending");

const orders: Order[] = [
  { id: 1, status: "active", amount: 100 },
  { id: 2, status: "pending", amount: 200 },
  { id: 3, status: "active", amount: 150 },
];

console.log(getActiveOrders(orders));
// [{ id: 1, status: "active", amount: 100 }, { id: 3, status: "active", amount: 150 }]
console.log(getPendingOrders(orders));
// [{ id: 2, status: "pending", amount: 200 }]
```

### コード例 2: Rust での部分適用

```rust
// Rust ではクロージャで部分適用を実現する

fn main() {
    // クロージャによる部分適用
    let multiply = |a: i32| move |b: i32| a * b;
    let double = multiply(2);
    let triple = multiply(3);

    println!("double(5) = {}", double(5));  // 10
    println!("triple(5) = {}", triple(5));  // 15

    // 実用例: 文字列フォーマット
    let format_price = |currency: &str| {
        let currency = currency.to_string();
        move |amount: f64| format!("{}{:.2}", currency, amount)
    };

    let format_usd = format_price("$");
    let format_yen = format_price("¥");

    println!("{}", format_usd(19.99));  // $19.99
    println!("{}", format_yen(2000.0)); // ¥2000.00

    // 実用例: バリデーション関数の生成
    let min_length = |min: usize| {
        move |s: &str| -> Result<&str, String> {
            if s.len() >= min {
                Ok(s)
            } else {
                Err(format!("{}文字以上必要です（現在{}文字）", min, s.len()))
            }
        }
    };

    let validate_username = min_length(3);
    let validate_password = min_length(8);

    println!("{:?}", validate_username("ab"));      // Err("3文字以上必要です（現在2文字）")
    println!("{:?}", validate_username("alice"));    // Ok("alice")
    println!("{:?}", validate_password("pass"));     // Err("8文字以上必要です（現在4文字）")
    println!("{:?}", validate_password("password123")); // Ok("password123")
}
```

### 1.2 カリー化と部分適用の違い

```
┌────────────────────────────────────────────────────────┐
│  カリー化 vs 部分適用                                   │
│                                                        │
│  ■ カリー化 (Currying):                                │
│    f(a, b, c) → f(a)(b)(c)                             │
│    全引数を1つずつ受け取る関数に変換する                │
│    元の関数の形を変える                                 │
│                                                        │
│  ■ 部分適用 (Partial Application):                     │
│    f(a, b, c) → g(b, c)   [a を固定]                   │
│    一部の引数を固定した新しい関数を作る                 │
│    元の関数を呼び出す際に一部の引数を事前に渡す         │
│                                                        │
│  ■ 関係:                                               │
│    カリー化された関数に引数を1つ渡す                    │
│    = 部分適用の一形態                                   │
│                                                        │
│  ■ 実用上の違い:                                       │
│    カリー化: 関数合成パイプラインで便利                 │
│    部分適用: 設定値の注入で便利                         │
└────────────────────────────────────────────────────────┘
```

```typescript
// 部分適用（カリー化なし）
function partial<A, B, C>(fn: (a: A, b: B) => C, a: A): (b: B) => C {
  return (b: B) => fn(a, b);
}

// bind による部分適用（JavaScript 標準）
function greet(greeting: string, name: string): string {
  return `${greeting}, ${name}!`;
}
const sayHello = greet.bind(null, "Hello");
const sayHi = greet.bind(null, "Hi");

console.log(sayHello("Taro"));  // "Hello, Taro!"
console.log(sayHi("Hanako"));   // "Hi, Hanako!"
```

---

## 2. パイプラインパターン

### 2.1 パイプラインとは — WHY から理解する

```
パイプラインの考え方
=====================

データ --> [変換1] --> [変換2] --> [変換3] --> 結果

Unix パイプ:
  cat file | grep "error" | sort | uniq -c

関数パイプライン:
  pipe(getData, filter(isActive), map(toDTO), sortBy('name'))

WHY: なぜパイプラインか？
  1. 宣言的: 「何をするか」を順序通りに記述
  2. 合成可能: 各ステップが独立、差し替え容易
  3. テスト容易: 各変換関数を単独でテスト可能
  4. 読みやすい: データの流れが左→右（上→下）で自然
```

```
┌────────────────────────────────────────────────────────┐
│  pipe vs compose                                       │
│                                                        │
│  pipe:    f → g → h   (左から右、データの流れ順)       │
│  compose: h ∘ g ∘ f   (右から左、数学的な合成順)       │
│                                                        │
│  pipe(f, g, h)(x)    = h(g(f(x)))                     │
│  compose(h, g, f)(x) = h(g(f(x)))                     │
│                                                        │
│  実用上は pipe が読みやすい（データの流れが自然）       │
│  数学的な議論では compose が使われる                    │
└────────────────────────────────────────────────────────┘
```

### コード例 3: 型安全なパイプライン関数

```typescript
// === 型安全な pipe 関数（オーバーロード） ===

function pipe<A, B>(a: A, ab: (a: A) => B): B;
function pipe<A, B, C>(a: A, ab: (a: A) => B, bc: (b: B) => C): C;
function pipe<A, B, C, D>(
  a: A, ab: (a: A) => B, bc: (b: B) => C, cd: (c: C) => D
): D;
function pipe<A, B, C, D, E>(
  a: A, ab: (a: A) => B, bc: (b: B) => C, cd: (c: C) => D, de: (d: D) => E
): E;
function pipe(initial: unknown, ...fns: Function[]): unknown {
  return fns.reduce((acc, fn) => fn(acc), initial);
}

// === 遅延パイプライン（関数を返す版） ===

function pipeWith<A, B>(ab: (a: A) => B): (a: A) => B;
function pipeWith<A, B, C>(
  ab: (a: A) => B, bc: (b: B) => C
): (a: A) => C;
function pipeWith<A, B, C, D>(
  ab: (a: A) => B, bc: (b: B) => C, cd: (c: C) => D
): (a: A) => D;
function pipeWith(...fns: Function[]): Function {
  return (initial: unknown) => fns.reduce((acc, fn) => fn(acc), initial);
}

// compose: 右から左
function compose<A, B, C>(
  bc: (b: B) => C, ab: (a: A) => B
): (a: A) => C {
  return (a: A) => bc(ab(a));
}

// === 実用例: ユーザーデータ変換パイプライン ===

interface RawUser {
  first_name: string;
  last_name: string;
  age: number;
  status: string;
  email: string;
}

interface ProcessedUser {
  fullName: string;
  age: number;
  email: string;
}

// 各ステップを独立した純粋関数として定義
const filterActive = (users: RawUser[]) =>
  users.filter(u => u.status === "active");

const filterAdults = (users: RawUser[]) =>
  users.filter(u => u.age >= 18);

const toProcessedUser = (users: RawUser[]): ProcessedUser[] =>
  users.map(u => ({
    fullName: `${u.first_name} ${u.last_name}`,
    age: u.age,
    email: u.email.toLowerCase(),
  }));

const sortByName = (users: ProcessedUser[]) =>
  [...users].sort((a, b) => a.fullName.localeCompare(b.fullName));

// パイプラインとして合成
const processUsers = pipeWith(
  filterActive,
  filterAdults,
  toProcessedUser,
  sortByName,
);

// 使用
const rawUsers: RawUser[] = [
  { first_name: "Taro", last_name: "Yamada", age: 25, status: "active", email: "TARO@example.com" },
  { first_name: "Hanako", last_name: "Suzuki", age: 16, status: "active", email: "hanako@example.com" },
  { first_name: "Jiro", last_name: "Tanaka", age: 30, status: "inactive", email: "jiro@example.com" },
  { first_name: "Akiko", last_name: "Sato", age: 22, status: "active", email: "Akiko@Example.COM" },
];

const result = processUsers(rawUsers);
console.log(result);
// [
//   { fullName: "Akiko Sato", age: 22, email: "akiko@example.com" },
//   { fullName: "Taro Yamada", age: 25, email: "taro@example.com" },
// ]
```

### コード例 4: ミドルウェア合成パターン

```typescript
// === Express/Koa スタイルのミドルウェア合成 ===

type Middleware<T> = (ctx: T, next: () => Promise<void>) => Promise<void>;

function composeMiddleware<T>(...middlewares: Middleware<T>[]): Middleware<T> {
  return (ctx: T, next: () => Promise<void>) => {
    let index = -1;
    function dispatch(i: number): Promise<void> {
      if (i <= index) return Promise.reject(new Error("next() called multiple times"));
      index = i;
      const fn = i === middlewares.length ? next : middlewares[i];
      return fn(ctx, () => dispatch(i + 1));
    }
    return dispatch(0);
  };
}

// ミドルウェアの定義
interface Context {
  method: string;
  path: string;
  headers: Record<string, string>;
  body?: unknown;
  response?: unknown;
  startTime?: number;
  user?: { id: string; role: string };
}

const logger: Middleware<Context> = async (ctx, next) => {
  ctx.startTime = Date.now();
  console.log(`→ ${ctx.method} ${ctx.path}`);
  await next();
  console.log(`← ${ctx.method} ${ctx.path} - ${Date.now() - ctx.startTime}ms`);
};

const auth: Middleware<Context> = async (ctx, next) => {
  const token = ctx.headers.authorization;
  if (!token) throw new Error("Unauthorized: No token provided");
  // トークン検証（簡略化）
  ctx.user = { id: "user-1", role: "admin" };
  await next();
};

const errorHandler: Middleware<Context> = async (ctx, next) => {
  try {
    await next();
  } catch (error) {
    ctx.response = {
      status: 500,
      error: error instanceof Error ? error.message : "Unknown error",
    };
  }
};

const handler: Middleware<Context> = async (ctx, next) => {
  ctx.response = { status: 200, data: { message: "Hello!" } };
  await next();
};

// ミドルウェアの合成
const app = composeMiddleware(errorHandler, logger, auth, handler);

// 使用
const ctx: Context = {
  method: "GET",
  path: "/api/users",
  headers: { authorization: "Bearer token123" },
};

await app(ctx, async () => {});
// → GET /api/users
// ← GET /api/users - 5ms
console.log(ctx.response); // { status: 200, data: { message: "Hello!" } }
```

### 2.2 Railway Oriented Programming

エラーハンドリングをパイプラインに組み込むパターンです。

```
┌────────────────────────────────────────────────────────┐
│  Railway Oriented Programming                          │
│                                                        │
│  成功の線路:  ─────[処理1]─────[処理2]─────[処理3]──→  │
│              ╲           ╲           ╲                  │
│  失敗の線路:  ─────────────────────────────────────→    │
│                                                        │
│  各処理で:                                              │
│  - 成功 → 成功の線路を継続                              │
│  - 失敗 → 失敗の線路に切り替え（以降の処理はスキップ） │
│                                                        │
│  Result<E, A> を使って型安全に実現                      │
└────────────────────────────────────────────────────────┘
```

### コード例 5: Railway Oriented Programming の実装

```typescript
// === Result 型の実装 ===

type Result<E, A> =
  | { tag: "Ok"; value: A }
  | { tag: "Err"; error: E };

const ok = <E, A>(value: A): Result<E, A> => ({ tag: "Ok", value });
const err = <E, A>(error: E): Result<E, A> => ({ tag: "Err", error });

// map: 成功時のみ変換（ファンクタ）
function mapR<E, A, B>(result: Result<E, A>, fn: (a: A) => B): Result<E, B> {
  return result.tag === "Ok" ? ok(fn(result.value)) : result;
}

// flatMap: 成功時のみ次の計算を実行（モナド）
function flatMapR<E, A, B>(
  result: Result<E, A>,
  fn: (a: A) => Result<E, B>
): Result<E, B> {
  return result.tag === "Ok" ? fn(result.value) : result;
}

// mapError: エラーの変換
function mapError<E1, E2, A>(
  result: Result<E1, A>,
  fn: (e: E1) => E2
): Result<E2, A> {
  return result.tag === "Err" ? err(fn(result.error)) : result;
}

// === Railway パイプライン ===
function railway<E, A, B>(
  ...fns: Array<(a: any) => Result<E, any>>
): (a: A) => Result<E, B> {
  return (a: A) => {
    let result: Result<E, any> = ok(a);
    for (const fn of fns) {
      if (result.tag === "Err") return result;
      result = fn(result.value);
    }
    return result;
  };
}

// === 実用例: ユーザー登録パイプライン ===

interface RegistrationInput {
  username: string;
  email: string;
  password: string;
}

interface ValidatedInput {
  username: string;
  email: string;
  password: string;
}

interface HashedInput {
  username: string;
  email: string;
  passwordHash: string;
}

interface User {
  id: string;
  username: string;
  email: string;
  passwordHash: string;
  createdAt: Date;
}

type RegistrationError =
  | { type: "validation"; message: string }
  | { type: "duplicate"; field: string }
  | { type: "database"; message: string };

// 各ステップ
function validateInput(input: RegistrationInput): Result<RegistrationError, ValidatedInput> {
  if (input.username.length < 3) {
    return err({ type: "validation", message: "ユーザー名は3文字以上" });
  }
  if (!input.email.includes("@")) {
    return err({ type: "validation", message: "無効なメールアドレス" });
  }
  if (input.password.length < 8) {
    return err({ type: "validation", message: "パスワードは8文字以上" });
  }
  return ok(input);
}

function checkDuplicate(input: ValidatedInput): Result<RegistrationError, ValidatedInput> {
  // DB チェック（簡略化）
  const existingUsers = ["admin", "root"];
  if (existingUsers.includes(input.username)) {
    return err({ type: "duplicate", field: "username" });
  }
  return ok(input);
}

function hashPassword(input: ValidatedInput): Result<RegistrationError, HashedInput> {
  try {
    return ok({
      username: input.username,
      email: input.email,
      passwordHash: `hashed_${input.password}`, // 実際にはbcryptを使用
    });
  } catch {
    return err({ type: "database", message: "ハッシュ化に失敗" });
  }
}

function saveToDatabase(input: HashedInput): Result<RegistrationError, User> {
  try {
    return ok({
      id: `user_${Date.now()}`,
      ...input,
      createdAt: new Date(),
    });
  } catch {
    return err({ type: "database", message: "保存に失敗" });
  }
}

// Railway パイプライン
const registerUser = railway<RegistrationError, RegistrationInput, User>(
  validateInput,
  checkDuplicate,
  hashPassword,
  saveToDatabase,
);

// 使用例
const result1 = registerUser({
  username: "taro",
  email: "taro@example.com",
  password: "secure123",
});
console.log(result1); // { tag: "Ok", value: { id: "user_...", ... } }

const result2 = registerUser({
  username: "ab",
  email: "bad",
  password: "short",
});
console.log(result2); // { tag: "Err", error: { type: "validation", message: "ユーザー名は3文字以上" } }
```

---

## 3. 不変データ操作（レンズ）

### 3.1 レンズとは — WHY から理解する

```
レンズ: 不変データ構造の部分的な読み書き
=========================================

深いネストのオブジェクト更新の問題:

[NG] ミュータブルな直接更新
  user.address.city = "Tokyo";
  → 副作用、テスト困難、予測不可能な変更

[NG] スプレッド地獄
  { ...user, address: { ...user.address, city: "Tokyo" } }
  → ネストが深くなるほど醜くなる

[OK] レンズで宣言的に更新
  set(addressCityLens, "Tokyo", user)
  → 合成可能、型安全、宣言的

レンズの型:
  Lens<S, A>
    get: S -> A          (全体から部分を取得)
    set: (A, S) -> S     (部分を更新して新しい全体を返す)
```

```
┌────────────────────────────────────────────────────────┐
│  レンズの動作イメージ                                   │
│                                                        │
│  ┌──────────────────┐                                  │
│  │  User             │                                 │
│  │  ├─ name: "Taro"  │                                 │
│  │  └─ address       │ ← addressLens                   │
│  │     ├─ city: "大阪"│ ← cityLens                     │
│  │     └─ zip: "530"  │                                │
│  └──────────────────┘                                  │
│                                                        │
│  userCityLens = composeLens(addressLens, cityLens)     │
│                                                        │
│  get: user → "大阪"                                    │
│  set("東京", user) → 新しいUser（city: "東京"）        │
│  over(toUpperCase, user) → 新しいUser（city: "大阪"）  │
│                                                        │
│  ポイント:                                              │
│  - 元の user オブジェクトは変更されない（不変）         │
│  - レンズは合成できる                                   │
│  - 型安全（TypeScript の型推論が効く）                  │
└────────────────────────────────────────────────────────┘
```

### コード例 6: レンズの完全実装

```typescript
// === レンズの型と基本操作 ===

interface Lens<S, A> {
  get: (s: S) => A;
  set: (a: A, s: S) => S;
}

function lens<S, A>(
  get: (s: S) => A,
  set: (a: A, s: S) => S
): Lens<S, A> {
  return { get, set };
}

// レンズの合成
function composeLens<S, A, B>(outer: Lens<S, A>, inner: Lens<A, B>): Lens<S, B> {
  return {
    get: (s: S) => inner.get(outer.get(s)),
    set: (b: B, s: S) => outer.set(inner.set(b, outer.get(s)), s),
  };
}

// over: レンズを通して関数を適用
function over<S, A>(l: Lens<S, A>, fn: (a: A) => A, s: S): S {
  return l.set(fn(l.get(s)), s);
}

// view: レンズを通して値を取得（get のエイリアス）
function view<S, A>(l: Lens<S, A>, s: S): A {
  return l.get(s);
}

// set: レンズを通して値を設定
function setL<S, A>(l: Lens<S, A>, a: A, s: S): S {
  return l.set(a, s);
}

// === プロパティレンズの自動生成 ===
function prop<S, K extends keyof S>(key: K): Lens<S, S[K]> {
  return lens(
    (s: S) => s[key],
    (a: S[K], s: S) => ({ ...s, [key]: a })
  );
}

// === 使用例 ===

interface Address {
  city: string;
  zip: string;
  country: string;
}

interface User {
  name: string;
  age: number;
  address: Address;
}

// プロパティレンズ
const addressLens = prop<User, "address">("address");
const cityLens = prop<Address, "city">("city");
const nameLens = prop<User, "name">("name");
const ageLens = prop<User, "age">("age");

// レンズの合成
const userCityLens = composeLens(addressLens, cityLens);

const user: User = {
  name: "Taro",
  age: 30,
  address: { city: "Osaka", zip: "530-0001", country: "Japan" },
};

// get
console.log(view(userCityLens, user)); // "Osaka"

// set — 不変更新
const updated = setL(userCityLens, "Tokyo", user);
console.log(updated);
// { name: "Taro", age: 30, address: { city: "Tokyo", zip: "530-0001", country: "Japan" } }
console.log(user.address.city); // "Osaka" — 元のオブジェクトは変更されない!

// over — 関数適用
const uppercased = over(userCityLens, c => c.toUpperCase(), user);
console.log(uppercased.address.city); // "OSAKA"

// 複数の更新を合成
const birthday = (u: User): User => {
  const aged = over(ageLens, a => a + 1, u);
  return over(nameLens, n => `${n} (${aged.age}歳)`, aged);
};

console.log(birthday(user));
// { name: "Taro (31歳)", age: 31, address: { ... } }
```

### 3.2 レンズ vs Immer vs スプレッド構文

| 手法 | メリット | デメリット | 適用場面 |
|---|---|---|---|
| **スプレッド構文** | 追加ライブラリ不要、直感的 | ネスト深い場合に冗長 | 浅いネスト（1-2段） |
| **Immer** | ミュータブル風に書ける、構造共有 | ランタイムコスト | 中程度のネスト、Redux |
| **レンズ** | 合成可能、型安全、再利用可能 | 学習コスト、初期セットアップ | 深いネスト、頻繁な更新 |

```typescript
// === 3つの手法の比較 ===

const user = {
  name: "Taro",
  address: {
    city: "Osaka",
    location: {
      lat: 34.69,
      lng: 135.50,
    },
  },
};

// 1. スプレッド構文 — 読みにくい
const updated1 = {
  ...user,
  address: {
    ...user.address,
    location: {
      ...user.address.location,
      lat: 35.68, // 東京の緯度
    },
  },
};

// 2. Immer — 直感的だがランタイムコスト
import { produce } from "immer";
const updated2 = produce(user, draft => {
  draft.address.location.lat = 35.68;
});

// 3. レンズ — 合成可能で再利用可能
const locationLens = composeLens(
  prop<typeof user, "address">("address"),
  composeLens(
    prop<typeof user.address, "location">("location"),
    prop<typeof user.address.location, "lat">("lat")
  )
);
const updated3 = setL(locationLens, 35.68, user);
```

---

## 4. メモ化と最適化

### 4.1 メモ化 — WHY から理解する

**WHY**: 純粋関数は同じ入力に対して常に同じ出力を返すため、計算結果をキャッシュして再利用できます。これがメモ化の基本原理です。

```
┌────────────────────────────────────────────────────────┐
│  メモ化の動作                                           │
│                                                        │
│  初回呼び出し:                                          │
│  fibonacci(10) → 計算実行 → 結果 55 → キャッシュに保存 │
│                                                        │
│  2回目以降:                                             │
│  fibonacci(10) → キャッシュヒット → 結果 55（瞬時）    │
│                                                        │
│  前提条件:                                              │
│  - 純粋関数であること（副作用なし）                     │
│  - 参照透過性があること（同じ入力→同じ出力）           │
│  - 引数空間が有限または頻出パターンがあること           │
│                                                        │
│  注意:                                                  │
│  - メモリ使用量とのトレードオフ                         │
│  - LRU キャッシュでメモリ制限を設ける                   │
│  - 引数がオブジェクトの場合はキー生成に注意             │
└────────────────────────────────────────────────────────┘
```

### コード例 7: メモ化の実装

```typescript
// === 汎用メモ化関数 ===

function memoize<Args extends unknown[], R>(
  fn: (...args: Args) => R,
  keyFn: (...args: Args) => string = (...args) => JSON.stringify(args)
): (...args: Args) => R {
  const cache = new Map<string, R>();

  const memoized = (...args: Args): R => {
    const key = keyFn(...args);
    if (cache.has(key)) return cache.get(key)!;
    const result = fn(...args);
    cache.set(key, result);
    return result;
  };

  // キャッシュの管理メソッドを追加
  (memoized as any).cache = cache;
  (memoized as any).clear = () => cache.clear();

  return memoized;
}

// === LRU キャッシュ付きメモ化 ===

function memoizeLRU<Args extends unknown[], R>(
  fn: (...args: Args) => R,
  maxSize: number = 100,
  keyFn: (...args: Args) => string = (...args) => JSON.stringify(args)
): (...args: Args) => R {
  const cache = new Map<string, R>();

  return (...args: Args): R => {
    const key = keyFn(...args);
    if (cache.has(key)) {
      const value = cache.get(key)!;
      // LRU: アクセスしたら末尾に移動
      cache.delete(key);
      cache.set(key, value);
      return value;
    }
    const result = fn(...args);
    cache.set(key, result);
    if (cache.size > maxSize) {
      // 最も古いエントリを削除
      const oldest = cache.keys().next().value;
      if (oldest !== undefined) cache.delete(oldest);
    }
    return result;
  };
}

// === TTL（有効期限）付きメモ化 ===

function memoizeTTL<Args extends unknown[], R>(
  fn: (...args: Args) => R,
  ttlMs: number = 60000 // デフォルト1分
): (...args: Args) => R {
  const cache = new Map<string, { value: R; expiresAt: number }>();

  return (...args: Args): R => {
    const key = JSON.stringify(args);
    const now = Date.now();
    const cached = cache.get(key);

    if (cached && cached.expiresAt > now) {
      return cached.value;
    }

    const result = fn(...args);
    cache.set(key, { value: result, expiresAt: now + ttlMs });
    return result;
  };
}

// === 使用例 ===

// 1. フィボナッチ数列
const fibonacci = memoize((n: number): number =>
  n <= 1 ? n : fibonacci(n - 1) + fibonacci(n - 2)
);

console.log(fibonacci(50)); // 12586269025 — 瞬時に計算

// 2. 高コストなデータ変換
const processLargeDataset = memoizeLRU(
  (data: string[], threshold: number): string[] => {
    console.log("Computing...");
    return data.filter(d => d.length > threshold).sort();
  },
  50 // 最大50エントリをキャッシュ
);

const data = ["apple", "banana", "cherry", "date"];
processLargeDataset(data, 4); // Computing... → ["apple", "banana", "cherry"]
processLargeDataset(data, 4); // キャッシュヒット → ["apple", "banana", "cherry"]

// 3. API レスポンスのキャッシュ
const fetchUserCached = memoizeTTL(
  async (userId: string) => {
    const res = await fetch(`/api/users/${userId}`);
    return res.json();
  },
  30000 // 30秒キャッシュ
);
```

### コード例 8: Rust でのメモ化

```rust
use std::collections::HashMap;

// Rust でのメモ化（HashMap ベース）
struct Memoize<F, K, V>
where
    F: Fn(K) -> V,
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    func: F,
    cache: HashMap<K, V>,
}

impl<F, K, V> Memoize<F, K, V>
where
    F: Fn(K) -> V,
    K: std::hash::Hash + Eq + Clone,
    V: Clone,
{
    fn new(func: F) -> Self {
        Memoize {
            func,
            cache: HashMap::new(),
        }
    }

    fn call(&mut self, key: K) -> V {
        if let Some(value) = self.cache.get(&key) {
            return value.clone();
        }
        let value = (self.func)(key.clone());
        self.cache.insert(key, value.clone());
        value
    }
}

fn main() {
    // フィボナッチのメモ化（HashMap使用）
    let mut fib_cache: HashMap<u64, u64> = HashMap::new();

    fn fib(n: u64, cache: &mut HashMap<u64, u64>) -> u64 {
        if let Some(&result) = cache.get(&n) {
            return result;
        }
        let result = if n <= 1 { n } else { fib(n - 1, cache) + fib(n - 2, cache) };
        cache.insert(n, result);
        result
    }

    println!("fib(50) = {}", fib(50, &mut fib_cache));
    // fib(50) = 12586269025

    // 文字列処理のメモ化
    let mut processor = Memoize::new(|s: String| {
        println!("Processing: {}", s);
        s.chars().rev().collect::<String>()
    });

    println!("{}", processor.call("hello".to_string())); // Processing: hello → "olleh"
    println!("{}", processor.call("hello".to_string())); // キャッシュヒット → "olleh"
    println!("{}", processor.call("world".to_string())); // Processing: world → "dlrow"
}
```

### 4.2 トランスデューサー

```typescript
// === トランスデューサー: 中間配列なしの変換合成 ===

// 問題: 通常のチェーンは各ステップで中間配列を作る
const result1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  .filter(x => x % 2 === 0)  // [2, 4, 6, 8, 10] — 中間配列1
  .map(x => x * 10)           // [20, 40, 60, 80, 100] — 中間配列2
  .filter(x => x > 30);       // [40, 60, 80, 100] — 中間配列3
// 3回の配列走査、3つの中間配列

// 解決: トランスデューサーは変換を合成し、1回の走査で処理
type Reducer<A, B> = (acc: B, value: A) => B;
type Transducer<A, B> = <C>(reducer: Reducer<B, C>) => Reducer<A, C>;

function mapT<A, B>(fn: (a: A) => B): Transducer<A, B> {
  return <C>(reducer: Reducer<B, C>): Reducer<A, C> =>
    (acc, value) => reducer(acc, fn(value));
}

function filterT<A>(pred: (a: A) => boolean): Transducer<A, A> {
  return <C>(reducer: Reducer<A, C>): Reducer<A, C> =>
    (acc, value) => pred(value) ? reducer(acc, value) : acc;
}

function composeT<A, B, C>(
  t1: Transducer<A, B>,
  t2: Transducer<B, C>
): Transducer<A, C> {
  return <D>(reducer: Reducer<C, D>): Reducer<A, D> =>
    t1(t2(reducer));
}

function transduce<A, B, C>(
  transducer: Transducer<A, B>,
  reducer: Reducer<B, C>,
  initial: C,
  input: A[]
): C {
  const composed = transducer(reducer);
  return input.reduce(composed, initial);
}

// 使用例
const xform = composeT(
  filterT<number>(x => x % 2 === 0),
  composeT(
    mapT<number, number>(x => x * 10),
    filterT<number>(x => x > 30)
  )
);

const result2 = transduce(
  xform,
  (acc: number[], val: number) => [...acc, val],
  [],
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
);

console.log(result2); // [40, 60, 80, 100]
// 1回の走査、中間配列なし!
```

---

## 5. 実務での関数型パターン

### 5.1 React/Redux での関数型パターン

```typescript
// === React Hooks と関数型パターン ===

// 1. useMemo — メモ化
function ExpensiveComponent({ data, filter }: Props) {
  // data や filter が変わらない限り再計算しない
  const processed = useMemo(
    () => data.filter(filter).sort(compareFn).map(toDisplayItem),
    [data, filter]
  );

  return <List items={processed} />;
}

// 2. useReducer — 状態遷移の関数型モデル
type Action =
  | { type: "ADD_ITEM"; payload: Item }
  | { type: "REMOVE_ITEM"; payload: string }
  | { type: "UPDATE_ITEM"; payload: { id: string; changes: Partial<Item> } };

// Reducer は純粋関数: (State, Action) → State
function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "ADD_ITEM":
      return { ...state, items: [...state.items, action.payload] };
    case "REMOVE_ITEM":
      return {
        ...state,
        items: state.items.filter(i => i.id !== action.payload),
      };
    case "UPDATE_ITEM":
      return {
        ...state,
        items: state.items.map(i =>
          i.id === action.payload.id ? { ...i, ...action.payload.changes } : i
        ),
      };
    default:
      return state;
  }
}

// 3. カスタムフック — 関数の合成
function useFilteredSortedData<T>(
  data: T[],
  filterFn: (item: T) => boolean,
  sortFn: (a: T, b: T) => number,
) {
  return useMemo(() => {
    return pipe(
      data,
      (d: T[]) => d.filter(filterFn),
      (d: T[]) => [...d].sort(sortFn),
    );
  }, [data, filterFn, sortFn]);
}
```

### コード例 9: Redux Toolkit と関数型パターン

```typescript
// === Redux Toolkit（Immer ベース）===

import { createSlice, PayloadAction } from "@reduxjs/toolkit";

interface TodoState {
  items: Todo[];
  filter: "all" | "active" | "completed";
}

const todoSlice = createSlice({
  name: "todos",
  initialState: { items: [], filter: "all" } as TodoState,
  reducers: {
    // Immer により、ミュータブル風に書いても不変更新される
    addTodo: (state, action: PayloadAction<string>) => {
      state.items.push({
        id: Date.now().toString(),
        text: action.payload,
        completed: false,
      });
    },
    toggleTodo: (state, action: PayloadAction<string>) => {
      const todo = state.items.find(t => t.id === action.payload);
      if (todo) todo.completed = !todo.completed;
    },
    setFilter: (state, action: PayloadAction<TodoState["filter"]>) => {
      state.filter = action.payload;
    },
  },
});

// Selector — 純粋関数でデータを導出
const selectFilteredTodos = (state: RootState): Todo[] => {
  const { items, filter } = state.todos;
  switch (filter) {
    case "active":
      return items.filter(t => !t.completed);
    case "completed":
      return items.filter(t => t.completed);
    default:
      return items;
  }
};

// メモ化されたセレクタ（reselect パターン）
import { createSelector } from "@reduxjs/toolkit";

const selectTodos = (state: RootState) => state.todos.items;
const selectFilter = (state: RootState) => state.todos.filter;

const selectFilteredTodosMemoized = createSelector(
  [selectTodos, selectFilter],
  (todos, filter) => {
    switch (filter) {
      case "active": return todos.filter(t => !t.completed);
      case "completed": return todos.filter(t => t.completed);
      default: return todos;
    }
  }
);
```

### コード例 10: fp-ts を使った関数型プログラミング

```typescript
// === fp-ts: TypeScript の本格的な関数型ライブラリ ===

import { pipe } from "fp-ts/function";
import * as O from "fp-ts/Option";
import * as E from "fp-ts/Either";
import * as A from "fp-ts/Array";
import * as TE from "fp-ts/TaskEither";

// 1. Option（Maybe）の使用
const getUser = (id: string): O.Option<User> =>
  id === "1" ? O.some({ id: "1", name: "Taro", age: 30 }) : O.none;

const result1 = pipe(
  getUser("1"),
  O.map(u => u.name),
  O.map(n => n.toUpperCase()),
  O.getOrElse(() => "Unknown")
);
console.log(result1); // "TARO"

// 2. Either（Result）の使用
const parseAge = (s: string): E.Either<string, number> => {
  const n = parseInt(s, 10);
  return isNaN(n) ? E.left("無効な数値") : E.right(n);
};

const validateAge = (age: number): E.Either<string, number> =>
  age >= 0 && age <= 150 ? E.right(age) : E.left("年齢の範囲外");

const result2 = pipe(
  parseAge("25"),
  E.flatMap(validateAge),
  E.map(age => `年齢: ${age}歳`),
  E.getOrElse(err => `エラー: ${err}`)
);
console.log(result2); // "年齢: 25歳"

// 3. 配列操作
const users: User[] = [
  { id: "1", name: "Taro", age: 30 },
  { id: "2", name: "Hanako", age: 17 },
  { id: "3", name: "Jiro", age: 25 },
];

const adultNames = pipe(
  users,
  A.filter((u: User) => u.age >= 18),
  A.map((u: User) => u.name),
  A.sort({ compare: (a: string, b: string) => a < b ? -1 : a > b ? 1 : 0, equals: (a, b) => a === b }),
);
console.log(adultNames); // ["Jiro", "Taro"]

// 4. TaskEither（非同期 + エラーハンドリング）
const fetchUser = (id: string): TE.TaskEither<Error, User> =>
  TE.tryCatch(
    () => fetch(`/api/users/${id}`).then(r => r.json()),
    (err) => new Error(`Failed to fetch user: ${err}`)
  );

const fetchOrders = (userId: string): TE.TaskEither<Error, Order[]> =>
  TE.tryCatch(
    () => fetch(`/api/orders?user=${userId}`).then(r => r.json()),
    (err) => new Error(`Failed to fetch orders: ${err}`)
  );

// パイプラインで合成
const getUserWithOrders = (userId: string) =>
  pipe(
    fetchUser(userId),
    TE.flatMap(user =>
      pipe(
        fetchOrders(user.id),
        TE.map(orders => ({ user, orders }))
      )
    )
  );
```

---

## 関数型パターン一覧比較表

| パターン | 目的 | TypeScript での実現 | Rust での実現 | Haskell |
|---|---|---|---|---|
| **カリー化** | 部分適用で関数を再利用 | arrow function / curry() | クロージャ | デフォルト |
| **パイプライン** | データ変換の宣言的記述 | pipe() 関数 | メソッドチェーン | `$` / `|>` |
| **レンズ** | 不変データの部分更新 | 手動実装 / Ramda | なし（所有権で管理） | lens パッケージ |
| **メモ化** | 計算結果のキャッシュ | Map / useMemo | HashMap | MVar / IORef |
| **トランスデューサー** | 中間配列なしの変換合成 | reduce ベース | Iterator アダプタ | conduit |
| **パターンマッチ** | データ構造の分解と分岐 | 型ガード / switch | match 式 | case / pattern |
| **Railway** | エラーハンドリング | Result 型 | Result<T, E> | Either |

### 関数型 vs 手続き型比較表

| 側面 | 関数型 | 手続き型 |
|---|---|---|
| **状態管理** | 不変データ + 新しい値を返す | ミュータブル変数を直接変更 |
| **制御フロー** | 再帰、高階関数、パイプライン | ループ、条件分岐 |
| **副作用** | 分離して管理（IO モナド等） | どこでも発生 |
| **テスト容易性** | 高い（参照透過性、純粋関数） | 低い（状態依存、モック必要） |
| **並行性** | 安全（共有状態なし） | 危険（競合状態） |
| **デバッグ** | 値の追跡が容易 | 状態の追跡が困難 |
| **パフォーマンス** | GC 依存、構造共有で改善 | 直接メモリ操作で高速 |
| **学習コスト** | 高い（抽象概念が多い） | 低い（直感的） |

---

## アンチパターン

### 1. 過度なポイントフリースタイル

```typescript
// [NG] 引数名を省略しすぎて可読性が著しく低下
const process = pipe(
  filter(propEq("active", true)),
  map(pick(["id", "name"])),
  sortBy(prop("name"))
);
// 何を処理しているか読み取りにくい
// デバッグ時にどのステップで問題が起きたか追いにくい

// [OK] 適度な命名で意図を明確にする
const getActiveUserNames = (users: User[]) =>
  users
    .filter(u => u.active)
    .map(u => ({ id: u.id, name: u.name }))
    .sort((a, b) => a.name.localeCompare(b.name));
// 引数名があるので何を処理しているか一目瞭然
```

### 2. 不適切なメモ化

```typescript
// [NG] 副作用のある関数をメモ化
const badMemo = memoize((url: string) => {
  console.log(`Fetching: ${url}`);  // 副作用!
  return fetch(url).then(r => r.json());
});
// 2回目の呼び出しで console.log が実行されない → ログが不完全に

// [NG] 引数空間が巨大な関数をメモ化
const badMemo2 = memoize((data: number[]) => {
  return data.reduce((a, b) => a + b, 0);
});
// 異なる配列が毎回渡される → キャッシュが効かずメモリだけ消費

// [OK] 純粋関数をメモ化 + LRU でメモリ制限
const goodMemo = memoizeLRU(
  (n: number): number => {
    // 重い計算（副作用なし）
    return fibonacci(n);
  },
  1000 // 最大1000エントリ
);
```

### 3. 不必要な不変性強制

```typescript
// [NG] ループ内で毎回新しい配列を作成（パフォーマンス問題）
function buildLargeArray(n: number): number[] {
  let result: number[] = [];
  for (let i = 0; i < n; i++) {
    result = [...result, i]; // O(n) のコピーが n 回 = O(n²)
  }
  return result;
}

// [OK] ローカルスコープ内ではミュータブルを許可し、結果を不変として返す
function buildLargeArrayGood(n: number): readonly number[] {
  const result: number[] = []; // ローカルでミュータブル
  for (let i = 0; i < n; i++) {
    result.push(i); // O(1) の push が n 回 = O(n)
  }
  return result; // 不変な readonly として返す
}
```

---

## 実践演習

### 演習1（基礎）: カリー化とパイプラインの実装

**課題**: 以下の要件を満たす関数を実装してください。

1. `curry2` と `curry3` を実装する
2. `pipe` 関数を実装する（2〜5ステップ）
3. カリー化した関数をパイプラインで合成して、文字列変換パイプラインを作る

```typescript
// TODO: curry2, curry3 を実装

// TODO: pipe を実装

// テストケース
const trim = (s: string) => s.trim();
const toLower = (s: string) => s.toLowerCase();
const addPrefix = curry2((prefix: string, s: string) => `${prefix}${s}`);

const normalizeEmail = pipe(trim, toLower, addPrefix("mailto:"));

console.log(normalizeEmail("  TARO@Example.COM  "));
// "mailto:taro@example.com"
```

**期待される出力**:

```
normalizeEmail("  TARO@Example.COM  ") = "mailto:taro@example.com"
```

### 演習2（応用）: レンズによる不変データ更新

**課題**: レンズパターンを使って、ネストしたデータ構造の不変更新を実装してください。

要件:
1. `lens`, `composeLens`, `over`, `prop` を実装する
2. 以下のデータ構造に対して、レンズを使った操作を行う
3. 元のオブジェクトが変更されていないことをテストで確認する

```typescript
interface Company {
  name: string;
  ceo: {
    name: string;
    address: {
      city: string;
      country: string;
    };
  };
  employees: number;
}

// TODO: レンズを実装

// テストケース
const company: Company = {
  name: "TechCorp",
  ceo: {
    name: "Taro Yamada",
    address: { city: "Tokyo", country: "Japan" },
  },
  employees: 500,
};

// CEO の住所の都市名を更新
const updated = setL(ceoCityLens, "Osaka", company);
console.log(updated.ceo.address.city); // "Osaka"
console.log(company.ceo.address.city); // "Tokyo" ← 元は変更されていない
```

**期待される出力**:

```
更新後: Osaka
元データ: Tokyo（変更されていない）
over で大文字化: TOKYO
```

### 演習3（発展）: Railway Oriented Programming の実装

**課題**: Result 型と Railway パイプラインを実装し、EC サイトの注文処理フローを構築してください。

要件:
1. `Result<E, A>` 型と `ok`, `err`, `flatMapR` を実装
2. 以下の処理ステップを Result を返す関数として実装:
   - `validateOrder`: 注文内容の検証
   - `checkStock`: 在庫確認
   - `calculatePrice`: 価格計算（割引適用）
   - `processPayment`: 支払い処理
   - `createShipment`: 配送手配
3. railway パイプラインで全ステップを合成
4. 各ステップでのエラーが適切に伝播されることを確認

```typescript
// TODO: Result 型と Railway パイプラインを実装

// テストケース
const result1 = processOrder({
  items: [{ productId: "p1", quantity: 2 }],
  paymentMethod: "credit_card",
  shippingAddress: "Tokyo",
});
// 期待: Ok({ orderId: "...", status: "shipped", ... })

const result2 = processOrder({
  items: [],
  paymentMethod: "credit_card",
  shippingAddress: "Tokyo",
});
// 期待: Err({ step: "validation", message: "注文に商品が含まれていません" })
```

**期待される出力**:

```
テスト1 (正常注文):
  Ok: { orderId: "order_...", status: "shipped", total: 5000 }

テスト2 (空の注文):
  Err: { step: "validation", message: "注文に商品が含まれていません" }

テスト3 (在庫不足):
  Err: { step: "stock", message: "商品 p99 の在庫が不足しています" }
```

---

## FAQ

### Q1: カリー化は JavaScript/TypeScript で実用的ですか？

**A**: 部分適用は非常に実用的ですが、フルカリー化は TypeScript の型推論と相性が悪い場合があります。実用的なアプローチとして:

- `lodash/fp` や `ramda` の `curry` を使う
- arrow function での手動部分適用: `const double = multiply(2);`
- TypeScript 5.x 以降では const type parameter の改善により、カリー化の型推論が向上

ポイントは「必要な箇所だけ部分適用する」ことです。全ての関数をカリー化する必要はありません。

### Q2: パイプライン演算子 (`|>`) は使えますか？

**A**: JavaScript の TC39 提案（Stage 2）として進行中ですが、2026年時点では標準化されていません。代替手段:

1. `pipe()` ユーティリティ関数（fp-ts, ramda 等）
2. メソッドチェーン（Array.map().filter() 等）
3. Babel プラグインでの先行使用（本番環境では非推奨）

### Q3: Immutable.js や Immer は必要ですか？

**A**: プロジェクトの規模と複雑さに依存します:

- **小規模（浅いネスト）**: スプレッド構文で十分
- **中規模（Redux 使用）**: Immer が推奨（Redux Toolkit に内蔵）
- **大規模（深いネスト、頻繁な更新）**: レンズパターンまたは Immer
- **Immutable.js**: 構造共有による効率的な不変データ。ただし plain JS との変換コストがある

### Q4: 関数型プログラミングはパフォーマンスが悪いのでは？

**A**: 一般的なWebアプリケーションではパフォーマンスの差は無視できるレベルです。不変更新はオブジェクトのコピーコストがありますが、以下の最適化が可能です:

- **構造共有**: Immer や Immutable.js は変更されない部分を共有
- **メモ化**: 再計算を避けて高速化
- **遅延評価**: 必要になるまで計算を遅延
- **トランスデューサー**: 中間配列の削減

パフォーマンスが本当に問題になるのは、大量データの繰り返し処理やリアルタイム処理の場合のみです。

### Q5: 関数型パターンを段階的に導入するには？

**A**: 以下の順序で段階的に導入するのが効果的です:

1. **不変性**: `const` の使用、スプレッド構文、`Array.map/filter/reduce`
2. **純粋関数**: 副作用のない関数を意識的に書く
3. **パイプライン**: データ変換をパイプラインとして記述
4. **Result/Option**: null チェックやエラーハンドリングの型安全化
5. **レンズ/メモ化**: 必要に応じて高度なパターンを導入

一気に全てを導入する必要はありません。

### Q6: OOP と関数型プログラミングは共存できますか？

**A**: はい、多くの現代的な言語（TypeScript, Kotlin, Scala, Rust）は両方のパラダイムをサポートしています。実践的な指針:

- **データの変換**: 関数型（map, filter, pipe）
- **状態のカプセル化**: OOP（クラス、モジュール）
- **副作用の管理**: 関数型（純粋関数 + 副作用の分離）
- **抽象化**: 両方（インターフェース + 高階関数）

---

## まとめ

| 項目 | 要点 |
|---|---|
| カリー化 | 引数を1つずつ受け取る関数に変換。部分適用で再利用性向上 |
| 部分適用 | 一部の引数を固定した新しい関数を生成。設定値の注入に便利 |
| パイプライン | データ変換を宣言的に記述。可読性と保守性の向上 |
| compose | 右から左への関数合成。数学的な合成順序 |
| pipe | 左から右への関数合成。データの流れが自然 |
| Railway | Result 型を使ったエラーハンドリングパイプライン |
| レンズ | 不変データ構造の部分的な読み書きを合成可能に |
| メモ化 | 純粋関数の計算結果をキャッシュ。再計算を回避 |
| トランスデューサー | 中間配列なしの変換合成。メモリ効率と速度の改善 |
| ミドルウェア | 関数合成による横断的関心事の分離 |
| 実践指針 | 可読性とのバランスが重要。段階的導入がベスト |

---

## 次に読むべきガイド

- [モナド](./00-monad.md) — 関数合成のより高度な抽象化
- [ファンクタ・アプリカティブ](./01-functor-applicative.md) — map と ap の理論的基盤
- [クリーンコードの原則](../../clean-code-principles/docs/00-principles/) — 読みやすいコードの基本
- [ビヘイビアパターン](../02-behavioral/) — OOP パターンとの比較
- [アーキテクチャパターン](../04-architectural/) — 大規模設計での関数型アプローチ

---

## 参考文献

1. **Eric Elliott**: [Composing Software](https://medium.com/javascript-scene/composing-software-an-introduction-27b72500d6ea) — JavaScript での関数型プログラミングの実践的シリーズ
2. **Brian Lonsdorf**: [Professor Frisby's Mostly Adequate Guide](https://mostly-adequate.gitbook.io/mostly-adequate-guide/) — 関数型プログラミングの入門ガイド（JavaScript ベース）
3. **Ramda Documentation**: [Ramda](https://ramdajs.com/) — JavaScript の関数型ユーティリティライブラリ。カリー化とパイプラインの実践例が豊富
4. **Giulio Canti**: [fp-ts](https://gcanti.github.io/fp-ts/) — TypeScript の本格的な関数型プログラミングライブラリ
5. **Scott Wlaschin**: [Railway Oriented Programming](https://fsharpforfunandprofit.com/rop/) — エラーハンドリングの関数型アプローチ。F# だが概念は言語非依存
