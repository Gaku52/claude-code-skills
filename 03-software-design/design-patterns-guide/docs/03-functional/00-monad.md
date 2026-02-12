# モナド (Monad)

> Maybe/Either、IO、Promise などのモナドパターンを理解し、副作用を制御しつつ合成可能なデータフローを構築する

## この章で学ぶこと

1. **モナドの本質** --- bind (flatMap) による計算の連鎖と、コンテキスト内での値の変換がなぜ必要か
2. **実用的なモナド** --- Maybe/Option、Either/Result、Promise/Future、IO、List、Reader の具体的実装と活用
3. **モナド則** --- 3つの法則（左単位元・右単位元・結合則）の意味と検証方法
4. **鉄道指向プログラミング** --- Either/Result を用いたエラーハンドリングパイプラインの設計
5. **モナド合成** --- モナドトランスフォーマー、do 記法、async/await との関係

---

## 前提知識

| トピック | 必要レベル | 参照先 |
|---|---|---|
| TypeScript ジェネリクス | クラス・関数のジェネリクス定義 | [TypeScript Handbook](https://www.typescriptlang.org/docs/handbook/2/generics.html) |
| 高階関数 | map, filter, reduce の理解 | [関数型パターン](./02-fp-patterns.md) |
| Promise/async-await | 非同期処理の基礎 | [MDN Promise](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise) |
| 代数的データ型 | Union 型、判別共用体 | [TypeScript Union Types](https://www.typescriptlang.org/docs/handbook/2/narrowing.html) |
| ファンクタの概念 | map の意味 | [ファンクタ・アプリカティブ](./01-functor-applicative.md) |

---

## WHY --- なぜモナドが必要か

### 問題: 副作用と分岐の爆発

実際のプログラムでは、あらゆる操作が「失敗するかもしれない」「値がないかもしれない」「非同期かもしれない」というコンテキストを伴う。これらを愚直に扱うと、null チェック・エラーハンドリング・コールバックのネストが爆発する。

```
問題: コンテキスト付き計算の連鎖
====================================

[素朴な実装 --- ネストの爆発]

  getUser(id) ──→ null チェック ──→ getAddress(user) ──→ null チェック ──→ getCity(addr)
                    |                                      |
                    v                                      v
                  return default                        return default

  fetchUser(id) ──→ try/catch ──→ fetchOrders(user) ──→ try/catch ──→ process(orders)
                      |                                    |
                      v                                    v
                    handleError                          handleError

  // ネスト地獄
  if (user != null) {
    if (user.address != null) {
      if (user.address.city != null) {
        // ようやく本来のロジック
      }
    }
  }

[モナドによる解決 --- フラットな合成]

  Maybe.of(user)
    .flatMap(u => Maybe.fromNullable(u.address))
    .flatMap(a => Maybe.fromNullable(a.city))
    .getOrElse("Unknown")

  // 各ステップで「失敗したら自動スキップ」
  // ネストなし。直線的。合成可能。
```

### 解決: コンテキスト内での合成可能な計算

モナドは **コンテキスト(文脈)付きの値** に対する **合成可能な計算の連鎖** を提供する抽象化である。

- **Maybe/Option**: 「値がないかもしれない」コンテキスト。null ならスキップ
- **Either/Result**: 「失敗するかもしれない」コンテキスト。エラーならスキップ
- **Promise/Future**: 「まだ完了していない」コンテキスト。完了後に次を実行
- **IO**: 「副作用がある」コンテキスト。実行を遅延して合成
- **List/Array**: 「複数の値がある」コンテキスト。各要素に適用してフラットに
- **Reader**: 「環境に依存する」コンテキスト。依存を引き回す

すべてのモナドは同じインターフェース (`unit` + `bind`) を持ち、同じ法則に従う。この統一性により、異なるコンテキストの計算を同じパターンで扱える。

---

## モナドの定義

Philip Wadler (1992) "Monads for functional programming" より:

> A monad is a triple (M, unit, bind) where M is a type constructor, unit lifts a value into the monadic context, and bind sequences computations within the context.

日本語訳: モナドは三つ組 (M, unit, bind) であり、M は型コンストラクタ、unit は値をモナドコンテキストに持ち上げ、bind はコンテキスト内で計算を連鎖させる。

```
モナドの構成要素
==================

  型コンストラクタ M:
    通常の型 T を「コンテキスト付きの型」M<T> に変換する

  unit (return, of, pure):
    T  --->  M<T>
    値をコンテキストに入れる。副作用なし。

  bind (flatMap, >>=, then, and_then):
    M<T>  --->  (T -> M<U>)  --->  M<U>
    コンテキスト内の値を取り出し、次の計算に渡し、結果をフラットにする。

  ※ map との違い:
    map  : M<T> -> (T ->   U ) -> M<U>    (値を変換)
    bind : M<T> -> (T -> M<U>) -> M<U>    (コンテキストを変換)

  map は bind + unit で定義できる:
    m.map(f) === m.bind(x => unit(f(x)))

モナド則 (Monad Laws)
=======================

  1. 左単位元 (Left Identity):
     unit(a).bind(f) === f(a)
     「値をコンテキストに入れてから bind」= 「直接関数に渡す」

  2. 右単位元 (Right Identity):
     m.bind(unit) === m
     「コンテキストの値を取り出して unit で戻す」= 「何もしない」

  3. 結合則 (Associativity):
     m.bind(f).bind(g) === m.bind(x => f(x).bind(g))
     「左から順に bind」= 「入れ子にして bind」

  図解:
    値 a  --[unit]--> M<a>  --[bind(f)]--> M<b>  --[bind(g)]--> M<c>

  これらの法則により、モナドは合成の順序に依存しない
  一貫した振る舞いを保証する。
```

---

## コード例 1: Maybe/Option モナド --- 完全実装

```typescript
// =============================================================
// Maybe モナド: null 安全な計算の連鎖
// =============================================================

class Maybe<T> {
  private constructor(private readonly value: T | null) {}

  // --- unit (return, of) ---
  static of<T>(value: T): Maybe<T> {
    return new Maybe(value);
  }

  static nothing<T>(): Maybe<T> {
    return new Maybe<T>(null);
  }

  static fromNullable<T>(value: T | null | undefined): Maybe<T> {
    return value == null ? Maybe.nothing() : Maybe.of(value);
  }

  // --- map (Functor) ---
  map<U>(fn: (value: T) => U): Maybe<U> {
    return this.value == null ? Maybe.nothing() : Maybe.of(fn(this.value));
  }

  // --- bind (flatMap) ---
  flatMap<U>(fn: (value: T) => Maybe<U>): Maybe<U> {
    return this.value == null ? Maybe.nothing() : fn(this.value);
  }

  // --- apply (Applicative) ---
  apply<U>(maybeFn: Maybe<(value: T) => U>): Maybe<U> {
    return maybeFn.flatMap(fn => this.map(fn));
  }

  // --- ユーティリティ ---
  getOrElse(defaultValue: T): T {
    return this.value ?? defaultValue;
  }

  getOrThrow(message: string): T {
    if (this.value == null) throw new Error(message);
    return this.value;
  }

  filter(predicate: (value: T) => boolean): Maybe<T> {
    if (this.value == null) return Maybe.nothing();
    return predicate(this.value) ? this : Maybe.nothing();
  }

  isNothing(): boolean {
    return this.value == null;
  }

  isJust(): boolean {
    return this.value != null;
  }

  // パターンマッチ
  match<U>(patterns: { just: (value: T) => U; nothing: () => U }): U {
    return this.value != null ? patterns.just(this.value) : patterns.nothing();
  }

  // デバッグ用 (副作用を挿入しつつチェーンを維持)
  tap(fn: (value: T) => void): Maybe<T> {
    if (this.value != null) fn(this.value);
    return this;
  }

  toString(): string {
    return this.value != null ? `Just(${this.value})` : "Nothing";
  }
}

// =============================================================
// モナド則の検証
// =============================================================
const f = (x: number) => Maybe.of(x * 2);
const g = (x: number) => (x > 0 ? Maybe.of(x + 1) : Maybe.nothing<number>());

// 左単位元: unit(a).bind(f) === f(a)
console.log(Maybe.of(5).flatMap(f).toString());  // Just(10)
console.log(f(5).toString());                     // Just(10)

// 右単位元: m.bind(unit) === m
console.log(Maybe.of(5).flatMap(Maybe.of).toString());  // Just(5)
console.log(Maybe.of(5).toString());                     // Just(5)

// 結合則: m.bind(f).bind(g) === m.bind(x => f(x).bind(g))
const m = Maybe.of(3);
console.log(m.flatMap(f).flatMap(g).toString());              // Just(7)
console.log(m.flatMap(x => f(x).flatMap(g)).toString());      // Just(7)

// =============================================================
// 実践的な使用例: ネストされたオブジェクトの安全な探索
// =============================================================
interface Company {
  name: string;
  ceo?: {
    name: string;
    address?: {
      city?: string;
      country?: string;
    };
  };
}

function getCeoCity(company: Company | null): string {
  return Maybe.fromNullable(company)
    .flatMap(c => Maybe.fromNullable(c.ceo))
    .flatMap(ceo => Maybe.fromNullable(ceo.address))
    .flatMap(addr => Maybe.fromNullable(addr.city))
    .getOrElse("Unknown");
}

// テスト
const company1: Company = {
  name: "Acme",
  ceo: { name: "Alice", address: { city: "Tokyo", country: "JP" } }
};
const company2: Company = { name: "Beta" };
const company3: Company = { name: "Gamma", ceo: { name: "Bob" } };

console.log(getCeoCity(company1));  // "Tokyo"
console.log(getCeoCity(company2));  // "Unknown" (ceo なし)
console.log(getCeoCity(company3));  // "Unknown" (address なし)
console.log(getCeoCity(null));      // "Unknown" (company なし)

// =============================================================
// Maybe + 配列: 安全な検索
// =============================================================
function findFirst<T>(
  items: T[],
  predicate: (item: T) => boolean
): Maybe<T> {
  const found = items.find(predicate);
  return Maybe.fromNullable(found);
}

interface Product {
  id: number;
  name: string;
  discount?: number;
}

const products: Product[] = [
  { id: 1, name: "Widget", discount: 0.1 },
  { id: 2, name: "Gadget" },
  { id: 3, name: "Doohickey", discount: 0.25 },
];

// 割引価格を安全に計算
function getDiscountedPrice(
  products: Product[],
  productId: number,
  basePrice: number
): Maybe<number> {
  return findFirst(products, p => p.id === productId)
    .flatMap(product => Maybe.fromNullable(product.discount))
    .map(discount => basePrice * (1 - discount));
}

console.log(getDiscountedPrice(products, 1, 100).toString());  // Just(90)
console.log(getDiscountedPrice(products, 2, 100).toString());  // Nothing (割引なし)
console.log(getDiscountedPrice(products, 9, 100).toString());  // Nothing (商品なし)
```

---

## コード例 2: Either/Result モナド --- 鉄道指向プログラミング

```typescript
// =============================================================
// Either モナド: 型安全なエラーハンドリング
// =============================================================

type Either<L, R> =
  | { readonly tag: "Left"; readonly value: L }
  | { readonly tag: "Right"; readonly value: R };

// --- コンストラクタ ---
const Left = <L, R = never>(value: L): Either<L, R> => ({
  tag: "Left",
  value,
});
const Right = <R, L = never>(value: R): Either<L, R> => ({
  tag: "Right",
  value,
});

// --- モナド操作 (関数スタイル) ---
const EitherM = {
  // unit
  of<R>(value: R): Either<never, R> {
    return Right(value);
  },

  // map (Functor)
  map<L, R, U>(either: Either<L, R>, fn: (r: R) => U): Either<L, U> {
    return either.tag === "Right" ? Right(fn(either.value)) : either;
  },

  // bind (flatMap)
  flatMap<L, R, U>(
    either: Either<L, R>,
    fn: (r: R) => Either<L, U>
  ): Either<L, U> {
    return either.tag === "Right" ? fn(either.value) : either;
  },

  // mapLeft: エラー側を変換
  mapLeft<L, R, U>(either: Either<L, R>, fn: (l: L) => U): Either<U, R> {
    return either.tag === "Left" ? Left(fn(either.value)) : either;
  },

  // bimap: 両側を変換
  bimap<L, R, U, V>(
    either: Either<L, R>,
    leftFn: (l: L) => U,
    rightFn: (r: R) => V
  ): Either<U, V> {
    return either.tag === "Left"
      ? Left(leftFn(either.value))
      : Right(rightFn(either.value));
  },

  // fold (パターンマッチ)
  fold<L, R, U>(
    either: Either<L, R>,
    onLeft: (l: L) => U,
    onRight: (r: R) => U
  ): U {
    return either.tag === "Left"
      ? onLeft(either.value)
      : onRight(either.value);
  },

  // tryCatch: 例外を Either に変換
  tryCatch<R>(fn: () => R): Either<Error, R> {
    try {
      return Right(fn());
    } catch (e) {
      return Left(e instanceof Error ? e : new Error(String(e)));
    }
  },

  // fromNullable: null/undefined を Left に変換
  fromNullable<L, R>(
    value: R | null | undefined,
    error: L
  ): Either<L, R> {
    return value == null ? Left(error) : Right(value);
  },

  // 複数の Either を結合 (すべて Right なら成功)
  sequence<L, R>(eithers: Either<L, R>[]): Either<L, R[]> {
    const results: R[] = [];
    for (const either of eithers) {
      if (either.tag === "Left") return either;
      results.push(either.value);
    }
    return Right(results);
  },
};

// =============================================================
// 鉄道指向プログラミング: バリデーションパイプライン
// =============================================================

// 構造化エラー型
interface ValidationError {
  field: string;
  message: string;
  code: string;
}

const vError = (
  field: string,
  message: string,
  code: string
): ValidationError => ({ field, message, code });

// 各バリデーション関数は Either を返す
function validateEmail(
  email: string
): Either<ValidationError, string> {
  if (!email.includes("@")) {
    return Left(vError("email", "Must contain @", "INVALID_EMAIL"));
  }
  if (email.length < 5) {
    return Left(vError("email", "Too short", "EMAIL_TOO_SHORT"));
  }
  return Right(email.toLowerCase().trim());
}

function validatePassword(
  password: string
): Either<ValidationError, string> {
  if (password.length < 8) {
    return Left(vError("password", "Min 8 chars", "PASSWORD_TOO_SHORT"));
  }
  if (!/[A-Z]/.test(password)) {
    return Left(vError("password", "Need uppercase", "PASSWORD_NO_UPPER"));
  }
  if (!/[0-9]/.test(password)) {
    return Left(vError("password", "Need digit", "PASSWORD_NO_DIGIT"));
  }
  return Right(password);
}

function validateAge(age: number): Either<ValidationError, number> {
  if (!Number.isInteger(age)) {
    return Left(vError("age", "Must be integer", "AGE_NOT_INTEGER"));
  }
  if (age < 13 || age > 120) {
    return Left(vError("age", "Must be 13-120", "AGE_OUT_OF_RANGE"));
  }
  return Right(age);
}

function validateUsername(
  name: string
): Either<ValidationError, string> {
  if (name.length < 3) {
    return Left(vError("username", "Min 3 chars", "USERNAME_TOO_SHORT"));
  }
  if (!/^[a-zA-Z0-9_]+$/.test(name)) {
    return Left(vError("username", "Alphanumeric only", "USERNAME_INVALID"));
  }
  return Right(name);
}

// --- 鉄道: 最初のエラーで停止 ---
interface RegistrationInput {
  username: string;
  email: string;
  password: string;
  age: number;
}

interface ValidatedUser {
  username: string;
  email: string;
  password: string;
  age: number;
}

function validateRegistration(
  input: RegistrationInput
): Either<ValidationError, ValidatedUser> {
  // 各ステップが Left を返した時点でパイプライン全体が Left になる
  const username = validateUsername(input.username);
  const email = EitherM.flatMap(username, () => validateEmail(input.email));
  const password = EitherM.flatMap(email, () => validatePassword(input.password));
  const age = EitherM.flatMap(password, () => validateAge(input.age));

  return EitherM.map(age, () => ({
    username: input.username,
    email: input.email.toLowerCase().trim(),
    password: input.password,
    age: input.age,
  }));
}

// テスト
const good = validateRegistration({
  username: "alice",
  email: "alice@example.com",
  password: "Secret123",
  age: 25,
});
console.log(good);
// { tag: "Right", value: { username: "alice", email: "alice@example.com", ... } }

const bad = validateRegistration({
  username: "al",
  email: "alice@example.com",
  password: "Secret123",
  age: 25,
});
console.log(bad);
// { tag: "Left", value: { field: "username", message: "Min 3 chars", code: "USERNAME_TOO_SHORT" } }

// --- 鉄道: すべてのエラーを収集 (Validation モナド) ---
function validateAllErrors(
  input: RegistrationInput
): Either<ValidationError[], ValidatedUser> {
  const results = [
    EitherM.mapLeft(validateUsername(input.username), e => [e]),
    EitherM.mapLeft(validateEmail(input.email), e => [e]),
    EitherM.mapLeft(validatePassword(input.password), e => [e]),
    EitherM.mapLeft(validateAge(input.age), e => [e]),
  ];

  const errors: ValidationError[] = [];
  for (const result of results) {
    if (result.tag === "Left") {
      errors.push(...result.value);
    }
  }

  if (errors.length > 0) return Left(errors);

  return Right({
    username: input.username,
    email: input.email.toLowerCase().trim(),
    password: input.password,
    age: input.age,
  });
}

const allBad = validateAllErrors({
  username: "al",
  email: "bad",
  password: "weak",
  age: 5,
});
console.log(allBad);
// { tag: "Left", value: [
//   { field: "username", message: "Min 3 chars", ... },
//   { field: "email", message: "Must contain @", ... },
//   { field: "password", message: "Min 8 chars", ... },
//   { field: "age", message: "Must be 13-120", ... }
// ] }
```

---

## コード例 3: Result モナド --- TypeScript 型安全実装

```typescript
// =============================================================
// Result<T, E>: Rust スタイルの型安全エラーハンドリング
// =============================================================

class Result<T, E> {
  private constructor(
    private readonly _ok: boolean,
    private readonly _value: T | undefined,
    private readonly _error: E | undefined
  ) {}

  static ok<T, E = never>(value: T): Result<T, E> {
    return new Result<T, E>(true, value, undefined);
  }

  static err<E, T = never>(error: E): Result<T, E> {
    return new Result<T, E>(false, undefined, error);
  }

  static fromTryCatch<T>(fn: () => T): Result<T, Error> {
    try {
      return Result.ok(fn());
    } catch (e) {
      return Result.err(e instanceof Error ? e : new Error(String(e)));
    }
  }

  static async fromPromise<T>(promise: Promise<T>): Promise<Result<T, Error>> {
    try {
      return Result.ok(await promise);
    } catch (e) {
      return Result.err(e instanceof Error ? e : new Error(String(e)));
    }
  }

  isOk(): boolean {
    return this._ok;
  }

  isErr(): boolean {
    return !this._ok;
  }

  // --- Functor ---
  map<U>(fn: (value: T) => U): Result<U, E> {
    return this._ok
      ? Result.ok(fn(this._value as T))
      : Result.err(this._error as E);
  }

  mapErr<F>(fn: (error: E) => F): Result<T, F> {
    return this._ok
      ? Result.ok(this._value as T)
      : Result.err(fn(this._error as E));
  }

  // --- Monad ---
  flatMap<U>(fn: (value: T) => Result<U, E>): Result<U, E> {
    return this._ok
      ? fn(this._value as T)
      : Result.err(this._error as E);
  }

  // --- 値の取得 ---
  unwrap(): T {
    if (!this._ok) throw new Error("Called unwrap on Err");
    return this._value as T;
  }

  unwrapOr(defaultValue: T): T {
    return this._ok ? (this._value as T) : defaultValue;
  }

  unwrapOrElse(fn: (error: E) => T): T {
    return this._ok ? (this._value as T) : fn(this._error as E);
  }

  unwrapErr(): E {
    if (this._ok) throw new Error("Called unwrapErr on Ok");
    return this._error as E;
  }

  // --- パターンマッチ ---
  match<U>(patterns: { ok: (value: T) => U; err: (error: E) => U }): U {
    return this._ok
      ? patterns.ok(this._value as T)
      : patterns.err(this._error as E);
  }

  // --- 合成 ---
  and<U>(other: Result<U, E>): Result<U, E> {
    return this._ok ? other : Result.err(this._error as E);
  }

  or(other: Result<T, E>): Result<T, E> {
    return this._ok ? this : other;
  }

  // --- 変換 ---
  toMaybe(): Maybe<T> {
    return this._ok ? Maybe.of(this._value as T) : Maybe.nothing();
  }

  toString(): string {
    return this._ok
      ? `Ok(${JSON.stringify(this._value)})`
      : `Err(${JSON.stringify(this._error)})`;
  }
}

// =============================================================
// 実践例: JSON パース -> バリデーション -> ビジネスロジック
// =============================================================

interface Config {
  host: string;
  port: number;
  maxRetries: number;
}

function parseJson(raw: string): Result<unknown, string> {
  return Result.fromTryCatch(() => JSON.parse(raw))
    .mapErr(e => `JSON parse error: ${e.message}`);
}

function validateConfig(data: unknown): Result<Config, string> {
  if (typeof data !== "object" || data === null) {
    return Result.err("Config must be an object");
  }
  const obj = data as Record<string, unknown>;
  if (typeof obj.host !== "string") {
    return Result.err("host must be a string");
  }
  if (typeof obj.port !== "number" || obj.port < 1 || obj.port > 65535) {
    return Result.err("port must be 1-65535");
  }
  if (typeof obj.maxRetries !== "number" || obj.maxRetries < 0) {
    return Result.err("maxRetries must be non-negative");
  }
  return Result.ok({
    host: obj.host as string,
    port: obj.port as number,
    maxRetries: obj.maxRetries as number,
  });
}

function normalizeConfig(config: Config): Result<Config, string> {
  if (config.host === "localhost") {
    return Result.ok({ ...config, host: "127.0.0.1" });
  }
  if (config.host.length === 0) {
    return Result.err("host cannot be empty");
  }
  return Result.ok(config);
}

// flatMap の連鎖 = 鉄道指向パイプライン
function loadConfig(raw: string): Result<Config, string> {
  return parseJson(raw)
    .flatMap(validateConfig)
    .flatMap(normalizeConfig);
}

// テスト
console.log(loadConfig('{"host":"localhost","port":8080,"maxRetries":3}').toString());
// Ok({"host":"127.0.0.1","port":8080,"maxRetries":3})

console.log(loadConfig('{"host":"","port":8080,"maxRetries":3}').toString());
// Err("host cannot be empty")

console.log(loadConfig("not json").toString());
// Err("JSON parse error: ...")

console.log(loadConfig('{"host":"example.com","port":99999,"maxRetries":3}').toString());
// Err("port must be 1-65535")
```

---

## コード例 4: IO モナドと Promise

```typescript
// =============================================================
// IO モナド: 副作用の遅延実行と合成
// =============================================================

// IO は「副作用を持つ計算」を値として扱う
// 実行されるまで副作用は発生しない (参照透過性の維持)
class IO<T> {
  constructor(private readonly effect: () => T) {}

  // --- unit ---
  static of<T>(value: T): IO<T> {
    return new IO(() => value);
  }

  // --- Functor ---
  map<U>(fn: (value: T) => U): IO<U> {
    return new IO(() => fn(this.effect()));
  }

  // --- Monad ---
  flatMap<U>(fn: (value: T) => IO<U>): IO<U> {
    return new IO(() => fn(this.effect()).run());
  }

  // 副作用を実行
  run(): T {
    return this.effect();
  }

  // 2つの IO を順次実行
  andThen<U>(next: IO<U>): IO<U> {
    return this.flatMap(() => next);
  }

  // デバッグ用
  tap(fn: (value: T) => void): IO<T> {
    return new IO(() => {
      const result = this.effect();
      fn(result);
      return result;
    });
  }
}

// --- IO ユーティリティ ---
const ConsoleIO = {
  log(message: string): IO<void> {
    return new IO(() => console.log(message));
  },
  readLine(prompt: string): IO<string> {
    // 実際の readline 実装は省略
    return new IO(() => {
      console.log(prompt);
      return "simulated-input";
    });
  },
  currentTime(): IO<Date> {
    return new IO(() => new Date());
  },
  randomInt(min: number, max: number): IO<number> {
    return new IO(() => Math.floor(Math.random() * (max - min + 1)) + min);
  },
};

// =============================================================
// IO の合成: プログラム全体を純粋に記述
// =============================================================

// このプログラム定義自体は純粋 (副作用なし)
const greetProgram: IO<void> = ConsoleIO.currentTime()
  .map(date => date.toISOString())
  .flatMap(time =>
    ConsoleIO.log(`[${time}] Hello!`)
      .andThen(ConsoleIO.log(`[${time}] Welcome to IO Monad`))
  );

// run() を呼ぶまで何も起きない
// greetProgram.run();  // 実行時にのみ副作用が発生

// =============================================================
// 比較: IO vs 直接副作用
// =============================================================

// [NG] 直接副作用 --- テスト困難
function greetDirect(): void {
  const time = new Date().toISOString();  // 副作用: 現在時刻
  console.log(`[${time}] Hello!`);        // 副作用: コンソール出力
  console.log(`[${time}] Welcome`);       // 副作用: コンソール出力
}

// [OK] IO モナド --- テスト可能
function greetIO(
  getTime: () => IO<Date>,
  log: (msg: string) => IO<void>
): IO<void> {
  return getTime()
    .map(date => date.toISOString())
    .flatMap(time =>
      log(`[${time}] Hello!`)
        .andThen(log(`[${time}] Welcome`))
    );
}

// テスト時: モック IO を注入
const logs: string[] = [];
const mockGetTime = () => IO.of(new Date("2025-01-01T00:00:00Z"));
const mockLog = (msg: string) => new IO(() => { logs.push(msg); });

greetIO(mockGetTime, mockLog).run();
console.log(logs);
// ["[2025-01-01T00:00:00.000Z] Hello!", "[2025-01-01T00:00:00.000Z] Welcome"]

// =============================================================
// Promise = 非同期 IO モナド
// async/await = do 記法の糖衣構文
// =============================================================

// Promise のモナド的性質
// then = flatMap (bind)
// Promise.resolve = unit

// [モナディックスタイル (then チェーン)]
function fetchUserOrdersChain(userId: string): Promise<OrderSummary> {
  return fetchUser(userId)                                // Promise<User>
    .then(user => fetchOrders(user.id))                   // flatMap: User -> Promise<Order[]>
    .then(orders => orders.filter(o => o.active))         // map: Order[] -> Order[]
    .then(orders => ({
      userId,
      totalOrders: orders.length,
      totalAmount: orders.reduce((sum, o) => sum + o.amount, 0),
    }));
}

// [do 記法スタイル (async/await)]
async function fetchUserOrdersAwait(userId: string): Promise<OrderSummary> {
  const user = await fetchUser(userId);                    // bind
  const orders = await fetchOrders(user.id);               // bind
  const activeOrders = orders.filter(o => o.active);       // 純粋計算

  return {                                                 // unit (return)
    userId,
    totalOrders: activeOrders.length,
    totalAmount: activeOrders.reduce((sum, o) => sum + o.amount, 0),
  };
}

// 型定義 (上記コード用)
interface OrderSummary {
  userId: string;
  totalOrders: number;
  totalAmount: number;
}

declare function fetchUser(id: string): Promise<{ id: string; name: string }>;
declare function fetchOrders(userId: string): Promise<Array<{ active: boolean; amount: number }>>;
```

---

## コード例 5: List モナドと Reader モナド

```typescript
// =============================================================
// List モナド: 非決定性計算
// flatMap = 各要素に関数を適用してフラットに
// =============================================================

// Array は実は List モナド
// Array.prototype.flatMap が bind に相当

// 例: チェスのナイトの移動
type Position = [number, number];

function knightMoves([x, y]: Position): Position[] {
  const deltas: Position[] = [
    [2, 1], [2, -1], [-2, 1], [-2, -1],
    [1, 2], [1, -2], [-1, 2], [-1, -2],
  ];
  return deltas
    .map(([dx, dy]) => [x + dx, y + dy] as Position)
    .filter(([nx, ny]) => nx >= 1 && nx <= 8 && ny >= 1 && ny <= 8);
}

// 3手後に到達可能なすべての位置 (List モナドの flatMap)
function knightReachableIn3(start: Position): Position[] {
  return [start]
    .flatMap(knightMoves)   // 1手目: すべての可能な位置
    .flatMap(knightMoves)   // 2手目: さらにすべての可能な位置
    .flatMap(knightMoves);  // 3手目: さらに
}

console.log(`(1,1) から3手で到達可能: ${knightReachableIn3([1, 1]).length} 通り`);

// List Comprehension としての flatMap
// Haskell: [(x, y) | x <- [1..3], y <- [1..3], x /= y]
// Python:  [(x, y) for x in range(1,4) for y in range(1,4) if x != y]
const pairs = [1, 2, 3]
  .flatMap(x => [1, 2, 3]
    .flatMap(y => x !== y ? [[x, y]] : [])
  );
console.log(pairs); // [[1,2],[1,3],[2,1],[2,3],[3,1],[3,2]]

// =============================================================
// Reader モナド: 依存性注入
// =============================================================

// Reader<E, A> は E -> A の関数をラップしたモナド
// 環境 E を暗黙的に引き回す
class Reader<E, A> {
  constructor(public readonly run: (env: E) => A) {}

  static of<E, A>(value: A): Reader<E, A> {
    return new Reader(() => value);
  }

  // 環境そのものを取得
  static ask<E>(): Reader<E, E> {
    return new Reader(env => env);
  }

  // 環境の一部を取得
  static asks<E, A>(fn: (env: E) => A): Reader<E, A> {
    return new Reader(fn);
  }

  map<B>(fn: (a: A) => B): Reader<E, B> {
    return new Reader(env => fn(this.run(env)));
  }

  flatMap<B>(fn: (a: A) => Reader<E, B>): Reader<E, B> {
    return new Reader(env => fn(this.run(env)).run(env));
  }
}

// 実践例: 設定と DB 接続を引き回す
interface AppEnv {
  db: { query: (sql: string) => string[] };
  config: { maxResults: number; locale: string };
  logger: { info: (msg: string) => void };
}

const getUsers: Reader<AppEnv, string[]> =
  Reader.asks((env: AppEnv) => env.db.query("SELECT * FROM users"));

const getMaxResults: Reader<AppEnv, number> =
  Reader.asks((env: AppEnv) => env.config.maxResults);

const getLocale: Reader<AppEnv, string> =
  Reader.asks((env: AppEnv) => env.config.locale);

// flatMap で Reader を合成 --- 環境は暗黙的に引き回される
const getUserReport: Reader<AppEnv, string> = getUsers
  .flatMap(users =>
    getMaxResults.flatMap(max =>
      getLocale.map(locale => {
        const limited = users.slice(0, max);
        return `[${locale}] Found ${limited.length} users (max: ${max})`;
      })
    )
  );

// 実行: 環境を一箇所で注入
const env: AppEnv = {
  db: { query: () => ["Alice", "Bob", "Charlie", "Dave", "Eve"] },
  config: { maxResults: 3, locale: "ja-JP" },
  logger: { info: console.log },
};

console.log(getUserReport.run(env));
// "[ja-JP] Found 3 users (max: 3)"

// テスト: モック環境で実行
const testEnv: AppEnv = {
  db: { query: () => ["TestUser"] },
  config: { maxResults: 10, locale: "en-US" },
  logger: { info: () => {} },
};

console.log(getUserReport.run(testEnv));
// "[en-US] Found 1 users (max: 10)"
```

---

## コード例 6: Python でのモナドパターン

```python
"""
Python でのモナド実装
=====================
Python は静的型が弱いため、Haskell や TypeScript ほど厳密な
モナド実装は一般的でないが、同じ概念は dataclass + Protocol で表現できる。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TypeVar, Generic, Callable, Optional, Union
from abc import ABC, abstractmethod

T = TypeVar("T")
U = TypeVar("U")
E = TypeVar("E")

# =============================================================
# Maybe モナド
# =============================================================

class Maybe(ABC, Generic[T]):
    """Maybe モナド: 値がないかもしれない計算"""

    @staticmethod
    def of(value: T) -> "Just[T]":
        return Just(value)

    @staticmethod
    def nothing() -> "Nothing[T]":
        return Nothing()

    @staticmethod
    def from_optional(value: Optional[T]) -> "Maybe[T]":
        if value is None:
            return Nothing()
        return Just(value)

    @abstractmethod
    def map(self, fn: Callable[[T], U]) -> "Maybe[U]": ...

    @abstractmethod
    def flat_map(self, fn: Callable[[T], "Maybe[U]"]) -> "Maybe[U]": ...

    @abstractmethod
    def get_or_else(self, default: T) -> T: ...

    @abstractmethod
    def is_just(self) -> bool: ...


@dataclass(frozen=True)
class Just(Maybe[T]):
    _value: T

    def map(self, fn: Callable[[T], U]) -> Maybe[U]:
        return Just(fn(self._value))

    def flat_map(self, fn: Callable[[T], Maybe[U]]) -> Maybe[U]:
        return fn(self._value)

    def get_or_else(self, default: T) -> T:
        return self._value

    def is_just(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"Just({self._value!r})"


@dataclass(frozen=True)
class Nothing(Maybe[T]):
    def map(self, fn: Callable[[T], U]) -> Maybe[U]:
        return Nothing()

    def flat_map(self, fn: Callable[[T], Maybe[U]]) -> Maybe[U]:
        return Nothing()

    def get_or_else(self, default: T) -> T:
        return default

    def is_just(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "Nothing()"


# 使用例
def safe_div(a: float, b: float) -> Maybe[float]:
    if b == 0:
        return Nothing()
    return Just(a / b)

def safe_sqrt(x: float) -> Maybe[float]:
    if x < 0:
        return Nothing()
    return Just(x ** 0.5)

# flatMap の連鎖
result = (
    safe_div(100, 4)          # Just(25.0)
    .flat_map(safe_sqrt)      # Just(5.0)
    .map(lambda x: x * 2)    # Just(10.0)
)
print(result)  # Just(10.0)

result_err = (
    safe_div(100, 0)          # Nothing()
    .flat_map(safe_sqrt)      # Nothing() (スキップ)
    .map(lambda x: x * 2)    # Nothing() (スキップ)
)
print(result_err)  # Nothing()


# =============================================================
# Result モナド
# =============================================================

class Result(ABC, Generic[T, E]):
    """Result モナド: 成功か失敗かの計算"""

    @staticmethod
    def ok(value: T) -> "Ok[T, E]":
        return Ok(value)

    @staticmethod
    def err(error: E) -> "Err[T, E]":
        return Err(error)

    @staticmethod
    def from_try(fn: Callable[[], T]) -> "Result[T, Exception]":
        try:
            return Ok(fn())
        except Exception as e:
            return Err(e)

    @abstractmethod
    def map(self, fn: Callable[[T], U]) -> "Result[U, E]": ...

    @abstractmethod
    def flat_map(self, fn: Callable[[T], "Result[U, E]"]) -> "Result[U, E]": ...

    @abstractmethod
    def map_err(self, fn: Callable[[E], object]) -> "Result[T, object]": ...

    @abstractmethod
    def unwrap_or(self, default: T) -> T: ...

    @abstractmethod
    def is_ok(self) -> bool: ...


@dataclass(frozen=True)
class Ok(Result[T, E]):
    _value: T

    def map(self, fn: Callable[[T], U]) -> Result[U, E]:
        return Ok(fn(self._value))

    def flat_map(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
        return fn(self._value)

    def map_err(self, fn: Callable[[E], object]) -> Result[T, object]:
        return Ok(self._value)

    def unwrap_or(self, default: T) -> T:
        return self._value

    def is_ok(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"Ok({self._value!r})"


@dataclass(frozen=True)
class Err(Result[T, E]):
    _error: E

    def map(self, fn: Callable[[T], U]) -> Result[U, E]:
        return Err(self._error)

    def flat_map(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
        return Err(self._error)

    def map_err(self, fn: Callable[[E], object]) -> Result[T, object]:
        return Err(fn(self._error))

    def unwrap_or(self, default: T) -> T:
        return default

    def is_ok(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"Err({self._error!r})"


# 実践例: ファイル処理パイプライン
import json

def read_file(path: str) -> Result[str, str]:
    try:
        with open(path) as f:
            return Ok(f.read())
    except FileNotFoundError:
        return Err(f"File not found: {path}")
    except PermissionError:
        return Err(f"Permission denied: {path}")

def parse_json(content: str) -> Result[dict, str]:
    try:
        return Ok(json.loads(content))
    except json.JSONDecodeError as e:
        return Err(f"JSON error: {e}")

def extract_field(data: dict, field: str) -> Result[object, str]:
    if field not in data:
        return Err(f"Missing field: {field}")
    return Ok(data[field])

# 鉄道指向パイプライン
def get_config_value(path: str, field: str) -> Result[object, str]:
    return (
        read_file(path)
        .flat_map(parse_json)
        .flat_map(lambda data: extract_field(data, field))
    )

# テスト
# result = get_config_value("config.json", "database_url")
# print(result)


# =============================================================
# Python 特有: コンテキストマネージャもモナド的
# =============================================================
from contextlib import contextmanager

@contextmanager
def managed_connection(url: str):
    """コンテキストマネージャ = Resource モナドの一種"""
    conn = f"Connection({url})"
    print(f"Opening {conn}")
    try:
        yield conn
    finally:
        print(f"Closing {conn}")

# with 文 = do 記法の糖衣構文
# with managed_connection("db://localhost") as conn:
#     print(f"Using {conn}")
```

---

## コード例 7: モナド合成とモナドトランスフォーマー

```typescript
// =============================================================
// モナド合成: 複数のモナドを組み合わせる実践パターン
// =============================================================

// --- 基本型定義 ---
type Result<T, E = Error> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly error: E };

const Ok = <T>(value: T): Result<T, never> => ({ ok: true, value });
const Err = <E>(error: E): Result<never, E> => ({ ok: false, error });

// --- ResultT<Promise>: Promise + Result のモナドトランスフォーマー ---
// 「非同期で、失敗するかもしれない計算」を合成可能にする

class AsyncResult<T, E = Error> {
  constructor(private readonly promise: Promise<Result<T, E>>) {}

  // --- unit ---
  static of<T>(value: T): AsyncResult<T, never> {
    return new AsyncResult(Promise.resolve(Ok(value)));
  }

  static err<E>(error: E): AsyncResult<never, E> {
    return new AsyncResult(Promise.resolve(Err(error)));
  }

  // Promise<T> を安全にラップ
  static fromPromise<T>(promise: Promise<T>): AsyncResult<T, Error> {
    return new AsyncResult(
      promise
        .then(value => Ok(value) as Result<T, Error>)
        .catch(e => Err(e instanceof Error ? e : new Error(String(e))))
    );
  }

  // --- Functor ---
  map<U>(fn: (value: T) => U): AsyncResult<U, E> {
    return new AsyncResult(
      this.promise.then(result =>
        result.ok ? Ok(fn(result.value)) : result
      )
    );
  }

  mapErr<F>(fn: (error: E) => F): AsyncResult<T, F> {
    return new AsyncResult(
      this.promise.then(result =>
        result.ok ? result : Err(fn(result.error))
      )
    );
  }

  // --- Monad ---
  flatMap<U>(fn: (value: T) => AsyncResult<U, E>): AsyncResult<U, E> {
    return new AsyncResult(
      this.promise.then(result =>
        result.ok ? fn(result.value).toPromise() : result
      )
    );
  }

  // --- ユーティリティ ---
  async toPromise(): Promise<Result<T, E>> {
    return this.promise;
  }

  async unwrapOr(defaultValue: T): Promise<T> {
    const result = await this.promise;
    return result.ok ? result.value : defaultValue;
  }

  // タイムアウト付き
  withTimeout(ms: number): AsyncResult<T, E | Error> {
    const timeout = new Promise<Result<T, E | Error>>(resolve =>
      setTimeout(() => resolve(Err(new Error(`Timeout: ${ms}ms`))), ms)
    );
    return new AsyncResult(Promise.race([this.promise as Promise<Result<T, E | Error>>, timeout]));
  }

  // リトライ
  retry(times: number, delayMs: number = 0): AsyncResult<T, E> {
    return new AsyncResult(
      this.promise.then(async result => {
        if (result.ok || times <= 0) return result;
        if (delayMs > 0) {
          await new Promise(resolve => setTimeout(resolve, delayMs));
        }
        return this.retry(times - 1, delayMs).toPromise();
      })
    );
  }
}

// =============================================================
// 実践例: API クライアント
// =============================================================

interface User {
  id: string;
  name: string;
  teamId: string;
}

interface Team {
  id: string;
  name: string;
  projectIds: string[];
}

interface Project {
  id: string;
  title: string;
  budget: number;
}

// API 関数 (各呼び出しは非同期 + 失敗可能)
function fetchUserById(id: string): AsyncResult<User, string> {
  return AsyncResult.fromPromise(
    fetch(`/api/users/${id}`).then(r => {
      if (!r.ok) throw new Error(`User not found: ${id}`);
      return r.json();
    })
  ).mapErr(e => `Failed to fetch user: ${e.message}`);
}

function fetchTeamById(id: string): AsyncResult<Team, string> {
  return AsyncResult.fromPromise(
    fetch(`/api/teams/${id}`).then(r => {
      if (!r.ok) throw new Error(`Team not found: ${id}`);
      return r.json();
    })
  ).mapErr(e => `Failed to fetch team: ${e.message}`);
}

function fetchProjectById(id: string): AsyncResult<Project, string> {
  return AsyncResult.fromPromise(
    fetch(`/api/projects/${id}`).then(r => {
      if (!r.ok) throw new Error(`Project not found: ${id}`);
      return r.json();
    })
  ).mapErr(e => `Failed to fetch project: ${e.message}`);
}

// flatMap の連鎖: エラーは自動的に伝播
async function getUserTeamBudget(userId: string): Promise<Result<number, string>> {
  return fetchUserById(userId)
    .flatMap(user => fetchTeamById(user.teamId))
    .flatMap(team => {
      if (team.projectIds.length === 0) {
        return AsyncResult.err("Team has no projects");
      }
      return fetchProjectById(team.projectIds[0]);
    })
    .map(project => project.budget)
    .withTimeout(5000)
    .mapErr(e => typeof e === "string" ? e : e.message)
    .toPromise();
}

// =============================================================
// do 記法スタイル: ジェネレータベースの合成
// =============================================================

// ジェネレータを使った「do 記法」の疑似実装
function* doResult<T>(): Generator<Result<any, any>, T, any> {
  // ジェネレータ内で yield で Result を「束縛」
  // Left/Err が返った時点でジェネレータを中断
  return undefined as T;
}

// 実用的な do 記法 (runDo)
function runDo<T, E>(
  gen: () => Generator<Result<T, E>, T, T>
): Result<T, E> {
  const iterator = gen();
  let next = iterator.next();

  while (!next.done) {
    const result = next.value;
    if (!result.ok) {
      return result;  // エラーならジェネレータを中断
    }
    next = iterator.next(result.value);
  }

  return Ok(next.value);
}

// do 記法での使用例
function processOrder(orderId: string): Result<string, string> {
  return runDo(function* () {
    const parsed = yield parseOrderId(orderId);         // bind
    const order = yield findOrder(parsed);               // bind
    const validated = yield validateOrder(order);         // bind
    return `Processed: ${validated.id}`;                 // return
  });
}

// 型定義用ダミー
declare function parseOrderId(id: string): Result<number, string>;
declare function findOrder(id: number): Result<{ id: number; total: number }, string>;
declare function validateOrder(order: { id: number; total: number }): Result<{ id: number; total: number }, string>;

// =============================================================
// MaybeT: Maybe + Promise のトランスフォーマー
// =============================================================

class MaybeAsync<T> {
  constructor(private readonly promise: Promise<Maybe<T>>) {}

  static of<T>(value: T): MaybeAsync<T> {
    return new MaybeAsync(Promise.resolve(Maybe.of(value)));
  }

  static nothing<T>(): MaybeAsync<T> {
    return new MaybeAsync(Promise.resolve(Maybe.nothing()));
  }

  static fromPromise<T>(promise: Promise<T | null | undefined>): MaybeAsync<T> {
    return new MaybeAsync(
      promise
        .then(v => Maybe.fromNullable(v))
        .catch(() => Maybe.nothing())
    );
  }

  map<U>(fn: (value: T) => U): MaybeAsync<U> {
    return new MaybeAsync(this.promise.then(m => m.map(fn)));
  }

  flatMap<U>(fn: (value: T) => MaybeAsync<U>): MaybeAsync<U> {
    return new MaybeAsync(
      this.promise.then(m =>
        m.match({
          just: value => fn(value).toPromise(),
          nothing: () => Promise.resolve(Maybe.nothing<U>()),
        })
      )
    );
  }

  async toPromise(): Promise<Maybe<T>> {
    return this.promise;
  }

  async getOrElse(defaultValue: T): Promise<T> {
    const m = await this.promise;
    return m.getOrElse(defaultValue);
  }
}

// 使用例: 非同期 + null の両方を扱う
async function findUserAvatar(userId: string): Promise<string> {
  return MaybeAsync.fromPromise(fetchUserMaybe(userId))
    .flatMap(user =>
      MaybeAsync.fromPromise(fetchProfileMaybe(user.profileId))
    )
    .map(profile => profile.avatarUrl)
    .getOrElse("/default-avatar.png");
}

declare function fetchUserMaybe(id: string): Promise<{ profileId: string } | null>;
declare function fetchProfileMaybe(id: string): Promise<{ avatarUrl: string } | null>;
```

---

## 深掘り 1: モナドの階層 --- Functor, Applicative, Monad

```
型クラスの階層 (Haskell の型クラス体系)
=========================================

  Functor          map:    (a -> b)   -> F a -> F b
    |               「箱の中の値を変換」
    v
  Applicative      apply:  F (a -> b) -> F a -> F b
    |               「箱の中の関数を、箱の中の値に適用」
    v
  Monad            bind:   (a -> M b) -> M a -> M b
                    「箱の中の値を取り出し、新しい箱を作る関数に渡す」

Functor < Applicative < Monad の関係:
  - すべてのモナドはアプリカティブ
  - すべてのアプリカティブはファンクタ
  - モナドが最も強力 (前のステップの結果に基づいて次の計算を決定できる)

Applicative vs Monad の違い:
  Applicative: 計算の構造が静的 (事前に決まる)
    liftA2 (+) (Just 3) (Just 5) --> Just 8
    すべての計算が独立して実行される

  Monad: 計算の構造が動的 (前の結果に依存)
    Just 3 >>= (\x -> if x > 0 then Just (x + 1) else Nothing)
    前のステップの結果を見て、次の計算を選べる

TypeScript での対応:
  Functor     → Array.map, Promise.then (値を返す場合)
  Applicative → Promise.all (独立した計算を並列実行)
  Monad       → Array.flatMap, Promise.then (Promise を返す場合)
```

| 抽象化 | 操作 | 「できること」 | TypeScript での例 |
|---|---|---|---|
| **Functor** | `map` | 箱の中の値を変換 | `[1,2,3].map(x => x*2)` |
| **Applicative** | `apply` / `liftA2` | 独立した箱同士を結合 | `Promise.all([p1, p2])` |
| **Monad** | `flatMap` / `bind` | 前の結果に基づく次の計算 | `promise.then(x => fetch(x.url))` |

**使い分けの指針**:

- 各計算が **独立** しているなら **Applicative** で十分 (並列実行可能)
- 前の計算の **結果** に基づいて次の計算を **選択** する必要があるなら **Monad** が必要

```typescript
// Applicative: 独立した計算 → 並列実行可能
const [user, products, settings] = await Promise.all([
  fetchUser(id),
  fetchProducts(),
  fetchSettings(),
]);

// Monad: 依存のある計算 → 逐次実行
const user = await fetchUser(id);              // 1. ユーザー取得
const team = await fetchTeam(user.teamId);     // 2. チーム取得 (user に依存)
const projects = await fetchProjects(team.id); // 3. プロジェクト取得 (team に依存)
```

---

## 深掘り 2: 実世界のモナドパターン

### 2.1 Redux / useReducer: State モナドの具現化

```typescript
// Redux の dispatch チェーンは State モナドの bind に相当
// state -> action -> newState

// React の useReducer: State モナドを UI に統合
type State = { count: number; history: number[] };
type Action =
  | { type: "increment" }
  | { type: "decrement" }
  | { type: "reset" };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case "increment":
      return {
        count: state.count + 1,
        history: [...state.history, state.count],
      };
    case "decrement":
      return {
        count: state.count - 1,
        history: [...state.history, state.count],
      };
    case "reset":
      return { count: 0, history: [] };
  }
}

// State モナドの型:  State s a = s -> (a, s)
// reducer の型:      State -> Action -> State
// dispatch の型:     Action -> void (状態は暗黙的に管理)
```

### 2.2 Express/Koa ミドルウェア: Reader + Writer + IO モナドの合成

```typescript
// ミドルウェアは「環境(req/res) + ログ + 副作用」のモナド合成
// Reader: req/res コンテキストを読み取る
// Writer: ヘッダーやログを蓄積
// IO: レスポンス送信、DB アクセスなどの副作用

// Koa の ctx.state は Reader + State の合成
// next() は Continuation モナドの bind に相当
```

### 2.3 RxJS Observable: List + IO + Promise の合成モナド

```typescript
// Observable は以下のモナドの合成と見なせる:
// - List: 複数の値 (時間軸上に分散)
// - IO: 副作用 (subscribe するまで実行されない = lazy)
// - Promise: 非同期 (時間的に遅延)

// pipe = モナド合成のパイプライン
// switchMap/mergeMap/concatMap = flatMap のバリエーション
//   switchMap: 新しい値が来たら前の subscription をキャンセル
//   mergeMap:  並行して実行
//   concatMap: 順次実行 (待ち行列)
```

---

## 深掘り 3: Rust の ? 演算子 --- モナドの最良の糖衣構文

```rust
// Rust の ? 演算子は Result/Option の flatMap (and_then) の糖衣構文
// Haskell の do 記法、JavaScript の async/await と同じ役割

use std::fs;
use std::num::ParseIntError;

// --- エラー型の定義 ---
#[derive(Debug)]
enum AppError {
    Io(std::io::Error),
    Parse(ParseIntError),
    Validation(String),
}

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        AppError::Io(e)
    }
}

impl From<ParseIntError> for AppError {
    fn from(e: ParseIntError) -> Self {
        AppError::Parse(e)
    }
}

// --- ? 演算子を使ったフラットなエラーハンドリング ---
fn read_config(path: &str) -> Result<Config, AppError> {
    let content = fs::read_to_string(path)?;          // Io エラーを自動変換
    let port: u16 = content.trim().parse()?;          // Parse エラーを自動変換

    if port < 1024 {
        return Err(AppError::Validation(
            format!("Port {} is privileged", port)
        ));
    }

    Ok(Config { port })
}

// --- ? 演算子なしの等価コード (flatMap の手動展開) ---
fn read_config_verbose(path: &str) -> Result<Config, AppError> {
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => return Err(AppError::Io(e)),
    };

    let port: u16 = match content.trim().parse() {
        Ok(p) => p,
        Err(e) => return Err(AppError::Parse(e)),
    };

    if port < 1024 {
        return Err(AppError::Validation(
            format!("Port {} is privileged", port)
        ));
    }

    Ok(Config { port })
}

struct Config {
    port: u16,
}

// --- Option の ? 演算子 ---
struct User {
    address: Option<Address>,
}

struct Address {
    city: Option<String>,
}

fn get_city(user: &Option<User>) -> Option<&str> {
    // ? は None なら早期リターン
    let user = user.as_ref()?;
    let address = user.address.as_ref()?;
    let city = address.city.as_deref()?;
    Some(city)
}

// --- and_then チェーンスタイル ---
fn get_city_chain(user: &Option<User>) -> Option<&str> {
    user.as_ref()
        .and_then(|u| u.address.as_ref())
        .and_then(|a| a.city.as_deref())
}

// --- Iterator + Result の組み合わせ ---
fn parse_numbers(inputs: &[&str]) -> Result<Vec<i64>, ParseIntError> {
    inputs.iter()
        .map(|s| s.parse::<i64>())   // Iterator<Item = Result<i64, _>>
        .collect()                     // Result<Vec<i64>, _> に自動変換!
}

// collect() は Iterator<Item = Result<T, E>> -> Result<Vec<T>, E> に対応
// これも モナドの traverse/sequence の実現
```

---

## モナド比較表

### 主要モナド一覧

| モナド | コンテキスト | unit | bind の動作 | 言語での対応 |
|---|---|---|---|---|
| **Maybe/Option** | 値の有無 | `Some(x)` | None ならスキップ | Rust: `Option`, TS: `?.`, Python: `Optional` |
| **Either/Result** | 成功/失敗 | `Right(x)` / `Ok(x)` | Left/Err ならスキップ | Rust: `Result`, Go: `error`, TS: `Promise.catch` |
| **Promise/Future** | 非同期 | `Promise.resolve(x)` | 完了後に次を実行 | JS: `Promise`, Rust: `Future`, Python: `asyncio` |
| **IO** | 副作用 | `IO.of(x)` | 実行を遅延して合成 | Haskell: `IO`, Effect-TS: `Effect` |
| **List/Array** | 複数の値 | `[x]` | 各要素に適用・結合 | JS: `Array.flatMap`, Python: リスト内包表記 |
| **Reader** | 依存注入 | `Reader.of(x)` | 環境を暗黙的に引き回す | DI コンテナ, React Context |
| **Writer** | ログ蓄積 | `Writer.of(x)` | 値＋ログを伝搬 | ミドルウェアのログ蓄積 |
| **State** | 状態管理 | `State.of(x)` | 状態を引き回す | Redux, useReducer |

### 言語ごとの糖衣構文対応

| 概念 | Haskell | Rust | TypeScript | Python |
|---|---|---|---|---|
| Maybe bind | `>>=` / `do` | `?` / `and_then()` | `?.` / `flatMap` | `or None` / `if x is not None` |
| Either bind | `>>=` / `do` | `?` / `and_then()` | `then` / `catch` | 例外 / `try/except` |
| do 記法 | `do { x <- m; ... }` | `let x = m?;` | `const x = await m;` | `x = await m` |
| for 内包表記 | `[f x | x <- xs]` | `xs.iter().map(f)` | `xs.flatMap(f)` | `[f(x) for x in xs]` |
| 型制約 | 型クラス `Monad m =>` | トレイト `impl Monad` | インターフェース | Protocol / ABC |

### 糖衣構文の対応関係

| モナド操作 | Haskell do 記法 | Rust ? 演算子 | JS async/await |
|---|---|---|---|
| bind | `x <- action` | `let x = action?;` | `const x = await action;` |
| return | `return x` | `Ok(x)` | `return x` |
| sequence | `action1 >> action2` | `action1?; action2?;` | `await a1; await a2;` |
| 失敗 | `fail "msg"` | `Err(e)?` | `throw new Error()` |

---

## アンチパターン

### 1. モナドの過剰使用 --- 言語ネイティブ機能の無視

**問題**: すべてに自作 Maybe/Either を適用し、チームに馴染みのないライブラリ依存を増やす。TypeScript では Optional Chaining (`?.`) と Nullish Coalescing (`??`) で十分なケースが多い。

```typescript
// =============================================================
// [NG] 自作 Maybe で不必要に複雑化
// =============================================================
function getUserCityBad(user: User | null): string {
  return Maybe.fromNullable(user)
    .flatMap(u => Maybe.fromNullable(u.address))
    .flatMap(a => Maybe.fromNullable(a.city))
    .getOrElse("Unknown");
}

// =============================================================
// [OK] TypeScript のネイティブ機能で十分
// =============================================================
function getUserCityGood(user: User | null): string {
  return user?.address?.city ?? "Unknown";
}

// =============================================================
// [OK] ただし、複雑なロジックが絡む場合は Maybe が有効
// =============================================================
function getDiscountedPrice(user: User | null): Maybe<number> {
  return Maybe.fromNullable(user)
    .flatMap(u => findMembership(u.id))        // DB 検索 (nullable)
    .filter(m => m.isActive)                    // 条件フィルタ
    .flatMap(m => calculateDiscount(m.tier))    // ビジネスロジック (失敗可能)
    .map(discount => basePrice * (1 - discount));
}
// Optional Chaining だけではこの分岐を表現しきれない

// 判断基準:
// - 単純な null チェックの連鎖 → ?. と ?? を使う
// - 条件分岐 + ビジネスロジック + 失敗可能性 → Maybe/Result を使う
```

### 2. 安易な unwrap --- モナドの安全性の破壊

**問題**: `Option.unwrap()` や `Result.unwrap()` を安易に使うと、実行時パニック/例外が発生する。モナドが提供する型安全性が完全に台無しになる。

```rust
// =============================================================
// [NG] unwrap の乱用
// =============================================================
fn process_bad(input: &str) -> String {
    let value = input.parse::<i32>().unwrap();     // パニック!
    let item = find_item(value).unwrap();          // パニック!
    let result = validate(item).unwrap();          // パニック!
    format!("Done: {}", result)
}

// =============================================================
// [OK] unwrap の代替手段を使う
// =============================================================

// 方法1: ? 演算子 (推奨)
fn process_good(input: &str) -> Result<String, AppError> {
    let value = input.parse::<i32>()?;
    let item = find_item(value)?;
    let result = validate(item)?;
    Ok(format!("Done: {}", result))
}

// 方法2: パターンマッチ
fn process_match(input: &str) -> String {
    match input.parse::<i32>() {
        Ok(value) => match find_item(value) {
            Some(item) => format!("Found: {}", item),
            None => "Not found".to_string(),
        },
        Err(e) => format!("Parse error: {}", e),
    }
}

// 方法3: unwrap_or / unwrap_or_else (デフォルト値がある場合)
fn process_default(input: &str) -> String {
    let value = input.parse::<i32>().unwrap_or(0);
    let item = find_item(value).unwrap_or_default();
    format!("Result: {}", item)
}

// 方法4: expect (unwrap よりましだが、本番コードでは避ける)
// デバッグ情報を付与できるが、パニックは発生する
fn process_expect(input: &str) -> i32 {
    input.parse::<i32>()
        .expect(&format!("Failed to parse '{}' as i32", input))
}
```

```typescript
// TypeScript でも同様
// =============================================================
// [NG] 型ガードなしでのアクセス
// =============================================================
const result = validateUser(input);
console.log(result.value.name);  // result が Left/Err なら runtime error

// =============================================================
// [OK] 型を尊重したアクセス
// =============================================================
const result = validateUser(input);
if (result.tag === "Right") {
  console.log(result.value.name);  // 型安全
} else {
  console.error(result.value);     // エラーハンドリング
}

// [OK] fold/match でパターンマッチ
EitherM.fold(
  result,
  error => console.error(`Validation failed: ${error}`),
  user => console.log(`Welcome, ${user.name}`)
);
```

### 3. モナドの混在 --- Either と例外の混在

**問題**: Either/Result を使ったエラーハンドリングと、従来の try/catch を無秩序に混在させると、エラーの流れが追跡不能になる。

```typescript
// =============================================================
// [NG] Either と例外の混在
// =============================================================
function processBad(input: string): Either<string, Output> {
  const parsed = parseInput(input);  // Either を返す
  if (parsed.tag === "Left") return parsed;

  // ここで突然 throw! Either のパイプラインが壊れる
  const data = JSON.parse(parsed.value.raw);  // throws!

  const validated = validateData(data);  // Either を返す
  if (validated.tag === "Left") return validated;

  return Right(transform(validated.value));
}

// 呼び出し側: Either だけケアしても JSON.parse の例外を見落とす

// =============================================================
// [OK] エラーハンドリング戦略を統一
// =============================================================

// 方法1: すべて Either に統一 (推奨)
function processGood(input: string): Either<string, Output> {
  return flatMap(parseInput(input), parsed =>
    flatMap(EitherM.tryCatch(() => JSON.parse(parsed.raw))
      .mapLeft(e => `JSON error: ${e.message}`), data =>
        flatMap(validateData(data), validated =>
          Right(transform(validated))
        )
      )
  );
}

// 方法2: 境界で変換 (外部ライブラリとの接合点)
function safeJsonParse(raw: string): Either<string, unknown> {
  return EitherM.tryCatch(() => JSON.parse(raw))
    .mapLeft(e => `JSON parse error: ${e.message}`);
}

function processAlsoGood(input: string): Either<string, Output> {
  const parsed = parseInput(input);
  const json = flatMap(parsed, p => safeJsonParse(p.raw));
  const validated = flatMap(json, validateData);
  return map(validated, transform);
}

// 原則:
// - プロジェクト内部: Either/Result で統一
// - 外部ライブラリの境界: tryCatch で Either に変換
// - 決して Either のパイプライン内部で throw しない

declare function parseInput(input: string): Either<string, { raw: string }>;
declare function validateData(data: unknown): Either<string, { valid: true }>;
declare function transform(data: { valid: true }): Output;
type Output = { result: string };
```

---

## 演習問題

### 演習 1: Maybe モナドの活用 (基礎)

以下の型定義に対し、Maybe モナドを使って安全にネストされたデータにアクセスする関数を実装してください。

```typescript
interface Department {
  name: string;
  manager?: {
    name: string;
    contact?: {
      email?: string;
      phone?: string;
    };
  };
}

// 実装してください
function getManagerEmail(dept: Department | null): string {
  // Maybe を使って dept?.manager?.contact?.email を安全に取得
  // 見つからない場合は "N/A" を返す
}

function getManagerPhone(dept: Department | null): Maybe<string> {
  // Maybe を返す版。呼び出し側で処理を選択できるようにする
}
```

**期待される出力**:
```
getManagerEmail({ name: "Engineering", manager: { name: "Alice", contact: { email: "alice@co.com" } } })
// => "alice@co.com"

getManagerEmail({ name: "Sales" })
// => "N/A"

getManagerEmail(null)
// => "N/A"

getManagerPhone({ name: "HR", manager: { name: "Bob", contact: { phone: "090-1234-5678" } } })
// => Just("090-1234-5678")

getManagerPhone({ name: "HR", manager: { name: "Bob" } })
// => Nothing()
```

### 演習 2: Result で鉄道指向バリデーション (応用)

ユーザー入力のバリデーションパイプラインを Result モナドで構築してください。

```typescript
interface OrderInput {
  productId: string;   // 数値文字列であること
  quantity: string;    // 1-100 の整数文字列であること
  couponCode?: string; // 存在する場合は "DISCOUNT-" で始まること
}

interface ValidatedOrder {
  productId: number;
  quantity: number;
  couponCode: string | null;
  discountRate: number;  // クーポンがあれば 0.1、なければ 0
}

// 各バリデーション関数を実装してください
function validateProductId(raw: string): Result<number, string> { /* ... */ }
function validateQuantity(raw: string): Result<number, string> { /* ... */ }
function validateCoupon(code?: string): Result<string | null, string> { /* ... */ }

// 鉄道指向で連鎖させてください
function validateOrder(input: OrderInput): Result<ValidatedOrder, string> { /* ... */ }
```

**期待される出力**:
```
validateOrder({ productId: "42", quantity: "5" })
// => Ok({ productId: 42, quantity: 5, couponCode: null, discountRate: 0 })

validateOrder({ productId: "42", quantity: "5", couponCode: "DISCOUNT-SUMMER" })
// => Ok({ productId: 42, quantity: 5, couponCode: "DISCOUNT-SUMMER", discountRate: 0.1 })

validateOrder({ productId: "abc", quantity: "5" })
// => Err("productId must be a number")

validateOrder({ productId: "42", quantity: "200" })
// => Err("quantity must be 1-100")

validateOrder({ productId: "42", quantity: "5", couponCode: "INVALID" })
// => Err("coupon must start with DISCOUNT-")
```

### 演習 3: AsyncResult モナドトランスフォーマー (発展)

以下の仕様を満たす `AsyncResult` ベースの API クライアントを実装してください。

```typescript
// 要件:
// 1. 各 API 呼び出しは AsyncResult<T, AppError> を返す
// 2. ネットワークエラー、404、バリデーションエラーを構造化エラー型で表現
// 3. flatMap でパイプラインを構築
// 4. タイムアウト (3秒) とリトライ (最大2回) をサポート
// 5. 最終結果を Result<T, AppError> として返す

type AppError =
  | { type: "network"; message: string }
  | { type: "notFound"; resource: string; id: string }
  | { type: "validation"; errors: string[] };

// 実装してください
class ApiClient {
  // ユーザー取得 → チーム取得 → メンバー一覧取得 のパイプライン
  async getTeamMembers(userId: string): Promise<Result<TeamMember[], AppError>> {
    return fetchUser(userId)
      .flatMap(user => fetchTeam(user.teamId))
      .flatMap(team => fetchMembers(team.id))
      .withTimeout(3000)
      .retry(2, 500)
      .toPromise();
  }
}
```

**期待される出力**:
```
// 正常系
await client.getTeamMembers("user-1")
// => { ok: true, value: [{ id: "m1", name: "Alice" }, { id: "m2", name: "Bob" }] }

// ユーザーが見つからない
await client.getTeamMembers("nonexistent")
// => { ok: false, error: { type: "notFound", resource: "user", id: "nonexistent" } }

// タイムアウト (3秒後にリトライ2回後)
await client.getTeamMembers("slow-user")
// => { ok: false, error: { type: "network", message: "Timeout: 3000ms" } }
```

---

## FAQ

### Q1: モナドは実際の開発で役に立ちますか？

**A**: はい。ただし「モナド」という名前を意識する必要はありません。以下はすべてモナドパターンの実例です。

| 日常的なコード | モナドとしての正体 |
|---|---|
| `array.flatMap(fn)` | List モナドの bind |
| `promise.then(fn)` | IO モナドの bind |
| `async/await` | do 記法の糖衣構文 |
| `result?` (Rust) | Either モナドの bind |
| `user?.address?.city` | Maybe モナドの bind (部分的) |
| `ctx.state` (Koa) | Reader モナドの ask |

モナドの価値は「既に使っている概念に理論的基盤を与える」ことにある。理論を知ることで、新しいコンテキスト(例: 非同期ストリーム)にも同じパターンを適用できるようになる。

### Q2: Haskell を学ばないとモナドは理解できませんか？

**A**: いいえ。モナドの概念は言語非依存です。TypeScript や Rust で `flatMap` / `and_then` のパターンを実践すれば十分理解できます。Haskell は型クラスでモナドを最も体系的に表現しますが、実用的な理解には不要です。むしろ、Haskell の抽象的な表現から入ると「モナドは難しい」という誤解が生まれやすい。「Promise の then はモナドの bind である」という具体例から入るのが効果的です。

### Q3: モナドトランスフォーマーとは何ですか？

**A**: 複数のモナドを重ねて使う仕組みです。例えば `MaybeT (Either Error)` は「失敗するかもしれない計算で、さらに None の可能性がある」を表現します。

実用的には以下が身近なモナドトランスフォーマーです。

| 組み合わせ | TypeScript での表現 | 用途 |
|---|---|---|
| Promise + Result | `Promise<Result<T, E>>` | 非同期 + エラーハンドリング |
| Promise + Maybe | `Promise<T \| null>` | 非同期 + 値の有無 |
| Array + Maybe | `(T \| null)[]` → `T[]` (filter) | コレクション + 欠損値 |

コード例 7 の `AsyncResult` クラスがこの概念の TypeScript 実装です。

### Q4: Either/Result と try/catch のどちらを使うべきですか？

**A**: プロジェクトの方針を統一することが最も重要です。以下が判断基準です。

| 基準 | Either/Result が有利 | try/catch が有利 |
|---|---|---|
| エラーの型安全性 | 呼び出し側がエラー型を認識 | 型情報が失われる (any/unknown) |
| エラーの網羅性チェック | 判別共用体でコンパイル時チェック | ランタイムまで不明 |
| パフォーマンス | 値の生成のみ | スタックトレース生成のコスト |
| エコシステム | Rust, Haskell, Scala で主流 | Java, Python, JS/TS で主流 |
| 学習コスト | チームに概念の理解が必要 | 多くの開発者に馴染みがある |
| 回復不能エラー | 不向き (OOM, stack overflow) | 適切 (finally で cleanup) |

**推奨**: TypeScript では「関数の戻り値として Result を使い、外部ライブラリの境界で tryCatch を使って Result に変換する」のがバランスが良い。

### Q5: なぜ flatMap であって map ではだめなのですか？

**A**: `map` は `M<M<T>>` というネストを生むが、`flatMap` は `M<T>` にフラット化する。これが「モナド」が「ファンクタ」より強力な理由です。

```typescript
// map だと二重にラップされる
const result: Maybe<Maybe<string>> =
  Maybe.of(user).map(u => Maybe.fromNullable(u.name));
// Maybe<Maybe<string>> ← 使いにくい!

// flatMap なら自動的にフラットになる
const result: Maybe<string> =
  Maybe.of(user).flatMap(u => Maybe.fromNullable(u.name));
// Maybe<string> ← 直接使える

// Promise も同様
// then は map と flatMap を兼ねるが、内部で自動フラット化している
Promise.resolve(userId)
  .then(id => fetchUser(id))  // fetchUser は Promise を返す
  // .then が map なら: Promise<Promise<User>> (二重ラップ)
  // .then が flatMap だから: Promise<User> (フラット)
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| モナドの本質 | `unit` と `bind (flatMap)` によるコンテキスト付き計算の合成 |
| モナド則 | 左単位元・右単位元・結合則の3つを満たすこと |
| Maybe/Option | null 安全性。値がないかもしれない計算の連鎖。`?.` の一般化 |
| Either/Result | 型安全なエラーハンドリング。鉄道指向プログラミング |
| Promise/Future | 非同期処理。`async/await` は do 記法の糖衣構文 |
| IO | 副作用の分離。純粋関数と副作用の境界を明確化。テスト容易性 |
| List | 非決定性計算。`flatMap` で複数の可能性を探索 |
| Reader | 依存性注入。環境を暗黙的に引き回す |
| モナドトランスフォーマー | 複数モナドの合成。`AsyncResult = Promise + Result` |
| 階層 | Functor < Applicative < Monad。独立計算なら Applicative で十分 |
| 実践指針 | 言語のネイティブ機能を優先。エラー戦略を統一。unwrap を避ける |

---

## 次に読むべきガイド

- [ファンクタ・アプリカティブ](./01-functor-applicative.md) --- モナドの前提となる抽象化。map と apply の理解
- [関数型パターン](./02-fp-patterns.md) --- カリー化、パイプライン、合成とモナドの組み合わせ
- [Strategy パターン](../02-behavioral/01-strategy.md) --- 関数を値として扱うパターン (高階関数の OOP 版)
- [Iterator パターン](../02-behavioral/04-iterator.md) --- List モナドの OOP 実装。遅延評価との関係

---

## 参考文献

1. **Philip Wadler** (1992): [Monads for functional programming](https://homepages.inf.ed.ac.uk/wadler/papers/marktoberdorf/baastad.pdf) --- モナドの理論的基盤を確立した論文
2. **Bartosz Milewski**: [Category Theory for Programmers](https://bartoszmilewski.com/2014/10/28/category-theory-for-programmers-the-preface/) --- 圏論とモナドの理論的解説
3. **Scott Wlaschin**: [Railway Oriented Programming](https://fsharpforfunandprofit.com/rop/) --- Either モナドの実践的解説 (鉄道指向プログラミング)
4. **Rust by Example**: [Error Handling](https://doc.rust-lang.org/rust-by-example/error.html) --- Result モナドと ? 演算子の実践ガイド
5. **Giulio Canti**: [fp-ts](https://gcanti.github.io/fp-ts/) --- TypeScript の関数型プログラミングライブラリ (モナドの実装例)
