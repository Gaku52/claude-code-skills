# TypeScript エラーハンドリングパターン

> Result型、カスタムエラー階層、zodバリデーションを組み合わせた堅牢なエラー処理戦略

## この章で学ぶこと

1. **Result型パターン** -- 例外を使わずに型安全にエラーを表現する方法
2. **カスタムエラー階層** -- ドメイン固有のエラークラスを設計し、エラーの種別を型で判別する技法
3. **zodによるバリデーション** -- ランタイムバリデーションと型推論を統合し、外部入力を安全に処理する方法
4. **非同期エラーハンドリング** -- Promise と Result 型を組み合わせた非同期処理のエラー管理
5. **エラーの集約と変換** -- 複数のエラーを集約し、レイヤー間でエラーを変換する技法
6. **実務での統合パターン** -- Express/NestJS/tRPC でのエラーハンドリング統合
7. **テスト戦略** -- エラーパスのテスト手法とベストプラクティス

---

## 1. Result型パターン

### 1-1. なぜ例外ではなく Result 型か

```
+----------------------------------+
|        従来の例外フロー           |
+----------------------------------+
|  function parse(input) {         |
|    if (invalid) throw Error()  --+---> catch ブロックで
|    return value                  |     型情報が消失
|  }                               |
+----------------------------------+

+----------------------------------+
|        Result 型フロー            |
+----------------------------------+
|  function parse(input):          |
|    Result<Value, ParseError>     |
|    if (invalid) return Err(...)--+---> 型が保持される
|    return Ok(value)              |     パターンマッチ可能
|  }                               |
+----------------------------------+
```

例外ベースのエラーハンドリングには以下の根本的な問題があります。

1. **型情報の消失**: `catch` ブロックの `error` は TypeScript 4.4 以降 `unknown` 型であり、どのような種類のエラーが発生しうるかをコンパイラが把握できません。
2. **暗黙的な制御フロー**: 関数シグネチャを見ただけでは、その関数がどのような例外を throw するかがわかりません（Java の checked exception と異なり、TypeScript には throw 宣言がありません）。
3. **コンポジションの困難さ**: 例外は関数合成を妨げます。`try-catch` のネストは可読性を著しく低下させます。
4. **テストの複雑化**: 例外を throw する関数のテストは、`expect(() => fn()).toThrow()` のような間接的なアサーションが必要です。

Result 型はこれらの問題を解決し、**エラーを値として扱う**ことで型安全性と可読性を両立します。

```typescript
// 例外ベース: 何が throw されるかわからない
function parseJSON(input: string): unknown {
  return JSON.parse(input); // SyntaxError を throw する可能性
}

// Result 型ベース: エラーが型シグネチャに現れる
function parseJSON(input: string): Result<unknown, SyntaxError> {
  try {
    return Ok(JSON.parse(input));
  } catch (e) {
    return Err(e instanceof SyntaxError ? e : new SyntaxError(String(e)));
  }
}
```

### 1-2. 基本の Result 型定義

```typescript
// Result 型の定義
type Result<T, E> = Ok<T> | Err<E>;

interface Ok<T> {
  readonly _tag: "Ok";
  readonly value: T;
}

interface Err<E> {
  readonly _tag: "Err";
  readonly error: E;
}

// コンストラクタ関数
function Ok<T>(value: T): Ok<T> {
  return { _tag: "Ok", value };
}

function Err<E>(error: E): Err<E> {
  return { _tag: "Err", error };
}

// 型ガード
function isOk<T, E>(result: Result<T, E>): result is Ok<T> {
  return result._tag === "Ok";
}

function isErr<T, E>(result: Result<T, E>): result is Err<E> {
  return result._tag === "Err";
}
```

### 1-3. Result 型のユーティリティ

```typescript
// map: 成功値を変換
function map<T, U, E>(
  result: Result<T, E>,
  fn: (value: T) => U
): Result<U, E> {
  return isOk(result) ? Ok(fn(result.value)) : result;
}

// flatMap (chain): 成功値から新しい Result を返す
function flatMap<T, U, E>(
  result: Result<T, E>,
  fn: (value: T) => Result<U, E>
): Result<U, E> {
  return isOk(result) ? fn(result.value) : result;
}

// unwrapOr: エラー時にデフォルト値を返す
function unwrapOr<T, E>(result: Result<T, E>, defaultValue: T): T {
  return isOk(result) ? result.value : defaultValue;
}

// 使用例
const parsed = parseAge("25");           // Result<number, ParseError>
const doubled = map(parsed, (n) => n * 2); // Result<number, ParseError>
const age = unwrapOr(doubled, 0);        // number
```

### 1-4. 高度な Result ユーティリティ

```typescript
// mapErr: エラーを変換する
function mapErr<T, E, F>(
  result: Result<T, E>,
  fn: (error: E) => F
): Result<T, F> {
  return isErr(result) ? Err(fn(result.error)) : result;
}

// tap: 副作用を実行しつつ Result をそのまま返す
function tap<T, E>(
  result: Result<T, E>,
  fn: (value: T) => void
): Result<T, E> {
  if (isOk(result)) {
    fn(result.value);
  }
  return result;
}

// tapErr: エラー時に副作用を実行
function tapErr<T, E>(
  result: Result<T, E>,
  fn: (error: E) => void
): Result<T, E> {
  if (isErr(result)) {
    fn(result.error);
  }
  return result;
}

// match: パターンマッチ
function match<T, E, U>(
  result: Result<T, E>,
  handlers: {
    ok: (value: T) => U;
    err: (error: E) => U;
  }
): U {
  return isOk(result) ? handlers.ok(result.value) : handlers.err(result.error);
}

// fromPromise: Promise を Result に変換
async function fromPromise<T, E = Error>(
  promise: Promise<T>,
  errorFn?: (error: unknown) => E
): Promise<Result<T, E>> {
  try {
    const value = await promise;
    return Ok(value);
  } catch (error) {
    if (errorFn) {
      return Err(errorFn(error));
    }
    return Err(error as E);
  }
}

// fromThrowable: throw する関数を Result に変換
function fromThrowable<T, E = Error>(
  fn: () => T,
  errorFn?: (error: unknown) => E
): Result<T, E> {
  try {
    return Ok(fn());
  } catch (error) {
    if (errorFn) {
      return Err(errorFn(error));
    }
    return Err(error as E);
  }
}

// combine: 複数の Result をまとめる
function combine<T, E>(results: Result<T, E>[]): Result<T[], E> {
  const values: T[] = [];
  for (const result of results) {
    if (isErr(result)) {
      return result;
    }
    values.push(result.value);
  }
  return Ok(values);
}

// combineAll: 全エラーを収集する
function combineAll<T, E>(results: Result<T, E>[]): Result<T[], E[]> {
  const values: T[] = [];
  const errors: E[] = [];

  for (const result of results) {
    if (isErr(result)) {
      errors.push(result.error);
    } else {
      values.push(result.value);
    }
  }

  return errors.length > 0 ? Err(errors) : Ok(values);
}

// 使用例: combine
const validations = [
  validateName("John"),   // Result<string, ValidationError>
  validateEmail("a@b.c"), // Result<string, ValidationError>
  validateAge("25"),      // Result<number, ValidationError>
];

const combined = combine(validations);
// Result<(string | number)[], ValidationError>
```

### 1-5. メソッドチェーン対応の Result クラス

プレーンオブジェクトの関数型スタイルだけでなく、メソッドチェーンスタイルも有用です。

```typescript
class ResultClass<T, E> {
  private constructor(
    private readonly _tag: "Ok" | "Err",
    private readonly _value?: T,
    private readonly _error?: E
  ) {}

  static ok<T, E = never>(value: T): ResultClass<T, E> {
    return new ResultClass<T, E>("Ok", value);
  }

  static err<T = never, E = unknown>(error: E): ResultClass<T, E> {
    return new ResultClass<T, E>("Err", undefined, error);
  }

  isOk(): this is ResultClass<T, never> {
    return this._tag === "Ok";
  }

  isErr(): this is ResultClass<never, E> {
    return this._tag === "Err";
  }

  map<U>(fn: (value: T) => U): ResultClass<U, E> {
    if (this._tag === "Ok") {
      return ResultClass.ok(fn(this._value as T));
    }
    return ResultClass.err(this._error as E);
  }

  mapErr<F>(fn: (error: E) => F): ResultClass<T, F> {
    if (this._tag === "Err") {
      return ResultClass.err(fn(this._error as E));
    }
    return ResultClass.ok(this._value as T);
  }

  flatMap<U>(fn: (value: T) => ResultClass<U, E>): ResultClass<U, E> {
    if (this._tag === "Ok") {
      return fn(this._value as T);
    }
    return ResultClass.err(this._error as E);
  }

  unwrap(): T {
    if (this._tag === "Ok") {
      return this._value as T;
    }
    throw new Error(`Attempted to unwrap an Err: ${this._error}`);
  }

  unwrapOr(defaultValue: T): T {
    return this._tag === "Ok" ? (this._value as T) : defaultValue;
  }

  unwrapOrElse(fn: (error: E) => T): T {
    return this._tag === "Ok" ? (this._value as T) : fn(this._error as E);
  }

  match<U>(handlers: { ok: (value: T) => U; err: (error: E) => U }): U {
    if (this._tag === "Ok") {
      return handlers.ok(this._value as T);
    }
    return handlers.err(this._error as E);
  }

  tap(fn: (value: T) => void): ResultClass<T, E> {
    if (this._tag === "Ok") {
      fn(this._value as T);
    }
    return this;
  }

  tapErr(fn: (error: E) => void): ResultClass<T, E> {
    if (this._tag === "Err") {
      fn(this._error as E);
    }
    return this;
  }

  toPromise(): Promise<T> {
    if (this._tag === "Ok") {
      return Promise.resolve(this._value as T);
    }
    return Promise.reject(this._error);
  }
}

// 使用例: メソッドチェーン
const result = ResultClass.ok<string, Error>("42")
  .map((s) => parseInt(s, 10))
  .flatMap((n) =>
    n > 0
      ? ResultClass.ok(n)
      : ResultClass.err(new Error("Must be positive"))
  )
  .map((n) => n * 2)
  .tapErr((e) => console.error("Error:", e.message))
  .match({
    ok: (value) => `Success: ${value}`,
    err: (error) => `Failed: ${error.message}`,
  });
```

### 1-6. neverthrow を使った実装

```typescript
import { ok, err, Result, ResultAsync } from "neverthrow";

// 基本的な使い方
function divide(a: number, b: number): Result<number, Error> {
  if (b === 0) {
    return err(new Error("Division by zero"));
  }
  return ok(a / b);
}

// ResultAsync: 非同期版 Result
function fetchUser(id: string): ResultAsync<User, ApiError> {
  return ResultAsync.fromPromise(
    fetch(`/api/users/${id}`).then((r) => r.json()),
    (error) => new ApiError("FETCH_FAILED", String(error))
  );
}

// メソッドチェーンで処理を組み合わせ
const result = await fetchUser("123")
  .andThen((user) =>
    user.isActive
      ? ok(user)
      : err(new ApiError("INACTIVE_USER", "User is inactive"))
  )
  .map((user) => ({
    id: user.id,
    name: user.name,
    email: user.email,
  }))
  .mapErr((error) => ({
    code: error.code,
    message: error.message,
  }));

// match でパターンマッチ
result.match(
  (user) => console.log("User:", user),
  (error) => console.error("Error:", error)
);

// combine: 複数の Result を合成
const combinedResult = Result.combine([
  validateName(input.name),
  validateEmail(input.email),
  validateAge(input.age),
]);
// Result<[string, string, number], ValidationError>

// combineWithAllErrors: 全てのエラーを収集
const allErrors = Result.combineWithAllErrors([
  validateName(input.name),
  validateEmail(input.email),
  validateAge(input.age),
]);
// Result<[string, string, number], ValidationError[]>
```

### 1-7. pipe パターンによるチェーン

```typescript
// pipe 関数の定義
function pipe<A>(value: A): A;
function pipe<A, B>(value: A, fn1: (a: A) => B): B;
function pipe<A, B, C>(value: A, fn1: (a: A) => B, fn2: (b: B) => C): C;
function pipe<A, B, C, D>(
  value: A,
  fn1: (a: A) => B,
  fn2: (b: B) => C,
  fn3: (c: C) => D
): D;
function pipe(value: unknown, ...fns: Function[]): unknown {
  return fns.reduce((acc, fn) => fn(acc), value);
}

// Result 用の pipe 対応関数
const R = {
  map:
    <T, U, E>(fn: (value: T) => U) =>
    (result: Result<T, E>): Result<U, E> =>
      isOk(result) ? Ok(fn(result.value)) : result,

  flatMap:
    <T, U, E>(fn: (value: T) => Result<U, E>) =>
    (result: Result<T, E>): Result<U, E> =>
      isOk(result) ? fn(result.value) : result,

  mapErr:
    <T, E, F>(fn: (error: E) => F) =>
    (result: Result<T, E>): Result<T, F> =>
      isErr(result) ? Err(fn(result.error)) : result,

  unwrapOr:
    <T, E>(defaultValue: T) =>
    (result: Result<T, E>): T =>
      isOk(result) ? result.value : defaultValue,

  tap:
    <T, E>(fn: (value: T) => void) =>
    (result: Result<T, E>): Result<T, E> => {
      if (isOk(result)) fn(result.value);
      return result;
    },
};

// 使用例
const processedAge = pipe(
  parseAge("25"),
  R.map((n: number) => n + 1),
  R.flatMap((n: number) =>
    n > 0 && n < 150 ? Ok(n) : Err(new Error("Invalid age"))
  ),
  R.tap((n: number) => console.log("Valid age:", n)),
  R.unwrapOr(0)
);
```

---

## 2. カスタムエラー階層

### 2-1. エラークラスの設計

```
+---------------------+
|     AppError        |  基底クラス
+---------------------+
         |
    +----+--------+----------+
    |             |          |
+---------+ +---------+ +-----------+
|Validation| |NotFound | |Permission |
|Error     | |Error    | |Error      |
+---------+ +---------+ +-----------+
    |
+----------+
|FieldError|  さらに特化
+----------+
```

```typescript
// 基底エラークラス
abstract class AppError extends Error {
  abstract readonly code: string;
  abstract readonly statusCode: number;
  readonly timestamp: Date;

  constructor(message: string, public readonly cause?: unknown) {
    super(message);
    this.name = this.constructor.name;
    this.timestamp = new Date();
    // プロトタイプチェーンの修正
    Object.setPrototypeOf(this, new.target.prototype);
  }

  toJSON() {
    return {
      code: this.code,
      message: this.message,
      timestamp: this.timestamp.toISOString(),
    };
  }
}

// 具体的なエラークラス
class ValidationError extends AppError {
  readonly code = "VALIDATION_ERROR";
  readonly statusCode = 400;

  constructor(
    message: string,
    public readonly fields: Record<string, string[]>
  ) {
    super(message);
  }
}

class NotFoundError extends AppError {
  readonly code = "NOT_FOUND";
  readonly statusCode = 404;

  constructor(public readonly resource: string, public readonly id: string) {
    super(`${resource} with id ${id} not found`);
  }
}

class PermissionError extends AppError {
  readonly code = "PERMISSION_DENIED";
  readonly statusCode = 403;

  constructor(
    public readonly action: string,
    public readonly resource: string
  ) {
    super(`Permission denied: cannot ${action} on ${resource}`);
  }
}
```

### 2-2. 拡張エラー階層の設計

実務のプロジェクトでは、より細かいエラー分類が必要になります。

```typescript
// ─── インフラストラクチャエラー ───
class InfraError extends AppError {
  readonly code = "INFRA_ERROR";
  readonly statusCode = 500;

  constructor(
    message: string,
    public readonly service: string,
    cause?: unknown
  ) {
    super(message, cause);
  }
}

class DatabaseError extends InfraError {
  readonly code = "DATABASE_ERROR" as const;

  constructor(
    message: string,
    public readonly query?: string,
    cause?: unknown
  ) {
    super(message, "database", cause);
  }
}

class ExternalApiError extends InfraError {
  readonly code = "EXTERNAL_API_ERROR" as const;

  constructor(
    message: string,
    public readonly endpoint: string,
    public readonly responseStatus?: number,
    cause?: unknown
  ) {
    super(message, "external_api", cause);
  }
}

class CacheError extends InfraError {
  readonly code = "CACHE_ERROR" as const;

  constructor(
    message: string,
    public readonly key?: string,
    cause?: unknown
  ) {
    super(message, "cache", cause);
  }
}

// ─── ビジネスロジックエラー ───
class BusinessError extends AppError {
  abstract readonly statusCode: number;

  constructor(message: string, cause?: unknown) {
    super(message, cause);
  }
}

class ConflictError extends BusinessError {
  readonly code = "CONFLICT";
  readonly statusCode = 409;

  constructor(
    public readonly resource: string,
    public readonly conflictField: string,
    public readonly conflictValue: string
  ) {
    super(
      `${resource} with ${conflictField}=${conflictValue} already exists`
    );
  }
}

class RateLimitError extends BusinessError {
  readonly code = "RATE_LIMIT_EXCEEDED";
  readonly statusCode = 429;

  constructor(
    public readonly limit: number,
    public readonly windowMs: number,
    public readonly retryAfterMs: number
  ) {
    super(`Rate limit exceeded: ${limit} requests per ${windowMs}ms`);
  }
}

class InsufficientBalanceError extends BusinessError {
  readonly code = "INSUFFICIENT_BALANCE";
  readonly statusCode = 422;

  constructor(
    public readonly required: number,
    public readonly available: number,
    public readonly currency: string
  ) {
    super(
      `Insufficient balance: required ${required} ${currency}, available ${available} ${currency}`
    );
  }
}

class ExpiredError extends BusinessError {
  readonly code = "EXPIRED";
  readonly statusCode = 410;

  constructor(
    public readonly resource: string,
    public readonly expiredAt: Date
  ) {
    super(`${resource} expired at ${expiredAt.toISOString()}`);
  }
}
```

### 2-3. エラーコードのリテラル型による網羅性チェック

```typescript
// すべてのエラーコードをリテラル型のユニオンで定義
type ErrorCode =
  | "VALIDATION_ERROR"
  | "NOT_FOUND"
  | "PERMISSION_DENIED"
  | "CONFLICT"
  | "RATE_LIMIT_EXCEEDED"
  | "INSUFFICIENT_BALANCE"
  | "EXPIRED"
  | "DATABASE_ERROR"
  | "EXTERNAL_API_ERROR"
  | "CACHE_ERROR";

// ドメインエラーの判別共用体
type DomainError =
  | ValidationError
  | NotFoundError
  | PermissionError
  | ConflictError
  | RateLimitError
  | InsufficientBalanceError
  | ExpiredError;

// 網羅性チェック用のヘルパー
function assertNever(value: never): never {
  throw new Error(`Unexpected value: ${value}`);
}

// エラーコードでの分岐（網羅性チェック付き）
function handleDomainError(error: DomainError): HttpResponse {
  switch (error.code) {
    case "VALIDATION_ERROR":
      return {
        status: 400,
        body: {
          code: error.code,
          message: error.message,
          fields: error.fields,
        },
      };
    case "NOT_FOUND":
      return {
        status: 404,
        body: {
          code: error.code,
          message: error.message,
          resource: error.resource,
        },
      };
    case "PERMISSION_DENIED":
      return {
        status: 403,
        body: { code: error.code, message: error.message },
      };
    case "CONFLICT":
      return {
        status: 409,
        body: { code: error.code, message: error.message },
      };
    case "RATE_LIMIT_EXCEEDED":
      return {
        status: 429,
        body: {
          code: error.code,
          message: error.message,
          retryAfter: error.retryAfterMs,
        },
      };
    case "INSUFFICIENT_BALANCE":
      return {
        status: 422,
        body: { code: error.code, message: error.message },
      };
    case "EXPIRED":
      return {
        status: 410,
        body: { code: error.code, message: error.message },
      };
    default:
      // ここに到達する場合、DomainError に新しいケースが追加されている
      return assertNever(error);
  }
}
```

### 2-4. Result 型とカスタムエラーの組み合わせ

```typescript
type DomainError = ValidationError | NotFoundError | PermissionError;

type DomainResult<T> = Result<T, DomainError>;

async function getUser(id: string): Promise<DomainResult<User>> {
  const user = await db.users.findById(id);
  if (!user) {
    return Err(new NotFoundError("User", id));
  }
  return Ok(user);
}

async function updateUser(
  requesterId: string,
  targetId: string,
  data: unknown
): Promise<DomainResult<User>> {
  // 権限チェック
  if (requesterId !== targetId) {
    return Err(new PermissionError("update", "User"));
  }

  // バリデーション
  const validation = userSchema.safeParse(data);
  if (!validation.success) {
    return Err(
      new ValidationError("Invalid user data", formatZodErrors(validation.error))
    );
  }

  // 更新
  const result = await getUser(targetId);
  if (isErr(result)) return result;

  const updated = await db.users.update(targetId, validation.data);
  return Ok(updated);
}
```

### 2-5. エラーファクトリパターン

大規模プロジェクトでは、エラーの生成を集約するファクトリを用意すると便利です。

```typescript
// エラーファクトリ
class AppErrors {
  // ─── バリデーションエラー ───
  static validation(fields: Record<string, string[]>): ValidationError {
    return new ValidationError("Validation failed", fields);
  }

  static requiredField(field: string): ValidationError {
    return new ValidationError(`${field} is required`, {
      [field]: [`${field}は必須です`],
    });
  }

  static invalidFormat(field: string, expected: string): ValidationError {
    return new ValidationError(`${field} has invalid format`, {
      [field]: [`${field}は${expected}の形式である必要があります`],
    });
  }

  // ─── 404 エラー ───
  static notFound(resource: string, id: string): NotFoundError {
    return new NotFoundError(resource, id);
  }

  static userNotFound(id: string): NotFoundError {
    return new NotFoundError("User", id);
  }

  static orderNotFound(id: string): NotFoundError {
    return new NotFoundError("Order", id);
  }

  // ─── 権限エラー ───
  static forbidden(action: string, resource: string): PermissionError {
    return new PermissionError(action, resource);
  }

  // ─── コンフリクトエラー ───
  static duplicate(
    resource: string,
    field: string,
    value: string
  ): ConflictError {
    return new ConflictError(resource, field, value);
  }

  static emailAlreadyExists(email: string): ConflictError {
    return new ConflictError("User", "email", email);
  }

  // ─── インフラエラー ───
  static database(message: string, cause?: unknown): DatabaseError {
    return new DatabaseError(message, undefined, cause);
  }

  static externalApi(
    endpoint: string,
    status?: number,
    cause?: unknown
  ): ExternalApiError {
    return new ExternalApiError(
      `External API error: ${endpoint}`,
      endpoint,
      status,
      cause
    );
  }
}

// 使用例
function createUser(data: CreateUserInput): Promise<DomainResult<User>> {
  const existing = await db.users.findByEmail(data.email);
  if (existing) {
    return Err(AppErrors.emailAlreadyExists(data.email));
  }
  // ...
}
```

### 2-6. エラーのシリアライゼーションとデシリアライゼーション

API レスポンスやログ出力のためのエラーシリアライゼーション。

```typescript
// エラーレスポンスの型
interface ErrorResponse {
  code: string;
  message: string;
  timestamp: string;
  requestId?: string;
  details?: Record<string, unknown>;
}

// エラーシリアライザ
class ErrorSerializer {
  static toResponse(error: AppError, requestId?: string): ErrorResponse {
    const base: ErrorResponse = {
      code: error.code,
      message: error.message,
      timestamp: error.timestamp.toISOString(),
      requestId,
    };

    // エラータイプに応じた詳細情報の追加
    if (error instanceof ValidationError) {
      base.details = { fields: error.fields };
    } else if (error instanceof NotFoundError) {
      base.details = { resource: error.resource, id: error.id };
    } else if (error instanceof RateLimitError) {
      base.details = {
        limit: error.limit,
        windowMs: error.windowMs,
        retryAfterMs: error.retryAfterMs,
      };
    }

    return base;
  }

  // ログ出力用（スタックトレース付き）
  static toLogEntry(error: AppError, context?: Record<string, unknown>) {
    return {
      level: error.statusCode >= 500 ? "error" : "warn",
      code: error.code,
      message: error.message,
      statusCode: error.statusCode,
      timestamp: error.timestamp.toISOString(),
      stack: error.stack,
      cause: error.cause,
      ...context,
    };
  }
}

// 構造化ログとの統合
import { Logger } from "pino";

function logError(logger: Logger, error: AppError, context?: Record<string, unknown>) {
  const entry = ErrorSerializer.toLogEntry(error, context);
  if (entry.level === "error") {
    logger.error(entry, error.message);
  } else {
    logger.warn(entry, error.message);
  }
}
```

---

## 3. zod によるバリデーション

### 3-1. スキーマ定義とエラー変換

```typescript
import { z } from "zod";

// スキーマ定義
const UserCreateSchema = z.object({
  name: z
    .string()
    .min(1, "名前は必須です")
    .max(100, "名前は100文字以内です"),
  email: z
    .string()
    .email("有効なメールアドレスを入力してください"),
  age: z
    .number()
    .int("年齢は整数を指定してください")
    .min(0, "年齢は0以上です")
    .max(150, "年齢は150以下です"),
});

// 型を自動推論
type UserCreate = z.infer<typeof UserCreateSchema>;
// => { name: string; email: string; age: number }

// zod → Result 型への変換
function validate<T>(
  schema: z.ZodSchema<T>,
  data: unknown
): Result<T, ValidationError> {
  const result = schema.safeParse(data);
  if (result.success) {
    return Ok(result.data);
  }

  const fields: Record<string, string[]> = {};
  for (const issue of result.error.issues) {
    const path = issue.path.join(".");
    if (!fields[path]) fields[path] = [];
    fields[path].push(issue.message);
  }

  return Err(new ValidationError("Validation failed", fields));
}
```

### 3-2. 高度な zod パターン

```typescript
// discriminatedUnion でリクエスト種別を判別
const PaymentSchema = z.discriminatedUnion("method", [
  z.object({
    method: z.literal("credit_card"),
    cardNumber: z.string().regex(/^\d{16}$/),
    expiry: z.string().regex(/^\d{2}\/\d{2}$/),
    cvv: z.string().regex(/^\d{3,4}$/),
  }),
  z.object({
    method: z.literal("bank_transfer"),
    bankCode: z.string().length(4),
    accountNumber: z.string().min(7).max(8),
  }),
  z.object({
    method: z.literal("wallet"),
    walletId: z.string().uuid(),
  }),
]);

type Payment = z.infer<typeof PaymentSchema>;

// transform でデータ整形
const DateRangeSchema = z
  .object({
    start: z.coerce.date(),
    end: z.coerce.date(),
  })
  .refine((data) => data.start < data.end, {
    message: "開始日は終了日より前でなければなりません",
    path: ["start"],
  });
```

### 3-3. 再利用可能なカスタムバリデータ

```typescript
// ─── カスタムバリデータの定義 ───

// 日本の電話番号
const JapanesePhoneNumber = z
  .string()
  .regex(
    /^0[0-9]{1,4}-?[0-9]{1,4}-?[0-9]{4}$/,
    "有効な日本の電話番号を入力してください"
  )
  .transform((val) => val.replace(/-/g, ""));

// 郵便番号
const JapanesePostalCode = z
  .string()
  .regex(/^\d{3}-?\d{4}$/, "有効な郵便番号を入力してください（例: 123-4567）")
  .transform((val) => val.replace(/-/, ""));

// パスワード強度
const StrongPassword = z
  .string()
  .min(8, "パスワードは8文字以上である必要があります")
  .max(128, "パスワードは128文字以下である必要があります")
  .refine(
    (val) => /[A-Z]/.test(val),
    "パスワードには大文字を1文字以上含める必要があります"
  )
  .refine(
    (val) => /[a-z]/.test(val),
    "パスワードには小文字を1文字以上含める必要があります"
  )
  .refine(
    (val) => /[0-9]/.test(val),
    "パスワードには数字を1文字以上含める必要があります"
  )
  .refine(
    (val) => /[!@#$%^&*(),.?":{}|<>]/.test(val),
    "パスワードには記号を1文字以上含める必要があります"
  );

// URL バリデータ（特定のドメインのみ許可）
function urlWithDomain(...domains: string[]) {
  return z
    .string()
    .url("有効なURLを入力してください")
    .refine(
      (val) => {
        try {
          const url = new URL(val);
          return domains.some(
            (d) => url.hostname === d || url.hostname.endsWith(`.${d}`)
          );
        } catch {
          return false;
        }
      },
      `許可されたドメイン: ${domains.join(", ")}`
    );
}

// ISO 8601 日付文字列
const ISODateString = z
  .string()
  .datetime({ message: "ISO 8601形式の日付文字列を入力してください" })
  .transform((val) => new Date(val));

// 非負整数（ページネーション用）
const PositiveInt = z.coerce
  .number()
  .int("整数を指定してください")
  .min(1, "1以上の値を指定してください");

const NonNegativeInt = z.coerce
  .number()
  .int("整数を指定してください")
  .min(0, "0以上の値を指定してください");

// ─── スキーマの組み合わせ例 ───

const AddressSchema = z.object({
  postalCode: JapanesePostalCode,
  prefecture: z.string().min(1, "都道府県は必須です"),
  city: z.string().min(1, "市区町村は必須です"),
  street: z.string().min(1, "番地は必須です"),
  building: z.string().optional(),
  phone: JapanesePhoneNumber.optional(),
});

const UserRegistrationSchema = z
  .object({
    name: z.string().min(1).max(100),
    email: z.string().email(),
    password: StrongPassword,
    passwordConfirmation: z.string(),
    address: AddressSchema,
    profileUrl: urlWithDomain("github.com", "twitter.com").optional(),
  })
  .refine((data) => data.password === data.passwordConfirmation, {
    message: "パスワードが一致しません",
    path: ["passwordConfirmation"],
  });

type UserRegistration = z.infer<typeof UserRegistrationSchema>;
```

### 3-4. zod のエラーメッセージのカスタマイズ

```typescript
// エラーマップによるグローバルなメッセージカスタマイズ
const japaneseErrorMap: z.ZodErrorMap = (issue, ctx) => {
  switch (issue.code) {
    case z.ZodIssueCode.invalid_type:
      if (issue.expected === "string") {
        return { message: "文字列を入力してください" };
      }
      if (issue.expected === "number") {
        return { message: "数値を入力してください" };
      }
      return { message: `${issue.expected}型の値を入力してください` };

    case z.ZodIssueCode.too_small:
      if (issue.type === "string") {
        return { message: `${issue.minimum}文字以上入力してください` };
      }
      if (issue.type === "number") {
        return { message: `${issue.minimum}以上の値を入力してください` };
      }
      if (issue.type === "array") {
        return { message: `${issue.minimum}個以上の要素が必要です` };
      }
      return { message: ctx.defaultError };

    case z.ZodIssueCode.too_big:
      if (issue.type === "string") {
        return { message: `${issue.maximum}文字以内で入力してください` };
      }
      if (issue.type === "number") {
        return { message: `${issue.maximum}以下の値を入力してください` };
      }
      return { message: ctx.defaultError };

    case z.ZodIssueCode.invalid_string:
      if (issue.validation === "email") {
        return { message: "有効なメールアドレスを入力してください" };
      }
      if (issue.validation === "url") {
        return { message: "有効なURLを入力してください" };
      }
      if (issue.validation === "uuid") {
        return { message: "有効なUUIDを入力してください" };
      }
      return { message: ctx.defaultError };

    default:
      return { message: ctx.defaultError };
  }
};

// グローバルに設定
z.setErrorMap(japaneseErrorMap);

// フォーム向けのエラーフォーマッタ
function formatZodErrorForForm(
  error: z.ZodError
): Record<string, string[]> {
  const formatted: Record<string, string[]> = {};

  for (const issue of error.issues) {
    const path = issue.path.length > 0 ? issue.path.join(".") : "_root";
    if (!formatted[path]) {
      formatted[path] = [];
    }
    formatted[path].push(issue.message);
  }

  return formatted;
}

// フラットなエラーメッセージリスト
function flattenZodErrors(error: z.ZodError): string[] {
  return error.issues.map((issue) => {
    const path = issue.path.join(".");
    return path ? `${path}: ${issue.message}` : issue.message;
  });
}
```

### 3-5. zod と API バリデーションの統合

```typescript
// Express ミドルウェアとしての zod バリデーション
import { Request, Response, NextFunction } from "express";

function validateBody<T>(schema: z.ZodSchema<T>) {
  return (req: Request, res: Response, next: NextFunction) => {
    const result = schema.safeParse(req.body);
    if (!result.success) {
      const errors = formatZodErrorForForm(result.error);
      return res.status(400).json({
        code: "VALIDATION_ERROR",
        message: "入力データが不正です",
        errors,
      });
    }
    req.body = result.data;
    next();
  };
}

function validateQuery<T>(schema: z.ZodSchema<T>) {
  return (req: Request, res: Response, next: NextFunction) => {
    const result = schema.safeParse(req.query);
    if (!result.success) {
      const errors = formatZodErrorForForm(result.error);
      return res.status(400).json({
        code: "VALIDATION_ERROR",
        message: "クエリパラメータが不正です",
        errors,
      });
    }
    // 型安全なクエリパラメータ
    (req as any).validatedQuery = result.data;
    next();
  };
}

function validateParams<T>(schema: z.ZodSchema<T>) {
  return (req: Request, res: Response, next: NextFunction) => {
    const result = schema.safeParse(req.params);
    if (!result.success) {
      const errors = formatZodErrorForForm(result.error);
      return res.status(400).json({
        code: "VALIDATION_ERROR",
        message: "パスパラメータが不正です",
        errors,
      });
    }
    (req as any).validatedParams = result.data;
    next();
  };
}

// 使用例
const PaginationSchema = z.object({
  page: z.coerce.number().int().min(1).default(1),
  limit: z.coerce.number().int().min(1).max(100).default(20),
  sort: z.enum(["created_at", "updated_at", "name"]).default("created_at"),
  order: z.enum(["asc", "desc"]).default("desc"),
});

const UserIdSchema = z.object({
  id: z.string().uuid("有効なユーザーIDを指定してください"),
});

app.get(
  "/api/users",
  validateQuery(PaginationSchema),
  async (req, res) => {
    const query = (req as any).validatedQuery;
    // query は { page: number; limit: number; sort: string; order: string } 型
    const users = await userService.list(query);
    res.json(users);
  }
);

app.get(
  "/api/users/:id",
  validateParams(UserIdSchema),
  async (req, res) => {
    const { id } = (req as any).validatedParams;
    const result = await userService.getById(id);
    if (isErr(result)) {
      return res.status(result.error.statusCode).json(result.error.toJSON());
    }
    res.json(result.value);
  }
);
```

### 3-6. zod と環境変数バリデーション

```typescript
// 環境変数スキーマ
const EnvSchema = z.object({
  // 必須
  NODE_ENV: z.enum(["development", "staging", "production"]),
  PORT: z.coerce.number().int().min(1).max(65535).default(3000),
  DATABASE_URL: z.string().url(),

  // Redis
  REDIS_HOST: z.string().default("localhost"),
  REDIS_PORT: z.coerce.number().int().default(6379),
  REDIS_PASSWORD: z.string().optional(),

  // JWT
  JWT_SECRET: z.string().min(32, "JWT_SECRETは32文字以上必要です"),
  JWT_EXPIRES_IN: z
    .string()
    .regex(/^\d+[smhd]$/, "例: 1h, 30m, 7d")
    .default("1h"),

  // 外部API
  STRIPE_SECRET_KEY: z.string().startsWith("sk_"),
  STRIPE_WEBHOOK_SECRET: z.string().startsWith("whsec_"),

  // S3
  AWS_ACCESS_KEY_ID: z.string().min(1),
  AWS_SECRET_ACCESS_KEY: z.string().min(1),
  S3_BUCKET_NAME: z.string().min(1),
  S3_REGION: z.string().default("ap-northeast-1"),

  // メール
  SMTP_HOST: z.string().optional(),
  SMTP_PORT: z.coerce.number().int().optional(),
  SMTP_USER: z.string().optional(),
  SMTP_PASSWORD: z.string().optional(),
  FROM_EMAIL: z.string().email().optional(),

  // ロギング
  LOG_LEVEL: z
    .enum(["trace", "debug", "info", "warn", "error", "fatal"])
    .default("info"),
});

type Env = z.infer<typeof EnvSchema>;

// 起動時にバリデーション
function loadEnv(): Env {
  const result = EnvSchema.safeParse(process.env);

  if (!result.success) {
    const errors = result.error.issues.map(
      (issue) => `  ${issue.path.join(".")}: ${issue.message}`
    );
    console.error("❌ 環境変数の設定エラー:");
    console.error(errors.join("\n"));
    process.exit(1);
  }

  return result.data;
}

// シングルトンとしてエクスポート
export const env = loadEnv();
```

---

## 4. 非同期エラーハンドリング

### 4-1. Promise と Result の統合

```typescript
// AsyncResult 型の定義
type AsyncResult<T, E> = Promise<Result<T, E>>;

// 非同期 Result ユーティリティ
const AsyncR = {
  // Promise<Result> の map
  map: async <T, U, E>(
    asyncResult: AsyncResult<T, E>,
    fn: (value: T) => U
  ): AsyncResult<U, E> => {
    const result = await asyncResult;
    return isOk(result) ? Ok(fn(result.value)) : result;
  },

  // Promise<Result> の flatMap
  flatMap: async <T, U, E>(
    asyncResult: AsyncResult<T, E>,
    fn: (value: T) => AsyncResult<U, E>
  ): AsyncResult<U, E> => {
    const result = await asyncResult;
    return isOk(result) ? fn(result.value) : result;
  },

  // Promise を Result に変換
  fromPromise: async <T, E>(
    promise: Promise<T>,
    errorMapper: (error: unknown) => E
  ): AsyncResult<T, E> => {
    try {
      const value = await promise;
      return Ok(value);
    } catch (error) {
      return Err(errorMapper(error));
    }
  },

  // 並列実行
  all: async <T, E>(
    results: AsyncResult<T, E>[]
  ): AsyncResult<T[], E> => {
    const resolved = await Promise.all(results);
    return combine(resolved);
  },

  // settled: すべての結果を返す（エラーも含む）
  allSettled: async <T, E>(
    results: AsyncResult<T, E>[]
  ): Promise<Result<T, E>[]> => {
    return Promise.all(results);
  },
};

// 使用例: 非同期処理のチェーン
async function processOrder(
  orderId: string,
  userId: string
): AsyncResult<OrderConfirmation, DomainError> {
  // 1. ユーザー取得
  const userResult = await getUser(userId);
  if (isErr(userResult)) return userResult;
  const user = userResult.value;

  // 2. 注文取得
  const orderResult = await getOrder(orderId);
  if (isErr(orderResult)) return orderResult;
  const order = orderResult.value;

  // 3. 権限チェック
  if (order.userId !== user.id) {
    return Err(new PermissionError("process", "Order"));
  }

  // 4. 支払い処理
  const paymentResult = await processPayment(order, user);
  if (isErr(paymentResult)) return paymentResult;

  // 5. 確認メール送信
  const emailResult = await sendConfirmation(user.email, order);
  if (isErr(emailResult)) {
    // メール送信失敗は致命的ではないのでログのみ
    console.warn("Failed to send confirmation email:", emailResult.error);
  }

  return Ok({
    orderId: order.id,
    status: "confirmed",
    paymentId: paymentResult.value.id,
    processedAt: new Date(),
  });
}
```

### 4-2. do記法風のパイプライン

early return が多くなる問題を解決する do 記法風のパターンです。

```typescript
// ResultBuilder: do記法風のチェーン
class ResultBuilder<E> {
  private steps: Map<string, unknown> = new Map();

  async bind<K extends string, T>(
    key: K,
    fn: () => AsyncResult<T, E>
  ): Promise<
    | { success: true; value: T }
    | { success: false; error: Err<E> }
  > {
    const result = await fn();
    if (isErr(result)) {
      return { success: false, error: result };
    }
    this.steps.set(key, result.value);
    return { success: true, value: result.value };
  }

  get<K extends string>(key: K): unknown {
    return this.steps.get(key);
  }
}

// Do記法ヘルパー
async function Do<T, E>(
  fn: (ctx: {
    bind: <V>(result: AsyncResult<V, E>) => Promise<V>;
  }) => AsyncResult<T, E>
): AsyncResult<T, E> {
  try {
    const ctx = {
      bind: async <V>(result: AsyncResult<V, E>): Promise<V> => {
        const r = await result;
        if (isErr(r)) {
          throw { _tag: "ResultError" as const, error: r };
        }
        return r.value;
      },
    };
    return await fn(ctx);
  } catch (e: unknown) {
    if (
      typeof e === "object" &&
      e !== null &&
      "_tag" in e &&
      (e as any)._tag === "ResultError"
    ) {
      return (e as any).error;
    }
    throw e;
  }
}

// 使用例: Do記法で平坦化
async function processOrder(
  orderId: string,
  userId: string
): AsyncResult<OrderConfirmation, DomainError> {
  return Do(async ({ bind }) => {
    const user = await bind(getUser(userId));
    const order = await bind(getOrder(orderId));

    if (order.userId !== user.id) {
      return Err(new PermissionError("process", "Order"));
    }

    const payment = await bind(processPayment(order, user));

    // メール送信は失敗しても続行
    await sendConfirmation(user.email, order).catch(() => {});

    return Ok({
      orderId: order.id,
      status: "confirmed" as const,
      paymentId: payment.id,
      processedAt: new Date(),
    });
  });
}
```

### 4-3. リトライパターン

```typescript
interface RetryOptions {
  maxRetries: number;
  initialDelayMs: number;
  maxDelayMs: number;
  backoffMultiplier: number;
  retryableErrors?: (error: unknown) => boolean;
}

const DEFAULT_RETRY_OPTIONS: RetryOptions = {
  maxRetries: 3,
  initialDelayMs: 100,
  maxDelayMs: 5000,
  backoffMultiplier: 2,
};

async function withRetry<T, E>(
  fn: () => AsyncResult<T, E>,
  options: Partial<RetryOptions> = {}
): AsyncResult<T, E> {
  const opts = { ...DEFAULT_RETRY_OPTIONS, ...options };
  let lastResult: Result<T, E> | undefined;
  let delay = opts.initialDelayMs;

  for (let attempt = 0; attempt <= opts.maxRetries; attempt++) {
    lastResult = await fn();

    if (isOk(lastResult)) {
      return lastResult;
    }

    // リトライ可能なエラーかチェック
    if (
      opts.retryableErrors &&
      !opts.retryableErrors(lastResult.error)
    ) {
      return lastResult;
    }

    // 最後の試行ではリトライしない
    if (attempt < opts.maxRetries) {
      // ジッター付きの指数バックオフ
      const jitter = Math.random() * delay * 0.1;
      await new Promise((resolve) =>
        setTimeout(resolve, Math.min(delay + jitter, opts.maxDelayMs))
      );
      delay *= opts.backoffMultiplier;
    }
  }

  return lastResult!;
}

// サーキットブレーカーパターン
class CircuitBreaker<T, E> {
  private failures = 0;
  private lastFailure: Date | null = null;
  private state: "closed" | "open" | "half-open" = "closed";

  constructor(
    private readonly fn: () => AsyncResult<T, E>,
    private readonly options: {
      failureThreshold: number;
      resetTimeoutMs: number;
      fallback?: () => AsyncResult<T, E>;
    }
  ) {}

  async execute(): AsyncResult<T, E> {
    if (this.state === "open") {
      // リセットタイムアウトが経過したかチェック
      const now = new Date();
      if (
        this.lastFailure &&
        now.getTime() - this.lastFailure.getTime() > this.options.resetTimeoutMs
      ) {
        this.state = "half-open";
      } else {
        // フォールバックを返すか、エラーを返す
        if (this.options.fallback) {
          return this.options.fallback();
        }
        return Err({
          code: "CIRCUIT_OPEN",
          message: "Circuit breaker is open",
        } as any);
      }
    }

    const result = await this.fn();

    if (isErr(result)) {
      this.failures++;
      this.lastFailure = new Date();

      if (this.failures >= this.options.failureThreshold) {
        this.state = "open";
      }
    } else {
      this.failures = 0;
      this.state = "closed";
    }

    return result;
  }
}

// 使用例
const fetchUserWithRetry = (id: string) =>
  withRetry(() => fetchUser(id), {
    maxRetries: 3,
    retryableErrors: (error) => {
      // ネットワークエラーのみリトライ
      return error instanceof ExternalApiError;
    },
  });

const userCircuitBreaker = new CircuitBreaker(
  () => fetchUser("123"),
  {
    failureThreshold: 5,
    resetTimeoutMs: 30000,
    fallback: () => Promise.resolve(Ok(getCachedUser("123"))),
  }
);
```

### 4-4. タイムアウトパターン

```typescript
class TimeoutError extends AppError {
  readonly code = "TIMEOUT";
  readonly statusCode = 408;

  constructor(
    public readonly operationName: string,
    public readonly timeoutMs: number
  ) {
    super(`Operation "${operationName}" timed out after ${timeoutMs}ms`);
  }
}

async function withTimeout<T, E>(
  fn: () => AsyncResult<T, E>,
  timeoutMs: number,
  operationName: string = "unknown"
): AsyncResult<T, E | TimeoutError> {
  const timeoutPromise = new Promise<Result<never, TimeoutError>>((resolve) =>
    setTimeout(
      () => resolve(Err(new TimeoutError(operationName, timeoutMs))),
      timeoutMs
    )
  );

  return Promise.race([fn(), timeoutPromise]);
}

// AbortController を使ったキャンセル可能な処理
async function fetchWithAbort<T>(
  url: string,
  options: { timeoutMs: number }
): AsyncResult<T, ExternalApiError | TimeoutError> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), options.timeoutMs);

  try {
    const response = await fetch(url, { signal: controller.signal });
    clearTimeout(timeoutId);

    if (!response.ok) {
      return Err(
        new ExternalApiError(
          `HTTP ${response.status}: ${response.statusText}`,
          url,
          response.status
        )
      );
    }

    const data = await response.json();
    return Ok(data as T);
  } catch (error) {
    clearTimeout(timeoutId);
    if (error instanceof DOMException && error.name === "AbortError") {
      return Err(new TimeoutError("fetch", options.timeoutMs));
    }
    return Err(new ExternalApiError(String(error), url));
  }
}
```

---

## 5. エラーの集約と変換

### 5-1. レイヤー間のエラー変換

```
+------------------+     +------------------+     +------------------+
| プレゼンテーション |     | アプリケーション   |     | ドメイン          |
|       層         |     |       層         |     |       層         |
+------------------+     +------------------+     +------------------+
| HttpError        | <-- | ApplicationError | <-- | DomainError      |
| - statusCode     |     | - DomainError    |     | - ValidationError|
| - body           |     | - AuthError      |     | - NotFoundError  |
|                  |     | - InputError     |     | - BusinessError  |
+------------------+     +------------------+     +------------------+
        ^                        ^                        ^
        |                        |                        |
+------------------+     +------------------+     +------------------+
| インフラ          |     | 外部サービス      |     | データベース      |
|       層         |     |       層         |     |       層         |
+------------------+     +------------------+     +------------------+
| InfraError       |     | ExternalApiError |     | DatabaseError    |
+------------------+     +------------------+     +------------------+
```

```typescript
// ─── レイヤー間のエラー変換マッパー ───

// インフラエラー → ドメインエラー
function infraToDomainError(error: InfraError): DomainError {
  if (error instanceof DatabaseError) {
    // ユニーク制約違反
    if (error.message.includes("unique constraint")) {
      return new ConflictError("Resource", "unknown", "unknown");
    }
    // その他のDBエラーはNotFoundとして扱うか再throw
    return new NotFoundError("Resource", "unknown");
  }
  // キャッチオール
  throw error; // インフラエラーはドメイン層で処理できない
}

// ドメインエラー → HTTPレスポンス
function domainToHttpResponse(error: DomainError): {
  status: number;
  body: ErrorResponse;
} {
  return {
    status: error.statusCode,
    body: ErrorSerializer.toResponse(error),
  };
}

// エラー変換パイプライン
class ErrorTransformer {
  private transformers: Map<
    string,
    (error: any) => AppError
  > = new Map();

  register<E extends AppError>(
    code: string,
    transformer: (error: E) => AppError
  ): this {
    this.transformers.set(code, transformer);
    return this;
  }

  transform(error: AppError): AppError {
    const transformer = this.transformers.get(error.code);
    if (transformer) {
      return transformer(error);
    }
    return error;
  }
}

// 使用例
const errorTransformer = new ErrorTransformer()
  .register("DATABASE_ERROR", (error: DatabaseError) => {
    if (error.message.includes("unique constraint")) {
      return new ConflictError("Resource", "field", "value");
    }
    return error;
  })
  .register("EXTERNAL_API_ERROR", (error: ExternalApiError) => {
    if (error.responseStatus === 404) {
      return new NotFoundError("ExternalResource", error.endpoint);
    }
    return error;
  });
```

### 5-2. 複数バリデーションエラーの集約

```typescript
// バリデーションエラーを集約するコレクター
class ValidationCollector {
  private errors: Record<string, string[]> = {};

  add(field: string, message: string): this {
    if (!this.errors[field]) {
      this.errors[field] = [];
    }
    this.errors[field].push(message);
    return this;
  }

  addIf(condition: boolean, field: string, message: string): this {
    if (condition) {
      this.add(field, message);
    }
    return this;
  }

  merge(other: ValidationCollector): this {
    for (const [field, messages] of Object.entries(other.errors)) {
      for (const message of messages) {
        this.add(field, message);
      }
    }
    return this;
  }

  hasErrors(): boolean {
    return Object.keys(this.errors).length > 0;
  }

  toResult<T>(value: T): Result<T, ValidationError> {
    if (this.hasErrors()) {
      return Err(new ValidationError("Validation failed", this.errors));
    }
    return Ok(value);
  }

  toValidationError(): ValidationError | null {
    if (this.hasErrors()) {
      return new ValidationError("Validation failed", this.errors);
    }
    return null;
  }
}

// 使用例: ビジネスルールのバリデーション
async function validateOrder(
  order: OrderInput,
  user: User
): Result<OrderInput, ValidationError> {
  const collector = new ValidationCollector();

  // zodスキーマの基本バリデーション
  const schemaResult = OrderSchema.safeParse(order);
  if (!schemaResult.success) {
    for (const issue of schemaResult.error.issues) {
      collector.add(issue.path.join("."), issue.message);
    }
  }

  // ビジネスルールのバリデーション
  collector
    .addIf(
      order.items.length === 0,
      "items",
      "注文には1つ以上の商品が必要です"
    )
    .addIf(
      order.items.length > 100,
      "items",
      "一度に注文できる商品は100個までです"
    )
    .addIf(
      !user.isVerified,
      "_root",
      "メールアドレスの認証が完了していません"
    )
    .addIf(
      user.isSuspended,
      "_root",
      "アカウントが停止されています"
    );

  // 在庫チェック（非同期）
  for (const item of order.items) {
    const stock = await getStock(item.productId);
    if (!stock || stock.quantity < item.quantity) {
      collector.add(
        `items.${item.productId}`,
        `在庫が不足しています（残り: ${stock?.quantity ?? 0}個）`
      );
    }
  }

  return collector.toResult(order);
}
```

### 5-3. エラーの伝播と変換チェーン

```typescript
// リポジトリ層: DB固有のエラーをドメインエラーに変換
class UserRepository {
  async findById(id: string): AsyncResult<User, NotFoundError | DatabaseError> {
    try {
      const row = await this.db.query("SELECT * FROM users WHERE id = $1", [id]);
      if (!row) {
        return Err(new NotFoundError("User", id));
      }
      return Ok(this.mapToUser(row));
    } catch (error) {
      return Err(new DatabaseError("Failed to query user", undefined, error));
    }
  }

  async create(data: CreateUserInput): AsyncResult<User, ConflictError | DatabaseError> {
    try {
      const row = await this.db.query(
        "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING *",
        [data.name, data.email]
      );
      return Ok(this.mapToUser(row));
    } catch (error: unknown) {
      // PostgreSQL のユニーク制約違反
      if (
        error instanceof Error &&
        error.message.includes("unique_violation")
      ) {
        return Err(new ConflictError("User", "email", data.email));
      }
      return Err(new DatabaseError("Failed to create user", undefined, error));
    }
  }

  private mapToUser(row: any): User {
    return {
      id: row.id,
      name: row.name,
      email: row.email,
      createdAt: row.created_at,
      updatedAt: row.updated_at,
    };
  }
}

// サービス層: リポジトリのエラーをアプリケーションエラーに変換
class UserService {
  constructor(
    private readonly userRepo: UserRepository,
    private readonly emailService: EmailService
  ) {}

  async register(
    input: unknown
  ): AsyncResult<User, ValidationError | ConflictError | InfraError> {
    // 1. バリデーション
    const validated = validate(UserRegistrationSchema, input);
    if (isErr(validated)) return validated;

    // 2. ユーザー作成
    const created = await this.userRepo.create(validated.value);
    if (isErr(created)) return created;

    // 3. ウェルカムメール（失敗しても続行）
    const emailResult = await this.emailService.sendWelcome(created.value.email);
    if (isErr(emailResult)) {
      // メール送信失敗はログに記録するが、ユーザー作成は成功とする
      console.warn("Welcome email failed:", emailResult.error);
    }

    return created;
  }
}

// コントローラ層: サービスのエラーをHTTPレスポンスに変換
class UserController {
  constructor(private readonly userService: UserService) {}

  async register(req: Request, res: Response): Promise<void> {
    const result = await this.userService.register(req.body);

    result.match({
      ok: (user) => {
        res.status(201).json(user);
      },
      err: (error) => {
        const response = domainToHttpResponse(error);
        res.status(response.status).json(response.body);
      },
    });
  }
}
```

---

## 6. 実務での統合パターン

### 6-1. Express でのグローバルエラーハンドリング

```typescript
import express, { Request, Response, NextFunction, ErrorRequestHandler } from "express";

// グローバルエラーハンドラ
const globalErrorHandler: ErrorRequestHandler = (
  error: unknown,
  req: Request,
  res: Response,
  _next: NextFunction
) => {
  // リクエストIDの取得
  const requestId = req.headers["x-request-id"] as string | undefined;

  // AppError のインスタンスかチェック
  if (error instanceof AppError) {
    const response = ErrorSerializer.toResponse(error, requestId);

    // 5xx エラーの場合はスタックトレースもログ
    if (error.statusCode >= 500) {
      console.error("Internal error:", {
        ...response,
        stack: error.stack,
        cause: error.cause,
      });
    }

    res.status(error.statusCode).json(response);
    return;
  }

  // 予期しないエラー
  console.error("Unexpected error:", error);
  res.status(500).json({
    code: "INTERNAL_ERROR",
    message: "An unexpected error occurred",
    requestId,
  });
};

// 非同期ルートハンドラのラッパー
function asyncHandler(
  fn: (req: Request, res: Response, next: NextFunction) => Promise<void>
) {
  return (req: Request, res: Response, next: NextFunction) => {
    fn(req, res, next).catch(next);
  };
}

// Result を使ったルートハンドラ
function resultHandler<T>(
  fn: (req: Request) => AsyncResult<T, AppError>
) {
  return asyncHandler(async (req, res) => {
    const result = await fn(req);

    if (isOk(result)) {
      res.json(result.value);
    } else {
      const error = result.error;
      const response = ErrorSerializer.toResponse(error);
      res.status(error.statusCode).json(response);
    }
  });
}

// アプリケーション設定
const app = express();
app.use(express.json());

// ルート定義
app.post(
  "/api/users",
  resultHandler(async (req) => {
    return userService.register(req.body);
  })
);

app.get(
  "/api/users/:id",
  resultHandler(async (req) => {
    return userService.getById(req.params.id);
  })
);

// グローバルエラーハンドラの登録（最後に）
app.use(globalErrorHandler);
```

### 6-2. NestJS でのエラーハンドリング

```typescript
import {
  ExceptionFilter,
  Catch,
  ArgumentsHost,
  HttpException,
  HttpStatus,
} from "@nestjs/common";
import { Response } from "express";

// NestJS 用の例外フィルタ
@Catch()
export class AppExceptionFilter implements ExceptionFilter {
  catch(exception: unknown, host: ArgumentsHost) {
    const ctx = host.switchToHttp();
    const response = ctx.getResponse<Response>();
    const request = ctx.getRequest();

    if (exception instanceof AppError) {
      response.status(exception.statusCode).json(
        ErrorSerializer.toResponse(exception, request.id)
      );
      return;
    }

    if (exception instanceof HttpException) {
      const status = exception.getStatus();
      const exceptionResponse = exception.getResponse();
      response.status(status).json(
        typeof exceptionResponse === "string"
          ? { code: "HTTP_ERROR", message: exceptionResponse }
          : exceptionResponse
      );
      return;
    }

    // 予期しないエラー
    console.error("Unhandled exception:", exception);
    response.status(HttpStatus.INTERNAL_SERVER_ERROR).json({
      code: "INTERNAL_ERROR",
      message: "Internal server error",
    });
  }
}

// Result を返すサービスの NestJS インターセプタ
import {
  Injectable,
  NestInterceptor,
  ExecutionContext,
  CallHandler,
} from "@nestjs/common";
import { Observable, map } from "rxjs";

@Injectable()
export class ResultInterceptor implements NestInterceptor {
  intercept(context: ExecutionContext, next: CallHandler): Observable<any> {
    return next.handle().pipe(
      map((result) => {
        if (result && typeof result === "object" && "_tag" in result) {
          if (result._tag === "Err") {
            const error = result.error;
            if (error instanceof AppError) {
              throw error; // AppExceptionFilter が処理
            }
          }
          if (result._tag === "Ok") {
            return result.value;
          }
        }
        return result;
      })
    );
  }
}

// コントローラでの使用
@Controller("users")
@UseFilters(AppExceptionFilter)
@UseInterceptors(ResultInterceptor)
export class UserController {
  constructor(private readonly userService: UserService) {}

  @Post()
  async create(@Body() body: unknown) {
    return this.userService.register(body);
    // Result<User, AppError> が返却される
    // ResultInterceptor が Ok → value, Err → throw に変換
  }

  @Get(":id")
  async findOne(@Param("id") id: string) {
    return this.userService.getById(id);
  }
}
```

### 6-3. tRPC でのエラーハンドリング

```typescript
import { initTRPC, TRPCError } from "@trpc/server";
import { z } from "zod";

const t = initTRPC.context<Context>().create({
  // zod のグローバルエラーフォーマッタ
  errorFormatter({ shape, error }) {
    return {
      ...shape,
      data: {
        ...shape.data,
        zodError:
          error.cause instanceof z.ZodError
            ? error.cause.flatten()
            : null,
        appError:
          error.cause instanceof AppError
            ? error.cause.toJSON()
            : null,
      },
    };
  },
});

// Result を TRPCError に変換するヘルパー
function resultToTRPC<T>(result: Result<T, AppError>): T {
  if (isOk(result)) {
    return result.value;
  }

  const error = result.error;
  const codeMap: Record<string, TRPCError["code"]> = {
    VALIDATION_ERROR: "BAD_REQUEST",
    NOT_FOUND: "NOT_FOUND",
    PERMISSION_DENIED: "FORBIDDEN",
    CONFLICT: "CONFLICT",
    RATE_LIMIT_EXCEEDED: "TOO_MANY_REQUESTS",
    INSUFFICIENT_BALANCE: "PRECONDITION_FAILED",
    EXPIRED: "PRECONDITION_FAILED",
  };

  throw new TRPCError({
    code: codeMap[error.code] ?? "INTERNAL_SERVER_ERROR",
    message: error.message,
    cause: error,
  });
}

// ルーター定義
const userRouter = t.router({
  create: t.procedure
    .input(UserCreateSchema)
    .mutation(async ({ input, ctx }) => {
      const result = await ctx.userService.register(input);
      return resultToTRPC(result);
    }),

  getById: t.procedure
    .input(z.object({ id: z.string().uuid() }))
    .query(async ({ input, ctx }) => {
      const result = await ctx.userService.getById(input.id);
      return resultToTRPC(result);
    }),

  list: t.procedure
    .input(PaginationSchema)
    .query(async ({ input, ctx }) => {
      const result = await ctx.userService.list(input);
      return resultToTRPC(result);
    }),
});
```

### 6-4. GraphQL でのエラーハンドリング

```typescript
import { GraphQLError } from "graphql";

// Result を GraphQL レスポンスに変換
function resultToGraphQL<T>(
  result: Result<T, AppError>
): T {
  if (isOk(result)) {
    return result.value;
  }

  const error = result.error;

  // GraphQL のエラーコード拡張
  throw new GraphQLError(error.message, {
    extensions: {
      code: error.code,
      statusCode: error.statusCode,
      ...(error instanceof ValidationError
        ? { fields: error.fields }
        : {}),
    },
  });
}

// Union 型によるエラー表現（GraphQL スタイル）
// GraphQL Schema:
// union UserResult = User | ValidationErrorPayload | NotFoundErrorPayload

interface UserResultSuccess {
  __typename: "User";
  id: string;
  name: string;
  email: string;
}

interface ValidationErrorPayload {
  __typename: "ValidationError";
  message: string;
  fields: Array<{ path: string; messages: string[] }>;
}

interface NotFoundErrorPayload {
  __typename: "NotFoundError";
  message: string;
  resource: string;
}

type UserResult = UserResultSuccess | ValidationErrorPayload | NotFoundErrorPayload;

function domainResultToGraphQLUnion(
  result: Result<User, DomainError>
): UserResult {
  if (isOk(result)) {
    return {
      __typename: "User",
      ...result.value,
    };
  }

  const error = result.error;
  if (error instanceof ValidationError) {
    return {
      __typename: "ValidationError",
      message: error.message,
      fields: Object.entries(error.fields).map(([path, messages]) => ({
        path,
        messages,
      })),
    };
  }
  if (error instanceof NotFoundError) {
    return {
      __typename: "NotFoundError",
      message: error.message,
      resource: error.resource,
    };
  }

  throw new GraphQLError("Internal error");
}
```

---

## 7. テスト戦略

### 7-1. Result 型のテスト

```typescript
import { describe, it, expect } from "vitest";

describe("Result utilities", () => {
  describe("map", () => {
    it("should transform Ok value", () => {
      const result = Ok(5);
      const mapped = map(result, (n) => n * 2);
      expect(mapped).toEqual(Ok(10));
    });

    it("should pass through Err", () => {
      const result = Err("error");
      const mapped = map(result, (n: number) => n * 2);
      expect(mapped).toEqual(Err("error"));
    });
  });

  describe("flatMap", () => {
    it("should chain Ok values", () => {
      const result = Ok(10);
      const chained = flatMap(result, (n) =>
        n > 0 ? Ok(n) : Err("must be positive")
      );
      expect(chained).toEqual(Ok(10));
    });

    it("should short-circuit on Err", () => {
      const result = Err("first error");
      const chained = flatMap(result, (n: number) => Ok(n * 2));
      expect(chained).toEqual(Err("first error"));
    });
  });

  describe("combine", () => {
    it("should combine all Ok results", () => {
      const results = [Ok(1), Ok(2), Ok(3)];
      expect(combine(results)).toEqual(Ok([1, 2, 3]));
    });

    it("should return first Err", () => {
      const results = [Ok(1), Err("error"), Ok(3)];
      expect(combine(results)).toEqual(Err("error"));
    });
  });
});

// カスタムマッチャー
expect.extend({
  toBeOk(received: Result<any, any>) {
    const pass = isOk(received);
    return {
      pass,
      message: () =>
        pass
          ? `Expected Result not to be Ok, got Ok(${JSON.stringify(received.value)})`
          : `Expected Result to be Ok, got Err(${JSON.stringify(
              (received as Err<any>).error
            )})`,
    };
  },
  toBeErr(received: Result<any, any>) {
    const pass = isErr(received);
    return {
      pass,
      message: () =>
        pass
          ? `Expected Result not to be Err, got Err(${JSON.stringify(received.error)})`
          : `Expected Result to be Err, got Ok(${JSON.stringify(
              (received as Ok<any>).value
            )})`,
    };
  },
  toBeOkWith(received: Result<any, any>, expected: any) {
    const pass = isOk(received) && JSON.stringify(received.value) === JSON.stringify(expected);
    return {
      pass,
      message: () =>
        pass
          ? `Expected Result not to be Ok(${JSON.stringify(expected)})`
          : `Expected Ok(${JSON.stringify(expected)}), got ${JSON.stringify(received)}`,
    };
  },
  toBeErrWith(received: Result<any, any>, expectedCode: string) {
    const pass =
      isErr(received) &&
      received.error instanceof AppError &&
      received.error.code === expectedCode;
    return {
      pass,
      message: () =>
        pass
          ? `Expected Result not to be Err with code "${expectedCode}"`
          : `Expected Err with code "${expectedCode}", got ${JSON.stringify(received)}`,
    };
  },
});

// 型拡張
declare module "vitest" {
  interface Assertion<T = any> {
    toBeOk(): void;
    toBeErr(): void;
    toBeOkWith(expected: any): void;
    toBeErrWith(expectedCode: string): void;
  }
}
```

### 7-2. サービス層のエラーテスト

```typescript
describe("UserService", () => {
  let userService: UserService;
  let mockUserRepo: jest.Mocked<UserRepository>;
  let mockEmailService: jest.Mocked<EmailService>;

  beforeEach(() => {
    mockUserRepo = {
      findById: jest.fn(),
      create: jest.fn(),
      findByEmail: jest.fn(),
    } as any;
    mockEmailService = {
      sendWelcome: jest.fn(),
    } as any;
    userService = new UserService(mockUserRepo, mockEmailService);
  });

  describe("register", () => {
    const validInput = {
      name: "Test User",
      email: "test@example.com",
      password: "Password123!",
      passwordConfirmation: "Password123!",
    };

    it("should return ValidationError for invalid input", async () => {
      const result = await userService.register({});
      expect(result).toBeErr();
      expect(result).toBeErrWith("VALIDATION_ERROR");
    });

    it("should return ConflictError for duplicate email", async () => {
      mockUserRepo.findByEmail.mockResolvedValue(Ok({ id: "existing" } as User));
      const result = await userService.register(validInput);
      expect(result).toBeErrWith("CONFLICT");
    });

    it("should create user successfully", async () => {
      mockUserRepo.findByEmail.mockResolvedValue(Ok(null as any));
      mockUserRepo.create.mockResolvedValue(
        Ok({ id: "new-id", ...validInput } as User)
      );
      mockEmailService.sendWelcome.mockResolvedValue(Ok(undefined));

      const result = await userService.register(validInput);
      expect(result).toBeOk();
    });

    it("should succeed even if welcome email fails", async () => {
      mockUserRepo.findByEmail.mockResolvedValue(Ok(null as any));
      mockUserRepo.create.mockResolvedValue(
        Ok({ id: "new-id", ...validInput } as User)
      );
      mockEmailService.sendWelcome.mockResolvedValue(
        Err(new InfraError("SMTP error", "email"))
      );

      const result = await userService.register(validInput);
      // メール送信失敗でもユーザー作成は成功
      expect(result).toBeOk();
    });

    it("should return DatabaseError on DB failure", async () => {
      mockUserRepo.findByEmail.mockResolvedValue(Ok(null as any));
      mockUserRepo.create.mockResolvedValue(
        Err(new DatabaseError("Connection refused"))
      );

      const result = await userService.register(validInput);
      expect(result).toBeErrWith("DATABASE_ERROR");
    });
  });
});
```

### 7-3. プロパティベーステスト

```typescript
import * as fc from "fast-check";

describe("Result laws", () => {
  // Functor則: map(id) === id
  it("should satisfy functor identity law", () => {
    fc.assert(
      fc.property(fc.integer(), (n) => {
        const result: Result<number, string> = Ok(n);
        const mapped = map(result, (x) => x);
        expect(mapped).toEqual(result);
      })
    );
  });

  // Functor則: map(f . g) === map(f) . map(g)
  it("should satisfy functor composition law", () => {
    fc.assert(
      fc.property(fc.integer(), (n) => {
        const f = (x: number) => x * 2;
        const g = (x: number) => x + 1;

        const result: Result<number, string> = Ok(n);
        const lhs = map(result, (x) => f(g(x)));
        const rhs = map(map(result, g), f);

        expect(lhs).toEqual(rhs);
      })
    );
  });

  // Monad則: flatMap(Ok) === id
  it("should satisfy monad left identity", () => {
    fc.assert(
      fc.property(fc.integer(), (n) => {
        const f = (x: number): Result<number, string> =>
          x > 0 ? Ok(x) : Err("negative");

        const lhs = flatMap(Ok(n), f);
        const rhs = f(n);

        expect(lhs).toEqual(rhs);
      })
    );
  });
});

// バリデーションのプロパティテスト
describe("Validation properties", () => {
  it("should accept any valid email", () => {
    fc.assert(
      fc.property(fc.emailAddress(), (email) => {
        const result = z.string().email().safeParse(email);
        expect(result.success).toBe(true);
      })
    );
  });

  it("should reject non-email strings", () => {
    fc.assert(
      fc.property(
        fc.string().filter((s) => !s.includes("@")),
        (notEmail) => {
          const result = z.string().email().safeParse(notEmail);
          expect(result.success).toBe(false);
        }
      )
    );
  });
});
```

### 7-4. E2E テストでのエラーレスポンス検証

```typescript
import request from "supertest";

describe("POST /api/users", () => {
  it("should return 400 with validation errors for invalid input", async () => {
    const response = await request(app)
      .post("/api/users")
      .send({ name: "", email: "invalid" })
      .expect(400);

    expect(response.body).toMatchObject({
      code: "VALIDATION_ERROR",
      message: expect.any(String),
    });
    expect(response.body.details.fields).toHaveProperty("name");
    expect(response.body.details.fields).toHaveProperty("email");
  });

  it("should return 409 for duplicate email", async () => {
    // まずユーザーを作成
    await request(app)
      .post("/api/users")
      .send({
        name: "Test",
        email: "dup@example.com",
        password: "Password123!",
        passwordConfirmation: "Password123!",
      })
      .expect(201);

    // 同じメールで再度作成
    const response = await request(app)
      .post("/api/users")
      .send({
        name: "Test2",
        email: "dup@example.com",
        password: "Password123!",
        passwordConfirmation: "Password123!",
      })
      .expect(409);

    expect(response.body.code).toBe("CONFLICT");
  });

  it("should return 404 for non-existent user", async () => {
    const response = await request(app)
      .get("/api/users/non-existent-id")
      .expect(404);

    expect(response.body.code).toBe("NOT_FOUND");
  });
});
```

---

## 8. パフォーマンス考慮事項

### 8-1. Result 型のオーバーヘッド

```typescript
// ─── ベンチマーク: Result 型 vs 例外 ───

// 例外ベース
function divideWithThrow(a: number, b: number): number {
  if (b === 0) throw new Error("Division by zero");
  return a / b;
}

// Result ベース
function divideWithResult(a: number, b: number): Result<number, string> {
  if (b === 0) return Err("Division by zero");
  return Ok(a / b);
}

// ベンチマーク結果（概算）:
// 正常系:
//   例外ベース:  ~5ns/op
//   Result ベース: ~20ns/op (オブジェクト生成のオーバーヘッド)
//
// エラー系:
//   例外ベース:  ~10,000ns/op (スタックトレース生成が重い)
//   Result ベース: ~20ns/op (一定)
//
// 結論: エラーが頻繁に発生するパスでは Result 型が圧倒的に高速

// ─── パフォーマンス最適化のテクニック ───

// 1. シングルトン Err パターン（同じエラーを繰り返し返す場合）
const DIVISION_BY_ZERO_ERR = Err("Division by zero") as Result<never, string>;

function divideFast(a: number, b: number): Result<number, string> {
  if (b === 0) return DIVISION_BY_ZERO_ERR;
  return Ok(a / b);
}

// 2. Error クラスのスタックトレース無効化
class LightweightError extends AppError {
  constructor(
    message: string,
    public readonly code: string,
    public readonly statusCode: number
  ) {
    super(message);
    // スタックトレースの生成をスキップ
    // パフォーマンスは向上するが、デバッグ情報は失われる
    this.stack = undefined;
  }
}

// 3. 遅延エラー生成
function divideWithLazy(a: number, b: number): Result<number, () => Error> {
  if (b === 0) return Err(() => new Error("Division by zero"));
  return Ok(a / b);
}
// エラーの詳細が必要な時だけ生成
const result = divideWithLazy(1, 0);
if (isErr(result)) {
  const error = result.error(); // ここで初めて Error が生成される
  console.error(error.stack);
}
```

### 8-2. zod のパフォーマンス最適化

```typescript
// 1. スキーマの事前コンパイル
// Bad: リクエストごとにスキーマを作成
app.post("/api/users", (req, res) => {
  const schema = z.object({ name: z.string() }); // 毎回作成される
  schema.parse(req.body);
});

// Good: スキーマはモジュールレベルで定義
const UserSchema = z.object({ name: z.string() });
app.post("/api/users", (req, res) => {
  UserSchema.parse(req.body); // 再利用される
});

// 2. 大量データのバリデーション
// Bad: 配列全体を一度にバリデーション
const LargeArraySchema = z.array(UserSchema);

// Good: サイズ制限付き + ストリーム処理
const BoundedArraySchema = z.array(UserSchema).max(1000);

// 3. coerce は必要な場合のみ
// Bad: 全フィールドに coerce
const schema1 = z.object({
  id: z.coerce.string(),   // 不要な変換
  name: z.coerce.string(), // 不要な変換
  age: z.coerce.number(),  // これは必要
});

// Good: 必要なフィールドのみ coerce
const schema2 = z.object({
  id: z.string(),
  name: z.string(),
  age: z.coerce.number(), // クエリパラメータからの変換に必要
});

// 4. typia との比較（コンパイル時コード生成）
// zodよりも10-100倍高速だが、ランタイムスキーマ操作はできない
import typia from "typia";

interface UserInput {
  name: string;
  email: string & typia.tags.Format<"email">;
  age: number & typia.tags.Minimum<0> & typia.tags.Maximum<150>;
}

// コンパイル時にバリデーションコードが生成される
const validateUser = typia.createValidate<UserInput>();
const result = validateUser(input);
```

---

## 比較表

### エラー処理戦略の比較

| 戦略 | 型安全性 | コスト | 可読性 | 適用場面 |
|------|---------|--------|--------|---------|
| try-catch | 低 (unknown) | 低 | 中 | 外部ライブラリ呼び出し |
| Result 型 | 高 | 中 | 高 | ドメインロジック |
| Either (fp-ts) | 高 | 高 | 中 | 関数型スタイル全体 |
| zod safeParse | 高 | 低 | 高 | 入力バリデーション |
| Effect-ts | 最高 | 高 | 低〜中 | 大規模エフェクト管理 |

### Result 型ライブラリ比較

| ライブラリ | バンドルサイズ | API スタイル | チェーン | パターンマッチ |
|-----------|-------------|-------------|---------|--------------|
| 自前実装 | 0 KB | 関数型 | 手動 | switch/if |
| neverthrow | ~2 KB | メソッドチェーン | `.andThen()` | `.match()` |
| ts-results | ~1 KB | Rust風 | `.map()` | `.match()` |
| fp-ts Either | ~15 KB | 関数型(pipe) | `pipe()` | `fold()` |
| effect/Either | ~50 KB+ | Effect風 | `Effect.map` | `Effect.match` |

### バリデーションライブラリ比較

| ライブラリ | バンドルサイズ | 速度 | 型推論 | スキーマ記述 | エコシステム |
|-----------|-------------|------|--------|------------|------------|
| zod | ~13 KB | 中 | 優秀 | メソッドチェーン | 豊富 |
| yup | ~15 KB | 中 | 良好 | メソッドチェーン | 豊富 |
| io-ts | ~8 KB | 中 | 優秀 | 関数型 | 中程度 |
| typia | 0 KB (コンパイル時) | 最速 | 完全 | TypeScript型 | 少ない |
| valibot | ~1 KB | 高速 | 優秀 | 関数型 | 成長中 |
| arktype | ~6 KB | 高速 | 優秀 | テンプレートリテラル | 少ない |

---

## アンチパターン

### AP-1: catch で型情報を握りつぶす

```typescript
// NG: error が unknown のまま処理
async function fetchUser(id: string): Promise<User> {
  try {
    return await api.get(`/users/${id}`);
  } catch (error) {
    // error は unknown -- 型情報なし
    console.log(error.message); // コンパイルエラーにならないが危険
    throw error; // 呼び出し元も unknown
  }
}

// OK: Result 型で型を保持
async function fetchUser(id: string): Promise<Result<User, ApiError>> {
  try {
    const user = await api.get(`/users/${id}`);
    return Ok(user);
  } catch (error) {
    return Err(ApiError.fromUnknown(error));
  }
}
```

### AP-2: エラーを文字列で判別する

```typescript
// NG: 文字列比較は脆い
try {
  await saveUser(data);
} catch (e) {
  if (e.message.includes("duplicate")) {  // タイポしても気づけない
    // ...
  }
}

// OK: 型で判別
const result = await saveUser(data);
if (isErr(result)) {
  switch (result.error.code) {
    case "VALIDATION_ERROR":   // リテラル型で補完が効く
      handleValidation(result.error);
      break;
    case "NOT_FOUND":
      handleNotFound(result.error);
      break;
  }
}
```

### AP-3: バリデーションなしで外部入力を信頼する

```typescript
// NG: req.body をそのまま型アサーション
app.post("/users", (req, res) => {
  const data = req.body as UserCreate; // 実行時チェックなし
  db.users.create(data);              // 不正データが DB に入る
});

// OK: zod で検証
app.post("/users", (req, res) => {
  const result = validate(UserCreateSchema, req.body);
  if (isErr(result)) {
    return res.status(400).json(result.error.toJSON());
  }
  db.users.create(result.value); // 検証済みデータ
});
```

### AP-4: 過剰な try-catch のネスト

```typescript
// NG: ネストが深く可読性が低い
async function processOrder(orderId: string) {
  try {
    const order = await getOrder(orderId);
    try {
      const payment = await processPayment(order);
      try {
        await sendReceipt(order, payment);
      } catch (e) {
        console.error("Receipt failed:", e);
      }
    } catch (e) {
      console.error("Payment failed:", e);
      throw e;
    }
  } catch (e) {
    console.error("Order failed:", e);
    throw e;
  }
}

// OK: Result 型でフラットに
async function processOrder(
  orderId: string
): AsyncResult<OrderConfirmation, DomainError> {
  const order = await getOrder(orderId);
  if (isErr(order)) return order;

  const payment = await processPayment(order.value);
  if (isErr(payment)) return payment;

  // メール送信は失敗しても続行
  await sendReceipt(order.value, payment.value);

  return Ok({ orderId, paymentId: payment.value.id });
}
```

### AP-5: エラーの握りつぶし

```typescript
// NG: エラーを完全に無視
async function fetchData() {
  try {
    return await api.get("/data");
  } catch {
    return null; // 何が起きたか不明
  }
}

// OK: エラーを適切に処理・変換
async function fetchData(): AsyncResult<Data, ApiError> {
  return AsyncR.fromPromise(
    api.get("/data"),
    (error) => new ApiError("FETCH_FAILED", String(error))
  );
}
```

### AP-6: unwrap の乱用

```typescript
// NG: unwrap はパニックを起こす
const user = (await getUser(id)).unwrap(); // エラー時に throw

// OK: パターンマッチで安全に処理
const result = await getUser(id);
if (isErr(result)) {
  return handleError(result.error);
}
const user = result.value;

// または unwrapOr でデフォルト値
const user = unwrapOr(await getUser(id), defaultUser);
```

### AP-7: 汎用的すぎるエラー型

```typescript
// NG: 全てを Error クラスで表現
function createUser(data: unknown): Result<User, Error> {
  // Error では何が起きたかわからない
}

// OK: 具体的なエラー型を使用
function createUser(
  data: unknown
): Result<User, ValidationError | ConflictError | DatabaseError> {
  // 呼び出し元で適切にハンドリングできる
}
```

---

## エラー処理フロー全体像

```
外部入力 (HTTP, File, ENV)
    |
    v
+------------------+
| zod バリデーション |---Err---> 400 Bad Request
+------------------+
    | Ok
    v
+------------------+
| ドメインロジック   |---Err---> DomainError
| (Result<T, E>)   |           |
+------------------+           +---> NotFoundError    -> 404
    | Ok                       +---> PermissionError  -> 403
    v                          +---> ConflictError    -> 409
+------------------+           +---> BusinessError    -> 422
| 永続化 / 外部API  |---Err---> InfraError -> 500
+------------------+           |
    | Ok                       +---> DatabaseError    -> 500
    v                          +---> ExternalApiError -> 502
 成功レスポンス 200            +---> TimeoutError     -> 408
                               +---> RateLimitError   -> 429
```

### 詳細フロー: ユーザー登録

```
POST /api/users { name, email, password }
    |
    v
+------------------------------+
| Express Middleware            |
| - JSON パース                 |
| - リクエストID付与            |
| - レート制限チェック           |
+------------------------------+
    |
    v
+------------------------------+
| Validation Layer              |
| - zod スキーマバリデーション    |
| - パスワード強度チェック        |
+------------------------------+
    |
    v (ValidationError → 400)
+------------------------------+
| Application Service           |
| - 重複メールチェック           |
| - ユーザー作成                 |
| - ウェルカムメール送信          |
+------------------------------+
    |
    v (ConflictError → 409)
+------------------------------+
| Repository Layer              |
| - SQL INSERT                  |
| - ユニーク制約チェック          |
+------------------------------+
    |
    v (DatabaseError → 500)
+------------------------------+
| Email Service                 |
| - SMTP 送信                   |
| - 失敗時は warn ログのみ       |
+------------------------------+
    |
    v
201 Created { id, name, email }
```

---

## 設計ガイドライン

### エラー設計のチェックリスト

| チェック項目 | 説明 |
|------------|------|
| エラーコードが一意か | 同じコードが異なる意味で使われていないか |
| エラーメッセージがユーザーフレンドリーか | 技術的な詳細は details に、メッセージは理解しやすく |
| HTTP ステータスコードが適切か | 4xx と 5xx の使い分けが正しいか |
| 機密情報が漏れていないか | スタックトレースやDB情報がクライアントに返されていないか |
| ログレベルが適切か | 4xx は warn、5xx は error |
| エラーがテスト可能か | エラーパスのテストが書きやすいか |
| 網羅性チェックがあるか | 新しいエラーを追加した時にコンパイルエラーになるか |
| リトライ可能性が明確か | クライアントがリトライすべきかどうかわかるか |

### レイヤー別エラー処理方針

| レイヤー | エラー処理方針 |
|---------|--------------|
| プレゼンテーション層 | エラーをHTTPレスポンスに変換、ログ出力 |
| アプリケーション層 | ドメインエラーの集約、トランザクション管理 |
| ドメイン層 | Result 型でビジネスルール違反を表現、throw 禁止 |
| インフラ層 | 外部サービスの例外を Result 型に変換 |
| 共通/横断 | グローバルエラーハンドラ、エラー監視 |

### エラーメッセージの国際化

```typescript
// エラーコードとメッセージの分離
const ERROR_MESSAGES: Record<string, Record<ErrorCode, string>> = {
  ja: {
    VALIDATION_ERROR: "入力データが不正です",
    NOT_FOUND: "リソースが見つかりません",
    PERMISSION_DENIED: "アクセス権限がありません",
    CONFLICT: "リソースが既に存在します",
    RATE_LIMIT_EXCEEDED: "リクエスト数が上限を超えました",
    INSUFFICIENT_BALANCE: "残高が不足しています",
    EXPIRED: "有効期限が切れています",
    DATABASE_ERROR: "データベースエラーが発生しました",
    EXTERNAL_API_ERROR: "外部サービスとの通信に失敗しました",
    CACHE_ERROR: "キャッシュエラーが発生しました",
  },
  en: {
    VALIDATION_ERROR: "Invalid input data",
    NOT_FOUND: "Resource not found",
    PERMISSION_DENIED: "Permission denied",
    CONFLICT: "Resource already exists",
    RATE_LIMIT_EXCEEDED: "Rate limit exceeded",
    INSUFFICIENT_BALANCE: "Insufficient balance",
    EXPIRED: "Resource has expired",
    DATABASE_ERROR: "Database error occurred",
    EXTERNAL_API_ERROR: "External service communication failed",
    CACHE_ERROR: "Cache error occurred",
  },
};

function getLocalizedMessage(code: ErrorCode, locale: string = "ja"): string {
  return ERROR_MESSAGES[locale]?.[code] ?? ERROR_MESSAGES["en"][code] ?? code;
}
```

---

## FAQ

### Q1: Result 型を使うと try-catch は完全に不要になりますか？

いいえ。外部ライブラリ（DB ドライバ、HTTP クライアントなど）は例外を throw するため、境界層（アダプター層）では try-catch が必要です。そこで例外を catch し、Result 型に変換する「境界パターン」を使います。ドメインロジック内部では Result 型のみを使い、throw を禁止するのがベストプラクティスです。

### Q2: neverthrow と自前 Result 型のどちらを使うべきですか？

小規模プロジェクトやライブラリでは自前実装で十分です。チームで統一したい場合やメソッドチェーン（`.andThen()`, `.map()`, `.match()`）を多用する場合は neverthrow が便利です。neverthrow は ~2KB と軽量で、TypeScript 専用に設計されています。

### Q3: zod のパフォーマンスは問題になりませんか？

通常の Web アプリケーションでは問題になりません。zod の検証は 1 リクエストあたりマイクロ秒〜ミリ秒単位です。ただし、大量データ（数万件の配列）を検証する場合は、`.parse()` の前にサイズチェックを入れるか、ストリーム処理を検討してください。パフォーマンスが問題になる場合は、typia（コンパイル時コード生成）や valibot（軽量バリデーション）も選択肢です。

### Q4: エラーのログ出力はどの層で行うべきですか？

エラーのログ出力は原則としてアプリケーション層（コントローラーやハンドラー）で一括して行います。ドメインロジック内でログ出力すると、テスト時にノイズになるうえ、同じエラーが複数回ログに出力される問題が発生します。

### Q5: Result 型を使う場合、Promise.reject は使わないべきですか？

はい。Result 型を使っている場合、`Promise.reject` は使わず `Promise<Result<T, E>>` を返すようにします。`Promise.reject` を使うと、呼び出し元で `try-catch` が必要になり、Result 型のメリットが失われます。ただし、ライブラリ層では `Promise.reject` が使われることもあるため、境界層で `fromPromise` を使って変換します。

### Q6: Effect-ts と neverthrow のどちらを選ぶべきですか？

neverthrow は Result 型に特化した軽量ライブラリで、既存のプロジェクトに段階的に導入できます。Effect-ts はより包括的なエフェクトシステムで、依存性注入、スケジューリング、並行処理なども含みます。新規プロジェクトで関数型プログラミングを全面的に採用するなら Effect-ts、既存プロジェクトのエラーハンドリング改善なら neverthrow がおすすめです。

### Q7: フロントエンドでも Result 型を使うべきですか？

フロントエンドでも Result 型は有効ですが、使い方が異なります。API 呼び出しの結果を Result 型で表現し、コンポーネント内でパターンマッチすることで、エラー状態の表示を型安全に行えます。ただし、React のエラーバウンダリや Vue のエラーハンドリングなど、フレームワーク固有の仕組みとの統合も考慮する必要があります。

### Q8: Branded 型とバリデーションの関係は？

Branded 型は「バリデーション済み」であることを型レベルで保証するパターンです。zod でバリデーションした後に Branded 型に変換することで、「このデータはバリデーション済みである」ことを型システムで表現できます。詳しくは [Branded Types パターン](./03-branded-types.md) を参照してください。

### Q9: マイクロサービス間のエラー伝播はどうすべきですか？

マイクロサービス間では、エラーコードとメッセージを標準化したエラーレスポンス（RFC 7807 Problem Details など）で伝播します。受信側で外部サービスのエラーレスポンスを自サービスのエラー型に変換するアダプターを実装します。gRPC の場合は Status Code を使用し、REST API の場合は HTTP ステータスコード + エラーコードの組み合わせが一般的です。

### Q10: テストで特定のエラーを期待する場合のベストプラクティスは？

Result 型を使っている場合、カスタムマッチャー（`toBeErrWith`）を定義して、エラーコードで検証するのがベストです。例外ベースの場合は `expect(fn).rejects.toThrow(SpecificError)` を使います。いずれの場合も、エラーメッセージの文字列比較ではなく、エラーの型やコードで検証してください。

---

## まとめ表

| 概念 | 要点 |
|------|------|
| Result 型 | `Ok<T> \| Err<E>` で成功/失敗を型安全に表現 |
| カスタムエラー | `code` リテラル型で判別可能なエラー階層を設計 |
| 境界パターン | try-catch は外部ライブラリとの境界のみで使用 |
| zod バリデーション | `safeParse` + Result 変換で入力を安全に処理 |
| 網羅性チェック | switch + `never` 型でエラーケースの漏れを防止 |
| エラー変換 | 各層で適切なエラー型に変換しながら伝播 |
| 非同期統合 | `AsyncResult<T, E>` で Promise と Result を統合 |
| リトライ | 指数バックオフ + サーキットブレーカーで耐障害性向上 |
| テスト戦略 | カスタムマッチャーでエラーパスを明確にテスト |
| パフォーマンス | エラー頻度が高い場合、Result 型は例外より高速 |

---

## 次に読むべきガイド

- [判別共用体パターン](./02-discriminated-unions.md) -- Result 型の基盤となる判別共用体の詳細
- [Branded Types パターン](./03-branded-types.md) -- バリデーション済みの値を型で保証する技法
- [Zod バリデーション](../04-ecosystem/00-zod-validation.md) -- zod の全機能と高度なパターン
- [Effect-ts](../04-ecosystem/03-effect-ts.md) -- より高度なエフェクトシステムによるエラー管理
- [依存性注入](./04-dependency-injection.md) -- テスタビリティとエラーハンドリングの統合

---

## 参考文献

1. **neverthrow** -- Type-Safe Error Handling in TypeScript
   https://github.com/supermacro/neverthrow

2. **Zod Documentation** -- TypeScript-first schema validation
   https://zod.dev/

3. **Rust Error Handling** -- The Rust Programming Language, Chapter 9
   https://doc.rust-lang.org/book/ch09-00-error-handling.html

4. **Parse, don't validate** -- Alexis King (2019)
   https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/

5. **Effect-ts Documentation** -- TypeScript Effect System
   https://effect.website/

6. **RFC 7807** -- Problem Details for HTTP APIs
   https://datatracker.ietf.org/doc/html/rfc7807

7. **typia** -- Super-fast Runtime Validators
   https://typia.io/

8. **valibot** -- Modular and type-safe schema validation
   https://valibot.dev/

9. **Railway Oriented Programming** -- Scott Wlaschin
   https://fsharpforfunandprofit.com/rop/

10. **Functional Error Handling** -- Matt Pocock
    https://www.mattpocock.com/
