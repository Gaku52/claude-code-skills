# TypeScript エラーハンドリングパターン

> Result型、カスタムエラー階層、zodバリデーションを組み合わせた堅牢なエラー処理戦略

## この章で学ぶこと

1. **Result型パターン** -- 例外を使わずに型安全にエラーを表現する方法
2. **カスタムエラー階層** -- ドメイン固有のエラークラスを設計し、エラーの種別を型で判別する技法
3. **zodによるバリデーション** -- ランタイムバリデーションと型推論を統合し、外部入力を安全に処理する方法

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

### 2-2. Result 型とカスタムエラーの組み合わせ

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
+------------------+           +---> NotFoundError  -> 404
    | Ok                       +---> PermissionError -> 403
    v                          +---> ConflictError   -> 409
+------------------+
| 永続化 / 外部API  |---Err---> InfraError -> 500
+------------------+
    | Ok
    v
 成功レスポンス 200
```

---

## FAQ

### Q1: Result 型を使うと try-catch は完全に不要になりますか？

いいえ。外部ライブラリ（DB ドライバ、HTTP クライアントなど）は例外を throw するため、境界層（アダプター層）では try-catch が必要です。そこで例外を catch し、Result 型に変換する「境界パターン」を使います。ドメインロジック内部では Result 型のみを使い、throw を禁止するのがベストプラクティスです。

### Q2: neverthrow と自前 Result 型のどちらを使うべきですか？

小規模プロジェクトやライブラリでは自前実装で十分です。チームで統一したい場合やメソッドチェーン（`.andThen()`, `.map()`, `.match()`）を多用する場合は neverthrow が便利です。neverthrow は ~2KB と軽量で、TypeScript 専用に設計されています。

### Q3: zod のパフォーマンスは問題になりませんか？

通常の Web アプリケーションでは問題になりません。zod の検証は 1 リクエストあたりマイクロ秒〜ミリ秒単位です。ただし、大量データ（数万件の配列）を検証する場合は、`.parse()` の前にサイズチェックを入れるか、ストリーム処理を検討してください。パフォーマンスが問題になる場合は、typia（コンパイル時コード生成）も選択肢です。

### Q4: エラーのログ出力はどの層で行うべきですか？

エラーのログ出力は原則としてアプリケーション層（コントローラーやハンドラー）で一括して行います。ドメインロジック内でログ出力すると、テスト時にノイズになるうえ、同じエラーが複数回ログに出力される問題が発生します。

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

---

## 次に読むべきガイド

- [判別共用体パターン](./02-discriminated-unions.md) -- Result 型の基盤となる判別共用体の詳細
- [Zod バリデーション](../04-ecosystem/00-zod-validation.md) -- zod の全機能と高度なパターン
- [Effect-ts](../04-ecosystem/03-effect-ts.md) -- より高度なエフェクトシステムによるエラー管理

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
