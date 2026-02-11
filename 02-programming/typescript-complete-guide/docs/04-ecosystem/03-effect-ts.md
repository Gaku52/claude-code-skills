# Effect-ts 完全ガイド

> TypeScript のためのエフェクトシステム -- 依存性注入、エラー処理、並行処理を型レベルで管理する

## この章で学ぶこと

1. **Effect の基本概念** -- `Effect<A, E, R>` 型の意味、基本的なパイプライン構築、実行方法
2. **エラー管理とサービス** -- 型付きエラー、Layer による DI、Resource 管理のパターン
3. **並行処理とストリーム** -- Fiber、Schedule、Stream を使った高度な非同期処理パターン

---

## 1. Effect の基本

### 1-1. Effect<A, E, R> 型

```
Effect<A, E, R> の3つの型パラメータ:

  Effect<A, E, R>
          |  |  |
          |  |  +--- R: Requirements (必要な依存)
          |  +------ E: Error (起こりうるエラーの型)
          +--------- A: Success (成功時の値の型)

  例:
  Effect<User, NotFoundError | DbError, UserRepo & Logger>

  意味:
  - 成功すると User を返す
  - NotFoundError または DbError が発生しうる
  - 実行には UserRepo と Logger が必要
```

```typescript
import { Effect, pipe } from "effect";

// 基本的な Effect の作成
// 成功する Effect
const succeed: Effect.Effect<number> = Effect.succeed(42);

// 失敗する Effect
const fail: Effect.Effect<never, Error> = Effect.fail(new Error("boom"));

// 同期処理を Effect に変換
const sync: Effect.Effect<number> = Effect.sync(() => {
  return Math.random();
});

// 非同期処理を Effect に変換
const async_: Effect.Effect<string, Error> = Effect.tryPromise({
  try: () => fetch("/api/data").then((r) => r.text()),
  catch: (error) => new Error(`Fetch failed: ${error}`),
});
```

### 1-2. パイプラインの構築

```
Effect パイプライン:

  Effect.succeed(42)
       |
  .pipe(Effect.map(n => n * 2))        → Effect<84>
       |
  .pipe(Effect.flatMap(n => ...))      → Effect<string, Error>
       |
  .pipe(Effect.catchTag("NotFound",    → エラーを回復
        () => Effect.succeed("default")))
       |
  .pipe(Effect.tap(v =>                → 副作用（値は変わらない）
        Effect.log(`value: ${v}`)))
       |
  Effect.runPromise(...)               → Promise<string>
```

```typescript
import { Effect, pipe } from "effect";

// pipe によるパイプライン構築
const program = pipe(
  Effect.succeed(10),
  Effect.map((n) => n * 2),           // 20
  Effect.flatMap((n) =>
    n > 15
      ? Effect.succeed(`Large: ${n}`)
      : Effect.fail(new Error("Too small"))
  ),
  Effect.tap((value) =>
    Effect.log(`Result: ${value}`)
  )
);

// メソッドチェーンスタイル
const program2 = Effect.succeed(10)
  .pipe(
    Effect.map((n) => n * 2),
    Effect.flatMap((n) =>
      n > 15
        ? Effect.succeed(`Large: ${n}`)
        : Effect.fail(new Error("Too small"))
    ),
  );

// 実行
const result = await Effect.runPromise(program);
// "Large: 20"
```

### 1-3. 型安全なエラー処理

```typescript
import { Effect, Data } from "effect";

// 型付きエラーの定義
class NotFoundError extends Data.TaggedError("NotFoundError")<{
  readonly resource: string;
  readonly id: string;
}> {}

class ValidationError extends Data.TaggedError("ValidationError")<{
  readonly message: string;
  readonly fields: readonly string[];
}> {}

class DatabaseError extends Data.TaggedError("DatabaseError")<{
  readonly cause: unknown;
}> {}

// エラーを返す Effect
function findUser(
  id: string
): Effect.Effect<User, NotFoundError | DatabaseError> {
  return pipe(
    Effect.tryPromise({
      try: () => db.users.findById(id),
      catch: (cause) => new DatabaseError({ cause }),
    }),
    Effect.flatMap((user) =>
      user
        ? Effect.succeed(user)
        : Effect.fail(new NotFoundError({ resource: "User", id }))
    )
  );
}

// 特定のエラーだけ処理（catchTag）
const userOrDefault = findUser("123").pipe(
  Effect.catchTag("NotFoundError", (error) =>
    Effect.succeed({ id: error.id, name: "Anonymous" } as User)
  )
  // DatabaseError はそのまま伝播
);
// 型: Effect<User, DatabaseError>
```

---

## 2. サービスと Layer

### 2-1. サービスの定義

```
Layer アーキテクチャ:

  +---------------------+
  | Application Layer   |  Effect<A, E, UserRepo & Logger>
  +---------------------+
           |
           | requires
           v
  +---------------------+
  | Service Layer       |  Layer<UserRepo & Logger>
  +---------------------+
       |           |
       v           v
  +----------+ +---------+
  | UserRepo | | Logger  |  具体的な実装
  +----------+ +---------+
       |
       v
  +----------+
  | Database |  さらに下位の依存
  +----------+
```

```typescript
import { Effect, Context, Layer } from "effect";

// サービスインターフェースの定義
class UserRepository extends Context.Tag("UserRepository")<
  UserRepository,
  {
    readonly findById: (id: string) => Effect.Effect<User | null, DatabaseError>;
    readonly save: (user: User) => Effect.Effect<void, DatabaseError>;
    readonly delete: (id: string) => Effect.Effect<void, DatabaseError>;
  }
>() {}

class EmailService extends Context.Tag("EmailService")<
  EmailService,
  {
    readonly send: (
      to: string,
      subject: string,
      body: string
    ) => Effect.Effect<void, EmailError>;
  }
>() {}

class Logger extends Context.Tag("Logger")<
  Logger,
  {
    readonly info: (message: string) => Effect.Effect<void>;
    readonly error: (message: string, cause?: unknown) => Effect.Effect<void>;
  }
>() {}

// サービスの使用
function createUser(
  data: CreateUserDto
): Effect.Effect<User, ValidationError | DatabaseError | EmailError, UserRepository & EmailService & Logger> {
  return Effect.gen(function* () {
    const userRepo = yield* UserRepository;
    const emailService = yield* EmailService;
    const logger = yield* Logger;

    yield* logger.info(`Creating user: ${data.email}`);

    const user: User = {
      id: crypto.randomUUID(),
      name: data.name,
      email: data.email,
      createdAt: new Date(),
    };

    yield* userRepo.save(user);
    yield* emailService.send(user.email, "Welcome!", `Hello ${user.name}`);
    yield* logger.info(`User created: ${user.id}`);

    return user;
  });
}
```

### 2-2. Layer の実装

```typescript
// Logger の実装 Layer
const ConsoleLoggerLive = Layer.succeed(Logger, {
  info: (message) => Effect.log(`[INFO] ${message}`),
  error: (message, cause) =>
    Effect.logError(`[ERROR] ${message}`, cause),
});

// UserRepository の実装 Layer
const PrismaUserRepoLive = Layer.effect(
  UserRepository,
  Effect.gen(function* () {
    // 他のサービスに依存可能
    const logger = yield* Logger;

    return {
      findById: (id) =>
        Effect.tryPromise({
          try: () => prisma.user.findUnique({ where: { id } }),
          catch: (cause) => new DatabaseError({ cause }),
        }),
      save: (user) =>
        pipe(
          Effect.tryPromise({
            try: () => prisma.user.create({ data: user }),
            catch: (cause) => new DatabaseError({ cause }),
          }),
          Effect.tap(() => logger.info(`Saved user: ${user.id}`)),
          Effect.asVoid
        ),
      delete: (id) =>
        pipe(
          Effect.tryPromise({
            try: () => prisma.user.delete({ where: { id } }),
            catch: (cause) => new DatabaseError({ cause }),
          }),
          Effect.asVoid
        ),
    };
  })
);

// EmailService の実装 Layer
const SmtpEmailServiceLive = Layer.succeed(EmailService, {
  send: (to, subject, body) =>
    Effect.tryPromise({
      try: () => sendSmtpEmail(to, subject, body),
      catch: (cause) => new EmailError({ cause }),
    }),
});

// Layer の合成
const AppLive = Layer.mergeAll(
  ConsoleLoggerLive,
  SmtpEmailServiceLive,
  PrismaUserRepoLive.pipe(Layer.provide(ConsoleLoggerLive))
);

// 実行
const program = createUser({ name: "Alice", email: "alice@example.com" });
const result = await Effect.runPromise(
  program.pipe(Effect.provide(AppLive))
);
```

---

## 3. 並行処理

### 3-1. 基本的な並行パターン

```typescript
import { Effect } from "effect";

// 並列実行
const parallel = Effect.all(
  [fetchUser("1"), fetchUser("2"), fetchUser("3")],
  { concurrency: "unbounded" }
);
// 型: Effect<[User, User, User], NotFoundError | DatabaseError, UserRepository>

// 制限付き並列実行
const limited = Effect.all(
  urls.map((url) => fetchData(url)),
  { concurrency: 5 } // 最大5並列
);

// 最初に成功した結果を使用
const fastest = Effect.raceAll([
  fetchFromCDN1(key),
  fetchFromCDN2(key),
  fetchFromCDN3(key),
]);

// forEach: 配列の各要素に Effect を適用
const results = Effect.forEach(
  userIds,
  (id) => fetchUser(id),
  { concurrency: 10 }
);
```

### 3-2. Schedule（リトライ / 繰り返し）

```typescript
import { Effect, Schedule } from "effect";

// リトライ戦略
const retryPolicy = Schedule.exponential("100 millis").pipe(
  Schedule.compose(Schedule.recurs(5)),        // 最大5回
  Schedule.jittered,                            // ジッターを追加
);

const resilientFetch = fetchData(url).pipe(
  Effect.retry(retryPolicy)
);

// タイムアウト付き
const withTimeout = fetchData(url).pipe(
  Effect.timeout("5 seconds")
);

// リトライ + タイムアウト
const robust = fetchData(url).pipe(
  Effect.timeout("3 seconds"),
  Effect.retry(
    Schedule.exponential("200 millis").pipe(
      Schedule.compose(Schedule.recurs(3))
    )
  )
);
```

---

## 4. Generator 構文（Effect.gen）

```typescript
// Effect.gen で同期的なスタイルで書く
const program = Effect.gen(function* () {
  // yield* で Effect の値を取り出す
  const user = yield* findUser("123");

  // 条件分岐
  if (user.role !== "ADMIN") {
    yield* Effect.fail(
      new PermissionError({ action: "delete", resource: "User" })
    );
  }

  // 並列実行
  const [posts, comments] = yield* Effect.all(
    [fetchPosts(user.id), fetchComments(user.id)],
    { concurrency: 2 }
  );

  // ログ
  yield* Effect.log(`User ${user.name} has ${posts.length} posts`);

  return { user, posts, comments };
});
```

---

## 比較表

### Effect-ts と他のアプローチ比較

| 特性 | Effect-ts | fp-ts | 素のTS | neverthrow |
|------|-----------|-------|--------|-----------|
| エラー型追跡 | 自動 | 手動 | なし | 手動 |
| DI | Layer | Reader | 手動/ライブラリ | なし |
| 並行処理 | Fiber | Task | Promise | Promise |
| リトライ | Schedule | 手動 | 手動 | 手動 |
| リソース管理 | Scope | Bracket | try-finally | try-finally |
| バンドルサイズ | ~50KB+ | ~15KB | 0KB | ~2KB |
| 学習コスト | 高 | 高 | 最低 | 低 |

### Effect-ts の実行関数

| 関数 | 戻り値 | エラー時 | 用途 |
|------|--------|---------|------|
| `runSync` | A | throw | 同期 Effect |
| `runPromise` | Promise<A> | reject | 非同期 Effect |
| `runPromiseExit` | Promise<Exit<A, E>> | 安全 | エラーハンドリング |
| `runFork` | Fiber<A, E> | - | バックグラウンド実行 |

---

## アンチパターン

### AP-1: Effect と Promise を混在させる

```typescript
// NG: Effect 内で直接 await
const program = Effect.gen(function* () {
  const data = await fetch("/api"); // NG: await は使えない
  return data;
});

// OK: tryPromise で Promise を Effect に変換
const program = Effect.gen(function* () {
  const data = yield* Effect.tryPromise({
    try: () => fetch("/api").then((r) => r.json()),
    catch: (error) => new FetchError({ cause: error }),
  });
  return data;
});
```

### AP-2: 全てを Effect で書こうとする

```typescript
// NG: 純粋関数まで Effect にする
const add = (a: number, b: number) =>
  Effect.succeed(a + b); // 不要な Effect ラッピング

// OK: 副作用のない関数はそのまま
const add = (a: number, b: number): number => a + b;

// Effect にすべきもの:
// - I/O（DB, HTTP, ファイル）
// - 失敗しうる操作
// - 依存を注入したい操作
// - リトライ/タイムアウトが必要な操作
```

---

## FAQ

### Q1: Effect-ts は本番プロジェクトで使えるレベルですか？

はい。Effect-ts は 2024 年に v3 (stable) がリリースされ、商用プロジェクトでの採用実績も増えています。ただし、学習コストが高いため、チーム全員が関数型プログラミングの基礎を理解している必要があります。

### Q2: Effect-ts を部分的に導入できますか？

はい。既存プロジェクトの特定のモジュール（エラーハンドリングが複雑な部分、リトライが必要な部分）のみに Effect を導入し、境界で `Effect.runPromise` で通常の Promise に変換できます。全面採用は段階的に行えます。

### Q3: Effect-ts のバンドルサイズは問題になりませんか？

バックエンドでは問題ありません。フロントエンドでは、Effect のコアだけで ~50KB (gzip 後 ~15KB) 程度です。Tree-shaking が効くため、使用する機能によってサイズは変わります。フロントエンドでは本当に必要な箇所のみ使用することを推奨します。

---

## まとめ表

| 概念 | 要点 |
|------|------|
| Effect<A, E, R> | 成功型/エラー型/依存型 の3つを追跡 |
| pipe / Effect.gen | パイプライン / generator の2つの構築スタイル |
| Layer | 依存性注入の仕組み、テスト時に差し替え可能 |
| catchTag | 特定のエラータグのみ処理 |
| Schedule | リトライ、繰り返し、ジッター等 |
| Fiber | 軽量な並行処理プリミティブ |

---

## 次に読むべきガイド

- [エラーハンドリング](../02-patterns/00-error-handling.md) -- Effect-ts のエラー処理と従来の Result 型の比較
- [DI パターン](../02-patterns/04-dependency-injection.md) -- Effect の Layer と従来の DI の比較
- [判別共用体](../02-patterns/02-discriminated-unions.md) -- TaggedError の基盤となる判別共用体

---

## 参考文献

1. **Effect Documentation**
   https://effect.website/docs/introduction

2. **Effect GitHub Repository**
   https://github.com/Effect-TS/effect

3. **Michael Arnaldi - Why Effect?**
   Effect-ts の設計思想とモチベーション
