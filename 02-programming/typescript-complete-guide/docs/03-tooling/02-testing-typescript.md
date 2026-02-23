# TypeScript テスト完全ガイド

> Vitest, Jest, 型テスト（tsd / expect-type）を使い、ランタイムの振る舞いと型の正しさの両方を検証する

## この章で学ぶこと

1. **Vitest によるテスト** -- Vite エコシステムと統合した高速テストランナーのセットアップと活用
2. **Jest + TypeScript** -- ts-jest / @swc/jest の設定と、既存プロジェクトでの Jest 運用
3. **型テスト** -- `expectTypeOf`（Vitest 組込み）や tsd を使い、ライブラリの型定義が正しいことを検証する技法
4. **テスト設計パターン** -- AAA パターン、テストダブル、統合テスト、E2E テストの構成方法
5. **テストのパフォーマンスと保守性** -- 大規模プロジェクトでのテスト戦略と最適化手法

---

## 1. Vitest

### 1-1. セットアップ

```
Vitest のアーキテクチャ:

  vite.config.ts
       |
       v
  +----------+     +---------+     +----------+
  | テスト    | --> | Vite    | --> | esbuild  |
  | ファイル  |     | (変換)  |     | (高速TS) |
  +----------+     +---------+     +----------+
       |
       v
  +----------+
  | テスト   |
  | 実行結果 |
  +----------+

  Vitest の特徴:
  - Vite と設定を共有（resolve.alias, plugins など）
  - esbuild によるトランスパイルで高速
  - Jest 互換の API
  - 型テスト（expectTypeOf）が組込み
  - HMR 対応のウォッチモード
  - ブラウザモード（Playwright / WebDriverIO）
```

```typescript
// vitest.config.ts
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,           // describe, it, expect をグローバルに
    environment: "node",     // or "jsdom", "happy-dom"
    include: ["src/**/*.test.ts", "tests/**/*.test.ts"],
    exclude: ["node_modules", "dist", "e2e/**"],
    // カバレッジ設定
    coverage: {
      provider: "v8",        // or "istanbul"
      reporter: ["text", "html", "lcov", "json-summary"],
      include: ["src/**/*.ts"],
      exclude: [
        "src/**/*.test.ts",
        "src/**/*.d.ts",
        "src/**/index.ts",    // re-export のみのファイル
        "src/types/**",
      ],
      thresholds: {
        branches: 80,
        functions: 80,
        lines: 80,
        statements: 80,
      },
    },
    // 型テストの設定
    typecheck: {
      enabled: true,         // 型テストを有効化
      tsconfig: "./tsconfig.test.json",
      include: ["src/**/*.typetest.ts"],
    },
    // テストのタイムアウト
    testTimeout: 10000,
    hookTimeout: 10000,
    // セットアップファイル
    setupFiles: ["./tests/setup.ts"],
    // グローバルセットアップ（テストスイート全体で1回）
    globalSetup: ["./tests/global-setup.ts"],
    // スナップショット
    snapshotFormat: {
      printBasicPrototype: false,
    },
    // モック自動クリーンアップ
    restoreMocks: true,
    clearMocks: true,
    mockReset: true,
    // 並列実行の設定
    pool: "threads",          // or "forks", "vmThreads"
    poolOptions: {
      threads: {
        minThreads: 1,
        maxThreads: 4,
      },
    },
  },
  resolve: {
    alias: {
      "@": "/src",
    },
  },
});
```

```json
// package.json
{
  "scripts": {
    "test": "vitest",
    "test:run": "vitest run",
    "test:coverage": "vitest run --coverage",
    "test:ui": "vitest --ui",
    "test:watch": "vitest --watch",
    "test:related": "vitest related",
    "test:changed": "vitest --changed"
  }
}
```

```typescript
// tsconfig.test.json
{
  "extends": "./tsconfig.json",
  "compilerOptions": {
    "types": ["vitest/globals"]
  },
  "include": ["src/**/*", "tests/**/*"]
}
```

### 1-2. テストの書き方

```typescript
// src/user-service.test.ts
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { UserService } from "./user-service";
import type { IUserRepository, IEmailService } from "./interfaces";

// モック作成（型安全）
function createMockUserRepo(): IUserRepository {
  return {
    findById: vi.fn(),
    findByEmail: vi.fn(),
    save: vi.fn(),
    delete: vi.fn(),
    findMany: vi.fn(),
  };
}

function createMockEmailService(): IEmailService {
  return {
    send: vi.fn(),
  };
}

describe("UserService", () => {
  let service: UserService;
  let mockRepo: ReturnType<typeof createMockUserRepo>;
  let mockEmail: ReturnType<typeof createMockEmailService>;

  beforeEach(() => {
    mockRepo = createMockUserRepo();
    mockEmail = createMockEmailService();
    service = new UserService(mockRepo, mockEmail);
  });

  describe("createUser", () => {
    it("should save user and send welcome email", async () => {
      // Arrange
      mockRepo.save.mockResolvedValue(undefined);
      mockEmail.send.mockResolvedValue(undefined);

      // Act
      const user = await service.createUser({
        name: "Alice",
        email: "alice@example.com",
      });

      // Assert
      expect(user.name).toBe("Alice");
      expect(mockRepo.save).toHaveBeenCalledOnce();
      expect(mockEmail.send).toHaveBeenCalledWith(
        "alice@example.com",
        expect.stringContaining("Welcome"),
        expect.any(String)
      );
    });

    it("should return error for invalid email", async () => {
      const result = await service.createUser({
        name: "Bob",
        email: "invalid",
      });

      expect(result).toMatchObject({
        _tag: "Err",
        error: { code: "VALIDATION_ERROR" },
      });
      expect(mockRepo.save).not.toHaveBeenCalled();
    });

    it("should handle database errors gracefully", async () => {
      mockRepo.save.mockRejectedValue(new Error("Connection refused"));

      const result = await service.createUser({
        name: "Charlie",
        email: "charlie@example.com",
      });

      expect(result).toMatchObject({
        _tag: "Err",
        error: { code: "DATABASE_ERROR" },
      });
    });
  });

  describe("deleteUser", () => {
    it("should delete existing user", async () => {
      mockRepo.findById.mockResolvedValue({
        id: "user-1",
        name: "Alice",
        email: "alice@example.com",
      });
      mockRepo.delete.mockResolvedValue(undefined);

      await service.deleteUser("user-1");

      expect(mockRepo.delete).toHaveBeenCalledWith("user-1");
    });

    it("should throw when user not found", async () => {
      mockRepo.findById.mockResolvedValue(null);

      await expect(service.deleteUser("nonexistent")).rejects.toThrow(
        "User not found"
      );
    });
  });
});
```

### 1-3. vi.mock によるモジュールモック

```typescript
// モジュール全体をモック
vi.mock("./database", () => ({
  db: {
    query: vi.fn(),
    transaction: vi.fn(),
  },
}));

// 部分モック（実装の一部だけ置き換え）
vi.mock("./utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./utils")>();
  return {
    ...actual,
    generateId: vi.fn(() => "fixed-id"),
    getCurrentTimestamp: vi.fn(() => new Date("2024-01-01T00:00:00Z")),
  };
});

// スパイ
import * as mathUtils from "./math-utils";
vi.spyOn(mathUtils, "calculate").mockReturnValue(42);

// 環境変数のモック
vi.stubEnv("NODE_ENV", "test");
vi.stubEnv("API_KEY", "test-key");

// タイマーのモック
vi.useFakeTimers();
vi.setSystemTime(new Date("2024-06-15T12:00:00Z"));
// テスト後にリストア
afterEach(() => {
  vi.useRealTimers();
});

// fetch のモック
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

mockFetch.mockResolvedValue({
  ok: true,
  status: 200,
  json: () => Promise.resolve({ data: "test" }),
});
```

### 1-4. スナップショットテスト

```typescript
import { describe, it, expect } from "vitest";

describe("Snapshot tests", () => {
  it("should match user object snapshot", () => {
    const user = createUser({ name: "Alice", email: "alice@test.com" });

    expect(user).toMatchSnapshot();
    // 初回実行時に __snapshots__ にスナップショットが保存される
  });

  it("should match inline snapshot", () => {
    const result = formatCurrency(1234.56, "JPY");

    expect(result).toMatchInlineSnapshot(`"¥1,235"`);
    // スナップショットがテストファイル内に埋め込まれる
  });

  it("should match serialized output", () => {
    const html = renderToString(<UserCard user={mockUser} />);

    expect(html).toMatchSnapshot();
    // スナップショットの更新: vitest --update
  });
});
```

### 1-5. パラメタライズドテスト

```typescript
import { describe, it, expect } from "vitest";

describe("calculateTax", () => {
  it.each([
    { price: 1000, rate: 0.1, expected: 1100 },
    { price: 2000, rate: 0.1, expected: 2200 },
    { price: 500, rate: 0.08, expected: 540 },
    { price: 0, rate: 0.1, expected: 0 },
    { price: 99.99, rate: 0.1, expected: 109.989 },
  ])(
    "should calculate $price with $rate% tax = $expected",
    ({ price, rate, expected }) => {
      expect(calculateTax(price, rate)).toBeCloseTo(expected);
    }
  );
});

// テーブル形式
describe("validateEmail", () => {
  it.each`
    email                | valid    | reason
    ${"user@example.com"} | ${true}  | ${"valid email"}
    ${"user@test.co.jp"} | ${true}  | ${"country TLD"}
    ${"invalid"}          | ${false} | ${"no @ symbol"}
    ${"@example.com"}     | ${false} | ${"no local part"}
    ${"user@"}            | ${false} | ${"no domain"}
  `("$email should be valid=$valid ($reason)", ({ email, valid }) => {
    expect(isValidEmail(email)).toBe(valid);
  });
});
```

---

## 2. Jest + TypeScript

### 2-1. ts-jest セットアップ

```typescript
// jest.config.ts
import type { Config } from "jest";

const config: Config = {
  preset: "ts-jest",
  testEnvironment: "node",
  roots: ["<rootDir>/src"],
  testMatch: ["**/*.test.ts"],
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1",
    "^@components/(.*)$": "<rootDir>/src/components/$1",
    "^@utils/(.*)$": "<rootDir>/src/utils/$1",
  },
  collectCoverageFrom: [
    "src/**/*.ts",
    "!src/**/*.d.ts",
    "!src/**/*.test.ts",
    "!src/**/index.ts",
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80,
    },
  },
  // セットアップファイル
  setupFilesAfterSetup: ["<rootDir>/tests/setup.ts"],
  // タイムアウト
  testTimeout: 10000,
};

export default config;
```

### 2-2. @swc/jest（高速版）

```typescript
// jest.config.ts -- SWC を使って高速化
import type { Config } from "jest";

const config: Config = {
  testEnvironment: "node",
  roots: ["<rootDir>/src"],
  testMatch: ["**/*.test.ts"],
  transform: {
    "^.+\\.tsx?$": [
      "@swc/jest",
      {
        jsc: {
          parser: {
            syntax: "typescript",
            tsx: true,
            decorators: true,
          },
          transform: {
            decoratorVersion: "2022-03",
            react: {
              runtime: "automatic",
            },
          },
        },
      },
    ],
  },
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1",
  },
};

export default config;
```

### 2-3. Jest から Vitest への移行

```typescript
// Jest のコードはほぼそのまま Vitest で動作する
// 主な変更点:

// 1. jest.fn() → vi.fn()
// 2. jest.mock() → vi.mock()
// 3. jest.spyOn() → vi.spyOn()

// 4. jest.config.ts → vitest.config.ts
// 5. @types/jest → vitest/globals

// 6. 非互換な機能:
//    - jest.requireActual() → await importOriginal()
//    - jest.useFakeTimers("modern") → vi.useFakeTimers()
//    - jest.runAllTimers() → vi.runAllTimers()

// 移行スクリプト（sed で一括置換）
// sed -i 's/jest\.fn/vi.fn/g' **/*.test.ts
// sed -i 's/jest\.mock/vi.mock/g' **/*.test.ts
// sed -i 's/jest\.spyOn/vi.spyOn/g' **/*.test.ts

// codemod ツールも利用可能:
// npx @vitest/codemod migrate-jest
```

---

## 3. 型テスト

### 3-1. Vitest の expectTypeOf

```
型テストの目的:

  ライブラリのパブリック API
       |
       v
  +------------------+
  | 型の正しさを検証   |
  |                  |
  | - 推論結果       |
  | - 代入可能性     |
  | - エラーになること |
  +------------------+
       |
       v
  型の回帰テスト（リファクタリングで型が壊れない保証）
```

```typescript
// src/result.typetest.ts
import { describe, it, expectTypeOf } from "vitest";
import { Ok, Err, type Result, map, isOk, flatMap } from "./result";

describe("Result type tests", () => {
  it("Ok should infer the correct type", () => {
    const ok = Ok(42);
    expectTypeOf(ok).toEqualTypeOf<{ _tag: "Ok"; value: number }>();
  });

  it("Err should infer the correct type", () => {
    const err = Err("not found");
    expectTypeOf(err).toEqualTypeOf<{ _tag: "Err"; error: string }>();
  });

  it("Result should be a union", () => {
    type R = Result<number, string>;
    expectTypeOf<R>().toEqualTypeOf<
      { _tag: "Ok"; value: number } | { _tag: "Err"; error: string }
    >();
  });

  it("map should transform the success type", () => {
    const result: Result<number, string> = Ok(42);
    const mapped = map(result, (n) => String(n));
    expectTypeOf(mapped).toEqualTypeOf<Result<string, string>>();
  });

  it("flatMap should compose Results", () => {
    const result: Result<number, string> = Ok(42);
    const composed = flatMap(result, (n) =>
      n > 0 ? Ok(String(n)) : Err("negative")
    );
    expectTypeOf(composed).toEqualTypeOf<Result<string, string>>();
  });

  it("isOk should narrow the type", () => {
    const result: Result<number, string> = Ok(42);
    if (isOk(result)) {
      expectTypeOf(result).toEqualTypeOf<{ _tag: "Ok"; value: number }>();
    }
  });

  it("should not accept wrong types", () => {
    // @ts-expect-error -- number は string に代入不可
    const bad: Result<string, string> = Ok(42);
  });

  // expectTypeOf の豊富なアサーション
  it("should demonstrate various type assertions", () => {
    // 型が一致するか
    expectTypeOf<string>().toEqualTypeOf<string>();

    // 代入可能か
    expectTypeOf<string>().toMatchTypeOf<string | number>();

    // 関数の引数の型
    function greet(name: string, age: number): string {
      return `${name} (${age})`;
    }
    expectTypeOf(greet).parameter(0).toBeString();
    expectTypeOf(greet).parameter(1).toBeNumber();
    expectTypeOf(greet).returns.toBeString();

    // 配列の要素型
    expectTypeOf<string[]>().items.toBeString();

    // オブジェクトのプロパティ
    interface User {
      name: string;
      age: number;
    }
    expectTypeOf<User>().toHaveProperty("name");
    expectTypeOf<User>().toHaveProperty("age");
  });
});
```

### 3-2. tsd を使った型テスト

```typescript
// test-d/index.test-d.ts（tsd 用）
import { expectType, expectError, expectAssignable, expectNotType } from "tsd";
import { createStore, type Store } from "../src";

// 型が正しく推論されること
const store = createStore({ count: 0, name: "test" });
expectType<Store<{ count: number; name: string }>>(store);

// get が正しい型を返すこと
const count = store.get("count");
expectType<number>(count);

// 存在しないキーでエラーになること
expectError(store.get("nonexistent"));

// 代入可能性のテスト
expectAssignable<{ count: number }>(store.getState());

// 型が一致しないこと
expectNotType<string>(store.get("count"));
```

```json
// package.json に tsd の設定
{
  "scripts": {
    "test:types": "tsd"
  },
  "tsd": {
    "directory": "test-d"
  }
}
```

### 3-3. @ts-expect-error による型テスト

```typescript
// コンパイルエラーになることを検証
describe("type safety", () => {
  it("should reject wrong argument types", () => {
    function add(a: number, b: number): number {
      return a + b;
    }

    // @ts-expect-error -- string は number に代入不可
    add("1", "2");

    // @ts-expect-error -- 引数が足りない
    add(1);

    // @ts-expect-error -- 引数が多すぎる
    add(1, 2, 3);
  });

  it("branded types should not be interchangeable", () => {
    type UserId = string & { __brand: "UserId" };
    type OrderId = string & { __brand: "OrderId" };

    function getUser(id: UserId): void {}
    const orderId = "order-1" as OrderId;

    // @ts-expect-error -- OrderId は UserId に代入不可
    getUser(orderId);
  });

  it("readonly properties should not be writable", () => {
    interface Config {
      readonly apiUrl: string;
      readonly port: number;
    }

    const config: Config = { apiUrl: "https://api.example.com", port: 3000 };

    // @ts-expect-error -- readonly プロパティに代入不可
    config.apiUrl = "https://other.com";
  });

  it("discriminated unions should be exhaustive", () => {
    type Shape =
      | { kind: "circle"; radius: number }
      | { kind: "square"; side: number };

    function area(shape: Shape): number {
      switch (shape.kind) {
        case "circle":
          return Math.PI * shape.radius ** 2;
        case "square":
          return shape.side ** 2;
        default:
          // never 型で網羅性チェック
          const _exhaustive: never = shape;
          return _exhaustive;
      }
    }
  });
});
```

---

## 4. テスト設計パターン

### 4-1. AAA パターン（Arrange-Act-Assert）

```typescript
it("should calculate total with tax", () => {
  // Arrange（準備）
  const items = [
    { price: 1000, quantity: 2 },
    { price: 500, quantity: 3 },
  ];
  const taxRate = 0.1;

  // Act（実行）
  const total = calculateTotal(items, taxRate);

  // Assert（検証）
  expect(total).toBe(3850); // (1000*2 + 500*3) * 1.1
});
```

### 4-2. テストデータビルダー

```typescript
// テストデータビルダーパターン
class UserBuilder {
  private data: Partial<User> = {
    id: "default-id",
    name: "Default User",
    email: "default@test.com",
    role: "USER",
    createdAt: new Date("2024-01-01"),
  };

  static create(): UserBuilder {
    return new UserBuilder();
  }

  withId(id: string): this {
    this.data.id = id;
    return this;
  }

  withName(name: string): this {
    this.data.name = name;
    return this;
  }

  withEmail(email: string): this {
    this.data.email = email;
    return this;
  }

  withRole(role: "USER" | "ADMIN"): this {
    this.data.role = role;
    return this;
  }

  build(): User {
    return this.data as User;
  }
}

// 使用例
it("admin should have special permissions", () => {
  const admin = UserBuilder.create()
    .withName("Admin Alice")
    .withRole("ADMIN")
    .build();

  expect(hasPermission(admin, "delete_users")).toBe(true);
});
```

### 4-3. テストコンテキストファクトリー

```typescript
// テストヘルパー
function createTestContext() {
  const userRepo = createMockUserRepo();
  const emailService = createMockEmailService();
  const logger = createMockLogger();
  const eventBus = createMockEventBus();

  const service = new UserService(userRepo, emailService, logger, eventBus);

  return { service, userRepo, emailService, logger, eventBus };
}

describe("UserService", () => {
  it("should handle concurrent creates", async () => {
    const { service, userRepo } = createTestContext();

    userRepo.save.mockResolvedValue(undefined);

    const results = await Promise.all([
      service.createUser({ name: "A", email: "a@test.com" }),
      service.createUser({ name: "B", email: "b@test.com" }),
    ]);

    expect(userRepo.save).toHaveBeenCalledTimes(2);
    results.forEach((r) => expect(r).toMatchObject({ _tag: "Ok" }));
  });

  it("should emit UserCreated event", async () => {
    const { service, userRepo, eventBus } = createTestContext();

    userRepo.save.mockResolvedValue(undefined);

    await service.createUser({ name: "Alice", email: "alice@test.com" });

    expect(eventBus.emit).toHaveBeenCalledWith(
      "UserCreated",
      expect.objectContaining({ email: "alice@test.com" })
    );
  });
});
```

### 4-4. HTTP モック（msw）

```typescript
// tests/mocks/handlers.ts
import { http, HttpResponse } from "msw";

export const handlers = [
  http.get("https://api.example.com/users", () => {
    return HttpResponse.json([
      { id: "1", name: "Alice", email: "alice@example.com" },
      { id: "2", name: "Bob", email: "bob@example.com" },
    ]);
  }),

  http.get("https://api.example.com/users/:id", ({ params }) => {
    const { id } = params;
    if (id === "404") {
      return new HttpResponse(null, { status: 404 });
    }
    return HttpResponse.json({
      id,
      name: "Alice",
      email: "alice@example.com",
    });
  }),

  http.post("https://api.example.com/users", async ({ request }) => {
    const body = await request.json();
    return HttpResponse.json(
      { id: "new-id", ...body },
      { status: 201 }
    );
  }),
];

// tests/setup.ts
import { setupServer } from "msw/node";
import { handlers } from "./mocks/handlers";

const server = setupServer(...handlers);

beforeAll(() => server.listen({ onUnhandledRequest: "error" }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

// テストでの使用
describe("UserApiClient", () => {
  it("should fetch users", async () => {
    const client = new UserApiClient("https://api.example.com");
    const users = await client.getUsers();

    expect(users).toHaveLength(2);
    expect(users[0].name).toBe("Alice");
  });

  it("should handle 404 errors", async () => {
    const client = new UserApiClient("https://api.example.com");

    await expect(client.getUser("404")).rejects.toThrow("User not found");
  });

  // テストごとにハンドラーを上書き
  it("should handle server errors", async () => {
    server.use(
      http.get("https://api.example.com/users", () => {
        return new HttpResponse(null, { status: 500 });
      })
    );

    const client = new UserApiClient("https://api.example.com");
    await expect(client.getUsers()).rejects.toThrow("Server error");
  });
});
```

### 4-5. データベーステスト（Prisma + テストコンテナ）

```typescript
// tests/helpers/database.ts
import { PrismaClient } from "@prisma/client";
import { PostgreSqlContainer, type StartedPostgreSqlContainer } from "@testcontainers/postgresql";
import { execSync } from "child_process";

let container: StartedPostgreSqlContainer;
let prisma: PrismaClient;

export async function setupTestDatabase(): Promise<PrismaClient> {
  // Docker コンテナでテスト用 PostgreSQL を起動
  container = await new PostgreSqlContainer("postgres:16")
    .withDatabase("testdb")
    .start();

  const databaseUrl = container.getConnectionUri();

  // マイグレーションを適用
  execSync(`DATABASE_URL="${databaseUrl}" npx prisma migrate deploy`, {
    stdio: "pipe",
  });

  prisma = new PrismaClient({
    datasources: { db: { url: databaseUrl } },
  });

  return prisma;
}

export async function teardownTestDatabase(): Promise<void> {
  await prisma.$disconnect();
  await container.stop();
}

export async function cleanDatabase(): Promise<void> {
  // トランザクションで全テーブルをクリア
  const tablenames = await prisma.$queryRaw<
    { tablename: string }[]
  >`SELECT tablename FROM pg_tables WHERE schemaname = 'public'`;

  for (const { tablename } of tablenames) {
    if (tablename !== "_prisma_migrations") {
      await prisma.$executeRawUnsafe(`TRUNCATE TABLE "${tablename}" CASCADE;`);
    }
  }
}

// テストでの使用
describe("UserRepository (integration)", () => {
  let prisma: PrismaClient;
  let repo: UserRepository;

  beforeAll(async () => {
    prisma = await setupTestDatabase();
    repo = new PrismaUserRepository(prisma);
  });

  afterAll(async () => {
    await teardownTestDatabase();
  });

  afterEach(async () => {
    await cleanDatabase();
  });

  it("should create and find user", async () => {
    const created = await repo.create({
      name: "Alice",
      email: "alice@test.com",
      role: "USER",
    });

    const found = await repo.findById(created.id);

    expect(found).toMatchObject({
      name: "Alice",
      email: "alice@test.com",
    });
  });

  it("should return null for non-existent user", async () => {
    const found = await repo.findById("non-existent-id");
    expect(found).toBeNull();
  });
});
```

---

## 5. テストの構成と命名規則

### 5-1. ファイル構成パターン

```
パターン A: コロケーション（同一ディレクトリ）
src/
├── user/
│   ├── user-service.ts
│   ├── user-service.test.ts      ← テストをソースの横に
│   ├── user-repository.ts
│   └── user-repository.test.ts
└── order/
    ├── order-service.ts
    └── order-service.test.ts

パターン B: 分離ディレクトリ
src/
├── user/
│   ├── user-service.ts
│   └── user-repository.ts
tests/
├── unit/
│   ├── user-service.test.ts
│   └── user-repository.test.ts
├── integration/
│   └── user-flow.test.ts
└── e2e/
    └── api.test.ts

パターン C: ハイブリッド（推奨）
src/
├── user/
│   ├── user-service.ts
│   ├── user-service.test.ts      ← ユニットテスト
│   └── user-repository.ts
tests/
├── integration/                   ← 統合テスト
│   └── user-creation.test.ts
├── e2e/                           ← E2E テスト
│   └── user-api.test.ts
├── helpers/                       ← テストヘルパー
│   ├── database.ts
│   ├── builders.ts
│   └── mocks.ts
└── setup.ts                       ← グローバルセットアップ
```

### 5-2. テスト命名規則

```typescript
// 命名パターン: should + 期待される振る舞い + when + 条件
describe("UserService.createUser", () => {
  // 正常系
  it("should create user and return Ok when valid data is provided", async () => {
    // ...
  });

  it("should send welcome email when user is created successfully", async () => {
    // ...
  });

  // 異常系
  it("should return ValidationError when email is invalid", async () => {
    // ...
  });

  it("should return DuplicateError when email already exists", async () => {
    // ...
  });

  // 境界値
  it("should accept name with exactly 100 characters", async () => {
    // ...
  });

  it("should reject name with 101 characters", async () => {
    // ...
  });
});
```

---

## 6. テストピラミッドとカバレッジ戦略

```
テストピラミッド:

         /\
        /  \     E2E テスト（少数）
       /    \    - Playwright / Cypress
      /------\   - ユーザーシナリオ全体
     /        \  - 実行時間: 長い
    /   統合    \
   /   テスト    \  統合テスト（中程度）
  /--------------\  - DB / API の結合
 /                \ - テストコンテナ
/   ユニットテスト   \
+-------------------+ ユニットテスト（多数）
                       - 純粋な関数
                       - モック使用
                       - 実行時間: 短い

目安:
  ユニットテスト: 70%
  統合テスト:     20%
  E2E テスト:     10%
```

```typescript
// カバレッジの除外設定
// vitest.config.ts
{
  test: {
    coverage: {
      exclude: [
        // テストファイル自体
        "**/*.test.ts",
        "**/*.spec.ts",
        // 型定義
        "**/*.d.ts",
        // 設定ファイル
        "*.config.ts",
        // re-export のみのファイル
        "**/index.ts",
        // 生成コード
        "src/generated/**",
        // テストヘルパー
        "tests/**",
      ],
    },
  },
}
```

---

## 比較表

### テストランナー比較

| 特性 | Vitest | Jest | Node.js test runner |
|------|--------|------|---------------------|
| 速度 | 非常に速い | 普通 | 速い |
| TypeScript | Vite で変換 | ts-jest / @swc/jest | --loader |
| 型テスト | expectTypeOf 組込み | tsd 別途 | なし |
| HMR | あり | なし | なし |
| UI | vitest --ui | jest-stare 等 | なし |
| Watch | 最適化済み | あり | あり |
| Coverage | v8 / istanbul | istanbul | v8 |
| エコシステム | 成長中 | 最大 | 最小 |
| ブラウザテスト | Playwright/WebDriverIO | jsdom のみ | なし |
| セットアップ | 簡単 | 中程度 | 最小 |
| スナップショット | あり | あり | あり |

### モック手法の比較

| 手法 | 用途 | 型安全性 | 柔軟性 | 推奨場面 |
|------|------|---------|--------|---------|
| vi.fn() / jest.fn() | 関数モック | 中 | 高 | コールバック |
| vi.mock() / jest.mock() | モジュールモック | 低 | 最高 | 外部依存 |
| vi.spyOn() / jest.spyOn() | スパイ | 高 | 中 | 既存関数の監視 |
| 手動モック | DI ベース | 最高 | 中 | サービス層 |
| msw | HTTP モック | 高 | 高 | API クライアント |
| testcontainers | 実 DB テスト | 最高 | 最高 | リポジトリ層 |

### テスト環境の比較

| 環境 | 用途 | DOM | パフォーマンス |
|------|------|-----|-------------|
| node | バックエンド | なし | 最速 |
| jsdom | フロントエンド | シミュレート | 速い |
| happy-dom | フロントエンド | シミュレート | 速い |
| playwright | E2E / ブラウザ | 実ブラウザ | 遅い |

---

## アンチパターン

### AP-1: テストで any を使う

```typescript
// NG: any で型安全性を破壊
it("should process data", () => {
  const mockData: any = { foo: "bar" };
  const result = processUser(mockData); // 型チェックが効かない
  expect(result).toBeDefined();
});

// OK: 正しい型でテストデータを作成
it("should process data", () => {
  const user = UserBuilder.create()
    .withName("Alice")
    .withEmail("alice@test.com")
    .build();
  const result = processUser(user); // 型チェックが効く
  expect(result.name).toBe("Alice");
});
```

### AP-2: 実装の詳細をテストする

```typescript
// NG: 内部実装に依存したテスト（壊れやすい）
it("should call repository save then email send", async () => {
  // save が email より先に呼ばれることをテスト
  const callOrder: string[] = [];
  mockRepo.save.mockImplementation(() => {
    callOrder.push("save");
    return Promise.resolve();
  });
  mockEmail.send.mockImplementation(() => {
    callOrder.push("email");
    return Promise.resolve();
  });

  await service.createUser(data);
  expect(callOrder).toEqual(["save", "email"]); // 内部実装に依存
});

// OK: 振る舞い（結果）をテスト
it("should create user and send welcome email", async () => {
  const result = await service.createUser(data);
  expect(result).toMatchObject({ _tag: "Ok" });
  expect(mockEmail.send).toHaveBeenCalledWith(
    data.email,
    expect.any(String),
    expect.any(String)
  );
});
```

### AP-3: テスト間の依存関係

```typescript
// NG: テスト間で状態を共有
let userId: string;

it("should create user", async () => {
  const user = await service.createUser(data);
  userId = user.id; // 次のテストで使う ← 危険!
  expect(user).toBeDefined();
});

it("should get user", async () => {
  const user = await service.getUser(userId); // 前のテストに依存
  expect(user.name).toBe("Alice");
});

// OK: 各テストが独立
it("should get created user", async () => {
  // Arrange: テスト内で完結
  const created = await service.createUser(data);

  // Act
  const found = await service.getUser(created.id);

  // Assert
  expect(found.name).toBe(data.name);
});
```

### AP-4: 過度なモック

```typescript
// NG: 全てをモックして何も検証していない
it("should work", async () => {
  mockRepo.findById.mockResolvedValue(mockUser);
  mockRepo.save.mockResolvedValue(undefined);
  mockEmail.send.mockResolvedValue(undefined);
  mockLogger.info.mockReturnValue(undefined);
  mockCache.get.mockResolvedValue(null);
  mockCache.set.mockResolvedValue(undefined);

  const result = await service.updateUser("id", { name: "New" });

  expect(result).toBeDefined(); // 何を検証してるのか不明
});

// OK: 本当に重要な振る舞いに焦点を当てる
it("should update user name and invalidate cache", async () => {
  mockRepo.findById.mockResolvedValue(existingUser);
  mockRepo.save.mockResolvedValue(undefined);

  const result = await service.updateUser("user-1", { name: "New Name" });

  expect(result.name).toBe("New Name");
  expect(mockRepo.save).toHaveBeenCalledWith(
    expect.objectContaining({ id: "user-1", name: "New Name" })
  );
  expect(mockCache.delete).toHaveBeenCalledWith("user:user-1");
});
```

---

## FAQ

### Q1: Vitest と Jest のどちらを選ぶべきですか？

新規プロジェクトでは Vitest を推奨します。Vite エコシステムとの統合、型テストの組込みサポート、高速な実行が利点です。既存の Jest プロジェクトを無理に移行する必要はありませんが、Vitest は Jest 互換の API を提供しているため、移行は比較的容易です。@vitest/codemod で自動移行も可能です。

### Q2: 型テストはどのくらい書くべきですか？

ライブラリやユーティリティ型を公開する場合は必須です。アプリケーションコードでは、複雑なジェネリクス関数やユーティリティ型に対して書くと効果的です。全ての関数に型テストを書く必要はなく、「型の推論結果が重要な部分」に集中してください。

### Q3: テストカバレッジの目標は何%が適切ですか？

80% が一般的な目標です。ただし、カバレッジ% だけを追うのではなく、「クリティカルパス（正常系・異常系の主要フロー）がカバーされているか」を重視してください。100% を目指すとテストの保守コストが膨大になります。

### Q4: 統合テストで本物の DB を使うべきですか？

可能であれば testcontainers で本物の DB を使うことを推奨します。モックでは検出できない SQL の問題やデータの整合性エラーを発見できます。CI 環境では Docker が必要になりますが、GitHub Actions 等では容易に設定できます。

### Q5: テストが遅い場合の対処法は？

1. `vitest --pool=threads` で並列実行
2. `vitest --changed` で変更ファイルに関連するテストのみ実行
3. モジュールモックの最小化（モックが多いほどオーバーヘッドが大きい）
4. 統合テストとユニットテストの分離（`vitest --project unit`）
5. CI では `--shard` で並列ジョブに分割

---

## まとめ表

| 概念 | 要点 |
|------|------|
| Vitest | Vite ベース、高速、型テスト組込み |
| Jest | 最大のエコシステム、@swc/jest で高速化可能 |
| 型テスト | expectTypeOf / tsd でライブラリの型を検証 |
| @ts-expect-error | コンパイルエラーになることを検証 |
| AAA パターン | Arrange-Act-Assert で構造化 |
| DI + モック | インターフェースベースで差し替え容易に |
| msw | HTTP リクエストのモック（サービスワーカー） |
| テストピラミッド | ユニット 70%、統合 20%、E2E 10% |

---

## 7. React コンポーネントのテスト

### 7-1. Testing Library + Vitest

```typescript
// src/components/UserCard.test.tsx
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { UserCard } from "./UserCard";

describe("UserCard", () => {
  const mockUser = {
    id: "1",
    name: "Alice",
    email: "alice@example.com",
    role: "admin" as const,
  };

  it("should render user information", () => {
    render(<UserCard user={mockUser} />);

    expect(screen.getByText("Alice")).toBeInTheDocument();
    expect(screen.getByText("alice@example.com")).toBeInTheDocument();
    expect(screen.getByRole("badge")).toHaveTextContent("admin");
  });

  it("should call onEdit when edit button is clicked", async () => {
    const onEdit = vi.fn();
    render(<UserCard user={mockUser} onEdit={onEdit} />);

    fireEvent.click(screen.getByRole("button", { name: /edit/i }));

    expect(onEdit).toHaveBeenCalledWith(mockUser.id);
  });

  it("should show delete confirmation dialog", async () => {
    const onDelete = vi.fn();
    render(<UserCard user={mockUser} onDelete={onDelete} />);

    fireEvent.click(screen.getByRole("button", { name: /delete/i }));

    await waitFor(() => {
      expect(screen.getByText("本当に削除しますか？")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /confirm/i }));
    expect(onDelete).toHaveBeenCalledWith(mockUser.id);
  });
});
```

### 7-2. カスタムフックのテスト

```typescript
// src/hooks/useUsers.test.ts
import { renderHook, waitFor } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useUsers } from "./useUsers";

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    );
  };
}

describe("useUsers", () => {
  it("should fetch and return users", async () => {
    const { result } = renderHook(() => useUsers(), {
      wrapper: createWrapper(),
    });

    // 初期状態
    expect(result.current.isLoading).toBe(true);

    // データ取得完了
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    expect(result.current.data).toHaveLength(2);
    expect(result.current.data?.[0].name).toBe("Alice");
  });

  it("should handle error state", async () => {
    // msw でエラーレスポンスを返す設定に上書き
    server.use(
      http.get("https://api.example.com/users", () => {
        return new HttpResponse(null, { status: 500 });
      })
    );

    const { result } = renderHook(() => useUsers(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toContain("Server error");
  });
});
```

---

## 8. CI/CD でのテスト設定

### 8-1. GitHub Actions

```yaml
# .github/workflows/test.yml
name: Test
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        shard: [1/4, 2/4, 3/4, 4/4]  # テストを4分割
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'npm'

      - run: npm ci

      - name: Run tests
        run: vitest run --shard=${{ matrix.shard }} --reporter=junit --outputFile=test-results.xml

      - name: Upload test results
        uses: actions/upload-artifact@v4
        if: always()
        with:
          name: test-results-${{ matrix.shard }}
          path: test-results.xml

  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: 'npm'

      - run: npm ci
      - run: vitest run --coverage

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: coverage/lcov.info
```

### 8-2. テストのパフォーマンス最適化

```typescript
// vitest.config.ts でパフォーマンスを最適化
export default defineConfig({
  test: {
    // スレッドプールで並列実行
    pool: "threads",
    poolOptions: {
      threads: {
        // CPU コア数に合わせる
        minThreads: 2,
        maxThreads: 8,
      },
    },
    // 失敗したテストを先に実行（フィードバック高速化）
    sequence: {
      shuffle: false,
    },
    // テスト分離（メモリリーク防止）
    isolate: true,
    // ファイルごとのタイムアウト
    testTimeout: 10000,
    // 重いテストの特定
    slowTestThreshold: 1000,
    // レポーター設定
    reporters: ["default", "junit"],
    outputFile: {
      junit: "test-results/junit.xml",
    },
  },
});
```

---

## 次に読むべきガイド

- [ビルドツール](./01-build-tools.md) -- Vitest と Vite の連携設定
- [DI パターン](../02-patterns/04-dependency-injection.md) -- テスタブルな設計と DI
- [ESLint + TypeScript](./04-eslint-typescript.md) -- テストコードの lint ルール

---

## 参考文献

1. **Vitest** -- Next Generation Testing Framework
   https://vitest.dev/

2. **Jest** -- Delightful JavaScript Testing
   https://jestjs.io/

3. **tsd** -- Check TypeScript type definitions
   https://github.com/tsdjs/tsd

4. **msw** -- Mock Service Worker
   https://mswjs.io/

5. **testcontainers** -- Integration testing with real services
   https://testcontainers.com/
