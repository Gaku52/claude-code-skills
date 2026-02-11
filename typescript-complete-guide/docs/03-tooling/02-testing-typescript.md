# TypeScript テスト完全ガイド

> Vitest, Jest, 型テスト（tsd / expect-type）を使い、ランタイムの振る舞いと型の正しさの両方を検証する

## この章で学ぶこと

1. **Vitest によるテスト** -- Vite エコシステムと統合した高速テストランナーのセットアップと活用
2. **Jest + TypeScript** -- ts-jest / @swc/jest の設定と、既存プロジェクトでの Jest 運用
3. **型テスト** -- `expectTypeOf`（Vitest 組込み）や tsd を使い、ライブラリの型定義が正しいことを検証する技法

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
```

```typescript
// vitest.config.ts
import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,           // describe, it, expect をグローバルに
    environment: "node",     // or "jsdom", "happy-dom"
    include: ["src/**/*.test.ts", "tests/**/*.test.ts"],
    coverage: {
      provider: "v8",        // or "istanbul"
      reporter: ["text", "html", "lcov"],
      thresholds: {
        branches: 80,
        functions: 80,
        lines: 80,
        statements: 80,
      },
    },
    typecheck: {
      enabled: true,         // 型テストを有効化
      tsconfig: "./tsconfig.test.json",
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
    "test:ui": "vitest --ui"
  }
}
```

### 1-2. テストの書き方

```typescript
// src/user-service.test.ts
import { describe, it, expect, vi, beforeEach } from "vitest";
import { UserService } from "./user-service";
import type { IUserRepository, IEmailService } from "./interfaces";

// モック作成
function createMockUserRepo(): IUserRepository {
  return {
    findById: vi.fn(),
    save: vi.fn(),
    delete: vi.fn(),
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
  };
});

// スパイ
import * as mathUtils from "./math-utils";
vi.spyOn(mathUtils, "calculate").mockReturnValue(42);
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
  },
  collectCoverageFrom: [
    "src/**/*.ts",
    "!src/**/*.d.ts",
    "!src/**/*.test.ts",
  ],
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
            decorators: true,
          },
          transform: {
            decoratorVersion: "2022-03",
          },
        },
      },
    ],
  },
};

export default config;
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
import { Ok, Err, type Result, map, isOk } from "./result";

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
});
```

### 3-2. tsd を使った型テスト

```typescript
// test-d/index.test-d.ts（tsd 用）
import { expectType, expectError, expectAssignable } from "tsd";
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
  });

  it("branded types should not be interchangeable", () => {
    type UserId = string & { __brand: "UserId" };
    type OrderId = string & { __brand: "OrderId" };

    function getUser(id: UserId): void {}
    const orderId = "order-1" as OrderId;

    // @ts-expect-error -- OrderId は UserId に代入不可
    getUser(orderId);
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

### 4-2. テストデータビルダー + DI

```typescript
// テストヘルパー
function createTestContext() {
  const userRepo = createMockUserRepo();
  const emailService = createMockEmailService();
  const logger = createMockLogger();

  const service = new UserService(userRepo, emailService, logger);

  return { service, userRepo, emailService, logger };
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
});
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

### モック手法の比較

| 手法 | 用途 | 型安全性 | 柔軟性 |
|------|------|---------|--------|
| vi.fn() / jest.fn() | 関数モック | 中 | 高 |
| vi.mock() / jest.mock() | モジュールモック | 低 | 最高 |
| vi.spyOn() / jest.spyOn() | スパイ | 高 | 中 |
| 手動モック | DI ベース | 最高 | 中 |
| msw | HTTP モック | 高 | 高 |

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
it("should call repository save then email send", () => {
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

// OK: 振る舞い（結果）をテスト
it("should create user and send welcome email", () => {
  const result = await service.createUser(data);
  expect(result).toMatchObject({ _tag: "Ok" });
  expect(mockEmail.send).toHaveBeenCalledWith(
    data.email,
    expect.any(String),
    expect.any(String)
  );
});
```

---

## FAQ

### Q1: Vitest と Jest のどちらを選ぶべきですか？

新規プロジェクトでは Vitest を推奨します。Vite エコシステムとの統合、型テストの組込みサポート、高速な実行が利点です。既存の Jest プロジェクトを無理に移行する必要はありませんが、Vitest は Jest 互換の API を提供しているため、移行は比較的容易です。

### Q2: 型テストはどのくらい書くべきですか？

ライブラリやユーティリティ型を公開する場合は必須です。アプリケーションコードでは、複雑なジェネリクス関数やユーティリティ型に対して書くと効果的です。全ての関数に型テストを書く必要はなく、「型の推論結果が重要な部分」に集中してください。

### Q3: テストカバレッジの目標は何%が適切ですか？

80% が一般的な目標です。ただし、カバレッジ% だけを追うのではなく、「クリティカルパス（正常系・異常系の主要フロー）がカバーされているか」を重視してください。100% を目指すとテストの保守コストが膨大になります。

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
