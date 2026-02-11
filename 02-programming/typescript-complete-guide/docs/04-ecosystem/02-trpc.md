# tRPC 完全ガイド

> TypeScript の型推論だけでクライアント-サーバー間の型安全性を実現する、スキーマ不要の RPC フレームワーク

## この章で学ぶこと

1. **tRPC の基本** -- ルーター定義、プロシージャ、入力バリデーション（zod）の統合
2. **クライアント統合** -- React（@trpc/react-query）、Next.js、バニラクライアントでの利用方法
3. **高度なパターン** -- ミドルウェア、エラーハンドリング、サブスクリプション、テスト戦略

---

## 1. tRPC の基本

### 1-1. コンセプト

```
tRPC の型共有モデル:

  従来の REST API:
  +---------+    HTTP    +---------+
  | Client  | ---------> | Server  |
  +---------+    JSON    +---------+
  型なし(any)              型あり
  → 型定義を手動同期 or OpenAPI コード生成

  tRPC:
  +---------+    HTTP    +---------+
  | Client  | ---------> | Server  |
  +---------+            +---------+
       |                      |
       +--- TypeScript 型推論 -+
  型を直接共有（コード生成不要!）

  サーバーのルーター型がそのまま
  クライアントの型として推論される
```

### 1-2. サーバーセットアップ

```typescript
// server/trpc.ts -- tRPC の初期化
import { initTRPC, TRPCError } from "@trpc/server";
import { z } from "zod";

// コンテキスト型
interface Context {
  userId: string | null;
  db: PrismaClient;
}

const t = initTRPC.context<Context>().create({
  errorFormatter({ shape, error }) {
    return {
      ...shape,
      data: {
        ...shape.data,
        zodError:
          error.cause instanceof z.ZodError
            ? error.cause.flatten()
            : null,
      },
    };
  },
});

// エクスポート
export const router = t.router;
export const publicProcedure = t.procedure;
export const middleware = t.middleware;
```

### 1-3. ルーター定義

```typescript
// server/routers/user.ts
import { z } from "zod";
import { router, publicProcedure } from "../trpc";

const UserCreateInput = z.object({
  name: z.string().min(1).max(100),
  email: z.string().email(),
  age: z.number().int().min(0).optional(),
});

const UserUpdateInput = z.object({
  name: z.string().min(1).max(100).optional(),
  email: z.string().email().optional(),
});

export const userRouter = router({
  // Query: データ取得
  list: publicProcedure
    .input(
      z.object({
        page: z.number().int().min(1).default(1),
        limit: z.number().int().min(1).max(100).default(20),
      })
    )
    .query(async ({ ctx, input }) => {
      const users = await ctx.db.user.findMany({
        skip: (input.page - 1) * input.limit,
        take: input.limit,
        orderBy: { createdAt: "desc" },
      });
      const total = await ctx.db.user.count();
      return { users, total, page: input.page };
    }),

  // Query: 単一取得
  byId: publicProcedure
    .input(z.object({ id: z.string().uuid() }))
    .query(async ({ ctx, input }) => {
      const user = await ctx.db.user.findUnique({
        where: { id: input.id },
        include: { profile: true },
      });
      if (!user) {
        throw new TRPCError({
          code: "NOT_FOUND",
          message: `User ${input.id} not found`,
        });
      }
      return user;
    }),

  // Mutation: 作成
  create: publicProcedure
    .input(UserCreateInput)
    .mutation(async ({ ctx, input }) => {
      return ctx.db.user.create({ data: input });
    }),

  // Mutation: 更新
  update: publicProcedure
    .input(
      z.object({
        id: z.string().uuid(),
        data: UserUpdateInput,
      })
    )
    .mutation(async ({ ctx, input }) => {
      return ctx.db.user.update({
        where: { id: input.id },
        data: input.data,
      });
    }),

  // Mutation: 削除
  delete: publicProcedure
    .input(z.object({ id: z.string().uuid() }))
    .mutation(async ({ ctx, input }) => {
      await ctx.db.user.delete({ where: { id: input.id } });
      return { success: true };
    }),
});
```

### 1-4. ルートルーター

```typescript
// server/routers/_app.ts
import { router } from "../trpc";
import { userRouter } from "./user";
import { postRouter } from "./post";

export const appRouter = router({
  user: userRouter,
  post: postRouter,
});

// 型をエクスポート（クライアントで使用）
export type AppRouter = typeof appRouter;
```

---

## 2. ミドルウェアと認証

### 2-1. 認証ミドルウェア

```
ミドルウェアチェーン:

  リクエスト
     |
     v
  +--------------------+
  | publicProcedure    |  誰でもアクセス可
  +--------------------+
     |
     v
  +--------------------+
  | isAuthed           |  ログイン必須
  | (middleware)       |  ctx.userId を保証
  +--------------------+
     |
     v
  +--------------------+
  | isAdmin            |  管理者のみ
  | (middleware)       |  ctx.user.role を保証
  +--------------------+
     |
     v
  プロシージャ実行
```

```typescript
// 認証ミドルウェア
const isAuthed = middleware(async ({ ctx, next }) => {
  if (!ctx.userId) {
    throw new TRPCError({
      code: "UNAUTHORIZED",
      message: "You must be logged in",
    });
  }

  const user = await ctx.db.user.findUnique({
    where: { id: ctx.userId },
  });

  if (!user) {
    throw new TRPCError({
      code: "UNAUTHORIZED",
      message: "User not found",
    });
  }

  return next({
    ctx: {
      ...ctx,
      userId: ctx.userId, // string (non-null が保証)
      user,               // User 型が ctx に追加
    },
  });
});

// 管理者ミドルウェア
const isAdmin = middleware(async ({ ctx, next }) => {
  // isAuthed の後に使用する前提
  if ((ctx as any).user?.role !== "ADMIN") {
    throw new TRPCError({
      code: "FORBIDDEN",
      message: "Admin access required",
    });
  }
  return next({ ctx });
});

// プロシージャのベース
const protectedProcedure = publicProcedure.use(isAuthed);
const adminProcedure = protectedProcedure.use(isAdmin);

// 使用例
export const adminRouter = router({
  listAllUsers: adminProcedure.query(async ({ ctx }) => {
    // ctx.user は User 型で保証されている
    return ctx.db.user.findMany();
  }),
});
```

---

## 3. クライアント統合

### 3-1. React + React Query

```typescript
// utils/trpc.ts
import { createTRPCReact } from "@trpc/react-query";
import type { AppRouter } from "../server/routers/_app";

export const trpc = createTRPCReact<AppRouter>();
```

```typescript
// app/providers.tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { httpBatchLink } from "@trpc/client";
import { trpc } from "../utils/trpc";

const queryClient = new QueryClient();
const trpcClient = trpc.createClient({
  links: [
    httpBatchLink({
      url: "http://localhost:3000/api/trpc",
      headers: () => ({
        Authorization: `Bearer ${getToken()}`,
      }),
    }),
  ],
});

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <trpc.Provider client={trpcClient} queryClient={queryClient}>
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    </trpc.Provider>
  );
}
```

```typescript
// components/UserList.tsx
import { trpc } from "../utils/trpc";

export function UserList() {
  // Query: 型が自動推論される
  const { data, isLoading, error } = trpc.user.list.useQuery({
    page: 1,
    limit: 20,
  });
  // data の型: { users: User[]; total: number; page: number } | undefined

  // Mutation
  const createUser = trpc.user.create.useMutation({
    onSuccess: () => {
      // キャッシュを無効化して再取得
      utils.user.list.invalidate();
    },
  });

  const utils = trpc.useUtils();

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <div>
      {data.users.map((user) => (
        <div key={user.id}>
          {user.name} ({user.email})
        </div>
      ))}
      <button
        onClick={() =>
          createUser.mutate({
            name: "New User",
            email: "new@example.com",
          })
        }
      >
        Add User
      </button>
    </div>
  );
}
```

### 3-2. Next.js App Router 統合

```typescript
// app/api/trpc/[trpc]/route.ts
import { fetchRequestHandler } from "@trpc/server/adapters/fetch";
import { appRouter } from "../../../../server/routers/_app";

const handler = (req: Request) =>
  fetchRequestHandler({
    endpoint: "/api/trpc",
    req,
    router: appRouter,
    createContext: async () => {
      // リクエストからコンテキストを作成
      const userId = await getUserIdFromRequest(req);
      return { userId, db: prisma };
    },
  });

export { handler as GET, handler as POST };
```

### 3-3. サーバーサイド呼び出し

```typescript
// Server Components で直接呼び出し
import { appRouter } from "../server/routers/_app";

// サーバーサイド caller
const caller = appRouter.createCaller({
  userId: null,
  db: prisma,
});

export default async function UsersPage() {
  // サーバーサイドで直接呼び出し（HTTP 不要）
  const { users } = await caller.user.list({ page: 1, limit: 20 });

  return (
    <ul>
      {users.map((user) => (
        <li key={user.id}>{user.name}</li>
      ))}
    </ul>
  );
}
```

---

## 4. テスト

```typescript
// server/routers/user.test.ts
import { describe, it, expect, vi } from "vitest";
import { appRouter } from "./_app";

describe("userRouter", () => {
  const mockDb = {
    user: {
      findMany: vi.fn(),
      findUnique: vi.fn(),
      create: vi.fn(),
      count: vi.fn(),
    },
  } as unknown as PrismaClient;

  const caller = appRouter.createCaller({
    userId: "user-1",
    db: mockDb,
  });

  it("should list users", async () => {
    const mockUsers = [
      { id: "1", name: "Alice", email: "alice@test.com" },
    ];
    (mockDb.user.findMany as any).mockResolvedValue(mockUsers);
    (mockDb.user.count as any).mockResolvedValue(1);

    const result = await caller.user.list({ page: 1, limit: 10 });

    expect(result.users).toEqual(mockUsers);
    expect(result.total).toBe(1);
  });

  it("should throw NOT_FOUND for invalid id", async () => {
    (mockDb.user.findUnique as any).mockResolvedValue(null);

    await expect(
      caller.user.byId({ id: "00000000-0000-0000-0000-000000000000" })
    ).rejects.toThrow("NOT_FOUND");
  });
});
```

---

## 比較表

### API フレームワーク比較

| 特性 | tRPC | REST (Express) | GraphQL | gRPC |
|------|------|----------------|---------|------|
| 型安全性 | 最高(推論) | 手動/OpenAPI | 中(codegen) | 高(protobuf) |
| コード生成 | 不要 | OpenAPI→型 | 必要 | 必要 |
| クライアント | TypeScript only | 言語問わず | 言語問わず | 言語問わず |
| バンドルサイズ | 小 | - | 中 | 大 |
| 学習コスト | 低 | 最低 | 中 | 高 |
| エコシステム | 成長中 | 最大 | 大 | 中 |

### tRPC リンク比較

| リンク | 用途 | バッチ | WebSocket |
|--------|------|--------|-----------|
| httpBatchLink | 標準 | あり | なし |
| httpLink | 単一リクエスト | なし | なし |
| wsLink | リアルタイム | なし | あり |
| splitLink | 条件分岐 | - | - |
| loggerLink | デバッグ | - | - |

---

## アンチパターン

### AP-1: ルーターが肥大化する

```typescript
// NG: 1 ファイルに全プロシージャを詰め込む
export const appRouter = router({
  getUser: publicProcedure.query(/* 50行 */),
  createUser: publicProcedure.mutation(/* 80行 */),
  updateUser: publicProcedure.mutation(/* 60行 */),
  deleteUser: publicProcedure.mutation(/* 30行 */),
  getPost: publicProcedure.query(/* 50行 */),
  // ... 100以上のプロシージャ
});

// OK: ドメインごとにルーターを分割
export const appRouter = router({
  user: userRouter,     // server/routers/user.ts
  post: postRouter,     // server/routers/post.ts
  comment: commentRouter, // server/routers/comment.ts
  admin: adminRouter,   // server/routers/admin.ts
});
```

### AP-2: クライアントで型を手動定義する

```typescript
// NG: サーバーの型を手動で複製
interface User {
  id: string;
  name: string;
  email: string;
}
const { data } = trpc.user.byId.useQuery({ id: "1" });
const user = data as User; // 手動キャスト

// OK: 型は自動推論に任せる
const { data } = trpc.user.byId.useQuery({ id: "1" });
// data の型はサーバーのルーター定義から自動推論される
```

---

## FAQ

### Q1: tRPC は REST API の代替になりますか？

TypeScript のモノレポ（フロントエンド + バックエンド）では完全に REST の代替になります。ただし、モバイルアプリ（Swift/Kotlin）や他言語のクライアントがある場合は REST/GraphQL の方が適しています。tRPC は「TypeScript エコシステム内」で最大の威力を発揮します。

### Q2: tRPC v10 と v11 の違いは何ですか？

v11 では React Server Components との統合強化、新しいリンク API、パフォーマンス改善が含まれます。v10 からの移行は比較的容易で、破壊的変更は少ないです。

### Q3: tRPC と GraphQL は併用できますか？

技術的には可能ですが、通常は片方を選択します。社内ツールやフルスタック TypeScript プロジェクトでは tRPC、公開 API やマルチプラットフォームでは GraphQL が適しています。

---

## まとめ表

| 概念 | 要点 |
|------|------|
| tRPC | TypeScript 型推論で E2E 型安全な API |
| Router | プロシージャをネストして構成 |
| Query / Mutation | 読み取り / 書き込み操作の区分 |
| Middleware | 認証、ログ、エラー処理のチェーン |
| createCaller | サーバーサイドでの直接呼び出し |
| zod 統合 | 入力バリデーションとスキーマ定義 |

---

## 次に読むべきガイド

- [Zod バリデーション](./00-zod-validation.md) -- tRPC の入力スキーマを定義する zod の全機能
- [Prisma + TypeScript](./01-prisma-typescript.md) -- tRPC + Prisma でフルスタック型安全
- [エラーハンドリング](../02-patterns/00-error-handling.md) -- tRPC のエラーハンドリング設計

---

## 参考文献

1. **tRPC Documentation**
   https://trpc.io/docs

2. **tRPC GitHub Repository**
   https://github.com/trpc/trpc

3. **Create T3 App** -- tRPC + Next.js + Prisma のスターターキット
   https://create.t3.gg/
