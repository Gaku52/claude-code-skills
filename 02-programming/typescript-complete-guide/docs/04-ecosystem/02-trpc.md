# tRPC 完全ガイド

> TypeScript の型推論だけでクライアント-サーバー間の型安全性を実現する、スキーマ不要の RPC フレームワーク

## この章で学ぶこと

1. **tRPC の基本** -- ルーター定義、プロシージャ、入力バリデーション（zod）の統合
2. **クライアント統合** -- React（@trpc/react-query）、Next.js、バニラクライアントでの利用方法
3. **高度なパターン** -- ミドルウェア、エラーハンドリング、サブスクリプション、テスト戦略
4. **パフォーマンス最適化** -- バッチング、キャッシュ戦略、レスポンス最適化
5. **本番運用** -- デプロイ、モニタリング、セキュリティ、スケーリング

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

### 1-5. コンテキスト生成の詳細パターン

```typescript
// server/context.ts -- 高度なコンテキスト生成
import { inferAsyncReturnType } from "@trpc/server";
import { CreateNextContextOptions } from "@trpc/server/adapters/next";
import { getServerSession } from "next-auth";
import { authOptions } from "../auth";
import { prisma } from "../db";

/**
 * コンテキスト生成関数
 * リクエストごとに呼ばれ、全プロシージャから参照可能な共有状態を作る
 */
export async function createContext(opts: CreateNextContextOptions) {
  const session = await getServerSession(opts.req, opts.res, authOptions);

  return {
    session,
    userId: session?.user?.id ?? null,
    db: prisma,
    req: opts.req,
    res: opts.res,
  };
}

export type Context = inferAsyncReturnType<typeof createContext>;
```

```typescript
// server/context.ts -- Fetch API 用（App Router / Edge Runtime）
import { FetchCreateContextFnOptions } from "@trpc/server/adapters/fetch";
import { getToken } from "next-auth/jwt";

export async function createContext(opts: FetchCreateContextFnOptions) {
  // ヘッダーからトークンを取得
  const authHeader = opts.req.headers.get("authorization");
  const token = authHeader?.replace("Bearer ", "");

  let userId: string | null = null;
  if (token) {
    try {
      const decoded = await verifyJWT(token);
      userId = decoded.sub;
    } catch {
      // トークン無効 -- userId は null のまま
    }
  }

  return {
    userId,
    db: prisma,
    requestId: crypto.randomUUID(),
    ip: opts.req.headers.get("x-forwarded-for") ?? "unknown",
  };
}
```

### 1-6. 入力スキーマの高度な定義

```typescript
// server/schemas/user.ts -- 再利用可能なスキーマ
import { z } from "zod";

// 基本スキーマ
export const UserSchema = z.object({
  id: z.string().uuid(),
  name: z.string().min(1).max(100),
  email: z.string().email(),
  role: z.enum(["USER", "ADMIN", "MODERATOR"]),
  createdAt: z.date(),
  updatedAt: z.date(),
});

// 部分型（PATCH 更新用）
export const UserUpdateSchema = UserSchema.pick({
  name: true,
  email: true,
  role: true,
}).partial();

// 作成用（id, timestamps を除外）
export const UserCreateSchema = UserSchema.omit({
  id: true,
  createdAt: true,
  updatedAt: true,
});

// フィルター用
export const UserFilterSchema = z.object({
  search: z.string().optional(),
  role: z.enum(["USER", "ADMIN", "MODERATOR"]).optional(),
  createdAfter: z.date().optional(),
  createdBefore: z.date().optional(),
});

// ページネーション用（汎用）
export const PaginationSchema = z.object({
  page: z.number().int().min(1).default(1),
  limit: z.number().int().min(1).max(100).default(20),
  sortBy: z.string().optional(),
  sortOrder: z.enum(["asc", "desc"]).default("desc"),
});

// 組み合わせ
export const UserListInput = PaginationSchema.merge(UserFilterSchema);

// 型の抽出（他のファイルで使用可能）
export type UserCreate = z.infer<typeof UserCreateSchema>;
export type UserUpdate = z.infer<typeof UserUpdateSchema>;
export type UserFilter = z.infer<typeof UserFilterSchema>;
```

```typescript
// server/routers/user.ts -- 外部スキーマを利用したルーター
import { router, publicProcedure, protectedProcedure } from "../trpc";
import {
  UserCreateSchema,
  UserUpdateSchema,
  UserListInput,
} from "../schemas/user";

export const userRouter = router({
  list: publicProcedure
    .input(UserListInput)
    .query(async ({ ctx, input }) => {
      const { page, limit, sortBy, sortOrder, search, role } = input;

      const where = {
        ...(search && {
          OR: [
            { name: { contains: search, mode: "insensitive" as const } },
            { email: { contains: search, mode: "insensitive" as const } },
          ],
        }),
        ...(role && { role }),
      };

      const [users, total] = await Promise.all([
        ctx.db.user.findMany({
          where,
          skip: (page - 1) * limit,
          take: limit,
          orderBy: sortBy ? { [sortBy]: sortOrder } : { createdAt: sortOrder },
        }),
        ctx.db.user.count({ where }),
      ]);

      return {
        users,
        total,
        page,
        totalPages: Math.ceil(total / limit),
        hasMore: page * limit < total,
      };
    }),

  create: protectedProcedure
    .input(UserCreateSchema)
    .mutation(async ({ ctx, input }) => {
      // メールの重複チェック
      const existing = await ctx.db.user.findUnique({
        where: { email: input.email },
      });
      if (existing) {
        throw new TRPCError({
          code: "CONFLICT",
          message: "A user with this email already exists",
        });
      }
      return ctx.db.user.create({ data: input });
    }),
});
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

### 2-2. ロギングミドルウェア

```typescript
// server/middleware/logging.ts
import { middleware } from "../trpc";

/**
 * リクエストのタイミングとパスをログ出力するミドルウェア
 */
export const loggerMiddleware = middleware(async ({ path, type, next }) => {
  const start = Date.now();

  const result = await next();

  const durationMs = Date.now() - start;
  const meta = {
    path,
    type,         // "query" | "mutation" | "subscription"
    durationMs,
    ok: result.ok,
  };

  if (result.ok) {
    console.log(`[tRPC] ${type} ${path} - ${durationMs}ms OK`);
  } else {
    console.error(`[tRPC] ${type} ${path} - ${durationMs}ms ERROR`, meta);
  }

  return result;
});

// 全プロシージャに適用
export const publicProcedure = t.procedure.use(loggerMiddleware);
```

### 2-3. レート制限ミドルウェア

```typescript
// server/middleware/rateLimit.ts
import { middleware } from "../trpc";
import { TRPCError } from "@trpc/server";

// シンプルなインメモリレート制限（本番では Redis を使用）
const rateLimitMap = new Map<string, { count: number; resetAt: number }>();

interface RateLimitOptions {
  windowMs: number;  // ウィンドウ時間（ミリ秒）
  maxRequests: number;  // ウィンドウ内の最大リクエスト数
}

export function createRateLimitMiddleware(options: RateLimitOptions) {
  return middleware(async ({ ctx, next }) => {
    const key = ctx.userId ?? ctx.ip ?? "anonymous";
    const now = Date.now();

    let record = rateLimitMap.get(key);

    if (!record || now > record.resetAt) {
      record = { count: 0, resetAt: now + options.windowMs };
      rateLimitMap.set(key, record);
    }

    record.count++;

    if (record.count > options.maxRequests) {
      throw new TRPCError({
        code: "TOO_MANY_REQUESTS",
        message: `Rate limit exceeded. Try again in ${Math.ceil(
          (record.resetAt - now) / 1000
        )} seconds`,
      });
    }

    return next();
  });
}

// 使用例
const rateLimitedProcedure = publicProcedure.use(
  createRateLimitMiddleware({
    windowMs: 60_000,   // 1分間
    maxRequests: 100,    // 最大100リクエスト
  })
);

// Redis ベースのレート制限（本番推奨）
import { Redis } from "ioredis";

const redis = new Redis(process.env.REDIS_URL!);

export function createRedisRateLimitMiddleware(options: RateLimitOptions) {
  return middleware(async ({ ctx, next }) => {
    const key = `ratelimit:${ctx.userId ?? ctx.ip ?? "anon"}`;
    const current = await redis.incr(key);

    if (current === 1) {
      await redis.pexpire(key, options.windowMs);
    }

    if (current > options.maxRequests) {
      const ttl = await redis.pttl(key);
      throw new TRPCError({
        code: "TOO_MANY_REQUESTS",
        message: `Rate limit exceeded. Try again in ${Math.ceil(ttl / 1000)}s`,
      });
    }

    return next();
  });
}
```

### 2-4. 組織ベースのアクセス制御ミドルウェア

```typescript
// server/middleware/organization.ts
import { middleware } from "../trpc";
import { TRPCError } from "@trpc/server";
import { z } from "zod";

/**
 * 組織メンバーシップを検証するミドルウェア
 * protectedProcedure の後に使用する
 */
export const withOrganization = middleware(async ({ ctx, next, rawInput }) => {
  // rawInput から orgId を取得
  const parsed = z.object({ orgId: z.string() }).safeParse(rawInput);
  if (!parsed.success) {
    throw new TRPCError({
      code: "BAD_REQUEST",
      message: "Organization ID is required",
    });
  }

  const membership = await ctx.db.organizationMember.findUnique({
    where: {
      userId_organizationId: {
        userId: ctx.userId,
        organizationId: parsed.data.orgId,
      },
    },
    include: { organization: true },
  });

  if (!membership) {
    throw new TRPCError({
      code: "FORBIDDEN",
      message: "You are not a member of this organization",
    });
  }

  return next({
    ctx: {
      ...ctx,
      organization: membership.organization,
      memberRole: membership.role, // "OWNER" | "ADMIN" | "MEMBER"
    },
  });
});

// 組織管理者専用
export const withOrgAdmin = middleware(async ({ ctx, next }) => {
  if (!["OWNER", "ADMIN"].includes((ctx as any).memberRole)) {
    throw new TRPCError({
      code: "FORBIDDEN",
      message: "Organization admin access required",
    });
  }
  return next({ ctx });
});

// プロシージャ定義
const orgProcedure = protectedProcedure.use(withOrganization);
const orgAdminProcedure = orgProcedure.use(withOrgAdmin);

// 使用例
export const orgRouter = router({
  getMembers: orgProcedure
    .input(z.object({ orgId: z.string() }))
    .query(async ({ ctx }) => {
      return ctx.db.organizationMember.findMany({
        where: { organizationId: ctx.organization.id },
        include: { user: true },
      });
    }),

  removeMember: orgAdminProcedure
    .input(z.object({
      orgId: z.string(),
      memberId: z.string(),
    }))
    .mutation(async ({ ctx, input }) => {
      // OWNER は削除不可
      const target = await ctx.db.organizationMember.findUnique({
        where: {
          userId_organizationId: {
            userId: input.memberId,
            organizationId: ctx.organization.id,
          },
        },
      });

      if (target?.role === "OWNER") {
        throw new TRPCError({
          code: "FORBIDDEN",
          message: "Cannot remove the organization owner",
        });
      }

      await ctx.db.organizationMember.delete({
        where: {
          userId_organizationId: {
            userId: input.memberId,
            organizationId: ctx.organization.id,
          },
        },
      });

      return { success: true };
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

### 3-4. バニラクライアント（React 不使用）

```typescript
// client/vanilla.ts
import { createTRPCClient, httpBatchLink } from "@trpc/client";
import type { AppRouter } from "../server/routers/_app";

// React 不要の純粋な tRPC クライアント
const client = createTRPCClient<AppRouter>({
  links: [
    httpBatchLink({
      url: "http://localhost:3000/api/trpc",
      headers: () => ({
        Authorization: `Bearer ${localStorage.getItem("token")}`,
      }),
    }),
  ],
});

// Query の実行
async function fetchUsers() {
  const result = await client.user.list.query({
    page: 1,
    limit: 10,
  });
  // result の型は自動推論される
  console.log(result.users);
  console.log(result.total);
}

// Mutation の実行
async function createUser(name: string, email: string) {
  const newUser = await client.user.create.mutate({
    name,
    email,
  });
  // newUser の型もサーバーの戻り値から推論
  console.log(`Created user: ${newUser.id}`);
}

// Node.js スクリプトや CLI ツールからの利用
async function main() {
  try {
    await fetchUsers();
    await createUser("Alice", "alice@example.com");
  } catch (err) {
    if (err instanceof TRPCClientError) {
      console.error("tRPC Error:", err.message);
      console.error("Error code:", err.data?.code);
    }
  }
}
```

### 3-5. 楽観的更新（Optimistic Updates）

```typescript
// components/TodoList.tsx -- 楽観的更新の完全な例
import { trpc } from "../utils/trpc";

export function TodoList() {
  const utils = trpc.useUtils();

  const { data: todos } = trpc.todo.list.useQuery();

  const toggleTodo = trpc.todo.toggle.useMutation({
    // 楽観的更新: サーバーレスポンスを待たずに UI を更新
    onMutate: async (input) => {
      // 1. 進行中の再取得をキャンセル（楽観的更新を上書きしないため）
      await utils.todo.list.cancel();

      // 2. 現在のデータをスナップショット
      const previousTodos = utils.todo.list.getData();

      // 3. 楽観的にキャッシュを更新
      utils.todo.list.setData(undefined, (old) => {
        if (!old) return old;
        return old.map((todo) =>
          todo.id === input.id
            ? { ...todo, completed: !todo.completed }
            : todo
        );
      });

      // 4. ロールバック用のコンテキストを返す
      return { previousTodos };
    },

    // エラー時: スナップショットにロールバック
    onError: (_err, _input, context) => {
      if (context?.previousTodos) {
        utils.todo.list.setData(undefined, context.previousTodos);
      }
    },

    // 成功・失敗に関わらず、最終的にサーバーデータと同期
    onSettled: () => {
      utils.todo.list.invalidate();
    },
  });

  const deleteTodo = trpc.todo.delete.useMutation({
    onMutate: async (input) => {
      await utils.todo.list.cancel();
      const previousTodos = utils.todo.list.getData();

      // 楽観的に削除
      utils.todo.list.setData(undefined, (old) => {
        if (!old) return old;
        return old.filter((todo) => todo.id !== input.id);
      });

      return { previousTodos };
    },
    onError: (_err, _input, context) => {
      if (context?.previousTodos) {
        utils.todo.list.setData(undefined, context.previousTodos);
      }
    },
    onSettled: () => {
      utils.todo.list.invalidate();
    },
  });

  return (
    <ul>
      {todos?.map((todo) => (
        <li key={todo.id}>
          <label>
            <input
              type="checkbox"
              checked={todo.completed}
              onChange={() => toggleTodo.mutate({ id: todo.id })}
            />
            <span style={{
              textDecoration: todo.completed ? "line-through" : "none",
            }}>
              {todo.title}
            </span>
          </label>
          <button onClick={() => deleteTodo.mutate({ id: todo.id })}>
            Delete
          </button>
        </li>
      ))}
    </ul>
  );
}
```

### 3-6. 無限スクロール

```typescript
// components/InfinitePostList.tsx
import { trpc } from "../utils/trpc";
import { useCallback, useRef, useEffect } from "react";

// サーバー側: カーソルベースのページネーション
// server/routers/post.ts
export const postRouter = router({
  infiniteList: publicProcedure
    .input(
      z.object({
        limit: z.number().int().min(1).max(50).default(20),
        cursor: z.string().optional(),  // 最後のアイテムの ID
        category: z.string().optional(),
      })
    )
    .query(async ({ ctx, input }) => {
      const { limit, cursor, category } = input;

      const posts = await ctx.db.post.findMany({
        take: limit + 1,  // +1 で次ページの有無を判定
        ...(cursor && {
          cursor: { id: cursor },
          skip: 1,  // カーソル自体はスキップ
        }),
        ...(category && { where: { category } }),
        orderBy: { createdAt: "desc" },
        include: { author: { select: { name: true, avatar: true } } },
      });

      let nextCursor: string | undefined;
      if (posts.length > limit) {
        const nextItem = posts.pop();
        nextCursor = nextItem!.id;
      }

      return { posts, nextCursor };
    }),
});

// クライアント側
export function InfinitePostList() {
  const {
    data,
    fetchNextPage,
    hasNextPage,
    isFetchingNextPage,
    isLoading,
  } = trpc.post.infiniteList.useInfiniteQuery(
    { limit: 20 },
    {
      getNextPageParam: (lastPage) => lastPage.nextCursor,
    }
  );

  // Intersection Observer で自動読み込み
  const observerRef = useRef<IntersectionObserver>();
  const loadMoreRef = useCallback(
    (node: HTMLDivElement | null) => {
      if (isFetchingNextPage) return;
      if (observerRef.current) observerRef.current.disconnect();

      observerRef.current = new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting && hasNextPage) {
          fetchNextPage();
        }
      });

      if (node) observerRef.current.observe(node);
    },
    [isFetchingNextPage, hasNextPage, fetchNextPage]
  );

  if (isLoading) return <div>Loading...</div>;

  const allPosts = data?.pages.flatMap((page) => page.posts) ?? [];

  return (
    <div>
      {allPosts.map((post) => (
        <article key={post.id}>
          <h2>{post.title}</h2>
          <p>by {post.author.name}</p>
          <p>{post.content.slice(0, 200)}...</p>
        </article>
      ))}

      <div ref={loadMoreRef}>
        {isFetchingNextPage ? (
          <p>Loading more...</p>
        ) : hasNextPage ? (
          <p>Scroll to load more</p>
        ) : (
          <p>No more posts</p>
        )}
      </div>
    </div>
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

### 4-1. 統合テスト

```typescript
// server/routers/user.integration.test.ts
import { describe, it, expect, beforeAll, afterAll, beforeEach } from "vitest";
import { createTRPCClient, httpBatchLink } from "@trpc/client";
import { createHTTPServer } from "@trpc/server/adapters/standalone";
import { appRouter } from "./_app";
import type { AppRouter } from "./_app";
import { PrismaClient } from "@prisma/client";

describe("User Router Integration", () => {
  let server: ReturnType<typeof createHTTPServer>;
  let client: ReturnType<typeof createTRPCClient<AppRouter>>;
  let prisma: PrismaClient;

  beforeAll(async () => {
    prisma = new PrismaClient({
      datasources: { db: { url: process.env.TEST_DATABASE_URL } },
    });

    server = createHTTPServer({
      router: appRouter,
      createContext: () => ({
        userId: "test-user-id",
        db: prisma,
      }),
    });

    // ランダムポートでサーバー起動
    const { port } = server.listen(0);

    client = createTRPCClient<AppRouter>({
      links: [
        httpBatchLink({
          url: `http://localhost:${port}`,
        }),
      ],
    });
  });

  afterAll(async () => {
    server.server.close();
    await prisma.$disconnect();
  });

  beforeEach(async () => {
    // テストごとにデータベースをクリーン
    await prisma.user.deleteMany();
  });

  it("should create and retrieve a user", async () => {
    // ユーザー作成
    const created = await client.user.create.mutate({
      name: "Integration Test User",
      email: "integration@test.com",
    });

    expect(created.id).toBeDefined();
    expect(created.name).toBe("Integration Test User");

    // 作成したユーザーを取得
    const retrieved = await client.user.byId.query({ id: created.id });
    expect(retrieved.name).toBe("Integration Test User");
    expect(retrieved.email).toBe("integration@test.com");
  });

  it("should paginate users correctly", async () => {
    // 25件のユーザーを作成
    await Promise.all(
      Array.from({ length: 25 }, (_, i) =>
        client.user.create.mutate({
          name: `User ${i}`,
          email: `user${i}@test.com`,
        })
      )
    );

    const page1 = await client.user.list.query({ page: 1, limit: 10 });
    expect(page1.users).toHaveLength(10);
    expect(page1.total).toBe(25);

    const page3 = await client.user.list.query({ page: 3, limit: 10 });
    expect(page3.users).toHaveLength(5);
  });

  it("should handle concurrent mutations safely", async () => {
    const user = await client.user.create.mutate({
      name: "Concurrent User",
      email: "concurrent@test.com",
    });

    // 同時に更新と削除を実行
    const results = await Promise.allSettled([
      client.user.update.mutate({
        id: user.id,
        data: { name: "Updated Name" },
      }),
      client.user.delete.mutate({ id: user.id }),
    ]);

    // 少なくとも1つは成功するはず
    const successes = results.filter((r) => r.status === "fulfilled");
    expect(successes.length).toBeGreaterThanOrEqual(1);
  });
});
```

### 4-2. ミドルウェアのテスト

```typescript
// server/middleware/auth.test.ts
import { describe, it, expect, vi } from "vitest";
import { appRouter } from "../routers/_app";
import { TRPCError } from "@trpc/server";

describe("Auth Middleware", () => {
  const mockDb = {
    user: {
      findUnique: vi.fn(),
      findMany: vi.fn(),
    },
  } as unknown as PrismaClient;

  it("should reject unauthenticated requests to protected routes", async () => {
    const caller = appRouter.createCaller({
      userId: null, // 未認証
      db: mockDb,
    });

    await expect(
      caller.admin.listAllUsers()
    ).rejects.toThrow(TRPCError);

    await expect(
      caller.admin.listAllUsers()
    ).rejects.toMatchObject({
      code: "UNAUTHORIZED",
    });
  });

  it("should reject non-admin users from admin routes", async () => {
    (mockDb.user.findUnique as any).mockResolvedValue({
      id: "user-1",
      role: "USER", // 管理者ではない
    });

    const caller = appRouter.createCaller({
      userId: "user-1",
      db: mockDb,
    });

    await expect(
      caller.admin.listAllUsers()
    ).rejects.toMatchObject({
      code: "FORBIDDEN",
    });
  });

  it("should allow admin users to access admin routes", async () => {
    const mockUsers = [{ id: "1", name: "User" }];
    (mockDb.user.findUnique as any).mockResolvedValue({
      id: "admin-1",
      role: "ADMIN",
    });
    (mockDb.user.findMany as any).mockResolvedValue(mockUsers);

    const caller = appRouter.createCaller({
      userId: "admin-1",
      db: mockDb,
    });

    const result = await caller.admin.listAllUsers();
    expect(result).toEqual(mockUsers);
  });
});
```

---

## 5. サブスクリプション（リアルタイム通信）

### 5-1. WebSocket サブスクリプションの設定

```
tRPC サブスクリプションのアーキテクチャ:

  +-----------+    WebSocket     +------------+
  |  Client   | <=============> |   Server   |
  +-----------+    双方向通信     +------------+
       |                              |
       | subscribe('onMessage')       | EventEmitter
       |------------------------------>|  .emit('message', data)
       |                              |
       |      { type: 'data',         |
       |        data: Message }       |
       |<-----------------------------|
       |                              |
       |      { type: 'data',         |
       |        data: Message }       |
       |<-----------------------------|
       |                              |
```

```typescript
// server/routers/chat.ts
import { observable } from "@trpc/server/observable";
import { z } from "zod";
import { EventEmitter } from "events";
import { router, protectedProcedure } from "../trpc";

// イベントエミッター（本番では Redis Pub/Sub を推奨）
const ee = new EventEmitter();

interface ChatMessage {
  id: string;
  roomId: string;
  userId: string;
  userName: string;
  content: string;
  createdAt: Date;
}

export const chatRouter = router({
  // メッセージ送信
  sendMessage: protectedProcedure
    .input(
      z.object({
        roomId: z.string(),
        content: z.string().min(1).max(2000),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const message: ChatMessage = {
        id: crypto.randomUUID(),
        roomId: input.roomId,
        userId: ctx.userId,
        userName: ctx.user.name,
        content: input.content,
        createdAt: new Date(),
      };

      // DB に保存
      await ctx.db.message.create({ data: message });

      // イベントを発行（サブスクライバーに通知）
      ee.emit(`room:${input.roomId}`, message);

      return message;
    }),

  // リアルタイムメッセージ受信
  onMessage: protectedProcedure
    .input(z.object({ roomId: z.string() }))
    .subscription(({ input }) => {
      return observable<ChatMessage>((emit) => {
        const handler = (message: ChatMessage) => {
          emit.next(message);
        };

        // イベントリスナーを登録
        ee.on(`room:${input.roomId}`, handler);

        // クリーンアップ: クライアント切断時に呼ばれる
        return () => {
          ee.off(`room:${input.roomId}`, handler);
        };
      });
    }),

  // ユーザーのタイピング状態
  onTyping: protectedProcedure
    .input(z.object({ roomId: z.string() }))
    .subscription(({ input }) => {
      return observable<{ userId: string; userName: string }>((emit) => {
        const handler = (data: { userId: string; userName: string }) => {
          emit.next(data);
        };

        ee.on(`typing:${input.roomId}`, handler);
        return () => ee.off(`typing:${input.roomId}`, handler);
      });
    }),

  startTyping: protectedProcedure
    .input(z.object({ roomId: z.string() }))
    .mutation(({ ctx, input }) => {
      ee.emit(`typing:${input.roomId}`, {
        userId: ctx.userId,
        userName: ctx.user.name,
      });
      return { ok: true };
    }),
});
```

### 5-2. クライアント側の WebSocket 設定

```typescript
// utils/trpc.ts -- WebSocket リンク付き
import { createTRPCReact } from "@trpc/react-query";
import { createWSClient, wsLink, httpBatchLink, splitLink } from "@trpc/client";
import type { AppRouter } from "../server/routers/_app";

export const trpc = createTRPCReact<AppRouter>();

// WebSocket クライアント
const wsClient = createWSClient({
  url: "ws://localhost:3000/trpc",
  retryDelayMs: (attemptIndex) =>
    Math.min(1000 * 2 ** attemptIndex, 30_000), // 指数バックオフ
  onOpen: () => console.log("WebSocket connected"),
  onClose: () => console.log("WebSocket disconnected"),
});

export const trpcClient = trpc.createClient({
  links: [
    // splitLink でサブスクリプションだけ WebSocket を使う
    splitLink({
      condition: (op) => op.type === "subscription",
      true: wsLink({ client: wsClient }),
      false: httpBatchLink({
        url: "http://localhost:3000/api/trpc",
        headers: () => ({
          Authorization: `Bearer ${getToken()}`,
        }),
      }),
    }),
  ],
});
```

```typescript
// components/ChatRoom.tsx
import { trpc } from "../utils/trpc";
import { useState } from "react";

export function ChatRoom({ roomId }: { roomId: string }) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [typingUsers, setTypingUsers] = useState<string[]>([]);

  // サブスクリプション: 新しいメッセージをリアルタイム受信
  trpc.chat.onMessage.useSubscription(
    { roomId },
    {
      onData: (message) => {
        setMessages((prev) => [...prev, message]);
      },
      onError: (err) => {
        console.error("Subscription error:", err);
      },
    }
  );

  // タイピング表示
  trpc.chat.onTyping.useSubscription(
    { roomId },
    {
      onData: ({ userName }) => {
        setTypingUsers((prev) => [...new Set([...prev, userName])]);
        // 3秒後に消す
        setTimeout(() => {
          setTypingUsers((prev) => prev.filter((u) => u !== userName));
        }, 3000);
      },
    }
  );

  const sendMessage = trpc.chat.sendMessage.useMutation();
  const startTyping = trpc.chat.startTyping.useMutation();

  const handleSend = async () => {
    if (!input.trim()) return;
    await sendMessage.mutateAsync({ roomId, content: input });
    setInput("");
  };

  return (
    <div>
      <div className="messages">
        {messages.map((msg) => (
          <div key={msg.id}>
            <strong>{msg.userName}</strong>: {msg.content}
          </div>
        ))}
      </div>

      {typingUsers.length > 0 && (
        <p>{typingUsers.join(", ")} is typing...</p>
      )}

      <input
        value={input}
        onChange={(e) => {
          setInput(e.target.value);
          startTyping.mutate({ roomId });
        }}
        onKeyDown={(e) => e.key === "Enter" && handleSend()}
      />
      <button onClick={handleSend}>Send</button>
    </div>
  );
}
```

---

## 6. エラーハンドリング

### 6-1. tRPC エラーコード一覧

```
tRPC エラーコードと HTTP ステータスの対応:

  +------------------------+--------+-----------------------------+
  | tRPC Code              | HTTP   | 用途                        |
  +------------------------+--------+-----------------------------+
  | BAD_REQUEST            | 400    | 入力バリデーション失敗       |
  | UNAUTHORIZED           | 401    | 認証が必要                  |
  | FORBIDDEN              | 403    | 権限不足                    |
  | NOT_FOUND              | 404    | リソースが見つからない       |
  | METHOD_NOT_SUPPORTED   | 405    | 非対応メソッド              |
  | TIMEOUT                | 408    | タイムアウト                |
  | CONFLICT               | 409    | リソース競合                |
  | PRECONDITION_FAILED    | 412    | 前提条件失敗                |
  | PAYLOAD_TOO_LARGE      | 413    | ペイロード過大              |
  | UNPROCESSABLE_CONTENT  | 422    | 処理不能なコンテンツ        |
  | TOO_MANY_REQUESTS      | 429    | レート制限超過              |
  | CLIENT_CLOSED_REQUEST  | 499    | クライアントが接続を閉じた   |
  | INTERNAL_SERVER_ERROR  | 500    | 内部エラー                  |
  +------------------------+--------+-----------------------------+
```

### 6-2. カスタムエラークラス

```typescript
// server/errors.ts
import { TRPCError } from "@trpc/server";

/**
 * ビジネスロジック用のカスタムエラー
 */
export class BusinessError extends TRPCError {
  public readonly errorCode: string;

  constructor(opts: {
    code: TRPCError["code"];
    message: string;
    errorCode: string;
    cause?: Error;
  }) {
    super({
      code: opts.code,
      message: opts.message,
      cause: opts.cause,
    });
    this.errorCode = opts.errorCode;
  }
}

// 具体的なエラー
export class InsufficientBalanceError extends BusinessError {
  constructor(balance: number, required: number) {
    super({
      code: "BAD_REQUEST",
      message: `Insufficient balance: have ${balance}, need ${required}`,
      errorCode: "INSUFFICIENT_BALANCE",
    });
  }
}

export class DuplicateEmailError extends BusinessError {
  constructor(email: string) {
    super({
      code: "CONFLICT",
      message: `Email ${email} is already registered`,
      errorCode: "DUPLICATE_EMAIL",
    });
  }
}

export class SubscriptionExpiredError extends BusinessError {
  constructor(expiredAt: Date) {
    super({
      code: "FORBIDDEN",
      message: `Subscription expired at ${expiredAt.toISOString()}`,
      errorCode: "SUBSCRIPTION_EXPIRED",
    });
  }
}
```

### 6-3. エラーフォーマッターの高度な設定

```typescript
// server/trpc.ts -- 高度なエラーフォーマッター
import { initTRPC } from "@trpc/server";
import { ZodError } from "zod";
import { BusinessError } from "./errors";

const t = initTRPC.context<Context>().create({
  errorFormatter({ shape, error }) {
    return {
      ...shape,
      data: {
        ...shape.data,
        // Zod バリデーションエラーの詳細
        zodError:
          error.cause instanceof ZodError
            ? error.cause.flatten()
            : null,
        // ビジネスエラーコード
        businessCode:
          error instanceof BusinessError
            ? error.errorCode
            : null,
        // 開発環境のみスタックトレースを含める
        stack:
          process.env.NODE_ENV === "development"
            ? error.stack
            : undefined,
      },
    };
  },
});
```

### 6-4. クライアント側のエラーハンドリング

```typescript
// hooks/useTRPCError.ts
import { TRPCClientError } from "@trpc/client";
import type { AppRouter } from "../server/routers/_app";
import { toast } from "sonner";

/**
 * tRPC エラーを統一的にハンドリングするユーティリティ
 */
export function handleTRPCError(error: unknown) {
  if (!(error instanceof TRPCClientError<AppRouter>)) {
    toast.error("An unexpected error occurred");
    console.error("Non-tRPC error:", error);
    return;
  }

  const { data, message } = error;

  // Zod バリデーションエラー
  if (data?.zodError) {
    const fieldErrors = data.zodError.fieldErrors;
    const messages = Object.entries(fieldErrors)
      .map(([field, errors]) => `${field}: ${(errors as string[]).join(", ")}`)
      .join("\n");
    toast.error(`Validation Error:\n${messages}`);
    return;
  }

  // ビジネスエラー
  if (data?.businessCode) {
    switch (data.businessCode) {
      case "INSUFFICIENT_BALANCE":
        toast.error("Your balance is insufficient for this operation");
        break;
      case "DUPLICATE_EMAIL":
        toast.error("This email is already registered");
        break;
      case "SUBSCRIPTION_EXPIRED":
        toast.error("Your subscription has expired. Please renew.");
        break;
      default:
        toast.error(message);
    }
    return;
  }

  // HTTP ステータスベースのハンドリング
  switch (error.data?.httpStatus) {
    case 401:
      toast.error("Please log in to continue");
      // リダイレクト
      window.location.href = "/login";
      break;
    case 403:
      toast.error("You don't have permission to perform this action");
      break;
    case 429:
      toast.error("Too many requests. Please try again later.");
      break;
    default:
      toast.error(message || "Something went wrong");
  }
}

// React コンポーネントでの使用
function CreatePostForm() {
  const createPost = trpc.post.create.useMutation({
    onSuccess: (data) => {
      toast.success("Post created successfully!");
      router.push(`/posts/${data.id}`);
    },
    onError: handleTRPCError,
  });

  // ...
}
```

---

## 7. パフォーマンス最適化

### 7-1. バッチリクエストの制御

```typescript
// クライアント側: バッチリンクの最適化
import { httpBatchLink } from "@trpc/client";

const trpcClient = trpc.createClient({
  links: [
    httpBatchLink({
      url: "/api/trpc",
      // バッチの最大サイズ（デフォルトは無制限）
      maxURLLength: 2083,  // IE の URL 長制限に合わせる場合

      // カスタムヘッダー
      headers: () => ({
        Authorization: `Bearer ${getToken()}`,
      }),

      // リクエスト前のフック
      fetch: (url, options) => {
        // カスタム fetch 実装（タイムアウト付き）
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30_000);

        return fetch(url, {
          ...options,
          signal: controller.signal,
        }).finally(() => clearTimeout(timeoutId));
      },
    }),
  ],
});
```

```
バッチリクエストの動作:

  通常（httpLink）:
  Client  →  GET /api/trpc/user.list     →  Server
  Client  →  GET /api/trpc/post.list     →  Server
  Client  →  GET /api/trpc/comment.list  →  Server
  3 HTTP リクエスト

  バッチ（httpBatchLink）:
  Client  →  GET /api/trpc/user.list,post.list,comment.list  →  Server
  1 HTTP リクエスト（3つの呼び出しをまとめる）

  レスポンス:
  [
    { result: { data: { users: [...] } } },
    { result: { data: { posts: [...] } } },
    { result: { data: { comments: [...] } } }
  ]
```

### 7-2. データの選択的取得（Output バリデーション）

```typescript
// server/routers/user.ts -- 出力のフィルタリング
export const userRouter = router({
  // 公開プロフィール（機密情報を除外）
  publicProfile: publicProcedure
    .input(z.object({ userId: z.string() }))
    .output(
      z.object({
        id: z.string(),
        name: z.string(),
        avatar: z.string().nullable(),
        bio: z.string().nullable(),
        // email, phone などは含めない
      })
    )
    .query(async ({ ctx, input }) => {
      const user = await ctx.db.user.findUnique({
        where: { id: input.userId },
      });
      if (!user) throw new TRPCError({ code: "NOT_FOUND" });
      return user; // output バリデーションが自動で不要フィールドを除外
    }),

  // 管理者向け（全フィールド）
  adminDetail: adminProcedure
    .input(z.object({ userId: z.string() }))
    .query(async ({ ctx, input }) => {
      return ctx.db.user.findUniqueOrThrow({
        where: { id: input.userId },
        include: {
          profile: true,
          sessions: true,
          auditLogs: { take: 50, orderBy: { createdAt: "desc" } },
        },
      });
    }),
});
```

### 7-3. キャッシュ戦略

```typescript
// components/UserProfile.tsx -- staleTime と cacheTime の設定
export function UserProfile({ userId }: { userId: string }) {
  const { data: user } = trpc.user.byId.useQuery(
    { id: userId },
    {
      // データが「新鮮」とみなされる時間（この間は再取得しない）
      staleTime: 5 * 60 * 1000, // 5分間

      // キャッシュがメモリに保持される時間
      gcTime: 30 * 60 * 1000, // 30分間（旧 cacheTime）

      // ウィンドウフォーカス時の再取得
      refetchOnWindowFocus: false,

      // マウント時の再取得
      refetchOnMount: "always",

      // リトライ設定
      retry: (failureCount, error) => {
        // 404 はリトライしない
        if (error.data?.httpStatus === 404) return false;
        return failureCount < 3;
      },
      retryDelay: (attemptIndex) =>
        Math.min(1000 * 2 ** attemptIndex, 30_000),
    }
  );

  // ...
}
```

```typescript
// グローバルなキャッシュ設定
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60 * 1000,         // デフォルト1分
      gcTime: 10 * 60 * 1000,       // デフォルト10分
      refetchOnWindowFocus: true,
      retry: 1,
      retryDelay: 1000,
    },
    mutations: {
      retry: false,                  // Mutation はデフォルトでリトライなし
    },
  },
});
```

### 7-4. プリフェッチとデータのプリロード

```typescript
// pages/users/index.tsx -- ページ遷移前のプリフェッチ
import { trpc } from "../../utils/trpc";
import Link from "next/link";

export function UserListPage() {
  const { data } = trpc.user.list.useQuery({ page: 1, limit: 20 });
  const utils = trpc.useUtils();

  return (
    <div>
      {data?.users.map((user) => (
        <Link
          key={user.id}
          href={`/users/${user.id}`}
          // ホバー時にプリフェッチ
          onMouseEnter={() => {
            utils.user.byId.prefetch({ id: user.id });
          }}
        >
          {user.name}
        </Link>
      ))}
    </div>
  );
}
```

```typescript
// Next.js SSR でのプリフェッチ
// app/users/page.tsx
import { createServerSideHelpers } from "@trpc/react-query/server";
import { appRouter } from "../../server/routers/_app";
import superjson from "superjson";

export default async function UsersPage() {
  const helpers = createServerSideHelpers({
    router: appRouter,
    ctx: { userId: null, db: prisma },
    transformer: superjson,
  });

  // サーバー側でデータをプリフェッチ
  await helpers.user.list.prefetch({ page: 1, limit: 20 });

  return (
    <HydrateClient state={helpers.dehydrate()}>
      <UserList />
    </HydrateClient>
  );
}
```

---

## 8. 高度なリンクの設定

### 8-1. loggerLink によるデバッグ

```typescript
import { loggerLink, httpBatchLink } from "@trpc/client";

const trpcClient = trpc.createClient({
  links: [
    // ロガーは他のリンクの前に配置
    loggerLink({
      enabled: (opts) =>
        process.env.NODE_ENV === "development" ||
        (opts.direction === "down" && opts.result instanceof Error),

      // カスタムログフォーマット
      colorMode: "ansi", // ターミナル用
    }),

    httpBatchLink({
      url: "/api/trpc",
    }),
  ],
});
```

### 8-2. splitLink による条件分岐

```typescript
import { splitLink, httpBatchLink, httpLink } from "@trpc/client";

const trpcClient = trpc.createClient({
  links: [
    splitLink({
      // 条件: ファイルアップロード系は個別リクエスト
      condition: (op) => op.path.startsWith("upload."),
      true: httpLink({
        url: "/api/trpc",
        // ファイルアップロード用の設定
      }),
      false: httpBatchLink({
        url: "/api/trpc",
      }),
    }),
  ],
});
```

### 8-3. カスタムリンクの作成

```typescript
// utils/retryLink.ts -- リトライリンクのカスタム実装
import { TRPCLink } from "@trpc/client";
import { observable } from "@trpc/server/observable";
import type { AppRouter } from "../server/routers/_app";

/**
 * 一定の条件でリクエストをリトライするカスタムリンク
 */
export function retryLink(opts: {
  maxRetries: number;
  retryableErrors: string[];
}): TRPCLink<AppRouter> {
  return () => {
    return ({ op, next }) => {
      return observable((observer) => {
        let attempts = 0;

        function attempt() {
          attempts++;
          const subscription = next(op).subscribe({
            next: (value) => observer.next(value),
            error: (err) => {
              if (
                attempts < opts.maxRetries &&
                opts.retryableErrors.includes(err.data?.code)
              ) {
                // リトライ可能なエラー -- 指数バックオフで再試行
                const delay = Math.min(1000 * 2 ** (attempts - 1), 10_000);
                console.warn(
                  `[retryLink] Retry attempt ${attempts} for ${op.path} in ${delay}ms`
                );
                setTimeout(attempt, delay);
              } else {
                observer.error(err);
              }
            },
            complete: () => observer.complete(),
          });

          return subscription;
        }

        const sub = attempt();
        return () => sub?.unsubscribe();
      });
    };
  };
}

// 使用例
const trpcClient = trpc.createClient({
  links: [
    retryLink({
      maxRetries: 3,
      retryableErrors: ["INTERNAL_SERVER_ERROR", "TIMEOUT"],
    }),
    httpBatchLink({ url: "/api/trpc" }),
  ],
});
```

---

## 9. ファイルアップロード

### 9-1. マルチパートアップロード

```typescript
// server/routers/upload.ts
import { z } from "zod";
import { router, protectedProcedure } from "../trpc";
import { S3Client, PutObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

const s3 = new S3Client({ region: process.env.AWS_REGION });

export const uploadRouter = router({
  // 署名付き URL を取得（クライアントから直接 S3 にアップロード）
  getPresignedUrl: protectedProcedure
    .input(
      z.object({
        filename: z.string(),
        contentType: z.string(),
        size: z.number().max(10 * 1024 * 1024), // 最大 10MB
      })
    )
    .mutation(async ({ ctx, input }) => {
      const key = `uploads/${ctx.userId}/${Date.now()}-${input.filename}`;

      const command = new PutObjectCommand({
        Bucket: process.env.S3_BUCKET!,
        Key: key,
        ContentType: input.contentType,
        ContentLength: input.size,
      });

      const presignedUrl = await getSignedUrl(s3, command, {
        expiresIn: 300, // 5分間有効
      });

      // DB にメタデータを保存
      const file = await ctx.db.file.create({
        data: {
          key,
          filename: input.filename,
          contentType: input.contentType,
          size: input.size,
          userId: ctx.userId,
          status: "PENDING",
        },
      });

      return { presignedUrl, fileId: file.id, key };
    }),

  // アップロード完了の確認
  confirmUpload: protectedProcedure
    .input(z.object({ fileId: z.string() }))
    .mutation(async ({ ctx, input }) => {
      const file = await ctx.db.file.update({
        where: {
          id: input.fileId,
          userId: ctx.userId,
        },
        data: { status: "COMPLETED" },
      });

      return { url: `${process.env.CDN_URL}/${file.key}` };
    }),
});
```

```typescript
// hooks/useFileUpload.ts -- クライアント側のアップロードフック
import { trpc } from "../utils/trpc";
import { useState } from "react";

export function useFileUpload() {
  const [progress, setProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);

  const getPresignedUrl = trpc.upload.getPresignedUrl.useMutation();
  const confirmUpload = trpc.upload.confirmUpload.useMutation();

  async function upload(file: File): Promise<string> {
    setIsUploading(true);
    setProgress(0);

    try {
      // 1. 署名付き URL を取得
      const { presignedUrl, fileId } = await getPresignedUrl.mutateAsync({
        filename: file.name,
        contentType: file.type,
        size: file.size,
      });

      // 2. S3 に直接アップロード（進捗追跡付き）
      await new Promise<void>((resolve, reject) => {
        const xhr = new XMLHttpRequest();

        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable) {
            setProgress(Math.round((e.loaded / e.total) * 100));
          }
        };

        xhr.onload = () => {
          if (xhr.status === 200) resolve();
          else reject(new Error(`Upload failed: ${xhr.status}`));
        };

        xhr.onerror = () => reject(new Error("Upload failed"));

        xhr.open("PUT", presignedUrl);
        xhr.setRequestHeader("Content-Type", file.type);
        xhr.send(file);
      });

      // 3. アップロード完了を通知
      const { url } = await confirmUpload.mutateAsync({ fileId });
      return url;
    } finally {
      setIsUploading(false);
    }
  }

  return { upload, progress, isUploading };
}
```

---

## 10. tRPC v11 の新機能

### 10-1. Server-Sent Events (SSE) トランスポート

```typescript
// tRPC v11 で追加された SSE リンク
import { unstable_httpBatchStreamLink } from "@trpc/client";

const trpcClient = trpc.createClient({
  links: [
    unstable_httpBatchStreamLink({
      url: "/api/trpc",
      // SSE でストリーミングレスポンス
    }),
  ],
});

// サーバー側: ストリーミングプロシージャ
export const aiRouter = router({
  generateText: protectedProcedure
    .input(z.object({ prompt: z.string() }))
    .query(async function* ({ input }) {
      // AsyncGenerator でストリーミングレスポンス
      const stream = await openai.chat.completions.create({
        model: "gpt-4",
        messages: [{ role: "user", content: input.prompt }],
        stream: true,
      });

      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content;
        if (content) {
          yield content;
        }
      }
    }),
});
```

### 10-2. FormData サポート

```typescript
// tRPC v11 の FormData 対応
import { experimental_formDataLink } from "@trpc/client";

// サーバー側
export const uploadRouter = router({
  uploadAvatar: protectedProcedure
    .input(
      z.object({
        file: z.instanceof(File),
        description: z.string().optional(),
      })
    )
    .mutation(async ({ ctx, input }) => {
      const buffer = Buffer.from(await input.file.arrayBuffer());
      // ファイル処理...
      return { url: "https://cdn.example.com/avatar.jpg" };
    }),
});
```

### 10-3. React Server Components との深い統合

```typescript
// app/users/[id]/page.tsx -- RSC + tRPC v11
import { createTRPCProxyClient, httpLink } from "@trpc/client";
import type { AppRouter } from "@/server/routers/_app";

// サーバーコンポーネント用クライアント
const serverClient = createTRPCProxyClient<AppRouter>({
  links: [
    httpLink({
      url: `${process.env.NEXT_PUBLIC_APP_URL}/api/trpc`,
      headers: () => {
        // サーバーコンポーネントではクッキーを直接アクセス
        const cookieStore = cookies();
        return {
          cookie: cookieStore.toString(),
        };
      },
    }),
  ],
});

// RSC で直接 tRPC を呼び出す
export default async function UserPage({
  params,
}: {
  params: { id: string };
}) {
  // サーバーサイドで直接データ取得（HTTP を介すが型安全）
  const user = await serverClient.user.byId.query({ id: params.id });

  return (
    <div>
      <h1>{user.name}</h1>
      <p>{user.email}</p>
      {/* クライアントコンポーネントに引き渡し */}
      <UserActions userId={user.id} />
    </div>
  );
}
```

---

## 11. 本番デプロイと運用

### 11-1. Express アダプター

```typescript
// server/index.ts -- Express サーバー
import express from "express";
import cors from "cors";
import { createExpressMiddleware } from "@trpc/server/adapters/express";
import { appRouter } from "./routers/_app";
import { createContext } from "./context";

const app = express();

app.use(cors());
app.use(express.json({ limit: "10mb" }));

// ヘルスチェック
app.get("/health", (_, res) => res.json({ status: "ok" }));

// tRPC ミドルウェア
app.use(
  "/api/trpc",
  createExpressMiddleware({
    router: appRouter,
    createContext,
    onError({ error, path }) {
      console.error(`[tRPC Error] ${path}:`, error);
      // エラーモニタリングサービスに送信
      Sentry.captureException(error);
    },
  })
);

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

### 11-2. Fastify アダプター

```typescript
// server/index.ts -- Fastify サーバー（高パフォーマンス）
import Fastify from "fastify";
import cors from "@fastify/cors";
import {
  fastifyTRPCPlugin,
  FastifyTRPCPluginOptions,
} from "@trpc/server/adapters/fastify";
import { appRouter, AppRouter } from "./routers/_app";
import { createContext } from "./context";

const server = Fastify({
  maxParamLength: 5000,
  logger: true,
});

await server.register(cors);

server.register(fastifyTRPCPlugin, {
  prefix: "/api/trpc",
  trpcOptions: {
    router: appRouter,
    createContext,
    onError({ path, error }) {
      server.log.error(`[tRPC] ${path}: ${error.message}`);
    },
  } satisfies FastifyTRPCPluginOptions<AppRouter>["trpcOptions"],
});

const start = async () => {
  try {
    await server.listen({ port: 3000, host: "0.0.0.0" });
  } catch (err) {
    server.log.error(err);
    process.exit(1);
  }
};

start();
```

### 11-3. Edge Runtime デプロイ（Cloudflare Workers）

```typescript
// worker/index.ts -- Cloudflare Workers での tRPC
import { fetchRequestHandler } from "@trpc/server/adapters/fetch";
import { appRouter } from "./routers/_app";

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    // CORS プリフライト
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": "*",
          "Access-Control-Allow-Methods": "GET, POST",
          "Access-Control-Allow-Headers": "Content-Type, Authorization",
        },
      });
    }

    return fetchRequestHandler({
      endpoint: "/api/trpc",
      req: request,
      router: appRouter,
      createContext: async () => ({
        userId: await getUserFromRequest(request),
        db: createD1Client(env.DB),
      }),
      responseMeta() {
        return {
          headers: {
            "Access-Control-Allow-Origin": "*",
          },
        };
      },
    });
  },
};
```

### 11-4. モニタリングとオブザーバビリティ

```typescript
// server/middleware/monitoring.ts
import { middleware } from "../trpc";
import { trace, SpanStatusCode } from "@opentelemetry/api";

const tracer = trace.getTracer("trpc");

/**
 * OpenTelemetry 対応のトレーシングミドルウェア
 */
export const tracingMiddleware = middleware(async ({ path, type, next }) => {
  return tracer.startActiveSpan(`trpc.${type}.${path}`, async (span) => {
    span.setAttribute("rpc.system", "trpc");
    span.setAttribute("rpc.method", path);
    span.setAttribute("rpc.type", type);

    try {
      const result = await next();

      if (!result.ok) {
        span.setStatus({
          code: SpanStatusCode.ERROR,
          message: "tRPC procedure failed",
        });
      }

      return result;
    } catch (error) {
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: (error as Error).message,
      });
      span.recordException(error as Error);
      throw error;
    } finally {
      span.end();
    }
  });
});

/**
 * メトリクス収集ミドルウェア（Prometheus 用）
 */
import { Counter, Histogram } from "prom-client";

const requestCounter = new Counter({
  name: "trpc_requests_total",
  help: "Total number of tRPC requests",
  labelNames: ["path", "type", "status"],
});

const requestDuration = new Histogram({
  name: "trpc_request_duration_seconds",
  help: "Duration of tRPC requests in seconds",
  labelNames: ["path", "type"],
  buckets: [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
});

export const metricsMiddleware = middleware(async ({ path, type, next }) => {
  const timer = requestDuration.startTimer({ path, type });

  try {
    const result = await next();
    requestCounter.inc({ path, type, status: result.ok ? "ok" : "error" });
    return result;
  } catch (error) {
    requestCounter.inc({ path, type, status: "exception" });
    throw error;
  } finally {
    timer();
  }
});
```

---

## 12. セキュリティベストプラクティス

### 12-1. 入力サニタイゼーション

```typescript
// server/schemas/sanitize.ts
import { z } from "zod";
import DOMPurify from "isomorphic-dompurify";

/**
 * HTML サニタイゼーション付きの文字列スキーマ
 */
export const sanitizedString = z.string().transform((val) => {
  return DOMPurify.sanitize(val, {
    ALLOWED_TAGS: [], // テキストのみ許可
    ALLOWED_ATTR: [],
  });
});

/**
 * リッチテキスト用（一部の HTML タグを許可）
 */
export const richText = z.string().transform((val) => {
  return DOMPurify.sanitize(val, {
    ALLOWED_TAGS: ["b", "i", "em", "strong", "a", "p", "br", "ul", "ol", "li"],
    ALLOWED_ATTR: ["href", "target", "rel"],
  });
});

// 使用例
export const postRouter = router({
  create: protectedProcedure
    .input(
      z.object({
        title: sanitizedString.pipe(z.string().min(1).max(200)),
        content: richText.pipe(z.string().min(1).max(50000)),
        tags: z.array(sanitizedString).max(10),
      })
    )
    .mutation(async ({ ctx, input }) => {
      return ctx.db.post.create({
        data: {
          ...input,
          authorId: ctx.userId,
        },
      });
    }),
});
```

### 12-2. CSRF 保護

```typescript
// server/middleware/csrf.ts
import { middleware } from "../trpc";
import { TRPCError } from "@trpc/server";

/**
 * CSRF 保護ミドルウェア
 * カスタムヘッダーの存在を検証する
 */
export const csrfProtection = middleware(async ({ ctx, next, type }) => {
  // Query（GET）は CSRF のリスクが低いためスキップ
  if (type === "query") {
    return next();
  }

  // Mutation の場合、カスタムヘッダーを要求
  const csrfToken = ctx.req?.headers?.["x-csrf-token"];
  if (!csrfToken) {
    throw new TRPCError({
      code: "FORBIDDEN",
      message: "CSRF token missing",
    });
  }

  // トークンの検証
  const expectedToken = await ctx.db.csrfToken.findUnique({
    where: { token: csrfToken as string },
  });

  if (!expectedToken || expectedToken.expiresAt < new Date()) {
    throw new TRPCError({
      code: "FORBIDDEN",
      message: "Invalid or expired CSRF token",
    });
  }

  return next();
});
```

### 12-3. データアクセスの制御

```typescript
// server/middleware/dataAccess.ts
import { middleware } from "../trpc";
import { TRPCError } from "@trpc/server";
import { z } from "zod";

/**
 * リソースの所有者チェックミドルウェア
 */
export function ownershipCheck<T extends z.ZodType>(
  resourceSchema: T,
  getResource: (id: string, db: PrismaClient) => Promise<any>,
  ownerField: string = "userId"
) {
  return middleware(async ({ ctx, rawInput, next }) => {
    const parsed = z.object({ id: z.string() }).safeParse(rawInput);
    if (!parsed.success) {
      throw new TRPCError({ code: "BAD_REQUEST" });
    }

    const resource = await getResource(parsed.data.id, ctx.db);
    if (!resource) {
      throw new TRPCError({ code: "NOT_FOUND" });
    }

    if (resource[ownerField] !== ctx.userId) {
      throw new TRPCError({
        code: "FORBIDDEN",
        message: "You can only modify your own resources",
      });
    }

    return next({
      ctx: { ...ctx, resource },
    });
  });
}

// 使用例
const ownPostProcedure = protectedProcedure.use(
  ownershipCheck(
    z.any(),
    (id, db) => db.post.findUnique({ where: { id } }),
    "authorId"
  )
);

export const postRouter = router({
  updateOwn: ownPostProcedure
    .input(z.object({
      id: z.string(),
      title: z.string().optional(),
      content: z.string().optional(),
    }))
    .mutation(async ({ ctx, input }) => {
      // ctx.resource に既にフェッチ済みの post が入っている
      return ctx.db.post.update({
        where: { id: input.id },
        data: {
          ...(input.title && { title: input.title }),
          ...(input.content && { content: input.content }),
        },
      });
    }),
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

### ミドルウェアパターン比較

| パターン | 用途 | 複雑度 | 再利用性 |
|----------|------|--------|----------|
| 認証チェック | ログインユーザー限定 | 低 | 高 |
| ロールベース RBAC | 権限レベル制御 | 中 | 高 |
| 組織ベースアクセス | マルチテナント | 高 | 中 |
| レート制限 | API 保護 | 中 | 高 |
| ロギング | デバッグ・監査 | 低 | 最高 |
| トレーシング | パフォーマンス分析 | 中 | 高 |
| CSRF 保護 | セキュリティ | 中 | 高 |
| 所有者チェック | リソース保護 | 中 | 中 |

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

### AP-3: コンテキストに不要なデータを詰め込む

```typescript
// NG: コンテキストが肥大化（全リクエストで全データを取得）
export async function createContext(opts: CreateNextContextOptions) {
  const session = await getServerSession(opts.req, opts.res, authOptions);
  const user = session ? await prisma.user.findUnique({
    where: { id: session.user.id },
    include: {
      profile: true,
      settings: true,
      organizations: { include: { organization: true } },
      notifications: { where: { read: false } },
    },
  }) : null;

  return { session, user, db: prisma };
}

// OK: コンテキストは最小限に。追加データはミドルウェアで取得
export async function createContext(opts: CreateNextContextOptions) {
  const session = await getServerSession(opts.req, opts.res, authOptions);
  return {
    userId: session?.user?.id ?? null,
    db: prisma,
  };
}
// 追加データが必要なルートだけミドルウェアで取得
```

### AP-4: エラーハンドリングを省略する

```typescript
// NG: エラーを握りつぶす
const createUser = trpc.user.create.useMutation();

// NG: catch なしの async 呼び出し
const handleCreate = async () => {
  await createUser.mutateAsync({ name: "", email: "invalid" });
  // エラー時にクラッシュする
};

// OK: 適切なエラーハンドリング
const createUser = trpc.user.create.useMutation({
  onSuccess: (data) => {
    toast.success("User created!");
    router.push(`/users/${data.id}`);
  },
  onError: (error) => {
    if (error.data?.zodError) {
      // バリデーションエラーをフォームに表示
      setFormErrors(error.data.zodError.fieldErrors);
    } else {
      toast.error(error.message);
    }
  },
});
```

### AP-5: 楽観的更新でロールバックを忘れる

```typescript
// NG: ロールバック処理がない
const toggleTodo = trpc.todo.toggle.useMutation({
  onMutate: async (input) => {
    utils.todo.list.setData(undefined, (old) =>
      old?.map((t) => (t.id === input.id ? { ...t, done: !t.done } : t))
    );
    // エラー時にデータが不整合になる!
  },
});

// OK: スナップショット + ロールバック + 再同期
const toggleTodo = trpc.todo.toggle.useMutation({
  onMutate: async (input) => {
    await utils.todo.list.cancel();
    const prev = utils.todo.list.getData();
    utils.todo.list.setData(undefined, (old) =>
      old?.map((t) => (t.id === input.id ? { ...t, done: !t.done } : t))
    );
    return { prev };
  },
  onError: (_, __, ctx) => {
    if (ctx?.prev) utils.todo.list.setData(undefined, ctx.prev);
  },
  onSettled: () => utils.todo.list.invalidate(),
});
```

---

## FAQ

### Q1: tRPC は REST API の代替になりますか？

TypeScript のモノレポ（フロントエンド + バックエンド）では完全に REST の代替になります。ただし、モバイルアプリ（Swift/Kotlin）や他言語のクライアントがある場合は REST/GraphQL の方が適しています。tRPC は「TypeScript エコシステム内」で最大の威力を発揮します。

### Q2: tRPC v10 と v11 の違いは何ですか？

v11 では React Server Components との統合強化、新しいリンク API、パフォーマンス改善が含まれます。v10 からの移行は比較的容易で、破壊的変更は少ないです。

### Q3: tRPC と GraphQL は併用できますか？

技術的には可能ですが、通常は片方を選択します。社内ツールやフルスタック TypeScript プロジェクトでは tRPC、公開 API やマルチプラットフォームでは GraphQL が適しています。

### Q4: tRPC はマイクロサービスで使えますか？

使えますが、注意が必要です。tRPC はモノレポ内での型共有を前提としているため、サービス間通信では型定義を共有パッケージとして公開する必要があります。サービス間のインターフェースが安定している場合は gRPC の方が適している場合もあります。

```typescript
// packages/shared-types/src/index.ts -- 共有型パッケージ
export type { AppRouter as ServiceARouter } from "@myapp/service-a/src/routers/_app";
export type { AppRouter as ServiceBRouter } from "@myapp/service-b/src/routers/_app";

// service-a/src/clients/serviceB.ts
import { createTRPCClient, httpBatchLink } from "@trpc/client";
import type { ServiceBRouter } from "@myapp/shared-types";

export const serviceBClient = createTRPCClient<ServiceBRouter>({
  links: [
    httpBatchLink({
      url: process.env.SERVICE_B_URL + "/trpc",
      headers: () => ({
        "x-service-auth": process.env.SERVICE_SECRET,
      }),
    }),
  ],
});
```

### Q5: tRPC のパフォーマンスはどうですか？

tRPC 自体のオーバーヘッドは最小限です。HTTP リクエスト/レスポンスのシリアライゼーションと zod バリデーションがコストの大部分を占めます。バッチリンクを使えばリクエスト数を大幅に削減できます。大規模なアプリケーションでも、ボトルネックは通常 DB クエリやビジネスロジックであり、tRPC 自体ではありません。

### Q6: tRPC で OpenAPI ドキュメントを生成できますか？

`trpc-openapi` パッケージを使うことで、tRPC ルーターから OpenAPI 仕様を自動生成できます。これにより、TypeScript 以外のクライアントにも REST API として公開できます。

```typescript
import { generateOpenApiDocument } from "trpc-openapi";

const openApiDoc = generateOpenApiDocument(appRouter, {
  title: "My API",
  version: "1.0.0",
  baseUrl: "https://api.example.com",
});
```

### Q7: テスト時にモックはどう作成しますか？

`createCaller` を使えばサーバーサイドのテストが容易です。クライアント側のテストでは、MSW（Mock Service Worker）を使って HTTP レベルでモックするか、tRPC フックを直接モックする方法があります。

```typescript
// MSW を使ったクライアントテスト
import { setupServer } from "msw/node";
import { http, HttpResponse } from "msw";

const server = setupServer(
  http.get("/api/trpc/user.list", () => {
    return HttpResponse.json({
      result: {
        data: {
          users: [{ id: "1", name: "Test User" }],
          total: 1,
          page: 1,
        },
      },
    });
  })
);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());
```

---

## まとめ表

| 概念 | 要点 |
|------|------|
| tRPC | TypeScript 型推論で E2E 型安全な API |
| Router | プロシージャをネストして構成 |
| Query / Mutation | 読み取り / 書き込み操作の区分 |
| Subscription | WebSocket によるリアルタイム通信 |
| Middleware | 認証、ログ、エラー処理のチェーン |
| createCaller | サーバーサイドでの直接呼び出し |
| zod 統合 | 入力バリデーションとスキーマ定義 |
| Links | リクエストパイプラインのカスタマイズ |
| Optimistic Updates | 楽観的更新でレスポンシブな UX |
| Infinite Queries | カーソルベースの無限スクロール |
| Output Validation | 出力のフィルタリングとセキュリティ |
| SSE / Streaming | v11 のストリーミングレスポンス |

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

4. **TanStack Query Documentation** -- tRPC React が内部で利用するデータフェッチライブラリ
   https://tanstack.com/query

5. **Zod Documentation** -- tRPC の入力バリデーションに使用するスキーマライブラリ
   https://zod.dev

6. **trpc-openapi** -- tRPC から OpenAPI ドキュメントを生成
   https://github.com/jlalmes/trpc-openapi
