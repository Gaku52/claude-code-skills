# Prisma + TypeScript 完全ガイド

> Prisma ORM で TypeScript の型安全性を最大限に活かし、データベース操作を堅牢に行う

## この章で学ぶこと

1. **Prisma の基本** -- スキーマ定義、マイグレーション、CRUD 操作の型安全な記述
2. **高度なクエリパターン** -- リレーション、トランザクション、生SQLの型安全な扱い
3. **実践的な設計** -- リポジトリパターン、テスト戦略、パフォーマンスチューニング
4. **運用とスケーリング** -- コネクションプーリング、Edge Runtime 対応、監視・ログ

---

## 1. Prisma の基本

### 1-1. セットアップ

```bash
# インストール
npm install prisma --save-dev
npm install @prisma/client

# 初期化
npx prisma init --datasource-provider postgresql
```

```
Prisma のアーキテクチャ:

  schema.prisma          npx prisma generate
  (スキーマ定義)  ─────────────────────────>  @prisma/client
       |                                       (型付きクライアント)
       |                                            |
  npx prisma migrate                                |
       |                                            |
       v                                            v
  データベース  <──── SQL クエリ ──── PrismaClient
  (PostgreSQL等)                     (ランタイム)
```

Prisma は 3 つの主要コンポーネントで構成される。

```
┌──────────────────────────────────────────────────┐
│                  Prisma Ecosystem                 │
├──────────────────────────────────────────────────┤
│                                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────┐│
│  │   Prisma     │  │   Prisma     │  │  Prisma  ││
│  │   Schema     │  │   Client     │  │  Studio  ││
│  │              │  │              │  │          ││
│  │ .prisma      │  │ @prisma/     │  │ GUI ツール││
│  │ ファイルで   │  │ client で    │  │ データ   ││
│  │ モデル定義   │  │ 型安全な     │  │ 閲覧/編集││
│  │              │  │ DB アクセス  │  │          ││
│  └──────┬───────┘  └──────┬───────┘  └──────────┘│
│         │                 │                       │
│         │  npx prisma     │                       │
│         │  generate       │                       │
│         └────────────────>┘                       │
│                                                   │
│  ┌─────────────┐  ┌──────────────┐               │
│  │   Prisma     │  │   Prisma     │               │
│  │   Migrate    │  │   Accelerate │               │
│  │              │  │              │               │
│  │ スキーマ     │  │ コネクション │               │
│  │ 差分から     │  │ プーリング + │               │
│  │ SQL 自動生成 │  │ グローバル   │               │
│  │              │  │ キャッシュ   │               │
│  └──────────────┘  └──────────────┘               │
└──────────────────────────────────────────────────┘
```

### 1-2. スキーマ定義

```prisma
// prisma/schema.prisma
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model User {
  id        String   @id @default(uuid())
  email     String   @unique
  name      String
  role      Role     @default(USER)
  posts     Post[]
  profile   Profile?
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt

  @@index([email])
  @@map("users")
}

model Post {
  id          String     @id @default(uuid())
  title       String
  content     String?
  published   Boolean    @default(false)
  author      User       @relation(fields: [authorId], references: [id])
  authorId    String
  categories  Category[]
  tags        Tag[]
  comments    Comment[]
  viewCount   Int        @default(0)
  createdAt   DateTime   @default(now())
  updatedAt   DateTime   @updatedAt

  @@index([authorId])
  @@index([published, createdAt])
  @@map("posts")
}

model Comment {
  id        String   @id @default(uuid())
  content   String
  post      Post     @relation(fields: [postId], references: [id], onDelete: Cascade)
  postId    String
  author    User     @relation(fields: [authorId], references: [id])
  authorId  String
  parent    Comment? @relation("CommentReplies", fields: [parentId], references: [id])
  parentId  String?
  replies   Comment[] @relation("CommentReplies")
  createdAt DateTime @default(now())

  @@index([postId])
  @@index([authorId])
  @@map("comments")
}

model Profile {
  id     String  @id @default(uuid())
  bio    String?
  avatar String?
  user   User    @relation(fields: [userId], references: [id])
  userId String  @unique

  @@map("profiles")
}

model Category {
  id    String @id @default(uuid())
  name  String @unique
  slug  String @unique
  posts Post[]

  @@map("categories")
}

model Tag {
  id    String @id @default(uuid())
  name  String @unique
  posts Post[]

  @@map("tags")
}

enum Role {
  USER
  ADMIN
  MODERATOR
}
```

#### スキーマ設計のベストプラクティス

| 項目 | 推奨 | 理由 |
|------|------|------|
| ID 型 | `uuid()` or `cuid()` | 連番は推測可能、分散システムで衝突リスク |
| タイムスタンプ | `createdAt` + `updatedAt` | 監査証跡、デバッグに必須 |
| テーブル名 | `@@map("snake_case")` | DB 慣習に合わせつつモデル名は PascalCase |
| インデックス | 検索条件に `@@index` | クエリパフォーマンスに直結 |
| 複合ユニーク | `@@unique([fieldA, fieldB])` | ビジネスルールの制約を DB レベルで保証 |
| onDelete | 明示指定 | `Cascade` / `SetNull` / `Restrict` を意図的に選択 |
| enum | Prisma enum | DB の enum と対応、型安全性を確保 |

#### リレーション設計パターン

```
リレーションの種類:

  1対1 (One-to-One)
  ┌──────┐     ┌─────────┐
  │ User │────>│ Profile  │
  └──────┘     └─────────┘
  userId: @unique で保証

  1対多 (One-to-Many)
  ┌──────┐     ┌──────┐
  │ User │────>│ Post │
  │      │────>│ Post │
  │      │────>│ Post │
  └──────┘     └──────┘
  posts Post[]  ←→  author User

  多対多 (Many-to-Many)
  ┌──────┐     ┌──────────┐
  │ Post │<──>>│ Category │
  │      │<──>>│ Category │
  └──────┘     └──────────┘
  暗黙的な中間テーブル _CategoryToPost が自動生成

  自己参照 (Self-Relation)
  ┌──────────┐
  │ Comment  │
  │  parent ─┼──┐
  │  replies ─┼──┘
  └──────────┘
  parentId で親コメントを参照
```

### 1-3. マイグレーション

```bash
# マイグレーション作成 + 適用
npx prisma migrate dev --name init

# 本番環境への適用
npx prisma migrate deploy

# スキーマの同期（開発用、マイグレーションなし）
npx prisma db push

# クライアント再生成
npx prisma generate

# GUI でデータ確認
npx prisma studio

# マイグレーションのリセット（開発環境のみ）
npx prisma migrate reset

# マイグレーションの差分を確認（適用せず SQL だけ表示）
npx prisma migrate diff --from-schema-datamodel prisma/schema.prisma --to-schema-datasource prisma/schema.prisma
```

#### マイグレーション戦略の比較

| 状況 | コマンド | 用途 |
|------|---------|------|
| 開発中の変更 | `prisma migrate dev` | マイグレーションファイルを生成し即適用 |
| プロトタイピング | `prisma db push` | マイグレーション不要で即反映 |
| CI/CD | `prisma migrate deploy` | 既存マイグレーションを順序通り適用 |
| 型だけ再生成 | `prisma generate` | DB に触れずクライアントコード再生成 |
| 全リセット | `prisma migrate reset` | 全マイグレーション再適用（データ削除） |

#### マイグレーションのよくある落とし穴

```typescript
// 問題: 本番 DB にカラム追加でデフォルト値がない
// → NOT NULL 制約違反で既存レコードが更新できない

// 解決: 段階的マイグレーション
// Step 1: NULL 許可でカラム追加
model User {
  displayName String? // まず nullable で追加
}

// Step 2: 既存データをバックフィル
// prisma/migrations/xxx_backfill_display_name.ts
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

async function backfill() {
  const users = await prisma.user.findMany({
    where: { displayName: null },
  });

  for (const user of users) {
    await prisma.user.update({
      where: { id: user.id },
      data: { displayName: user.name },
    });
  }
}

backfill();

// Step 3: NOT NULL 制約を追加
model User {
  displayName String @default("") // NOT NULL に変更
}
```

### 1-4. PrismaClient の初期化パターン

```typescript
// lib/prisma.ts -- シングルトンパターン（推奨）
import { PrismaClient } from "@prisma/client";

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined;
};

export const prisma =
  globalForPrisma.prisma ??
  new PrismaClient({
    log:
      process.env.NODE_ENV === "development"
        ? ["query", "info", "warn", "error"]
        : ["error"],
  });

if (process.env.NODE_ENV !== "production") {
  globalForPrisma.prisma = prisma;
}
```

```typescript
// なぜシングルトンが必要か？
// ─────────────────────────────────────────────
// Next.js の開発モードではホットリロードのたびにモジュールが
// 再評価される。new PrismaClient() が毎回実行されると、
// コネクションプールが増え続け、最終的に DB の接続上限に達する。
//
// globalThis にキャッシュすることで、ホットリロードをまたいで
// 同一インスタンスを再利用できる。
```

#### ログ設定の詳細

```typescript
const prisma = new PrismaClient({
  log: [
    { level: "query", emit: "event" },
    { level: "error", emit: "stdout" },
    { level: "info", emit: "stdout" },
    { level: "warn", emit: "stdout" },
  ],
});

// クエリログをカスタムロガーに送る
prisma.$on("query", (e) => {
  console.log(`[Prisma Query] ${e.query}`);
  console.log(`  Params: ${e.params}`);
  console.log(`  Duration: ${e.duration}ms`);

  // スロークエリ検出
  if (e.duration > 1000) {
    console.warn(`[SLOW QUERY] ${e.duration}ms: ${e.query}`);
  }
});
```

---

## 2. 型安全な CRUD 操作

### 2-1. 基本的な CRUD

```typescript
import { PrismaClient, Prisma } from "@prisma/client";

const prisma = new PrismaClient();

// ────────── Create ──────────
const user = await prisma.user.create({
  data: {
    email: "alice@example.com",
    name: "Alice",
    role: "ADMIN", // enum のリテラル型で補完が効く
  },
});
// 型: User

// createMany で一括作成
const result = await prisma.user.createMany({
  data: [
    { email: "bob@example.com", name: "Bob" },
    { email: "carol@example.com", name: "Carol" },
    { email: "dave@example.com", name: "Dave" },
  ],
  skipDuplicates: true, // 重複をスキップ
});
// 型: Prisma.BatchPayload { count: number }

// ────────── Read ──────────
const found = await prisma.user.findUnique({
  where: { email: "alice@example.com" },
});
// 型: User | null

// findUniqueOrThrow: 見つからない場合は例外
const mustExist = await prisma.user.findUniqueOrThrow({
  where: { email: "alice@example.com" },
});
// 型: User（null なし）

const users = await prisma.user.findMany({
  where: {
    role: "USER",
    createdAt: { gte: new Date("2024-01-01") },
  },
  orderBy: { createdAt: "desc" },
  take: 10,
  skip: 0,
});
// 型: User[]

// findFirst: 最初の1件を取得
const firstAdmin = await prisma.user.findFirst({
  where: { role: "ADMIN" },
  orderBy: { createdAt: "asc" },
});
// 型: User | null

// ────────── Update ──────────
const updated = await prisma.user.update({
  where: { id: user.id },
  data: { name: "Alice Smith" },
});
// 型: User

// upsert: 存在すれば更新、なければ作成
const upserted = await prisma.user.upsert({
  where: { email: "alice@example.com" },
  update: { name: "Alice Updated" },
  create: {
    email: "alice@example.com",
    name: "Alice",
  },
});

// updateMany: 条件に一致する全レコードを更新
const bulkUpdate = await prisma.user.updateMany({
  where: { role: "USER" },
  data: { role: "MODERATOR" },
});
// 型: Prisma.BatchPayload

// ────────── Delete ──────────
const deleted = await prisma.user.delete({
  where: { id: user.id },
});

// deleteMany: 条件に一致する全レコードを削除
const bulkDelete = await prisma.post.deleteMany({
  where: {
    published: false,
    createdAt: { lt: new Date("2023-01-01") },
  },
});

// ────────── Aggregate ──────────
const stats = await prisma.post.aggregate({
  _count: { _all: true },
  _avg: { viewCount: true },
  _max: { viewCount: true },
  _min: { viewCount: true },
  where: { published: true },
});
// 型: {
//   _count: { _all: number };
//   _avg: { viewCount: number | null };
//   _max: { viewCount: number | null };
//   _min: { viewCount: number | null };
// }

// groupBy: グループ集計
const postsByRole = await prisma.user.groupBy({
  by: ["role"],
  _count: { _all: true },
  having: {
    role: {
      _count: { gt: 5 },
    },
  },
  orderBy: {
    _count: { role: "desc" },
  },
});
```

### 2-2. フィルタリング演算子

```typescript
// Prisma の Where 条件は非常に表現力が高い

// 文字列フィルタ
const search = await prisma.user.findMany({
  where: {
    name: {
      contains: "ali",   // LIKE '%ali%'
      startsWith: "A",   // LIKE 'A%'
      endsWith: "ce",    // LIKE '%ce'
      mode: "insensitive", // 大文字小文字を無視（PostgreSQL）
    },
  },
});

// 数値フィルタ
const popular = await prisma.post.findMany({
  where: {
    viewCount: {
      gt: 100,    // > 100
      gte: 50,    // >= 50
      lt: 10000,  // < 10000
      lte: 500,   // <= 500
      not: 0,     // != 0
    },
  },
});

// リストフィルタ
const adminsOrMods = await prisma.user.findMany({
  where: {
    role: { in: ["ADMIN", "MODERATOR"] },
    // role: { notIn: ["USER"] }, // 逆条件
  },
});

// 論理演算子
const complex = await prisma.user.findMany({
  where: {
    OR: [
      { role: "ADMIN" },
      {
        AND: [
          { role: "MODERATOR" },
          { createdAt: { gte: new Date("2024-01-01") } },
        ],
      },
    ],
    NOT: {
      email: { contains: "test" },
    },
  },
});

// リレーション条件（some / every / none）
const usersWithPublishedPosts = await prisma.user.findMany({
  where: {
    posts: {
      some: { published: true },  // 少なくとも1件の公開記事がある
    },
  },
});

const usersWithAllPublished = await prisma.user.findMany({
  where: {
    posts: {
      every: { published: true }, // 全記事が公開済み
    },
  },
});

const usersWithNoPosts = await prisma.user.findMany({
  where: {
    posts: {
      none: {}, // 記事が1件もない
    },
  },
});
```

### 2-3. リレーションの取得

```
select と include の違い:

  include: 既存フィールド + リレーション
  +------------------+
  | id               |
  | email            |
  | name             |  ← 全フィールド保持
  | role             |
  | posts: Post[]    |  ← リレーション追加
  +------------------+

  select: 指定フィールドのみ
  +------------------+
  | name             |  ← 指定フィールドのみ
  | email            |
  | posts: { title } |  ← リレーションも選択可
  +------------------+

  注意: select と include は同時に使えない（トップレベル）
```

```typescript
// include でリレーション取得
const userWithPosts = await prisma.user.findUnique({
  where: { id: "user-1" },
  include: {
    posts: {
      where: { published: true },
      orderBy: { createdAt: "desc" },
      take: 5,
      include: {
        categories: true,   // ネストしたリレーション
        _count: {
          select: { comments: true },
        },
      },
    },
    profile: true,
  },
});
// 型: (User & {
//   posts: (Post & {
//     categories: Category[];
//     _count: { comments: number };
//   })[];
//   profile: Profile | null;
// }) | null

// select で必要なフィールドのみ
const userSummary = await prisma.user.findUnique({
  where: { id: "user-1" },
  select: {
    name: true,
    email: true,
    posts: {
      select: {
        title: true,
        createdAt: true,
      },
      where: { published: true },
    },
    _count: {
      select: { posts: true },
    },
  },
});
// 型: {
//   name: string;
//   email: string;
//   posts: { title: string; createdAt: Date }[];
//   _count: { posts: number };
// } | null
```

### 2-4. ネストした作成・更新

```typescript
// ユーザーと関連データを一括作成
const newUser = await prisma.user.create({
  data: {
    email: "bob@example.com",
    name: "Bob",
    profile: {
      create: {
        bio: "Hello, I'm Bob",
        avatar: "https://example.com/avatar.png",
      },
    },
    posts: {
      create: [
        {
          title: "First Post",
          content: "Hello World",
          published: true,
          categories: {
            connectOrCreate: [
              {
                where: { name: "General" },
                create: { name: "General", slug: "general" },
              },
            ],
          },
        },
        { title: "Draft", content: "WIP" },
      ],
    },
  },
  include: {
    profile: true,
    posts: { include: { categories: true } },
  },
});

// connectOrCreate: 既存があれば接続、なければ作成
const post = await prisma.post.create({
  data: {
    title: "TypeScript Tips",
    author: { connect: { email: "alice@example.com" } },
    categories: {
      connectOrCreate: [
        {
          where: { name: "TypeScript" },
          create: { name: "TypeScript", slug: "typescript" },
        },
        {
          where: { name: "Programming" },
          create: { name: "Programming", slug: "programming" },
        },
      ],
    },
  },
});

// ネストした更新
const updatedUser = await prisma.user.update({
  where: { id: "user-1" },
  data: {
    name: "Bob Updated",
    profile: {
      upsert: {
        create: { bio: "New bio" },
        update: { bio: "Updated bio" },
      },
    },
    posts: {
      updateMany: {
        where: { published: false },
        data: { published: true },
      },
      deleteMany: {
        createdAt: { lt: new Date("2023-01-01") },
      },
    },
  },
});
```

### 2-5. Prisma の型ユーティリティ

```typescript
import { Prisma } from "@prisma/client";

// ───── 生成された型を活用 ─────

// モデルの入力型
type UserCreateInput = Prisma.UserCreateInput;
// { email: string; name: string; role?: Role; ... }

type UserWhereInput = Prisma.UserWhereInput;
// { id?: StringFilter; email?: StringFilter; ... }

type UserOrderByInput = Prisma.UserOrderByWithRelationInput;
// { id?: SortOrder; email?: SortOrder; ... }

// select / include に基づく戻り値型の推論
type UserWithPosts = Prisma.UserGetPayload<{
  include: { posts: true; profile: true };
}>;
// User & { posts: Post[]; profile: Profile | null }

type UserSummary = Prisma.UserGetPayload<{
  select: { id: true; name: true; email: true };
}>;
// { id: string; name: string; email: string }

// ───── バリデーターとの統合 ─────

// Prisma の型から zod スキーマを生成するユーティリティ
import { z } from "zod";

// Prisma の UserCreateInput を参考に zod スキーマを定義
const userCreateSchema = z.object({
  email: z.string().email(),
  name: z.string().min(1).max(100),
  role: z.enum(["USER", "ADMIN", "MODERATOR"]).optional(),
}) satisfies z.ZodType<Omit<Prisma.UserCreateInput, "posts" | "profile">>;

// Prisma.validator を使った型安全なクエリオブジェクト
const userWithPostsQuery = Prisma.validator<Prisma.UserFindManyArgs>()({
  where: { role: "ADMIN" },
  include: {
    posts: {
      where: { published: true },
      select: { id: true, title: true },
    },
  },
});

// この定義を再利用可能にする
async function getAdminUsers() {
  return prisma.user.findMany(userWithPostsQuery);
}

// 戻り値型も自動推論される
type AdminUsersResult = Prisma.PromiseReturnType<typeof getAdminUsers>;
```

---

## 3. 高度なクエリパターン

### 3-1. トランザクション

```typescript
// ───── 方法1: インタラクティブトランザクション（推奨） ─────
const transfer = await prisma.$transaction(async (tx) => {
  // tx は PrismaClient と同じ API だがトランザクション内
  const sender = await tx.user.update({
    where: { id: senderId },
    data: { balance: { decrement: amount } },
  });

  if (sender.balance < 0) {
    throw new Error("Insufficient balance");
    // → トランザクション全体がロールバック
  }

  const receiver = await tx.user.update({
    where: { id: receiverId },
    data: { balance: { increment: amount } },
  });

  // 転送ログを記録
  await tx.transferLog.create({
    data: {
      senderId,
      receiverId,
      amount,
      timestamp: new Date(),
    },
  });

  return { sender, receiver };
}, {
  timeout: 5000,                    // タイムアウト
  maxWait: 2000,                    // トランザクション取得の最大待機時間
  isolationLevel: "Serializable",   // 分離レベル
});

// ───── 方法2: バッチトランザクション ─────
// 複数操作を配列で渡す（全て成功 or 全てロールバック）
const [user, post, comment] = await prisma.$transaction([
  prisma.user.create({ data: { email: "x@x.com", name: "X" } }),
  prisma.post.create({ data: { title: "P", authorId: "..." } }),
  prisma.comment.create({ data: { content: "C", postId: "...", authorId: "..." } }),
]);
// 戻り値は配列（各操作の結果）

// ───── 方法3: 楽観的ロックパターン ─────
async function updateWithOptimisticLock(
  postId: string,
  expectedVersion: number,
  newTitle: string
): Promise<Post> {
  const result = await prisma.post.updateMany({
    where: {
      id: postId,
      version: expectedVersion, // バージョン番号をチェック
    },
    data: {
      title: newTitle,
      version: { increment: 1 },
    },
  });

  if (result.count === 0) {
    throw new Error("Optimistic lock conflict: record was modified by another transaction");
  }

  return prisma.post.findUniqueOrThrow({ where: { id: postId } });
}
```

#### トランザクション分離レベルの比較

```
分離レベルと並行性の問題:

  ┌────────────────────┬──────────┬──────────────┬─────────────┐
  │ 分離レベル          │ Dirty    │ Non-repeatable│ Phantom    │
  │                    │ Read     │ Read          │ Read       │
  ├────────────────────┼──────────┼──────────────┼─────────────┤
  │ ReadUncommitted    │ あり     │ あり          │ あり        │
  │ ReadCommitted      │ なし     │ あり          │ あり        │
  │ RepeatableRead     │ なし     │ なし          │ あり        │
  │ Serializable       │ なし     │ なし          │ なし        │
  └────────────────────┴──────────┴──────────────┴─────────────┘

  上に行くほどパフォーマンスが良い
  下に行くほど整合性が高い
```

### 3-2. Prisma Client Extensions

```typescript
// ───── カスタムメソッドの追加 ─────
const xprisma = prisma.$extends({
  model: {
    user: {
      async findByEmail(email: string) {
        return prisma.user.findUnique({
          where: { email },
          include: { profile: true },
        });
      },

      async softDelete(id: string) {
        return prisma.user.update({
          where: { id },
          data: { deletedAt: new Date() },
        });
      },

      async exists(id: string): Promise<boolean> {
        const count = await prisma.user.count({
          where: { id },
        });
        return count > 0;
      },
    },

    post: {
      async publish(id: string) {
        return prisma.post.update({
          where: { id },
          data: {
            published: true,
            publishedAt: new Date(),
          },
        });
      },

      async incrementViewCount(id: string) {
        return prisma.post.update({
          where: { id },
          data: { viewCount: { increment: 1 } },
        });
      },
    },
  },

  // クエリに対するミドルウェア的な拡張
  query: {
    user: {
      // 全ての findMany にソフトデリート条件を自動追加
      async findMany({ model, operation, args, query }) {
        args.where = { ...args.where, deletedAt: null };
        return query(args);
      },
      // findUnique にも適用
      async findUnique({ args, query }) {
        args.where = { ...args.where, deletedAt: null } as any;
        return query(args);
      },
    },
    // 全モデル共通のクエリログ
    $allModels: {
      async $allOperations({ model, operation, args, query }) {
        const start = performance.now();
        const result = await query(args);
        const end = performance.now();
        console.log(`${model}.${operation} took ${end - start}ms`);
        return result;
      },
    },
  },

  // 結果に対する変換
  result: {
    user: {
      fullName: {
        needs: { name: true },
        compute(user) {
          return user.name.toUpperCase();
        },
      },
    },
  },
});

// 拡張メソッドの使用
const user = await xprisma.user.findByEmail("alice@example.com");
const exists = await xprisma.user.exists("user-123");
await xprisma.post.publish("post-456");
```

### 3-3. 型安全な生 SQL

```typescript
// Prisma の型付き SQL（Prisma 5.x+）
import { Prisma } from "@prisma/client";

// ───── $queryRaw: SELECT クエリ ─────
interface UserPostCount {
  id: string;
  name: string;
  post_count: bigint;
}

const minPosts = 5;
const users = await prisma.$queryRaw<UserPostCount[]>`
  SELECT u.id, u.name, COUNT(p.id) as post_count
  FROM users u
  LEFT JOIN posts p ON u.id = p."authorId"
  WHERE u.role = 'USER'
  GROUP BY u.id, u.name
  HAVING COUNT(p.id) > ${minPosts}
  ORDER BY post_count DESC
`;

// 注意: bigint は JSON シリアライズできないため変換が必要
const serializable = users.map((u) => ({
  ...u,
  post_count: Number(u.post_count),
}));

// ───── $executeRaw: INSERT/UPDATE/DELETE ─────
const affectedRows = await prisma.$executeRaw`
  UPDATE posts
  SET "viewCount" = "viewCount" + 1
  WHERE id = ${postId}
`;
// 型: number（影響を受けた行数）

// ───── Prisma.sql で安全にクエリを組み立てる ─────
function buildSearchQuery(
  searchTerm: string,
  role?: string,
  limit: number = 10
) {
  const conditions: Prisma.Sql[] = [
    Prisma.sql`u.name ILIKE ${`%${searchTerm}%`}`,
  ];

  if (role) {
    conditions.push(Prisma.sql`u.role = ${role}`);
  }

  const whereClause = Prisma.join(conditions, " AND ");

  return prisma.$queryRaw<{ id: string; name: string; role: string }[]>`
    SELECT u.id, u.name, u.role
    FROM users u
    WHERE ${whereClause}
    LIMIT ${limit}
  `;
}

// ───── TypedSQL（Prisma 5.9+）─────
// prisma/sql/getUserStats.sql を作成:
// SELECT u.id, u.name, COUNT(p.id) as "postCount"
// FROM users u LEFT JOIN posts p ON u.id = p."authorId"
// WHERE u.role = $1
// GROUP BY u.id

// npx prisma generate --sql で型生成
import { getUserStats } from "@prisma/client/sql";

const stats = await prisma.$queryRawTyped(getUserStats("ADMIN"));
// 型: { id: string; name: string; postCount: number }[]
```

### 3-4. ページネーション

```typescript
// ───── オフセットベース（伝統的） ─────
async function getPaginatedUsers(page: number, pageSize: number) {
  const [users, total] = await prisma.$transaction([
    prisma.user.findMany({
      skip: (page - 1) * pageSize,
      take: pageSize,
      orderBy: { createdAt: "desc" },
    }),
    prisma.user.count(),
  ]);

  return {
    data: users,
    meta: {
      total,
      page,
      pageSize,
      totalPages: Math.ceil(total / pageSize),
      hasNext: page * pageSize < total,
      hasPrev: page > 1,
    },
  };
}

// ───── カーソルベース（大量データ向け、推奨） ─────
async function getCursorPaginatedPosts(
  cursor?: string,
  take: number = 20
) {
  const posts = await prisma.post.findMany({
    take: take + 1, // 1件余分に取得して hasNext を判定
    ...(cursor
      ? {
          cursor: { id: cursor },
          skip: 1, // カーソル自体をスキップ
        }
      : {}),
    orderBy: { createdAt: "desc" },
    include: {
      author: { select: { name: true } },
      _count: { select: { comments: true } },
    },
  });

  const hasNext = posts.length > take;
  const data = hasNext ? posts.slice(0, -1) : posts;
  const nextCursor = hasNext ? data[data.length - 1].id : null;

  return {
    data,
    meta: {
      hasNext,
      nextCursor,
    },
  };
}
```

```
ページネーション方式の比較:

  オフセットベース:
  ┌────────────────────────────────────────────┐
  │ Page 1    │ Page 2    │ Page 3    │ ...    │
  │ skip=0    │ skip=20   │ skip=40   │        │
  │ take=20   │ take=20   │ take=20   │        │
  └────────────────────────────────────────────┘
  ✅ ページ番号でジャンプ可能
  ❌ 大量データで遅くなる（OFFSET が大きいほど遅い）
  ❌ データ挿入/削除時にページずれが発生

  カーソルベース:
  ┌────────────┐  cursor  ┌────────────┐  cursor  ┌──────┐
  │ Chunk 1    │───────>  │ Chunk 2    │───────>  │ ...  │
  │ after: null│          │ after: id20│          │      │
  └────────────┘          └────────────┘          └──────┘
  ✅ 大量データでも一定の速度
  ✅ データ変更の影響を受けにくい
  ❌ ページ番号ジャンプ不可
  ❌ 実装がやや複雑
```

### 3-5. フルテキスト検索

```typescript
// PostgreSQL のフルテキスト検索（Prisma 4.x+）
// schema.prisma に preview feature を追加:
// generator client {
//   provider        = "prisma-client-js"
//   previewFeatures = ["fullTextSearch", "fullTextIndex"]
// }

const results = await prisma.post.findMany({
  where: {
    // PostgreSQL のテキスト検索演算子
    title: { search: "TypeScript & Prisma" },
    content: { search: "型安全 | ORM" },
  },
  orderBy: {
    _relevance: {
      fields: ["title", "content"],
      search: "TypeScript Prisma",
      sort: "desc",
    },
  },
});
```

---

## 4. リポジトリパターン

```
リポジトリパターン:

  Controller / Service
       |
       v
  +-------------------+
  | IUserRepository   |  ← インターフェース
  +-------------------+
       |            |
       v            v
  +-----------+  +-----------+
  |PrismaUser |  |MockUser   |
  |Repository |  |Repository |
  +-----------+  +-----------+
  (本番)         (テスト)
```

### 4-1. 汎用リポジトリインターフェース

```typescript
// 汎用のリポジトリインターフェース
interface PaginationParams {
  page: number;
  pageSize: number;
}

interface PaginatedResult<T> {
  data: T[];
  meta: {
    total: number;
    page: number;
    pageSize: number;
    totalPages: number;
  };
}

interface IRepository<T, CreateInput, UpdateInput> {
  findById(id: string): Promise<T | null>;
  findMany(params?: PaginationParams): Promise<PaginatedResult<T>>;
  create(data: CreateInput): Promise<T>;
  update(id: string, data: UpdateInput): Promise<T>;
  delete(id: string): Promise<void>;
  count(where?: Partial<T>): Promise<number>;
}
```

### 4-2. User リポジトリの実装

```typescript
// インターフェース
interface IUserRepository extends IRepository<User, CreateUserDto, UpdateUserDto> {
  findByEmail(email: string): Promise<User | null>;
  findByRole(role: Role): Promise<User[]>;
  findWithPosts(id: string): Promise<(User & { posts: Post[] }) | null>;
  existsByEmail(email: string): Promise<boolean>;
}

type CreateUserDto = {
  email: string;
  name: string;
  role?: Role;
};

type UpdateUserDto = Partial<CreateUserDto>;

// Prisma 実装
class PrismaUserRepository implements IUserRepository {
  constructor(private readonly prisma: PrismaClient) {}

  async findById(id: string): Promise<User | null> {
    return this.prisma.user.findUnique({ where: { id } });
  }

  async findByEmail(email: string): Promise<User | null> {
    return this.prisma.user.findUnique({ where: { email } });
  }

  async findByRole(role: Role): Promise<User[]> {
    return this.prisma.user.findMany({ where: { role } });
  }

  async findWithPosts(id: string): Promise<(User & { posts: Post[] }) | null> {
    return this.prisma.user.findUnique({
      where: { id },
      include: {
        posts: {
          where: { published: true },
          orderBy: { createdAt: "desc" },
        },
      },
    });
  }

  async findMany(params?: PaginationParams): Promise<PaginatedResult<User>> {
    const page = params?.page ?? 1;
    const pageSize = params?.pageSize ?? 20;

    const [data, total] = await this.prisma.$transaction([
      this.prisma.user.findMany({
        skip: (page - 1) * pageSize,
        take: pageSize,
        orderBy: { createdAt: "desc" },
      }),
      this.prisma.user.count(),
    ]);

    return {
      data,
      meta: {
        total,
        page,
        pageSize,
        totalPages: Math.ceil(total / pageSize),
      },
    };
  }

  async create(data: CreateUserDto): Promise<User> {
    return this.prisma.user.create({ data });
  }

  async update(id: string, data: UpdateUserDto): Promise<User> {
    return this.prisma.user.update({ where: { id }, data });
  }

  async delete(id: string): Promise<void> {
    await this.prisma.user.delete({ where: { id } });
  }

  async count(where?: Partial<User>): Promise<number> {
    return this.prisma.user.count({ where: where as any });
  }

  async existsByEmail(email: string): Promise<boolean> {
    const count = await this.prisma.user.count({
      where: { email },
    });
    return count > 0;
  }
}

// モック実装（テスト用）
class InMemoryUserRepository implements IUserRepository {
  private users: User[] = [];

  async findById(id: string): Promise<User | null> {
    return this.users.find((u) => u.id === id) ?? null;
  }

  async findByEmail(email: string): Promise<User | null> {
    return this.users.find((u) => u.email === email) ?? null;
  }

  async findByRole(role: Role): Promise<User[]> {
    return this.users.filter((u) => u.role === role);
  }

  async findWithPosts(id: string): Promise<(User & { posts: Post[] }) | null> {
    const user = this.users.find((u) => u.id === id);
    if (!user) return null;
    return { ...user, posts: [] };
  }

  async findMany(params?: PaginationParams): Promise<PaginatedResult<User>> {
    const page = params?.page ?? 1;
    const pageSize = params?.pageSize ?? 20;
    const start = (page - 1) * pageSize;
    const data = this.users.slice(start, start + pageSize);

    return {
      data,
      meta: {
        total: this.users.length,
        page,
        pageSize,
        totalPages: Math.ceil(this.users.length / pageSize),
      },
    };
  }

  async create(data: CreateUserDto): Promise<User> {
    const user: User = {
      id: crypto.randomUUID(),
      ...data,
      role: data.role ?? "USER",
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    this.users.push(user);
    return user;
  }

  async update(id: string, data: UpdateUserDto): Promise<User> {
    const index = this.users.findIndex((u) => u.id === id);
    if (index === -1) throw new Error("User not found");
    this.users[index] = {
      ...this.users[index],
      ...data,
      updatedAt: new Date(),
    };
    return this.users[index];
  }

  async delete(id: string): Promise<void> {
    this.users = this.users.filter((u) => u.id !== id);
  }

  async count(): Promise<number> {
    return this.users.length;
  }

  async existsByEmail(email: string): Promise<boolean> {
    return this.users.some((u) => u.email === email);
  }

  // テスト用ヘルパー
  clear(): void {
    this.users = [];
  }

  seed(users: User[]): void {
    this.users = [...users];
  }
}
```

### 4-3. サービス層での使用

```typescript
class UserService {
  constructor(
    private readonly userRepo: IUserRepository,
    private readonly emailService: IEmailService,
    private readonly logger: ILogger
  ) {}

  async registerUser(data: CreateUserDto): Promise<User> {
    // ビジネスルールのバリデーション
    const existing = await this.userRepo.existsByEmail(data.email);
    if (existing) {
      throw new ConflictError(`Email ${data.email} is already registered`);
    }

    // ユーザー作成
    const user = await this.userRepo.create(data);

    // ウェルカムメール送信
    await this.emailService.send(
      user.email,
      "Welcome!",
      `Hello ${user.name}, welcome to our platform!`
    );

    this.logger.info("User registered", { userId: user.id });
    return user;
  }

  async getUserProfile(id: string): Promise<User & { posts: Post[] }> {
    const user = await this.userRepo.findWithPosts(id);
    if (!user) {
      throw new NotFoundError(`User ${id} not found`);
    }
    return user;
  }
}
```

---

## 5. テスト戦略

### 5-1. テスト環境構築

```typescript
// test/setup.ts -- テスト用 Prisma クライアント
import { PrismaClient } from "@prisma/client";
import { execSync } from "child_process";

const TEST_DATABASE_URL =
  process.env.TEST_DATABASE_URL ??
  "postgresql://postgres:postgres@localhost:5433/test_db";

let prisma: PrismaClient;

// テスト全体の前処理
beforeAll(async () => {
  // テスト用 DB のマイグレーション
  execSync("npx prisma migrate deploy", {
    env: {
      ...process.env,
      DATABASE_URL: TEST_DATABASE_URL,
    },
  });

  prisma = new PrismaClient({
    datasources: { db: { url: TEST_DATABASE_URL } },
  });

  await prisma.$connect();
});

// 各テストの前処理（テーブルクリーンアップ）
beforeEach(async () => {
  // テーブルの削除順序はリレーションに注意
  await prisma.$transaction([
    prisma.comment.deleteMany(),
    prisma.post.deleteMany(),
    prisma.profile.deleteMany(),
    prisma.user.deleteMany(),
    prisma.category.deleteMany(),
    prisma.tag.deleteMany(),
  ]);
});

afterAll(async () => {
  await prisma.$disconnect();
});

export { prisma };
```

```yaml
# docker-compose.test.yml
version: "3.8"
services:
  test-db:
    image: postgres:16-alpine
    ports:
      - "5433:5432"
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: test_db
    tmpfs:
      - /var/lib/postgresql/data  # メモリ上で高速化
```

### 5-2. インテグレーションテスト

```typescript
import { describe, it, expect, beforeEach } from "vitest";
import { prisma } from "../setup";

describe("User CRUD Integration Tests", () => {
  it("should create a user with profile", async () => {
    const user = await prisma.user.create({
      data: {
        email: "test@example.com",
        name: "Test User",
        profile: {
          create: { bio: "Hello" },
        },
      },
      include: { profile: true },
    });

    expect(user.email).toBe("test@example.com");
    expect(user.profile).not.toBeNull();
    expect(user.profile!.bio).toBe("Hello");
  });

  it("should enforce unique email constraint", async () => {
    await prisma.user.create({
      data: { email: "dup@example.com", name: "User 1" },
    });

    await expect(
      prisma.user.create({
        data: { email: "dup@example.com", name: "User 2" },
      })
    ).rejects.toThrow();
  });

  it("should cascade delete posts when user is deleted", async () => {
    const user = await prisma.user.create({
      data: {
        email: "author@example.com",
        name: "Author",
        posts: {
          create: [
            { title: "Post 1", published: true },
            { title: "Post 2", published: false },
          ],
        },
      },
    });

    // ユーザー削除前の投稿数
    const beforeCount = await prisma.post.count({
      where: { authorId: user.id },
    });
    expect(beforeCount).toBe(2);

    // ユーザー削除（Cascade が設定されている場合）
    await prisma.user.delete({ where: { id: user.id } });

    // 投稿も削除されていることを確認
    const afterCount = await prisma.post.count({
      where: { authorId: user.id },
    });
    expect(afterCount).toBe(0);
  });

  it("should correctly paginate results", async () => {
    // 15件のテストデータ作成
    await prisma.user.createMany({
      data: Array.from({ length: 15 }, (_, i) => ({
        email: `user${i}@example.com`,
        name: `User ${i}`,
      })),
    });

    // ページ1（10件）
    const page1 = await prisma.user.findMany({
      take: 10,
      skip: 0,
      orderBy: { email: "asc" },
    });
    expect(page1).toHaveLength(10);

    // ページ2（5件）
    const page2 = await prisma.user.findMany({
      take: 10,
      skip: 10,
      orderBy: { email: "asc" },
    });
    expect(page2).toHaveLength(5);
  });
});
```

### 5-3. リポジトリのユニットテスト

```typescript
import { describe, it, expect, beforeEach } from "vitest";

describe("UserService with InMemoryRepository", () => {
  let userService: UserService;
  let userRepo: InMemoryUserRepository;
  let emailService: MockEmailService;
  let logger: MockLogger;

  beforeEach(() => {
    userRepo = new InMemoryUserRepository();
    emailService = new MockEmailService();
    logger = new MockLogger();
    userService = new UserService(userRepo, emailService, logger);
  });

  it("should register a new user", async () => {
    const user = await userService.registerUser({
      email: "new@example.com",
      name: "New User",
    });

    expect(user.email).toBe("new@example.com");
    expect(user.name).toBe("New User");

    // メールが送信されたことを確認
    expect(emailService.sentEmails).toHaveLength(1);
    expect(emailService.sentEmails[0].to).toBe("new@example.com");

    // ログが記録されたことを確認
    expect(logger.infoMessages).toHaveLength(1);
  });

  it("should throw ConflictError for duplicate email", async () => {
    await userService.registerUser({
      email: "dup@example.com",
      name: "User 1",
    });

    await expect(
      userService.registerUser({
        email: "dup@example.com",
        name: "User 2",
      })
    ).rejects.toThrow(ConflictError);

    // メールは1回だけ送信されている
    expect(emailService.sentEmails).toHaveLength(1);
  });

  it("should return user with posts", async () => {
    const user = await userRepo.create({
      email: "author@example.com",
      name: "Author",
    });

    const profile = await userService.getUserProfile(user.id);
    expect(profile.posts).toEqual([]);
  });

  it("should throw NotFoundError for non-existent user", async () => {
    await expect(
      userService.getUserProfile("non-existent-id")
    ).rejects.toThrow(NotFoundError);
  });
});

// モック実装
class MockEmailService implements IEmailService {
  sentEmails: { to: string; subject: string; body: string }[] = [];

  async send(to: string, subject: string, body: string): Promise<void> {
    this.sentEmails.push({ to, subject, body });
  }
}

class MockLogger implements ILogger {
  infoMessages: string[] = [];
  errorMessages: string[] = [];

  info(message: string): void {
    this.infoMessages.push(message);
  }
  error(message: string): void {
    this.errorMessages.push(message);
  }
}
```

---

## 6. パフォーマンス最適化

### 6-1. N+1 問題の検出と解決

```typescript
// ───── NG: ループ内でクエリ（N+1 問題） ─────
const users = await prisma.user.findMany();
for (const user of users) {
  const posts = await prisma.post.findMany({
    where: { authorId: user.id },
  });
  // 1 + N 回のクエリが発行される
}

// ───── OK: include で一括取得 ─────
const usersWithPosts = await prisma.user.findMany({
  include: {
    posts: true,
  },
});
// 2回のクエリ（users + posts）で完了

// ───── OK: 事前に全投稿を取得してマッピング ─────
const users = await prisma.user.findMany();
const posts = await prisma.post.findMany({
  where: { authorId: { in: users.map((u) => u.id) } },
});

const postsByAuthor = new Map<string, Post[]>();
for (const post of posts) {
  const existing = postsByAuthor.get(post.authorId) ?? [];
  postsByAuthor.set(post.authorId, [...existing, post]);
}

const result = users.map((user) => ({
  ...user,
  posts: postsByAuthor.get(user.id) ?? [],
}));
```

### 6-2. select による必要フィールドの絞り込み

```typescript
// NG: 全フィールド取得（不要なデータも含む）
const users = await prisma.user.findMany();
// → id, email, name, role, createdAt, updatedAt 全て取得

// OK: 必要なフィールドだけ取得
const users = await prisma.user.findMany({
  select: {
    id: true,
    name: true,
    email: true,
  },
});
// → 3フィールドのみ。DBからの転送量が減り、メモリ消費も低い
```

### 6-3. インデックス戦略

```prisma
// 頻繁に使われるクエリに合わせたインデックス設計

model Post {
  id        String   @id @default(uuid())
  title     String
  content   String?
  published Boolean  @default(false)
  authorId  String
  createdAt DateTime @default(now())

  // 単一カラムインデックス
  @@index([authorId])

  // 複合インデックス（公開記事を新しい順に取得するクエリ用）
  @@index([published, createdAt(sort: Desc)])

  // カバリングインデックス（クエリに必要な全カラムを含む）
  @@index([authorId, published, createdAt])

  @@map("posts")
}

model User {
  id    String @id @default(uuid())
  email String @unique  // unique はインデックスを兼ねる
  name  String

  // 部分インデックス（PostgreSQL）
  // schema.prisma では直接サポートされないため、
  // マイグレーション SQL で手動追加:
  // CREATE INDEX idx_active_users ON users (email) WHERE deleted_at IS NULL;

  @@map("users")
}
```

### 6-4. コネクションプーリング

```typescript
// PrismaClient のコネクション設定
const prisma = new PrismaClient({
  datasources: {
    db: {
      url: process.env.DATABASE_URL,
    },
  },
});

// DATABASE_URL でプーリングパラメータを指定
// postgresql://user:pass@host:5432/db?connection_limit=10&pool_timeout=30
```

```
コネクションプーリングのアーキテクチャ:

  サーバーレス環境での課題:
  ┌─────────┐  ┌─────────┐  ┌─────────┐
  │ Lambda  │  │ Lambda  │  │ Lambda  │  × N インスタンス
  │ Instance│  │ Instance│  │ Instance│
  └────┬────┘  └────┬────┘  └────┬────┘
       │           │            │
       v           v            v
  ┌────────────────────────────────────┐
  │          PostgreSQL                │
  │   max_connections = 100            │ ← すぐ枯渇！
  └────────────────────────────────────┘

  Prisma Accelerate / PgBouncer で解決:
  ┌─────────┐  ┌─────────┐  ┌─────────┐
  │ Lambda  │  │ Lambda  │  │ Lambda  │
  └────┬────┘  └────┬────┘  └────┬────┘
       │           │            │
       v           v            v
  ┌────────────────────────────────────┐
  │     Connection Pooler              │  ← プール管理
  │  (Prisma Accelerate / PgBouncer)   │
  └──────────────┬─────────────────────┘
                 │
                 v
  ┌────────────────────────────────────┐
  │          PostgreSQL                │
  │   少数の接続で多数のクライアントを処理│
  └────────────────────────────────────┘
```

```typescript
// Prisma Accelerate の設定
// .env
// DATABASE_URL="prisma://accelerate.prisma-data.net/?api_key=..."
// DIRECT_URL="postgresql://user:pass@host:5432/db"  // マイグレーション用

// schema.prisma
// datasource db {
//   provider  = "postgresql"
//   url       = env("DATABASE_URL")
//   directUrl = env("DIRECT_URL")
// }

// Accelerate のキャッシュ機能
const users = await prisma.user.findMany({
  cacheStrategy: {
    ttl: 60,      // 60秒キャッシュ
    swr: 120,     // Stale While Revalidate: 120秒
  },
});
```

### 6-5. バッチ処理

```typescript
// 大量データの処理はバッチ分割で
async function processAllUsers(batchSize: number = 100) {
  let cursor: string | undefined;
  let processedCount = 0;

  while (true) {
    const users = await prisma.user.findMany({
      take: batchSize,
      ...(cursor
        ? { cursor: { id: cursor }, skip: 1 }
        : {}),
      orderBy: { id: "asc" },
    });

    if (users.length === 0) break;

    // バッチ処理
    await Promise.all(
      users.map(async (user) => {
        await processUser(user);
      })
    );

    processedCount += users.length;
    cursor = users[users.length - 1].id;

    console.log(`Processed ${processedCount} users...`);
  }

  return processedCount;
}

// createMany で一括挿入
async function bulkInsertUsers(users: CreateUserDto[]) {
  const CHUNK_SIZE = 1000;
  let totalInserted = 0;

  for (let i = 0; i < users.length; i += CHUNK_SIZE) {
    const chunk = users.slice(i, i + CHUNK_SIZE);
    const result = await prisma.user.createMany({
      data: chunk,
      skipDuplicates: true,
    });
    totalInserted += result.count;
  }

  return totalInserted;
}
```

---

## 7. 実践的な設計パターン

### 7-1. ソフトデリート

```typescript
// schema.prisma にソフトデリート用カラムを追加
// model User {
//   ...
//   deletedAt DateTime?
// }

// Client Extension でソフトデリートを透過的に処理
const prismaWithSoftDelete = prisma.$extends({
  query: {
    user: {
      async findMany({ args, query }) {
        args.where = { ...args.where, deletedAt: null };
        return query(args);
      },
      async findFirst({ args, query }) {
        args.where = { ...args.where, deletedAt: null };
        return query(args);
      },
      async findUnique({ args, query }) {
        // findUnique は where の制約が厳しいので注意
        return query(args);
      },
      async delete({ args }) {
        // 物理削除を論理削除に変換
        return prisma.user.update({
          where: args.where,
          data: { deletedAt: new Date() },
        });
      },
      async deleteMany({ args }) {
        return prisma.user.updateMany({
          where: args.where ?? {},
          data: { deletedAt: new Date() },
        });
      },
    },
  },
  model: {
    user: {
      // 完全な物理削除（管理者用）
      async hardDelete(id: string) {
        return prisma.user.delete({ where: { id } });
      },
      // 復元
      async restore(id: string) {
        return prisma.user.update({
          where: { id },
          data: { deletedAt: null },
        });
      },
    },
  },
});
```

### 7-2. 監査ログ（Audit Trail）

```typescript
// 変更履歴を自動記録する Extension
const prismaWithAudit = prisma.$extends({
  query: {
    $allModels: {
      async create({ model, args, query }) {
        const result = await query(args);
        await prisma.auditLog.create({
          data: {
            model: model as string,
            action: "CREATE",
            recordId: (result as any).id,
            newData: JSON.stringify(result),
            userId: getCurrentUserId(), // コンテキストから取得
            timestamp: new Date(),
          },
        });
        return result;
      },
      async update({ model, args, query }) {
        // 変更前のデータを取得
        const before = await (prisma as any)[model].findUnique({
          where: args.where,
        });
        const result = await query(args);
        await prisma.auditLog.create({
          data: {
            model: model as string,
            action: "UPDATE",
            recordId: (result as any).id,
            oldData: JSON.stringify(before),
            newData: JSON.stringify(result),
            userId: getCurrentUserId(),
            timestamp: new Date(),
          },
        });
        return result;
      },
      async delete({ model, args, query }) {
        const before = await (prisma as any)[model].findUnique({
          where: args.where,
        });
        const result = await query(args);
        await prisma.auditLog.create({
          data: {
            model: model as string,
            action: "DELETE",
            recordId: (before as any).id,
            oldData: JSON.stringify(before),
            userId: getCurrentUserId(),
            timestamp: new Date(),
          },
        });
        return result;
      },
    },
  },
});
```

### 7-3. マルチテナント

```typescript
// テナント ID を全クエリに自動付与

function createTenantPrisma(tenantId: string) {
  return prisma.$extends({
    query: {
      $allModels: {
        async findMany({ args, query }) {
          args.where = { ...args.where, tenantId };
          return query(args);
        },
        async findFirst({ args, query }) {
          args.where = { ...args.where, tenantId };
          return query(args);
        },
        async create({ args, query }) {
          args.data = { ...args.data, tenantId };
          return query(args);
        },
        async update({ args, query }) {
          args.where = { ...args.where, tenantId } as any;
          return query(args);
        },
        async delete({ args, query }) {
          args.where = { ...args.where, tenantId } as any;
          return query(args);
        },
      },
    },
  });
}

// 使用例（ミドルウェアで設定）
app.use((req, res, next) => {
  const tenantId = req.headers["x-tenant-id"] as string;
  req.prisma = createTenantPrisma(tenantId);
  next();
});
```

---

## 比較表

### ORM / クエリビルダー比較

| 特性 | Prisma | Drizzle | TypeORM | Kysely |
|------|--------|---------|---------|--------|
| 型安全性 | 最高(生成) | 最高(推論) | 中(デコレータ) | 高(推論) |
| スキーマ定義 | .prisma | TypeScript | デコレータ | TypeScript |
| マイグレーション | 組込み | drizzle-kit | 組込み | 別途 |
| 生SQL | $queryRaw | sql`` | query() | sql`` |
| バンドルサイズ | 大(Engine) | 小 | 大 | 小 |
| 学習コスト | 低 | 低 | 中 | 中 |
| Edge Runtime | 対応(Accelerate) | 対応 | 非対応 | 対応 |
| リレーション | 宣言的 | 宣言的/SQL | デコレータ | JOIN手動 |
| トランザクション | Interactive TX | SQL直接 | QueryRunner | SQL直接 |
| 公式GUI | Prisma Studio | Drizzle Studio | なし | なし |

### Prisma のクエリ手法比較

| 手法 | 型安全性 | 柔軟性 | パフォーマンス | 用途 |
|------|---------|--------|-------------|------|
| findMany / findUnique | 最高 | 中 | 良好 | 標準CRUD |
| include / select | 最高 | 中 | 要注意(N+1) | リレーション |
| $queryRaw | 中 | 最高 | 最高 | 複雑なクエリ |
| TypedSQL | 高 | 最高 | 最高 | SQL ファイル管理 |
| $transaction | 最高 | 高 | 良好 | 複数操作 |
| Client Extensions | 最高 | 高 | 良好 | カスタムロジック |
| Accelerate + cache | 最高 | 中 | 最高 | 読み取り頻度高 |

### データベースプロバイダー対応

| 機能 | PostgreSQL | MySQL | SQLite | MongoDB | SQL Server |
|------|-----------|-------|--------|---------|------------|
| フルテキスト検索 | あり | あり | なし | なし | なし |
| JSON フィルタ | あり | あり | なし | あり | あり |
| enum | あり | あり | なし | なし | なし |
| 配列型 | あり | なし | なし | あり | なし |
| インタラクティブTX | あり | あり | あり | あり | あり |

---

## アンチパターン

### AP-1: N+1 問題を放置する

```typescript
// NG: ループ内でクエリ（N+1 問題）
const users = await prisma.user.findMany();
for (const user of users) {
  const posts = await prisma.post.findMany({
    where: { authorId: user.id },
  });
  // 1 + N 回のクエリが発行される
}

// OK: include で一括取得
const users = await prisma.user.findMany({
  include: {
    posts: true,
  },
});
// 2回のクエリ（users + posts）で完了
```

### AP-2: PrismaClient をリクエスト毎に生成

```typescript
// NG: 毎回新しいインスタンス（接続プール枯渇）
app.get("/users", async (req, res) => {
  const prisma = new PrismaClient();
  const users = await prisma.user.findMany();
  await prisma.$disconnect();
  res.json(users);
});

// OK: シングルトンで共有
// lib/prisma.ts
const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined;
};
const prisma = globalForPrisma.prisma ?? new PrismaClient();
if (process.env.NODE_ENV !== "production") {
  globalForPrisma.prisma = prisma;
}
export { prisma };
```

### AP-3: include のネストが深すぎる

```typescript
// NG: 4段階以上のネスト（パフォーマンス悪化）
const user = await prisma.user.findUnique({
  where: { id },
  include: {
    posts: {
      include: {
        comments: {
          include: {
            author: {
              include: {
                profile: true,  // 4段階ネスト
              },
            },
          },
        },
      },
    },
  },
});

// OK: 必要なデータだけ取得して別途組み立て
const user = await prisma.user.findUnique({
  where: { id },
  include: {
    posts: {
      select: { id: true, title: true },
      take: 10,
    },
  },
});

// 必要に応じて追加クエリ
const postIds = user?.posts.map((p) => p.id) ?? [];
const comments = await prisma.comment.findMany({
  where: { postId: { in: postIds } },
  include: { author: { select: { name: true } } },
  take: 50,
});
```

### AP-4: マイグレーションファイルを手動編集する

```typescript
// NG: 生成されたマイグレーション SQL を直接書き換える
// → prisma migrate dev が差分を正しく検出できなくなる

// OK: カスタム SQL が必要な場合は空のマイグレーションを作成
// npx prisma migrate dev --create-only --name add_custom_index
// → 生成されたファイルに SQL を追加してから
// npx prisma migrate dev で適用
```

### AP-5: $queryRaw でユーザー入力を直接埋め込む

```typescript
// NG: SQL インジェクションの危険
const users = await prisma.$queryRaw`
  SELECT * FROM users WHERE name = '${userInput}'
`;

// OK: テンプレートリテラルのプレースホルダを使う
const users = await prisma.$queryRaw`
  SELECT * FROM users WHERE name = ${userInput}
`;
// Prisma が自動的にパラメータ化してくれる
```

---

## FAQ

### Q1: Prisma と Drizzle のどちらを選ぶべきですか？

Prisma はスキーマファーストの設計が直感的で、Studio、マイグレーション、型生成が一体化しています。Drizzle はより軽量で SQL に近く、Edge Runtime との相性が良いです。チームの SQL 習熟度が高ければ Drizzle、ORM の抽象化を好むなら Prisma が適しています。

大規模チームでは Prisma のスキーマが「Single Source of Truth」として機能し、バックエンドとフロントエンドの共通言語になる点が大きなメリットです。一方で、複雑なクエリが多い場合は Drizzle の SQL ライクな記法の方が自然に感じるでしょう。

### Q2: Prisma のパフォーマンスが遅い場合の対策は？

以下の対策を順番に検討してください:

1. `select` で必要なフィールドのみ取得
2. `include` のネストを最小化（3段階以内）
3. 適切なインデックスの追加（`@@index`）
4. N+1 問題の検出と解消（`prisma.$on("query")` でログ確認）
5. `$queryRaw` による複雑なクエリの最適化
6. Prisma Accelerate（コネクションプーリング + キャッシュ）の導入
7. 読み取りレプリカの活用（read replica 設定）

### Q3: テストではどうやって DB をモックしますか？

テスト用 DB（Docker の PostgreSQL）を使ったインテグレーションテストが最も信頼性が高いです。単体テストではリポジトリインターフェースのモック実装を DI で注入します。`prisma-mock` ライブラリもありますが、実 DB テストを推奨します。

テストの種類と使い分け:
- **ユニットテスト**: InMemoryRepository を使い、ビジネスロジックのみテスト
- **インテグレーションテスト**: Docker DB で Prisma Client を実際に使う
- **E2E テスト**: API エンドポイント経由で DB まで含めてテスト

### Q4: Prisma を Edge Runtime（Vercel Edge Functions, Cloudflare Workers）で使えますか？

Prisma Accelerate を使うことで Edge Runtime に対応できます。通常の Prisma Client は Node.js のネイティブバイナリに依存するため Edge では動作しませんが、Accelerate はHTTP 経由で接続するため、任意のランタイムで利用可能です。

```typescript
// Edge Runtime での使用
import { PrismaClient } from "@prisma/client/edge";
import { withAccelerate } from "@prisma/extension-accelerate";

const prisma = new PrismaClient().$extends(withAccelerate());
```

### Q5: スキーマ変更時にダウンタイムを避けるには？

段階的マイグレーション戦略を使います:

1. **拡張フェーズ**: 新カラムを nullable で追加、古いコードとの互換性を維持
2. **移行フェーズ**: 新旧両方のカラムに書き込み、バックフィルを実行
3. **収縮フェーズ**: 古いカラムへの書き込みを停止、NOT NULL 制約を追加
4. **クリーンアップ**: 古いカラムを削除

---

## まとめ表

| 概念 | 要点 |
|------|------|
| schema.prisma | データモデルの Single Source of Truth |
| prisma generate | スキーマから型付きクライアントを自動生成 |
| include / select | リレーションの型安全な取得 |
| $transaction | 複数操作のアトミック実行 |
| Client Extensions | カスタムメソッドの型安全な追加 |
| リポジトリパターン | DI でテスタビリティを確保 |
| Prisma Accelerate | コネクションプーリングとキャッシュ |
| TypedSQL | SQL ファイルから型安全なクエリを生成 |
| ソフトデリート | Extension で透過的に実装 |
| 楽観的ロック | version カラムで並行更新を検出 |

---

## 演習問題

### 演習1: スキーマ設計

以下の要件を満たす Prisma スキーマを設計してください。

- EC サイトの商品カタログシステム
- 商品（Product）は複数のカテゴリに属する（多対多）
- 商品には複数の SKU（バリエーション: サイズ、色など）がある（1対多）
- ユーザーは商品をお気に入りに追加できる（多対多）
- 商品レビュー（1対多、ユーザーと商品の両方にリレーション）
- 適切なインデックスと enum を含めること

### 演習2: 型安全なクエリ

Prisma.validator と Prisma.UserGetPayload を使って、以下の再利用可能なクエリ型を定義してください。

1. ユーザーのダッシュボード用データ（プロフィール + 最新5件の投稿 + 未読通知数）
2. 管理者用ユーザー一覧（全フィールド + 投稿数 + コメント数）
3. 公開記事の詳細ページ用データ（著者名 + カテゴリ + コメント上位10件）

### 演習3: リポジトリパターンの実装

以下の仕様でリポジトリを実装してください。

- `IPostRepository` インターフェースを設計
- `PrismaPostRepository` と `InMemoryPostRepository` の両方を実装
- 検索（タイトル/内容のテキスト検索）、フィルタ（カテゴリ、公開状態）、ソート（日付、閲覧数）をサポート
- カーソルベースのページネーションを含む

### 演習4: トランザクション

以下のビジネスロジックをトランザクションで実装してください。

- ユーザーが記事を「購入」する処理
- ユーザーの残高を確認し、記事の価格を差し引く
- 購入記録をログテーブルに挿入
- 著者の収益カラムを更新
- いずれかの操作が失敗したら全体をロールバック
- 楽観的ロックで並行購入を防止

### 演習5: パフォーマンスチューニング

以下のコードのパフォーマンス問題を特定し、改善してください。

```typescript
// このコードにはパフォーマンス問題がいくつあるか？
async function getPopularAuthors() {
  const users = await prisma.user.findMany();

  const results = [];
  for (const user of users) {
    const postCount = await prisma.post.count({
      where: { authorId: user.id, published: true },
    });

    if (postCount > 10) {
      const posts = await prisma.post.findMany({
        where: { authorId: user.id, published: true },
        include: {
          categories: true,
          comments: true,
        },
        orderBy: { viewCount: "desc" },
      });

      results.push({
        user,
        postCount,
        posts,
        totalViews: posts.reduce((sum, p) => sum + p.viewCount, 0),
      });
    }
  }

  return results.sort((a, b) => b.totalViews - a.totalViews).slice(0, 10);
}
```

ヒント: N+1 問題、不要なデータ取得、集計のアプローチ、ページネーション欠如に注目してください。

### 演習6: ソフトデリートの Extension

Prisma Client Extension を使って、以下の機能を持つソフトデリートシステムを実装してください。

- 全モデル共通（`$allModels`）で動作する
- `delete` / `deleteMany` を論理削除に変換
- `findMany` / `findFirst` / `findUnique` で削除済みレコードを自動除外
- `restore(id)` メソッドで復元可能
- `hardDelete(id)` メソッドで物理削除可能
- 削除から30日経過したレコードを自動物理削除するバッチ処理

---

## 次に読むべきガイド

- [tRPC](./02-trpc.md) -- Prisma + tRPC で型安全なフルスタック開発
- [Zod バリデーション](./00-zod-validation.md) -- Prisma スキーマと zod の連携
- [DI パターン](../02-patterns/04-dependency-injection.md) -- リポジトリの DI 設計

---

## 参考文献

1. **Prisma Documentation**
   https://www.prisma.io/docs

2. **Prisma GitHub Repository**
   https://github.com/prisma/prisma

3. **Prisma Best Practices**
   https://www.prisma.io/docs/guides

4. **Prisma Client Extensions**
   https://www.prisma.io/docs/concepts/components/prisma-client/client-extensions

5. **Prisma Accelerate**
   https://www.prisma.io/docs/accelerate
