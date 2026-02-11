# Prisma + TypeScript 完全ガイド

> Prisma ORM で TypeScript の型安全性を最大限に活かし、データベース操作を堅牢に行う

## この章で学ぶこと

1. **Prisma の基本** -- スキーマ定義、マイグレーション、CRUD 操作の型安全な記述
2. **高度なクエリパターン** -- リレーション、トランザクション、生SQLの型安全な扱い
3. **実践的な設計** -- リポジトリパターン、テスト戦略、パフォーマンスチューニング

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
  createdAt   DateTime   @default(now())

  @@index([authorId])
  @@map("posts")
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
  posts Post[]

  @@map("categories")
}

enum Role {
  USER
  ADMIN
  MODERATOR
}
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
```

---

## 2. 型安全な CRUD 操作

### 2-1. 基本的な CRUD

```typescript
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

// Create
const user = await prisma.user.create({
  data: {
    email: "alice@example.com",
    name: "Alice",
    role: "ADMIN", // enum のリテラル型で補完が効く
  },
});
// 型: User

// Read
const found = await prisma.user.findUnique({
  where: { email: "alice@example.com" },
});
// 型: User | null

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

// Update
const updated = await prisma.user.update({
  where: { id: user.id },
  data: { name: "Alice Smith" },
});
// 型: User

// Delete
const deleted = await prisma.user.delete({
  where: { id: user.id },
});
```

### 2-2. リレーションの取得

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
    },
    profile: true,
  },
});
// 型: (User & { posts: Post[]; profile: Profile | null }) | null

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

### 2-3. ネストした作成・更新

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
        { title: "First Post", content: "Hello World", published: true },
        { title: "Draft", content: "WIP" },
      ],
    },
  },
  include: {
    profile: true,
    posts: true,
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
          create: { name: "TypeScript" },
        },
        {
          where: { name: "Programming" },
          create: { name: "Programming" },
        },
      ],
    },
  },
});
```

---

## 3. 高度なクエリパターン

### 3-1. トランザクション

```typescript
// インタラクティブトランザクション
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

  return { sender, receiver };
}, {
  timeout: 5000,          // タイムアウト
  isolationLevel: "Serializable", // 分離レベル
});
```

### 3-2. Prisma Client Extensions

```typescript
// カスタムメソッドの追加
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
    },
  },
  query: {
    user: {
      // 全クエリにソフトデリート条件を追加
      async findMany({ model, operation, args, query }) {
        args.where = { ...args.where, deletedAt: null };
        return query(args);
      },
    },
  },
});

// カスタムメソッドの使用
const user = await xprisma.user.findByEmail("alice@example.com");
```

### 3-3. 型安全な生 SQL

```typescript
// Prisma の型付き SQL（Prisma 5.x+）
import { Prisma } from "@prisma/client";

// $queryRaw でパラメータ化クエリ
const users = await prisma.$queryRaw<
  { id: string; name: string; post_count: bigint }[]
>`
  SELECT u.id, u.name, COUNT(p.id) as post_count
  FROM users u
  LEFT JOIN posts p ON u.id = p."authorId"
  WHERE u.role = ${Prisma.sql`'USER'`}
  GROUP BY u.id, u.name
  HAVING COUNT(p.id) > ${minPosts}
  ORDER BY post_count DESC
`;
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

```typescript
// インターフェース
interface IUserRepository {
  findById(id: string): Promise<User | null>;
  findByEmail(email: string): Promise<User | null>;
  findMany(params: {
    where?: Partial<User>;
    skip?: number;
    take?: number;
  }): Promise<User[]>;
  create(data: Omit<User, "id" | "createdAt" | "updatedAt">): Promise<User>;
  update(id: string, data: Partial<User>): Promise<User>;
  delete(id: string): Promise<void>;
}

// Prisma 実装
class PrismaUserRepository implements IUserRepository {
  constructor(private readonly prisma: PrismaClient) {}

  async findById(id: string): Promise<User | null> {
    return this.prisma.user.findUnique({ where: { id } });
  }

  async findByEmail(email: string): Promise<User | null> {
    return this.prisma.user.findUnique({ where: { email } });
  }

  async findMany(params: {
    where?: Partial<User>;
    skip?: number;
    take?: number;
  }): Promise<User[]> {
    return this.prisma.user.findMany(params);
  }

  async create(
    data: Omit<User, "id" | "createdAt" | "updatedAt">
  ): Promise<User> {
    return this.prisma.user.create({ data });
  }

  async update(id: string, data: Partial<User>): Promise<User> {
    return this.prisma.user.update({ where: { id }, data });
  }

  async delete(id: string): Promise<void> {
    await this.prisma.user.delete({ where: { id } });
  }
}
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

### Prisma のクエリ手法比較

| 手法 | 型安全性 | 柔軟性 | パフォーマンス | 用途 |
|------|---------|--------|-------------|------|
| findMany / findUnique | 最高 | 中 | 良好 | 標準CRUD |
| include / select | 最高 | 中 | 要注意(N+1) | リレーション |
| $queryRaw | 中 | 最高 | 最高 | 複雑なクエリ |
| $transaction | 最高 | 高 | 良好 | 複数操作 |

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
const prisma = new PrismaClient();

// 開発環境でのホットリロード対策
const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined;
};
const prisma = globalForPrisma.prisma ?? new PrismaClient();
if (process.env.NODE_ENV !== "production") {
  globalForPrisma.prisma = prisma;
}
export { prisma };
```

---

## FAQ

### Q1: Prisma と Drizzle のどちらを選ぶべきですか？

Prisma はスキーマファーストの設計が直感的で、Studio、マイグレーション、型生成が一体化しています。Drizzle はより軽量で SQL に近く、Edge Runtime との相性が良いです。チームの SQL 習熟度が高ければ Drizzle、ORM の抽象化を好むなら Prisma が適しています。

### Q2: Prisma のパフォーマンスが遅い場合の対策は？

`select` で必要なフィールドのみ取得、`include` のネストを最小化、インデックスの追加、`$queryRaw` による複雑なクエリの最適化が有効です。Prisma Accelerate（コネクションプーリング + キャッシュ）の導入も検討してください。

### Q3: テストではどうやって DB をモックしますか？

テスト用 DB（Docker の PostgreSQL）を使ったインテグレーションテストが最も信頼性が高いです。単体テストではリポジトリインターフェースのモック実装を DI で注入します。`prisma-mock` ライブラリもありますが、実 DB テストを推奨します。

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
