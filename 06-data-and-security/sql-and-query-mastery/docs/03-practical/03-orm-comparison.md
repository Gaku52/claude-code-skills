# ORM 比較 — Prisma / TypeORM / Drizzle / SQLAlchemy

> 主要 4 つの ORM を機能・パフォーマンス・開発体験の観点から比較し、プロジェクトに最適な ORM を選択するための実践ガイド。

---

## この章で学ぶこと

1. **各 ORM のアーキテクチャ** と設計思想の違い
2. **CRUD 操作の実装比較** と型安全性の差
3. **パフォーマンス特性** とスケーラビリティの違い
4. **トランザクション管理** の実装パターン
5. **マイグレーション戦略** と本番運用の考慮事項

## 前提知識

- TypeScript または Python の基礎知識
- リレーショナルデータベースの基本概念（テーブル、リレーション、SQL）
- [02-performance-tuning.md](./02-performance-tuning.md) の接続プール知識があると望ましい

---

## 1. ORM 選択の全体像

### 1.1 設計思想の違い

```
┌────────────────────────────────────────────────────────┐
│               ORM の設計思想スペクトラム                 │
│                                                        │
│  SQL に近い ←──────────────────────────→ 抽象度が高い   │
│                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ Drizzle  │  │SQLAlchemy│  │  Prisma  │  │TypeORM │ │
│  │          │  │  Core    │  │          │  │        │ │
│  │ SQL-like │  │ 表現力   │  │ 独自DSL  │  │ AR/DM  │ │
│  │ TypeSafe │  │ 最大     │  │ 型生成   │  │ デコ   │ │
│  └──────────┘  └──────────┘  └──────────┘  │ レータ │ │
│                                            └────────┘ │
│                                                        │
│  型安全 ←────────────────────────────→ 柔軟性          │
│                                                        │
│  Drizzle ≈ Prisma > SQLAlchemy > TypeORM              │
└────────────────────────────────────────────────────────┘
```

### 1.2 各 ORM の立ち位置

```
                    型安全性
                      ▲
                      │
              Prisma  │  Drizzle
                 ●    │    ●
                      │
   ──────────────────┼──────────────────→ SQL制御度
                      │
            TypeORM   │   SQLAlchemy
                ●     │      ●
                      │
```

### 1.3 ORMの内部アーキテクチャ

```
┌──────── ORM の内部動作フロー ──────────────────┐
│                                                  │
│  アプリケーションコード                           │
│  │                                              │
│  ▼                                              │
│  ┌──────────────────────────────────┐           │
│  │ ORM レイヤー                      │           │
│  │  ┌─────────────────────────┐     │           │
│  │  │ クエリビルダー           │     │           │
│  │  │ (TypeSafe API → SQL生成)│     │           │
│  │  └───────┬─────────────────┘     │           │
│  │          │                        │           │
│  │  ┌───────▼─────────────────┐     │           │
│  │  │ マッピングレイヤー       │     │           │
│  │  │ (DB行 → オブジェクト)   │     │           │
│  │  └───────┬─────────────────┘     │           │
│  │          │                        │           │
│  │  ┌───────▼─────────────────┐     │           │
│  │  │ 接続プール管理           │     │           │
│  │  │ (Connection Pooling)    │     │           │
│  │  └───────┬─────────────────┘     │           │
│  └──────────┼────────────────────────┘           │
│             │                                    │
│  ┌──────────▼──────────────────────┐             │
│  │ データベースドライバ             │             │
│  │ (pg, mysql2, better-sqlite3等) │             │
│  └──────────┬──────────────────────┘             │
│             │                                    │
│  ┌──────────▼──────────────────────┐             │
│  │ RDBMS (PostgreSQL, MySQL等)    │             │
│  └─────────────────────────────────┘             │
└──────────────────────────────────────────────────┘
```

### 1.4 ORM パターンの分類

```
┌──────── ORM デザインパターン ──────────────────┐
│                                                  │
│  Active Record (AR):                             │
│  ┌──────────────────────────────────────┐       │
│  │ モデルクラス = テーブル + CRUD操作     │       │
│  │ user.save(), user.find()              │       │
│  │ 採用: TypeORM (一部), Rails AR        │       │
│  │ 長所: シンプル、直感的                │       │
│  │ 短所: ビジネスロジックとDB操作が混在  │       │
│  └──────────────────────────────────────┘       │
│                                                  │
│  Data Mapper (DM):                               │
│  ┌──────────────────────────────────────┐       │
│  │ モデル(Entity)とDB操作(Repository)を分離│     │
│  │ repository.save(user)                 │       │
│  │ 採用: TypeORM (DM mode), SQLAlchemy   │       │
│  │ 長所: 関心の分離、テスタビリティ      │       │
│  │ 短所: コード量が多い                  │       │
│  └──────────────────────────────────────┘       │
│                                                  │
│  Query Builder (QB):                             │
│  ┌──────────────────────────────────────┐       │
│  │ SQLに近いAPI、型安全なクエリ構築      │       │
│  │ db.select().from(users).where(...)    │       │
│  │ 採用: Drizzle, Knex.js               │       │
│  │ 長所: SQL知識が直接活かせる           │       │
│  │ 短所: ORMの利便性が少ない             │       │
│  └──────────────────────────────────────┘       │
│                                                  │
│  Schema-First (SF):                              │
│  ┌──────────────────────────────────────┐       │
│  │ 独自DSLでスキーマ定義 → 型自動生成    │       │
│  │ schema.prisma → prisma generate       │       │
│  │ 採用: Prisma                          │       │
│  │ 長所: スキーマが単一の真実の源        │       │
│  │ 短所: DSL学習コスト、柔軟性の制限     │       │
│  └──────────────────────────────────────┘       │
└──────────────────────────────────────────────────┘
```

---

## 2. 各 ORM の CRUD 実装比較

### 2.1 スキーマ / モデル定義

```typescript
// === Prisma (schema.prisma) ===
model User {
  id        String   @id @default(uuid())
  name      String
  email     String   @unique
  posts     Post[]
  createdAt DateTime @default(now())
}

model Post {
  id        String   @id @default(uuid())
  title     String
  body      String
  published Boolean  @default(false)
  author    User     @relation(fields: [authorId], references: [id])
  authorId  String
}
```

```typescript
// === TypeORM (Entity デコレータ) ===
import { Entity, PrimaryGeneratedColumn, Column, OneToMany, ManyToOne } from "typeorm";

@Entity()
export class User {
  @PrimaryGeneratedColumn("uuid")
  id: string;

  @Column()
  name: string;

  @Column({ unique: true })
  email: string;

  @OneToMany(() => Post, (post) => post.author)
  posts: Post[];

  @Column({ type: "timestamp", default: () => "NOW()" })
  createdAt: Date;
}

@Entity()
export class Post {
  @PrimaryGeneratedColumn("uuid")
  id: string;

  @Column()
  title: string;

  @Column("text")
  body: string;

  @Column({ default: false })
  published: boolean;

  @ManyToOne(() => User, (user) => user.posts)
  author: User;
}
```

```typescript
// === Drizzle (TypeScript スキーマ) ===
import { pgTable, uuid, varchar, text, boolean, timestamp } from "drizzle-orm/pg-core";
import { relations } from "drizzle-orm";

export const users = pgTable("users", {
  id: uuid("id").primaryKey().defaultRandom(),
  name: varchar("name", { length: 255 }).notNull(),
  email: varchar("email", { length: 255 }).notNull().unique(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export const posts = pgTable("posts", {
  id: uuid("id").primaryKey().defaultRandom(),
  title: varchar("title", { length: 255 }).notNull(),
  body: text("body").notNull(),
  published: boolean("published").default(false).notNull(),
  authorId: uuid("author_id").notNull().references(() => users.id),
});

// リレーション定義（Drizzleのrelational query用）
export const usersRelations = relations(users, ({ many }) => ({
  posts: many(posts),
}));

export const postsRelations = relations(posts, ({ one }) => ({
  author: one(users, {
    fields: [posts.authorId],
    references: [users.id],
  }),
}));
```

```python
# === SQLAlchemy (Mapped 型注釈) ===
from sqlalchemy import String, Boolean, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from datetime import datetime
import uuid

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255))
    email: Mapped[str] = mapped_column(String(255), unique=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)

    posts: Mapped[list["Post"]] = relationship(back_populates="author")

class Post(Base):
    __tablename__ = "posts"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(255))
    body: Mapped[str]
    published: Mapped[bool] = mapped_column(default=False)
    author_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))

    author: Mapped["User"] = relationship(back_populates="posts")
```

### 2.2 SELECT（リレーション含む）

```typescript
// === Prisma ===
const usersWithPosts = await prisma.user.findMany({
  where: { email: { contains: "@example.com" } },
  include: {
    posts: {
      where: { published: true },
      orderBy: { createdAt: "desc" },
      take: 5,
    },
  },
  take: 10,
});

// Prisma の生成SQL:
// SELECT "User"."id", "User"."name", "User"."email", "User"."createdAt"
// FROM "User"
// WHERE "User"."email" LIKE '%@example.com%'
// LIMIT 10;
//
// SELECT "Post"."id", "Post"."title", ...
// FROM "Post"
// WHERE "Post"."authorId" IN ($1, $2, ...) AND "Post"."published" = true
// ORDER BY "Post"."createdAt" DESC
// LIMIT 5;
// → 2クエリで実行（N+1ではない）

// === TypeORM ===
const usersWithPosts = await userRepository.find({
  where: { email: Like("%@example.com") },
  relations: { posts: true },
  take: 10,
});
// TypeORM ではリレーションのフィルタは QueryBuilder が必要
const usersWithPosts2 = await userRepository
  .createQueryBuilder("user")
  .leftJoinAndSelect("user.posts", "post", "post.published = :pub", { pub: true })
  .where("user.email LIKE :email", { email: "%@example.com" })
  .orderBy("post.createdAt", "DESC")
  .take(10)
  .getMany();

// === Drizzle ===
const usersWithPosts = await db.query.users.findMany({
  where: like(users.email, "%@example.com"),
  with: {
    posts: {
      where: eq(posts.published, true),
      orderBy: desc(posts.createdAt),
      limit: 5,
    },
  },
  limit: 10,
});

// Drizzle のSQL-likeクエリ構築（低レベルAPI）
const result = await db
  .select({
    userName: users.name,
    postTitle: posts.title,
  })
  .from(users)
  .leftJoin(posts, eq(users.id, posts.authorId))
  .where(like(users.email, "%@example.com"))
  .limit(10);
```

```python
# === SQLAlchemy ===
from sqlalchemy import select
from sqlalchemy.orm import selectinload

stmt = (
    select(User)
    .where(User.email.contains("@example.com"))
    .options(
        selectinload(User.posts).where(Post.published == True)
    )
    .limit(10)
)
users_with_posts = session.scalars(stmt).all()

# SQLAlchemy のローディング戦略比較
# 1. Lazy Loading（デフォルト）: アクセス時にクエリ → N+1問題の原因
# 2. Eager Loading (joinedload): JOINで一括取得
# 3. Subquery Loading (selectinload): サブクエリで一括取得（推奨）
# 4. Raise Loading (raiseload): アクセス時にエラー → N+1を検出

stmt_joined = (
    select(User)
    .options(joinedload(User.posts))  # LEFT JOINで取得
    .limit(10)
)

stmt_subquery = (
    select(User)
    .options(selectinload(User.posts))  # 別クエリで取得（推奨）
    .limit(10)
)

# raiseload: 明示的にロードしないとエラー（N+1検出用）
from sqlalchemy.orm import raiseload
stmt_strict = (
    select(User)
    .options(raiseload(User.posts))  # user.postsアクセスでエラー
)
```

### 2.3 INSERT（バルク）

```typescript
// === Prisma ===
const users = await prisma.user.createMany({
  data: [
    { name: "Alice", email: "alice@example.com" },
    { name: "Bob", email: "bob@example.com" },
  ],
  skipDuplicates: true,
});

// Prisma: ネストされたリレーション作成
const userWithPosts = await prisma.user.create({
  data: {
    name: "Charlie",
    email: "charlie@example.com",
    posts: {
      create: [
        { title: "First Post", body: "Hello World", published: true },
        { title: "Draft", body: "WIP", published: false },
      ],
    },
  },
  include: { posts: true },
});

// === Drizzle ===
const inserted = await db.insert(users).values([
  { name: "Alice", email: "alice@example.com" },
  { name: "Bob", email: "bob@example.com" },
]).onConflictDoNothing().returning();

// Drizzle: UPSERT (ON CONFLICT)
const upserted = await db.insert(users).values({
  name: "Alice",
  email: "alice@example.com",
}).onConflictDoUpdate({
  target: users.email,
  set: { name: "Alice Updated" },
}).returning();

// === TypeORM ===
const result = await userRepository.insert([
  { name: "Alice", email: "alice@example.com" },
  { name: "Bob", email: "bob@example.com" },
]);
// TypeORM: UPSERT
await userRepository.upsert(
  { name: "Alice", email: "alice@example.com" },
  ["email"]
);
```

```python
# === SQLAlchemy ===
session.add_all([
    User(name="Alice", email="alice@example.com"),
    User(name="Bob", email="bob@example.com"),
])
session.commit()

# SQLAlchemy: バルクINSERT（高速版）
from sqlalchemy import insert
stmt = insert(User).values([
    {"name": "Alice", "email": "alice@example.com"},
    {"name": "Bob", "email": "bob@example.com"},
])
session.execute(stmt)
session.commit()

# SQLAlchemy: UPSERT (PostgreSQL)
from sqlalchemy.dialects.postgresql import insert as pg_insert
stmt = pg_insert(User).values(name="Alice", email="alice@example.com")
stmt = stmt.on_conflict_do_update(
    index_elements=["email"],
    set_={"name": stmt.excluded.name},
)
session.execute(stmt)
session.commit()
```

### 2.4 トランザクション管理

```typescript
// === Prisma: Interactive Transaction ===
const result = await prisma.$transaction(async (tx) => {
  const user = await tx.user.create({
    data: { name: "Alice", email: "alice@example.com" },
  });
  const post = await tx.post.create({
    data: { title: "Hello", body: "World", authorId: user.id },
  });
  return { user, post };
});
// 例外が発生すると自動ロールバック

// Prisma: Sequential Transaction（複数操作をアトミックに）
const [user, post] = await prisma.$transaction([
  prisma.user.create({ data: { name: "Bob", email: "bob@example.com" } }),
  prisma.post.create({ data: { title: "Hi", body: "!", authorId: "..." } }),
]);

// === Drizzle ===
const result = await db.transaction(async (tx) => {
  const [user] = await tx.insert(users).values({
    name: "Alice", email: "alice@example.com"
  }).returning();

  await tx.insert(posts).values({
    title: "Hello", body: "World", authorId: user.id
  });

  return user;
});

// === TypeORM: QueryRunner ===
const queryRunner = dataSource.createQueryRunner();
await queryRunner.startTransaction();
try {
  const user = queryRunner.manager.create(User, {
    name: "Alice", email: "alice@example.com"
  });
  await queryRunner.manager.save(user);

  const post = queryRunner.manager.create(Post, {
    title: "Hello", body: "World", author: user
  });
  await queryRunner.manager.save(post);

  await queryRunner.commitTransaction();
} catch (err) {
  await queryRunner.rollbackTransaction();
  throw err;
} finally {
  await queryRunner.release();
}
```

```python
# === SQLAlchemy: Session ===
# 方法1: コンテキストマネージャ（推奨）
with Session(engine) as session:
    with session.begin():  # 自動commit/rollback
        user = User(name="Alice", email="alice@example.com")
        session.add(user)
        session.flush()  # IDを取得するためにflush

        post = Post(title="Hello", body="World", author=user)
        session.add(post)
    # ブロック終了時に自動commit

# 方法2: 明示的なcommit/rollback
session = Session(engine)
try:
    user = User(name="Alice", email="alice@example.com")
    session.add(user)
    session.flush()

    post = Post(title="Hello", body="World", author=user)
    session.add(post)
    session.commit()
except Exception:
    session.rollback()
    raise
finally:
    session.close()

# 方法3: ネストトランザクション（SAVEPOINT）
with Session(engine) as session:
    with session.begin():
        session.add(User(name="Alice", email="alice@example.com"))

        # ネストトランザクション（SAVEPOINT）
        with session.begin_nested():
            try:
                session.add(User(name="Alice", email="alice@example.com"))  # 重複
                session.flush()
            except IntegrityError:
                pass  # SAVEPOINTにロールバック、外側は続行

        session.add(User(name="Bob", email="bob@example.com"))
    # Alice + Bob がコミット（重複Aliceはロールバック済み）
```

---

## 3. マイグレーション比較

```
┌────────────────────────────────────────────────────────┐
│              マイグレーション方式                        │
│                                                        │
│  Prisma:    スキーマファイル → prisma migrate dev       │
│             → SQL を自動生成 → prisma migrate deploy    │
│             ※ 宣言的 (Desired State)                    │
│                                                        │
│  TypeORM:   Entity の変更を検知                         │
│             → typeorm migration:generate               │
│             → typeorm migration:run                    │
│             ※ synchronize:true は本番禁止               │
│                                                        │
│  Drizzle:   スキーマファイル → drizzle-kit generate     │
│             → drizzle-kit migrate                      │
│             ※ Prisma に近い宣言的アプローチ              │
│                                                        │
│  SQLAlchemy: Alembic を使用                             │
│             → alembic revision --autogenerate           │
│             → alembic upgrade head                     │
│             ※ 自動生成 + 手動調整                       │
└────────────────────────────────────────────────────────┘
```

### 3.1 マイグレーション実行例

```bash
# === Prisma ===
# 開発環境: スキーマの差分からマイグレーション生成
npx prisma migrate dev --name add_user_avatar

# 本番環境: マイグレーション適用のみ（生成はしない）
npx prisma migrate deploy

# スキーマの状態をDBにプッシュ（プロトタイプ用、マイグレーションなし）
npx prisma db push

# === Drizzle ===
# マイグレーションファイル生成
npx drizzle-kit generate

# マイグレーション適用
npx drizzle-kit migrate

# === TypeORM ===
# マイグレーション生成（Entityの差分を検知）
npx typeorm migration:generate -n AddUserAvatar

# マイグレーション適用
npx typeorm migration:run

# === SQLAlchemy + Alembic ===
# 初期化
alembic init alembic

# マイグレーション生成（自動検知）
alembic revision --autogenerate -m "add user avatar"

# マイグレーション適用
alembic upgrade head

# ロールバック
alembic downgrade -1
```

### マイグレーション機能比較

| 機能 | Prisma | TypeORM | Drizzle | Alembic (SQLAlchemy) |
|------|--------|---------|---------|---------------------|
| 自動生成 | ✓（スキーマ差分） | ✓（Entity差分） | ✓（スキーマ差分） | ✓（モデル差分） |
| ロールバック | ✗（手動） | ✓ | ✗（手動） | ✓ |
| シード | prisma db seed | 手動 | 手動 | 手動 |
| マルチDB | ✗ | ✓ | ✗ | ✓ |
| ベースライン | prisma migrate resolve | 手動 | 手動 | alembic stamp |
| SQL確認 | ✓（自動保存） | ✗ | ✓（自動保存） | ✓（--sql） |
| チーム運用 | 良好（ロックファイル） | 注意必要 | 良好 | 良好（ブランチ対応） |

---

## 4. 比較表

### 4.1 機能比較

| 機能 | Prisma | TypeORM | Drizzle | SQLAlchemy |
|------|--------|---------|---------|------------|
| **言語** | TypeScript/JS | TypeScript/JS | TypeScript/JS | Python |
| **パラダイム** | 独自 DSL + 型生成 | ActiveRecord / DataMapper | SQL-like TypeSafe | DataMapper (Unit of Work) |
| **型安全性** | 高（自動生成） | 中（デコレータ依存） | 高（推論ベース） | 中（Mapped 型注釈） |
| **生SQL** | `$queryRaw` | `query()` | `sql` テンプレート | `text()` |
| **リレーション** | `include` / `select` | `relations` / QueryBuilder | `with` (関係クエリ) | `relationship` + Loading Strategy |
| **マイグレーション** | Prisma Migrate | TypeORM CLI / synchronize | drizzle-kit | Alembic |
| **接続プール** | 内蔵（Rust Engine） | 内蔵 | 外部ドライバ依存 | 内蔵 (QueuePool) |
| **対応DB** | PostgreSQL, MySQL, SQLite, MongoDB | PostgreSQL, MySQL, SQLite, etc. | PostgreSQL, MySQL, SQLite | ほぼ全 RDBMS |
| **トランザクション** | `$transaction` | QueryRunner / デコレータ | `db.transaction()` | Session / begin() |
| **N+1 対策** | include で自動 | eager: true / QueryBuilder | with で自動 | selectinload / joinedload |
| **バッチ操作** | createMany, updateMany | insert, update (QueryBuilder) | insert, update (バッチ) | bulk_insert_mappings |
| **UPSERT** | upsert (4.0+) | upsert | onConflictDoUpdate | on_conflict_do_update |
| **サブクエリ** | 限定的 | QueryBuilder で可能 | SQL template | 完全対応 |
| **ウィンドウ関数** | $queryRaw のみ | QueryBuilder | sql template | over() で対応 |

### 4.2 開発体験比較

| 観点 | Prisma | TypeORM | Drizzle | SQLAlchemy |
|------|--------|---------|---------|------------|
| **学習コスト** | 低 | 中 | 低 | 高 |
| **ドキュメント** | 優秀 | 良好 | 良好 | 優秀 |
| **エラーメッセージ** | 明確 | 不明瞭な場合あり | 明確 | 詳細 |
| **IDE 補完** | 優秀（型生成） | 良好 | 優秀（型推論） | 良好（型注釈） |
| **デバッグ** | Prisma Studio | なし（外部ツール） | Drizzle Studio | SQLAlchemy echo |
| **バンドルサイズ** | 大（Rust エンジン ~15MB） | 中 (~3MB) | 小 (~500KB) | N/A（サーバー） |
| **コミュニティ** | 大 | 大（やや停滞） | 急成長中 | 巨大 |
| **本番実績** | 多い | 多い | 増加中 | 非常に多い |
| **テスタビリティ** | モック可能 | Repository パターン | 関数ベース | Session モック |

### 4.3 パフォーマンス特性比較

| 指標 | Prisma | TypeORM | Drizzle | SQLAlchemy |
|------|--------|---------|---------|------------|
| **Cold Start** | 遅い（Rust Engine起動） | 普通 | 速い | 普通 |
| **クエリ生成速度** | 速い（Rust） | 普通 | 速い | 普通 |
| **メモリ使用量** | 多い（Engine分） | 普通 | 少ない | 普通 |
| **接続プール効率** | 良好（内蔵） | 良好（内蔵） | ドライバ依存 | 優秀（QueuePool） |
| **バルクINSERT** | 良好（createMany） | やや遅い | 速い | 速い（Core API） |
| **大量SELECT** | 良好 | 普通 | 速い | 良好（yield_per） |
| **サーバーレス適性** | 中（Cold Start問題） | 良好 | 優秀 | 普通 |

---

## 5. 高度な使用パターン

### 5.1 生SQLの実行

```typescript
// === Prisma: $queryRaw ===
const users = await prisma.$queryRaw<User[]>`
  SELECT u.*, COUNT(p.id) as post_count
  FROM "User" u
  LEFT JOIN "Post" p ON p."authorId" = u.id
  GROUP BY u.id
  HAVING COUNT(p.id) > ${minPosts}
  ORDER BY post_count DESC
`;

// === Drizzle: sql テンプレート ===
import { sql } from "drizzle-orm";

const result = await db.execute(sql`
  SELECT ${users.name}, COUNT(${posts.id}) as post_count
  FROM ${users}
  LEFT JOIN ${posts} ON ${posts.authorId} = ${users.id}
  GROUP BY ${users.name}
  HAVING COUNT(${posts.id}) > ${minPosts}
`);

// === TypeORM: Query ===
const result = await dataSource.query(
  `SELECT u.*, COUNT(p.id) as post_count
   FROM "user" u
   LEFT JOIN "post" p ON p."authorId" = u.id
   GROUP BY u.id
   HAVING COUNT(p.id) > $1`,
  [minPosts]
);
```

```python
# === SQLAlchemy: text() ===
from sqlalchemy import text

stmt = text("""
    SELECT u.*, COUNT(p.id) as post_count
    FROM users u
    LEFT JOIN posts p ON p.author_id = u.id
    GROUP BY u.id
    HAVING COUNT(p.id) > :min_posts
""")
result = session.execute(stmt, {"min_posts": min_posts}).all()

# SQLAlchemy: ハイブリッド（Core + ORM）
from sqlalchemy import func, select

stmt = (
    select(User, func.count(Post.id).label("post_count"))
    .outerjoin(Post)
    .group_by(User.id)
    .having(func.count(Post.id) > min_posts)
    .order_by(func.count(Post.id).desc())
)
results = session.execute(stmt).all()
```

### 5.2 複雑なクエリパターン

```typescript
// === Drizzle: サブクエリ ===
import { sql, eq, gt, and } from "drizzle-orm";

// 各部署の平均給与以上の社員
const deptAvg = db
  .select({
    deptId: employees.departmentId,
    avgSalary: sql`AVG(${employees.salary})`.as("avg_salary"),
  })
  .from(employees)
  .groupBy(employees.departmentId)
  .as("dept_avg");

const result = await db
  .select()
  .from(employees)
  .innerJoin(deptAvg, eq(employees.departmentId, deptAvg.deptId))
  .where(gt(employees.salary, deptAvg.avgSalary));

// === Prisma: fluent API ===
// Prismaでは複雑なサブクエリは$queryRawが必要
const result = await prisma.$queryRaw`
  SELECT e.*
  FROM employees e
  INNER JOIN (
    SELECT department_id, AVG(salary) as avg_salary
    FROM employees GROUP BY department_id
  ) da ON e.department_id = da.department_id
  WHERE e.salary > da.avg_salary
`;
```

```python
# === SQLAlchemy: 複雑なサブクエリ ===
from sqlalchemy import select, func

# 各部署の平均給与以上の社員
dept_avg = (
    select(
        Employee.department_id,
        func.avg(Employee.salary).label("avg_salary")
    )
    .group_by(Employee.department_id)
    .subquery()
)

stmt = (
    select(Employee)
    .join(dept_avg, Employee.department_id == dept_avg.c.department_id)
    .where(Employee.salary > dept_avg.c.avg_salary)
)
results = session.scalars(stmt).all()

# SQLAlchemy: ウィンドウ関数
from sqlalchemy import over

stmt = (
    select(
        Employee.name,
        Employee.salary,
        func.rank().over(
            partition_by=Employee.department_id,
            order_by=Employee.salary.desc()
        ).label("salary_rank")
    )
)
```

---

## 6. アンチパターン

### 6.1 TypeORM の synchronize: true を本番で使う

```typescript
// NG: 本番でスキーマ自動同期
const dataSource = new DataSource({
  type: "postgres",
  synchronize: true,  // ← 本番で絶対NG！
  // テーブルが自動変更される → データ消失リスク
});

// OK: マイグレーションファイルで明示的に管理
const dataSource = new DataSource({
  type: "postgres",
  synchronize: false,
  migrations: ["dist/migrations/*.js"],
  migrationsRun: true,  // 起動時にマイグレーション実行
});
```

**問題点**: synchronize はテーブルを削除して再作成する場合があり、本番データが消失する。開発環境でのみ使用し、本番ではマイグレーションファイルで管理する。

### 6.2 Prisma で N+1 を発生させる

```typescript
// NG: ループ内で個別クエリ
const users = await prisma.user.findMany();
for (const user of users) {
  // ユーザーごとに1クエリ = N+1問題!
  const posts = await prisma.post.findMany({
    where: { authorId: user.id },
  });
  console.log(`${user.name}: ${posts.length} posts`);
}

// OK: include で1回のクエリに統合
const users = await prisma.user.findMany({
  include: {
    posts: {
      select: { id: true },  // 必要なフィールドだけ
    },
  },
});
for (const user of users) {
  console.log(`${user.name}: ${user.posts.length} posts`);
}
```

### 6.3 SQLAlchemy の暗黙的遅延ロード

```python
# NG: async コンテキストでの遅延ロード（SQLAlchemy 2.0）
async def get_users():
    async with async_session() as session:
        result = await session.execute(select(User))
        users = result.scalars().all()

    # セッション外でリレーションアクセス → エラー or N+1
    for user in users:
        print(user.posts)  # DetachedInstanceError or lazy load

# OK: 明示的にローディング戦略を指定
async def get_users_with_posts():
    async with async_session() as session:
        result = await session.execute(
            select(User).options(selectinload(User.posts))
        )
        users = result.scalars().all()
        for user in users:
            print(user.posts)  # 既にロード済み
```

### 6.4 Drizzle での型安全性の無視

```typescript
// NG: sql テンプレートで型を無視
const result = await db.execute(sql`SELECT * FROM users WHERE id = ${userId}`);
// result の型は不明確

// OK: 型安全なクエリビルダーを使用
const result = await db
  .select()
  .from(users)
  .where(eq(users.id, userId));
// result の型は推論される: { id: string, name: string, ... }[]
```

---

## 7. エッジケース

### エッジケース1: 大量データのストリーミング

```python
# SQLAlchemy: yield_per でメモリ効率的な大量読み取り
stmt = select(User).execution_options(yield_per=1000)
for user in session.scalars(stmt):
    process(user)
# → 1000行ずつDBから取得（全行をメモリに載せない）
```

```typescript
// Prisma: カーソルベースのページネーション
let cursor: string | undefined;
while (true) {
  const users = await prisma.user.findMany({
    take: 100,
    ...(cursor ? { skip: 1, cursor: { id: cursor } } : {}),
    orderBy: { id: "asc" },
  });
  if (users.length === 0) break;
  cursor = users[users.length - 1].id;
  for (const user of users) {
    await process(user);
  }
}
```

### エッジケース2: マルチテナント

```typescript
// Prisma: RLSとクライアント拡張
const prismaWithTenant = prisma.$extends({
  query: {
    $allModels: {
      async $allOperations({ args, query }) {
        // 全クエリにテナントフィルタを自動追加
        args.where = { ...args.where, tenantId: currentTenantId };
        return query(args);
      },
    },
  },
});
```

```python
# SQLAlchemy: イベントフックでマルチテナント
from sqlalchemy import event

@event.listens_for(Session, "do_orm_execute")
def _add_tenant_filter(execute_state):
    if execute_state.is_select:
        execute_state.statement = execute_state.statement.where(
            User.tenant_id == get_current_tenant_id()
        )
```

### エッジケース3: 楽観的ロック

```typescript
// Prisma: バージョンフィールドで楽観的ロック
const updated = await prisma.product.update({
  where: { id: productId, version: currentVersion },
  data: { name: "New Name", version: { increment: 1 } },
});
// version が一致しない場合は RecordNotFoundError
```

```python
# SQLAlchemy: version_id_col で楽観的ロック
class Product(Base):
    __tablename__ = "products"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    version: Mapped[int] = mapped_column(default=1)

    __mapper_args__ = {
        "version_id_col": version,
    }

# 更新時にバージョン不一致でStaleDataError
product = session.get(Product, product_id)
product.name = "New Name"
session.commit()  # UPDATE ... WHERE id = ? AND version = ?
```

---

## 8. 演習

### 演習1（基礎）: CRUD操作の実装

以下の要件を、好みのORMで実装せよ。

**要件**:
- ユーザー（name, email）と投稿（title, body, published）のCRUD
- 未公開の投稿を持つユーザー一覧を取得するクエリ
- ユーザー削除時に投稿もカスケード削除

### 演習2（応用）: N+1問題の検出と修正

以下のコードのN+1問題を特定し、修正せよ。

```typescript
// 問題のあるコード
const departments = await prisma.department.findMany();
for (const dept of departments) {
  const employees = await prisma.employee.findMany({
    where: { departmentId: dept.id },
  });
  const avgSalary = employees.reduce((sum, e) => sum + e.salary, 0) / employees.length;
  console.log(`${dept.name}: avg salary = ${avgSalary}`);
}
```

### 演習3（発展）: パフォーマンス比較

同じクエリ（1000ユーザーの一覧 + 各ユーザーの投稿数）を4つのORM全てで実装し、実行時間とメモリ使用量を比較せよ。

---

## 9. FAQ

### Q1. 新規プロジェクトではどの ORM を選ぶべき？

**A.** 言語とチームの経験で選ぶのが最良。

- **TypeScript + 型安全重視**: Drizzle（SQL に近く、型推論が優秀）
- **TypeScript + 生産性重視**: Prisma（スキーマファースト、ドキュメント充実）
- **TypeScript + レガシープロジェクト**: TypeORM（既存の Express/NestJS との統合実績）
- **Python**: SQLAlchemy（デファクトスタンダード、2.0 で大幅改善）

### Q2. Prisma の Rust エンジンはパフォーマンスに影響する？

**A.** Prisma は Query Engine として Rust バイナリを使用する。これにより初回起動が遅く（Cold Start）、バンドルサイズが大きい（約 15MB）。サーバーレス環境（Lambda）では Cold Start が問題になる場合がある。Prisma 6.x 以降では軽量化が進んでいるが、Lambda で問題が出る場合は Drizzle や直接 SQL（sqlc 等）を検討する。

### Q3. SQLAlchemy 1.x から 2.0 への移行は大変？

**A.** 段階的な移行が可能。SQLAlchemy 1.4 が橋渡しバージョンで、`future=True` フラグで 2.0 スタイルを段階的に導入できる。主な変更点は:
- `session.query()` → `select()` ステートメント
- `Column` → `mapped_column`
- 暗黙的な遅延読み込み → 明示的なローディング戦略

移行ガイドが公式ドキュメントに用意されており、`SQLALCHEMY_WARN_20=1` で非推奨警告を有効にして段階的に対応できる。

### Q4. ORMを使わずに直接SQLを書くべきケースは？

**A.** 以下の場合はORMより直接SQLが適している:
- **極めて複雑なクエリ**（再帰CTE、ウィンドウ関数の組み合わせ等）
- **バッチ処理**（数百万行のバルク操作）
- **パフォーマンスが最重要**（マイクロ秒単位の最適化）
- **DBスペシフィックな機能**（PostgreSQLのLISTEN/NOTIFY等）
この場合でも、sqlc（Go）やkysely（TypeScript）のような型安全なSQLツールの利用を推奨する。

### Q5. ORM間の移行はどの程度大変か？

**A.** ORM間の移行は一般的に大きなコストがかかる。リポジトリパターンを採用していれば、データアクセス層のみの変更で済む。直接ORMのAPIを呼んでいる場合はビジネスロジック層まで影響が及ぶ。移行コストを最小化するには、ORMのAPIをビジネスロジックから分離するアーキテクチャ（リポジトリパターン、DAO パターン）を採用すべき。

### Q6. テスト環境でのORMの使い方は？

**A.** 各ORMのテスト戦略:
- **Prisma**: `prisma migrate reset` でテストDB初期化、またはモックライブラリ使用
- **Drizzle**: テスト用のSQLite接続を使用、またはpg-memなどのインメモリDB
- **TypeORM**: `synchronize: true` でテストDB自動生成、`dropDatabase` で初期化
- **SQLAlchemy**: テストごとにトランザクション作成 → ロールバックでクリーンアップ

---

## 10. トラブルシューティング

| 問題 | ORM | 原因 | 対処法 |
|------|-----|------|--------|
| Cold Startが遅い | Prisma | Rustエンジンの起動 | Prisma Accelerate、またはDrizzleへの移行 |
| N+1クエリ | 全ORM | リレーションの遅延ロード | include/with/selectinloadで明示的にロード |
| マイグレーション競合 | TypeORM | synchronize使用 | マイグレーションファイルに切り替え |
| メモリリーク | SQLAlchemy | Sessionの未クローズ | コンテキストマネージャを使用 |
| 型エラー | TypeORM | デコレータの不足 | strict: true + experimental decorators |
| 接続枯渇 | 全ORM | プールサイズ不足 | max接続数の調整、pgBouncer導入 |

---

## 11. セキュリティ考慮事項

```
┌──────── ORM セキュリティチェックリスト ────────┐
│                                                  │
│  1. SQLインジェクション防止                      │
│     ✓ ORM のクエリビルダーを使用                │
│     ✓ 生SQLはパラメータバインドを使用           │
│     ✗ 文字列連結でSQLを構築しない              │
│                                                  │
│  2. 接続文字列の管理                            │
│     ✓ 環境変数で管理                            │
│     ✓ シークレットマネージャを使用              │
│     ✗ ソースコードにハードコードしない          │
│                                                  │
│  3. 権限の最小化                                │
│     ✓ アプリ用DBユーザーにはSELECT/INSERT/      │
│       UPDATE/DELETE のみ付与                     │
│     ✗ スーパーユーザーで接続しない              │
│                                                  │
│  4. マイグレーションの権限分離                   │
│     ✓ マイグレーション用とアプリ用のDB          │
│       ユーザーを分離                            │
│     ✓ DDL権限はマイグレーション用のみ           │
└──────────────────────────────────────────────────┘
```

---

## 12. まとめ

| 項目 | ポイント |
|------|---------|
| **Prisma** | スキーマファースト、型安全、Rust エンジン、学習コスト低 |
| **TypeORM** | デコレータベース、NestJS 統合、機能豊富だがメンテナンス停滞気味 |
| **Drizzle** | SQL-like、型推論、軽量、急成長中、サーバーレス最適 |
| **SQLAlchemy** | Python のデファクト、表現力最大、2.0 で型安全性向上 |
| **選定基準** | 言語 → チーム経験 → 型安全性 → パフォーマンス要件 |
| **N+1対策** | 全ORMで明示的なローディングが必要 |
| **マイグレーション** | 本番では必ずファイルベースのマイグレーション管理 |
| **テスト** | リポジトリパターンでORM依存を分離 |

---

## 次に読むべきガイド

- [02-performance-tuning.md](./02-performance-tuning.md) — 接続プールとキャッシュの最適化
- マイグレーション運用ガイド — 本番環境でのスキーマ変更戦略
- N+1 問題完全ガイド — ORM ごとの対策パターン

---

## 参考文献

1. **Prisma 公式ドキュメント** — https://www.prisma.io/docs
2. **Drizzle ORM 公式ドキュメント** — https://orm.drizzle.team/docs/overview
3. **SQLAlchemy 公式ドキュメント** — "SQLAlchemy 2.0 Tutorial" — https://docs.sqlalchemy.org/en/20/tutorial/
4. **TypeORM 公式ドキュメント** — https://typeorm.io/
5. Fowler, M. (2002). *Patterns of Enterprise Application Architecture*. Addison-Wesley. (Active Record, Data Mapper パターン)
6. **HikariCP Wiki** — "About Pool Sizing" — https://github.com/brettwooldridge/HikariCP/wiki/About-Pool-Sizing
