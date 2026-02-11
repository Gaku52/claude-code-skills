# ORM 比較 — Prisma / TypeORM / Drizzle / SQLAlchemy

> 主要 4 つの ORM を機能・パフォーマンス・開発体験の観点から比較し、プロジェクトに最適な ORM を選択するための実践ガイド。

---

## この章で学ぶこと

1. **各 ORM のアーキテクチャ** と設計思想の違い
2. **CRUD 操作の実装比較** と型安全性の差
3. **パフォーマンス特性** とスケーラビリティの違い

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

// === Drizzle ===
const inserted = await db.insert(users).values([
  { name: "Alice", email: "alice@example.com" },
  { name: "Bob", email: "bob@example.com" },
]).onConflictDoNothing().returning();
```

```python
# === SQLAlchemy ===
session.add_all([
    User(name="Alice", email="alice@example.com"),
    User(name="Bob", email="bob@example.com"),
])
session.commit()
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
| **接続プール** | 内蔵 | 内蔵 | 外部ドライバ依存 | 内蔵 (QueuePool) |
| **対応DB** | PostgreSQL, MySQL, SQLite, MongoDB | PostgreSQL, MySQL, SQLite, etc. | PostgreSQL, MySQL, SQLite | ほぼ全 RDBMS |
| **トランザクション** | `$transaction` | QueryRunner / デコレータ | `db.transaction()` | Session / begin() |
| **N+1 対策** | include で自動 | eager: true / QueryBuilder | with で自動 | selectinload / joinedload |

### 4.2 開発体験比較

| 観点 | Prisma | TypeORM | Drizzle | SQLAlchemy |
|------|--------|---------|---------|------------|
| **学習コスト** | 低 | 中 | 低 | 高 |
| **ドキュメント** | 優秀 | 良好 | 良好 | 優秀 |
| **エラーメッセージ** | 明確 | 不明瞭な場合あり | 明確 | 詳細 |
| **IDE 補完** | 優秀（型生成） | 良好 | 優秀（型推論） | 良好（型注釈） |
| **デバッグ** | Prisma Studio | なし（外部ツール） | Drizzle Studio | SQLAlchemy echo |
| **バンドルサイズ** | 大（Rust エンジン） | 中 | 小 | N/A（サーバー） |
| **コミュニティ** | 大 | 大（やや停滞） | 急成長中 | 巨大 |
| **本番実績** | 多い | 多い | 増加中 | 非常に多い |

---

## 5. アンチパターン

### 5.1 TypeORM の synchronize: true を本番で使う

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

### 5.2 Prisma で N+1 を発生させる

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

---

## 6. FAQ

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

---

## 7. まとめ

| 項目 | ポイント |
|------|---------|
| **Prisma** | スキーマファースト、型安全、Rust エンジン、学習コスト低 |
| **TypeORM** | デコレータベース、NestJS 統合、機能豊富だがメンテナンス停滞気味 |
| **Drizzle** | SQL-like、型推論、軽量、急成長中 |
| **SQLAlchemy** | Python のデファクト、表現力最大、2.0 で型安全性向上 |
| **選定基準** | 言語 → チーム経験 → 型安全性 → パフォーマンス要件 |

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
