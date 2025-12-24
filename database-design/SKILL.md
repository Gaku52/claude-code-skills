---
name: database-design
description: データベース設計ガイド。正規化、インデックス、リレーション設計、SQL最適化、マイグレーション、Prisma/TypeORMなど、効率的なデータベース設計のベストプラクティス。
---

# Database Design Skill

## 📋 目次

1. [概要](#概要)
2. [いつ使うか](#いつ使うか)
3. [スキーマ設計](#スキーマ設計)
4. [正規化](#正規化)
5. [インデックス](#インデックス)
6. [リレーション](#リレーション)
7. [Prisma](#prisma)
8. [SQL最適化](#sql最適化)
9. [実践例](#実践例)
10. [Agent連携](#agent連携)

---

## 概要

このSkillは、データベース設計をカバーします：

- **スキーマ設計** - テーブル構造、データ型
- **正規化** - 第1〜第3正規形
- **インデックス** - パフォーマンス最適化
- **リレーション** - 1対多、多対多
- **マイグレーション** - Prisma, TypeORM
- **SQL最適化** - クエリ最適化

---

## いつ使うか

### 🎯 必須のタイミング

- [ ] 新規プロジェクト開始時（スキーマ設計）
- [ ] テーブル追加時
- [ ] パフォーマンス問題発生時（インデックス追加）
- [ ] データモデル変更時（マイグレーション）

---

## スキーマ設計

### 基本原則

1. **主キー（Primary Key）** - 全テーブルに必須
2. **外部キー（Foreign Key）** - リレーション定義
3. **NOT NULL制約** - 必須フィールド
4. **UNIQUE制約** - 重複禁止
5. **デフォルト値** - created_at, updated_at等

### ユーザーテーブル例

```sql
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR(100) NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  password_hash VARCHAR(255) NOT NULL,
  role VARCHAR(20) DEFAULT 'user',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_users_email ON users(email);
```

### データ型選択

| データ型 | 用途 | 例 |
|---------|------|-----|
| **UUID** | ID（PostgreSQL） | `id UUID` |
| **VARCHAR(n)** | 可変長文字列 | `name VARCHAR(100)` |
| **TEXT** | 長文 | `content TEXT` |
| **INTEGER** | 整数 | `age INTEGER` |
| **DECIMAL(p,s)** | 金額 | `price DECIMAL(10,2)` |
| **BOOLEAN** | 真偽値 | `is_active BOOLEAN` |
| **TIMESTAMP** | 日時 | `created_at TIMESTAMP` |
| **JSON/JSONB** | JSON（PostgreSQL） | `metadata JSONB` |

---

## 正規化

### 第1正規形（1NF）

**ルール：** 各セルに1つの値のみ

```sql
-- ❌ 悪い例（複数の値）
CREATE TABLE users (
  id INT,
  name VARCHAR(100),
  hobbies VARCHAR(255) -- "Reading, Gaming, Cooking"
);

-- ✅ 良い例（別テーブルに分離）
CREATE TABLE users (
  id INT PRIMARY KEY,
  name VARCHAR(100)
);

CREATE TABLE hobbies (
  id INT PRIMARY KEY,
  user_id INT REFERENCES users(id),
  hobby VARCHAR(100)
);
```

### 第2正規形（2NF）

**ルール：** 非キー属性が主キー全体に依存

```sql
-- ❌ 悪い例
CREATE TABLE order_items (
  order_id INT,
  product_id INT,
  product_name VARCHAR(100),  -- product_idのみに依存
  product_price DECIMAL(10,2), -- product_idのみに依存
  quantity INT,
  PRIMARY KEY (order_id, product_id)
);

-- ✅ 良い例
CREATE TABLE products (
  id INT PRIMARY KEY,
  name VARCHAR(100),
  price DECIMAL(10,2)
);

CREATE TABLE order_items (
  order_id INT,
  product_id INT REFERENCES products(id),
  quantity INT,
  PRIMARY KEY (order_id, product_id)
);
```

### 第3正規形（3NF）

**ルール：** 非キー属性が他の非キー属性に依存しない

```sql
-- ❌ 悪い例
CREATE TABLE employees (
  id INT PRIMARY KEY,
  name VARCHAR(100),
  department_id INT,
  department_name VARCHAR(100) -- department_idに依存
);

-- ✅ 良い例
CREATE TABLE departments (
  id INT PRIMARY KEY,
  name VARCHAR(100)
);

CREATE TABLE employees (
  id INT PRIMARY KEY,
  name VARCHAR(100),
  department_id INT REFERENCES departments(id)
);
```

---

## インデックス

### インデックスの種類

```sql
-- B-Treeインデックス（デフォルト）
CREATE INDEX idx_users_email ON users(email);

-- ユニークインデックス
CREATE UNIQUE INDEX idx_users_email ON users(email);

-- 複合インデックス
CREATE INDEX idx_posts_user_created ON posts(user_id, created_at);

-- 部分インデックス（PostgreSQL）
CREATE INDEX idx_active_users ON users(email) WHERE is_active = true;

-- GINインデックス（全文検索、PostgreSQL）
CREATE INDEX idx_posts_content ON posts USING GIN(to_tsvector('english', content));
```

### インデックスの使い所

**✅ インデックスを作成すべき場合：**
- WHERE句で頻繁に使用されるカラム
- JOIN条件のカラム
- ORDER BY / GROUP BYで使用されるカラム
- UNIQUE制約が必要なカラム

**❌ インデックスを避けるべき場合：**
- 小さなテーブル（< 1000行）
- 頻繁に更新されるカラム
- カーディナリティが低いカラム（例: boolean）

---

## リレーション

### 1対多（One-to-Many）

```sql
-- ユーザー（1） : 投稿（多）
CREATE TABLE users (
  id UUID PRIMARY KEY,
  name VARCHAR(100)
);

CREATE TABLE posts (
  id UUID PRIMARY KEY,
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title VARCHAR(200),
  content TEXT
);

CREATE INDEX idx_posts_user_id ON posts(user_id);
```

### 多対多（Many-to-Many）

```sql
-- ユーザー（多） : ロール（多）
CREATE TABLE users (
  id UUID PRIMARY KEY,
  name VARCHAR(100)
);

CREATE TABLE roles (
  id UUID PRIMARY KEY,
  name VARCHAR(50)
);

-- 中間テーブル
CREATE TABLE user_roles (
  user_id UUID REFERENCES users(id) ON DELETE CASCADE,
  role_id UUID REFERENCES roles(id) ON DELETE CASCADE,
  PRIMARY KEY (user_id, role_id)
);

CREATE INDEX idx_user_roles_user ON user_roles(user_id);
CREATE INDEX idx_user_roles_role ON user_roles(role_id);
```

### ON DELETE動作

| 動作 | 説明 |
|------|------|
| **CASCADE** | 親削除時に子も削除 |
| **SET NULL** | 親削除時に子の外部キーをNULLに |
| **RESTRICT** | 子が存在する場合、親の削除を拒否 |
| **NO ACTION** | RESTRICTと同様（PostgreSQL） |

---

## Prisma

### スキーマ定義

```prisma
// schema.prisma
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model User {
  id        String   @id @default(uuid())
  name      String
  email     String   @unique
  password  String
  role      Role     @default(USER)
  posts     Post[]
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt

  @@index([email])
}

model Post {
  id        String   @id @default(uuid())
  title     String
  content   String
  published Boolean  @default(false)
  author    User     @relation(fields: [authorId], references: [id], onDelete: Cascade)
  authorId  String
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt

  @@index([authorId])
  @@index([published, createdAt])
}

enum Role {
  USER
  ADMIN
  MODERATOR
}
```

### マイグレーション

```bash
# 初期マイグレーション作成
pnpm prisma migrate dev --name init

# マイグレーション適用
pnpm prisma migrate deploy

# スキーマからクライアント生成
pnpm prisma generate

# データベースリセット
pnpm prisma migrate reset

# Prisma Studio起動（GUI）
pnpm prisma studio
```

### クエリ

```typescript
import { PrismaClient } from '@prisma/client'

const prisma = new PrismaClient()

// 作成
const user = await prisma.user.create({
  data: {
    name: 'John',
    email: 'john@example.com',
    password: 'hashed_password'
  }
})

// 取得
const users = await prisma.user.findMany({
  where: { role: 'ADMIN' },
  include: { posts: true },
  orderBy: { createdAt: 'desc' }
})

// 更新
await prisma.user.update({
  where: { id: '123' },
  data: { name: 'John Doe' }
})

// 削除
await prisma.user.delete({
  where: { id: '123' }
})

// トランザクション
await prisma.$transaction([
  prisma.user.create({ data: { name: 'User 1', email: 'user1@example.com', password: 'pass' } }),
  prisma.user.create({ data: { name: 'User 2', email: 'user2@example.com', password: 'pass' } })
])
```

---

## SQL最適化

### N+1問題

```typescript
// ❌ 悪い例（N+1問題）
const users = await prisma.user.findMany()
for (const user of users) {
  const posts = await prisma.post.findMany({ where: { authorId: user.id } })
  // N回のクエリ
}

// ✅ 良い例（includeで1クエリ）
const users = await prisma.user.findMany({
  include: { posts: true }
})
```

### SELECT文の最適化

```typescript
// ❌ 悪い例（全カラム取得）
const users = await prisma.user.findMany()

// ✅ 良い例（必要なカラムのみ）
const users = await prisma.user.findMany({
  select: { id: true, name: true, email: true }
})
```

### ページネーション

```typescript
// カーソルベースページネーション（推奨）
const posts = await prisma.post.findMany({
  take: 20,
  cursor: lastPostId ? { id: lastPostId } : undefined,
  orderBy: { createdAt: 'desc' }
})

// オフセットベースページネーション
const posts = await prisma.post.findMany({
  skip: (page - 1) * 20,
  take: 20,
  orderBy: { createdAt: 'desc' }
})
```

---

## 実践例

### Example 1: ブログシステム

```prisma
// schema.prisma
model User {
  id        String   @id @default(uuid())
  name      String
  email     String   @unique
  password  String
  posts     Post[]
  comments  Comment[]
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt

  @@index([email])
}

model Post {
  id        String    @id @default(uuid())
  title     String
  slug      String    @unique
  content   String
  published Boolean   @default(false)
  author    User      @relation(fields: [authorId], references: [id], onDelete: Cascade)
  authorId  String
  tags      PostTag[]
  comments  Comment[]
  createdAt DateTime  @default(now())
  updatedAt DateTime  @updatedAt

  @@index([authorId])
  @@index([published, createdAt])
  @@index([slug])
}

model Tag {
  id    String    @id @default(uuid())
  name  String    @unique
  posts PostTag[]
}

model PostTag {
  post   Post   @relation(fields: [postId], references: [id], onDelete: Cascade)
  postId String
  tag    Tag    @relation(fields: [tagId], references: [id], onDelete: Cascade)
  tagId  String

  @@id([postId, tagId])
  @@index([postId])
  @@index([tagId])
}

model Comment {
  id        String   @id @default(uuid())
  content   String
  post      Post     @relation(fields: [postId], references: [id], onDelete: Cascade)
  postId    String
  author    User     @relation(fields: [authorId], references: [id], onDelete: Cascade)
  authorId  String
  createdAt DateTime @default(now())

  @@index([postId])
  @@index([authorId])
}
```

---

## Agent連携

### 📖 Agentへの指示例

**Prismaスキーマ作成**
```
以下のPrismaスキーマを作成してください：
- User（name, email, password）
- Post（title, content, authorId）
- Comment（content, postId, authorId）

適切なリレーションとインデックスを含めてください。
```

**マイグレーション実行**
```
Prismaマイグレーションを作成して適用してください。
マイグレーション名は "add_comments_table" にしてください。
```

---

## まとめ

### データベース設計のベストプラクティス

1. **正規化** - 第3正規形まで
2. **インデックス** - 頻繁に検索されるカラムに作成
3. **リレーション** - 適切な外部キー制約
4. **型安全性** - Prisma活用

---

_Last updated: 2025-12-24_
