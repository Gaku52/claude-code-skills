# データベース基礎 - 完全初心者ガイド

## 目次

1. [概要](#概要)
2. [データベースとは](#データベースとは)
3. [リレーショナルデータベース](#リレーショナルデータベース)
4. [SQL基礎](#sql基礎)
5. [ORM入門](#orm入門)
6. [データベース設計](#データベース設計)
7. [実装例](#実装例)
8. [演習問題](#演習問題)
9. [次のステップ](#次のステップ)

---

## 概要

### 何を学ぶか

- データベースの基本概念
- SQLの基礎（SELECT、INSERT、UPDATE、DELETE）
- ORM（Object-Relational Mapping）
- テーブル設計の基本
- FastAPI + SQLAlchemyでのDB操作

### なぜ重要か

バックエンド開発の中核はデータベース操作です。適切なデータベース設計とクエリの最適化が、アプリケーションのパフォーマンスとスケーラビリティを決定します。

### 学習時間：1〜2時間

---

## データベースとは

### 定義

**データベース**は、データを構造化して保存・管理するシステムです。

### なぜファイルではなくデータベースか

| 項目 | ファイル | データベース |
|------|---------|------------|
| **同時アクセス** | ❌ 困難 | ✅ 可能 |
| **データ整合性** | ❌ 保証なし | ✅ トランザクションで保証 |
| **検索速度** | ❌ 遅い | ✅ インデックスで高速 |
| **バックアップ** | ❌ 手動 | ✅ 自動バックアップ可能 |

### データベースの種類

#### 1. リレーショナルデータベース（SQL）

**特徴**：テーブル形式、SQL言語、ACID保証

**代表例**：
- PostgreSQL（推奨）
- MySQL
- SQLite（開発用）

#### 2. NoSQLデータベース

**特徴**：柔軟なスキーマ、水平スケール

**代表例**：
- MongoDB（ドキュメント）
- Redis（キーバリュー）
- Elasticsearch（検索エンジン）

---

## リレーショナルデータベース

### テーブル構造

```
users テーブル
┌────┬────────┬─────────────────────┬─────┐
│ id │ name   │ email               │ age │
├────┼────────┼─────────────────────┼─────┤
│ 1  │ 太郎   │ taro@example.com    │ 25  │
│ 2  │ 花子   │ hanako@example.com  │ 30  │
│ 3  │ 次郎   │ jiro@example.com    │ 22  │
└────┴────────┴─────────────────────┴─────┘
```

### リレーション（関連）

#### 1対多（One-to-Many）

```
users (1) ──< posts (多)

1人のユーザーが複数の記事を持つ

users テーブル
┌────┬────────┐
│ id │ name   │
├────┼────────┤
│ 1  │ 太郎   │
└────┴────────┘

posts テーブル
┌────┬───────────┬─────────┐
│ id │ title     │ user_id │
├────┼───────────┼─────────┤
│ 1  │ 記事1     │ 1       │
│ 2  │ 記事2     │ 1       │
└────┴───────────┴─────────┘
```

#### 多対多（Many-to-Many）

```
users (多) ──< user_tags >── (多) tags

中間テーブルを使う

users テーブル
┌────┬────────┐
│ id │ name   │
├────┼────────┤
│ 1  │ 太郎   │
│ 2  │ 花子   │
└────┴────────┘

tags テーブル
┌────┬────────┐
│ id │ name   │
├────┼────────┤
│ 1  │ Python │
│ 2  │ React  │
└────┴────────┘

user_tags テーブル（中間テーブル）
┌─────────┬────────┐
│ user_id │ tag_id │
├─────────┼────────┤
│ 1       │ 1      │
│ 1       │ 2      │
│ 2       │ 1      │
└─────────┴────────┘
```

---

## SQL基礎

### CREATE（テーブル作成）

```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    age INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### INSERT（データ挿入）

```sql
-- 単一行
INSERT INTO users (name, email, age)
VALUES ('太郎', 'taro@example.com', 25);

-- 複数行
INSERT INTO users (name, email, age)
VALUES
    ('花子', 'hanako@example.com', 30),
    ('次郎', 'jiro@example.com', 22);
```

### SELECT（データ取得）

```sql
-- すべて取得
SELECT * FROM users;

-- 特定のカラムのみ
SELECT name, email FROM users;

-- 条件付き
SELECT * FROM users WHERE age >= 25;

-- 並び替え
SELECT * FROM users ORDER BY age DESC;

-- 件数制限
SELECT * FROM users LIMIT 10 OFFSET 0;

-- 集計
SELECT COUNT(*) FROM users;
SELECT AVG(age) FROM users;
SELECT MAX(age), MIN(age) FROM users;
```

### UPDATE（データ更新）

```sql
-- 特定のユーザーを更新
UPDATE users
SET age = 26
WHERE id = 1;

-- 複数カラム更新
UPDATE users
SET name = '山田太郎', age = 26
WHERE id = 1;
```

### DELETE（データ削除）

```sql
-- 特定のユーザーを削除
DELETE FROM users WHERE id = 1;

-- 条件付き削除
DELETE FROM users WHERE age < 18;
```

### JOIN（テーブル結合）

```sql
-- ユーザーと記事を結合
SELECT
    users.name,
    posts.title,
    posts.created_at
FROM users
INNER JOIN posts ON users.id = posts.user_id;

-- LEFT JOIN（ユーザーが記事を持っていなくても表示）
SELECT
    users.name,
    COUNT(posts.id) as post_count
FROM users
LEFT JOIN posts ON users.id = posts.user_id
GROUP BY users.id, users.name;
```

---

## ORM入門

### ORMとは

**ORM（Object-Relational Mapping）**は、データベースのテーブルをプログラムのオブジェクトとしてマッピングする技術です。

**利点**：
- SQLを直接書かなくて良い
- 型安全
- コードが読みやすい

### SQLAlchemy（Python）

```python
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

# モデル定義
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    age = Column(Integer)

# データベース接続
engine = create_engine("postgresql://user:password@localhost/dbname")
SessionLocal = sessionmaker(bind=engine)
db = SessionLocal()

# CRUD操作

# Create
new_user = User(name="太郎", email="taro@example.com", age=25)
db.add(new_user)
db.commit()

# Read
user = db.query(User).filter(User.id == 1).first()
users = db.query(User).filter(User.age >= 25).all()

# Update
user = db.query(User).filter(User.id == 1).first()
user.age = 26
db.commit()

# Delete
user = db.query(User).filter(User.id == 1).first()
db.delete(user)
db.commit()
```

---

## データベース設計

### 正規化

**第1正規形**：繰り返しグループをなくす

```
❌ 悪い設計
users テーブル
┌────┬────────┬──────────────────────┐
│ id │ name   │ hobbies              │
├────┼────────┼──────────────────────┤
│ 1  │ 太郎   │ 読書, 音楽, スポーツ │
└────┴────────┴──────────────────────┘

✅ 良い設計
users テーブル        hobbies テーブル
┌────┬────────┐      ┌────┬─────────┬────────┐
│ id │ name   │      │ id │ user_id │ hobby  │
├────┼────────┤      ├────┼─────────┼────────┤
│ 1  │ 太郎   │      │ 1  │ 1       │ 読書   │
└────┴────────┘      │ 2  │ 1       │ 音楽   │
                      │ 3  │ 1       │スポーツ│
                      └────┴─────────┴────────┘
```

### インデックス

**インデックス**は、検索を高速化するための索引です。

```sql
-- インデックス作成
CREATE INDEX idx_users_email ON users(email);

-- 複合インデックス
CREATE INDEX idx_posts_user_created ON posts(user_id, created_at);

-- 検索が速くなる
SELECT * FROM users WHERE email = 'taro@example.com';  -- インデックス使用
```

---

## 実装例

### FastAPI + SQLAlchemy

```python
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel

# データベース設定
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# モデル
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)

Base.metadata.create_all(bind=engine)

# Pydanticスキーマ
class UserCreate(BaseModel):
    name: str
    email: str

class UserResponse(BaseModel):
    id: int
    name: str
    email: str

    class Config:
        from_attributes = True

# FastAPI
app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/users", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(name=user.name, email=user.email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "ユーザーが見つかりません")
    return user
```

---

## 演習問題

### 問題：ブログシステムのテーブル設計

以下の要件を満たすテーブルを設計してください：
- ユーザーが記事を投稿できる
- 記事にはカテゴリーが設定できる（複数可）
- 記事にコメントができる

**解答例**：

```sql
-- ユーザーテーブル
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL
);

-- 記事テーブル
CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    content TEXT,
    user_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- カテゴリーテーブル
CREATE TABLE categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL
);

-- 記事とカテゴリーの中間テーブル
CREATE TABLE post_categories (
    post_id INTEGER REFERENCES posts(id),
    category_id INTEGER REFERENCES categories(id),
    PRIMARY KEY (post_id, category_id)
);

-- コメントテーブル
CREATE TABLE comments (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    post_id INTEGER REFERENCES posts(id),
    user_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 次のステップ

### このガイドで学んだこと

- ✅ データベースの基本概念
- ✅ SQL基礎（CRUD操作）
- ✅ ORM（SQLAlchemy）
- ✅ テーブル設計の基本

### 次に学ぶべきガイド

**次のガイド**：[05-authentication-basics.md](./05-authentication-basics.md) - 認証の基礎

---

**前のガイド**：[03-rest-api-intro.md](./03-rest-api-intro.md)

**親ガイド**：[Backend Development - SKILL.md](../../SKILL.md)
