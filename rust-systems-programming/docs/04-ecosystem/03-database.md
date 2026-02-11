# データベース — sqlx / diesel / SeaORM

> Rust でデータベースを扱うための主要 3 クレートを比較し、プロジェクトに最適な ORM / クエリビルダーを選択するための実践ガイド。

---

## この章で学ぶこと

1. **sqlx** — コンパイル時 SQL 検証を持つ非同期クエリライブラリ
2. **diesel** — 型安全な DSL ベースの同期 ORM
3. **SeaORM** — ActiveRecord パターンの非同期 ORM

---

## 1. Rust データベースクレートの全体像

### 1.1 レイヤー構成

```
┌─────────────────────────────────────────────────┐
│           Application Code                      │
├─────────────────────────────────────────────────┤
│  ORM / Query Layer                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │  diesel   │  │  SeaORM  │  │  sqlx    │      │
│  │ (DSL型)   │  │ (AR型)   │  │ (Raw SQL)│      │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘      │
│        │             │             │            │
├────────┼─────────────┼─────────────┼────────────┤
│  Connection Pool                                │
│  ┌──────────────────────────────────────┐       │
│  │  sqlx::Pool / deadpool / bb8         │       │
│  └──────────────────┬───────────────────┘       │
│                     │                           │
├─────────────────────┼───────────────────────────┤
│  Driver                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ tokio-    │  │ sqlx-    │  │ libpq /  │      │
│  │ postgres  │  │ sqlite   │  │ mysqlclient│    │
│  └──────────┘  └──────────┘  └──────────┘      │
└─────────────────────────────────────────────────┘
```

### 1.2 選択フロー

```
Rust でDBを使いたい
       │
       ├── 生の SQL を書きたい？
       │      │
       │      ├── Yes → sqlx
       │      │         (コンパイル時SQL検証が魅力)
       │      │
       │      └── No ──┐
       │               │
       ├── 型安全な DSL が欲しい？
       │      │
       │      ├── Yes → diesel
       │      │         (コンパイル時に全クエリを型検査)
       │      │
       │      └── No ──┐
       │               │
       └── ActiveRecord パターンが好み？
              │
              └── Yes → SeaORM
                        (Rails/Laravel 的な使い心地)
```

---

## 2. sqlx — コンパイル時 SQL 検証

### 2.1 セットアップ

```toml
# Cargo.toml
[dependencies]
sqlx = { version = "0.8", features = [
    "runtime-tokio",     # 非同期ランタイム
    "tls-rustls",        # TLS
    "postgres",          # PostgreSQL ドライバ
    "macros",            # query! マクロ
    "migrate",           # マイグレーション
    "chrono",            # 日時型サポート
    "uuid",              # UUID 型サポート
] }
tokio = { version = "1", features = ["full"] }
```

### 2.2 基本的な CRUD 操作

```rust
use sqlx::{PgPool, FromRow};
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Debug, FromRow)]
struct User {
    id: Uuid,
    name: String,
    email: String,
    created_at: DateTime<Utc>,
}

// コネクションプール作成
async fn create_pool() -> Result<PgPool, sqlx::Error> {
    PgPool::builder()
        .max_connections(10)
        .connect("postgres://user:pass@localhost:5432/mydb")
        .await
}

// INSERT — query! マクロでコンパイル時に SQL を検証
async fn create_user(pool: &PgPool, name: &str, email: &str) -> Result<User, sqlx::Error> {
    sqlx::query_as!(
        User,
        r#"
        INSERT INTO users (id, name, email, created_at)
        VALUES ($1, $2, $3, NOW())
        RETURNING id, name, email, created_at
        "#,
        Uuid::new_v4(),
        name,
        email,
    )
    .fetch_one(pool)
    .await
}

// SELECT — 複数行取得
async fn list_users(pool: &PgPool, limit: i64) -> Result<Vec<User>, sqlx::Error> {
    sqlx::query_as!(
        User,
        "SELECT id, name, email, created_at FROM users ORDER BY created_at DESC LIMIT $1",
        limit,
    )
    .fetch_all(pool)
    .await
}

// UPDATE
async fn update_email(pool: &PgPool, user_id: Uuid, new_email: &str) -> Result<bool, sqlx::Error> {
    let result = sqlx::query!(
        "UPDATE users SET email = $1 WHERE id = $2",
        new_email,
        user_id,
    )
    .execute(pool)
    .await?;

    Ok(result.rows_affected() > 0)
}

// DELETE
async fn delete_user(pool: &PgPool, user_id: Uuid) -> Result<bool, sqlx::Error> {
    let result = sqlx::query!("DELETE FROM users WHERE id = $1", user_id)
        .execute(pool)
        .await?;

    Ok(result.rows_affected() > 0)
}
```

### 2.3 マイグレーション

```bash
# sqlx-cli のインストール
cargo install sqlx-cli --no-default-features --features postgres

# マイグレーション作成
sqlx migrate add create_users_table

# 生成されたファイルを編集
# migrations/20260101000000_create_users_table.sql
```

```sql
-- migrations/20260101000000_create_users_table.sql
CREATE TABLE users (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       VARCHAR(255) NOT NULL,
    email      VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users (email);
```

```bash
# マイグレーション実行
sqlx migrate run

# マイグレーション状態確認
sqlx migrate info
```

---

## 3. diesel — 型安全な DSL ベース ORM

### 3.1 セットアップ

```toml
# Cargo.toml
[dependencies]
diesel = { version = "2.2", features = ["postgres", "chrono", "uuid"] }
diesel_migrations = "2.2"
dotenvy = "0.15"
```

### 3.2 スキーマとモデル定義

```rust
// src/schema.rs (diesel CLI が自動生成)
diesel::table! {
    users (id) {
        id -> Uuid,
        name -> Varchar,
        email -> Varchar,
        created_at -> Timestamptz,
        updated_at -> Timestamptz,
    }
}

diesel::table! {
    posts (id) {
        id -> Uuid,
        user_id -> Uuid,
        title -> Varchar,
        body -> Text,
        published -> Bool,
        created_at -> Timestamptz,
    }
}

diesel::joinable!(posts -> users (user_id));
diesel::allow_tables_to_appear_in_same_query!(users, posts);
```

```rust
// src/models.rs
use diesel::prelude::*;
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Queryable, Selectable, Debug)]
#[diesel(table_name = crate::schema::users)]
pub struct User {
    pub id: Uuid,
    pub name: String,
    pub email: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Insertable)]
#[diesel(table_name = crate::schema::users)]
pub struct NewUser<'a> {
    pub name: &'a str,
    pub email: &'a str,
}
```

### 3.3 CRUD 操作

```rust
use diesel::prelude::*;
use crate::schema::users;
use crate::models::{User, NewUser};

// INSERT
fn create_user(conn: &mut PgConnection, name: &str, email: &str) -> QueryResult<User> {
    let new_user = NewUser { name, email };

    diesel::insert_into(users::table)
        .values(&new_user)
        .returning(User::as_returning())
        .get_result(conn)
}

// SELECT with filter
fn find_user_by_email(conn: &mut PgConnection, email_addr: &str) -> QueryResult<User> {
    users::table
        .filter(users::email.eq(email_addr))
        .select(User::as_select())
        .first(conn)
}

// SELECT with JOIN
fn get_user_with_posts(conn: &mut PgConnection, user_id: Uuid) -> QueryResult<Vec<(User, Post)>> {
    users::table
        .inner_join(posts::table)
        .filter(users::id.eq(user_id))
        .filter(posts::published.eq(true))
        .select((User::as_select(), Post::as_select()))
        .load(conn)
}

// UPDATE
fn update_user_email(conn: &mut PgConnection, user_id: Uuid, new_email: &str) -> QueryResult<User> {
    diesel::update(users::table.filter(users::id.eq(user_id)))
        .set(users::email.eq(new_email))
        .returning(User::as_returning())
        .get_result(conn)
}

// DELETE
fn delete_user(conn: &mut PgConnection, user_id: Uuid) -> QueryResult<usize> {
    diesel::delete(users::table.filter(users::id.eq(user_id)))
        .execute(conn)
}
```

---

## 4. SeaORM — ActiveRecord パターンの非同期 ORM

### 4.1 セットアップ

```toml
# Cargo.toml
[dependencies]
sea-orm = { version = "1.0", features = [
    "sqlx-postgres",
    "runtime-tokio-rustls",
    "macros",
] }
```

### 4.2 エンティティ定義

```rust
// src/entities/user.rs
use sea_orm::entity::prelude::*;

#[derive(Clone, Debug, PartialEq, DeriveEntityModel)]
#[sea_orm(table_name = "users")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: Uuid,
    pub name: String,
    pub email: String,
    pub created_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(has_many = "super::post::Entity")]
    Posts,
}

impl Related<super::post::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Posts.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}
```

### 4.3 CRUD 操作

```rust
use sea_orm::*;
use crate::entities::{user, post};

// INSERT
async fn create_user(db: &DatabaseConnection, name: &str, email: &str) -> Result<user::Model, DbErr> {
    let new_user = user::ActiveModel {
        id: Set(Uuid::new_v4()),
        name: Set(name.to_string()),
        email: Set(email.to_string()),
        created_at: Set(chrono::Utc::now()),
        ..Default::default()
    };

    new_user.insert(db).await
}

// SELECT with filter and pagination
async fn list_users(
    db: &DatabaseConnection,
    page: u64,
    per_page: u64,
) -> Result<(Vec<user::Model>, u64), DbErr> {
    let paginator = user::Entity::find()
        .order_by_desc(user::Column::CreatedAt)
        .paginate(db, per_page);

    let total = paginator.num_pages().await?;
    let users = paginator.fetch_page(page).await?;

    Ok((users, total))
}

// SELECT with JOIN (Eager Loading)
async fn get_user_with_posts(
    db: &DatabaseConnection,
    user_id: Uuid,
) -> Result<Option<(user::Model, Vec<post::Model>)>, DbErr> {
    let result = user::Entity::find_by_id(user_id)
        .find_with_related(post::Entity)
        .all(db)
        .await?;

    Ok(result.into_iter().next())
}

// UPDATE
async fn update_email(
    db: &DatabaseConnection,
    user_id: Uuid,
    new_email: &str,
) -> Result<user::Model, DbErr> {
    let mut user: user::ActiveModel = user::Entity::find_by_id(user_id)
        .one(db)
        .await?
        .ok_or(DbErr::RecordNotFound("User not found".into()))?
        .into();

    user.email = Set(new_email.to_string());
    user.update(db).await
}

// DELETE
async fn delete_user(db: &DatabaseConnection, user_id: Uuid) -> Result<DeleteResult, DbErr> {
    user::Entity::delete_by_id(user_id).exec(db).await
}
```

---

## 5. 比較表

### 5.1 機能比較

| 機能 | sqlx | diesel | SeaORM |
|------|------|--------|--------|
| **パラダイム** | 生 SQL + マクロ | DSL ベース ORM | ActiveRecord ORM |
| **非同期対応** | ネイティブ | diesel-async で対応 | ネイティブ |
| **コンパイル時検証** | SQL 構文 + 型 | DSL レベルの型安全 | なし（ランタイム検証） |
| **マイグレーション** | SQL ファイル | DSL or SQL | SeaORM CLI |
| **PostgreSQL** | 対応 | 対応 | 対応 |
| **MySQL** | 対応 | 対応 | 対応 |
| **SQLite** | 対応 | 対応 | 対応 |
| **トランザクション** | 対応 | 対応 | 対応 |
| **接続プール** | 組み込み | 外部 (r2d2/deadpool) | 組み込み (sqlx ベース) |
| **学習コスト** | 低（SQL が書ければ OK） | 中（DSL の習得が必要） | 中（エンティティ定義の理解） |
| **コミュニティ** | 大 | 大 | 中 |

### 5.2 パフォーマンス特性比較

| 観点 | sqlx | diesel | SeaORM |
|------|------|--------|--------|
| **クエリ生成オーバーヘッド** | なし（生SQL） | 小（DSL→SQL変換） | 中（ORM抽象化） |
| **コンパイル時間** | 中（マクロ展開） | 長（型推論が重い） | 中 |
| **バイナリサイズ** | 小 | 中 | 中 |
| **N+1 問題対策** | 手動（SQL制御） | 手動（JOIN記述） | find_with_related |
| **バルクインサート** | query + unnest | insert_into().values(&vec) | insert_many() |
| **生SQLフォールバック** | デフォルト | sql_query() | FromQueryResult |

---

## 6. アンチパターン

### 6.1 N+1 クエリ問題

```rust
// NG: ユーザーごとに投稿を個別取得（N+1）
async fn bad_get_users_with_posts(pool: &PgPool) -> Result<Vec<(User, Vec<Post>)>> {
    let users = sqlx::query_as!(User, "SELECT * FROM users")
        .fetch_all(pool).await?;

    let mut result = Vec::new();
    for user in users {
        // ユーザー数だけクエリが発行される!
        let posts = sqlx::query_as!(Post,
            "SELECT * FROM posts WHERE user_id = $1", user.id
        ).fetch_all(pool).await?;
        result.push((user, posts));
    }
    Ok(result)
}

// OK: JOIN で1回のクエリで取得
async fn good_get_users_with_posts(pool: &PgPool) -> Result<Vec<UserWithPosts>> {
    sqlx::query_as!(UserWithPosts,
        r#"
        SELECT u.id, u.name, u.email,
               COALESCE(json_agg(p.*) FILTER (WHERE p.id IS NOT NULL), '[]') as "posts!: Json<Vec<Post>>"
        FROM users u
        LEFT JOIN posts p ON p.user_id = u.id
        GROUP BY u.id
        "#
    ).fetch_all(pool).await
}
```

### 6.2 トランザクション未使用での複数操作

```rust
// NG: トランザクションなしで関連データを操作
async fn bad_transfer(pool: &PgPool, from: Uuid, to: Uuid, amount: i64) -> Result<()> {
    sqlx::query!("UPDATE accounts SET balance = balance - $1 WHERE id = $2", amount, from)
        .execute(pool).await?;
    // ← ここで障害が発生すると残高が消失!
    sqlx::query!("UPDATE accounts SET balance = balance + $1 WHERE id = $2", amount, to)
        .execute(pool).await?;
    Ok(())
}

// OK: トランザクションで原子性を保証
async fn good_transfer(pool: &PgPool, from: Uuid, to: Uuid, amount: i64) -> Result<()> {
    let mut tx = pool.begin().await?;

    sqlx::query!("UPDATE accounts SET balance = balance - $1 WHERE id = $2", amount, from)
        .execute(&mut *tx).await?;
    sqlx::query!("UPDATE accounts SET balance = balance + $1 WHERE id = $2", amount, to)
        .execute(&mut *tx).await?;

    tx.commit().await?;  // 両方成功した場合のみコミット
    Ok(())
}
```

---

## 7. FAQ

### Q1. sqlx の `query!` マクロはどうやってコンパイル時に SQL を検証している？

**A.** `query!` マクロはコンパイル時に `DATABASE_URL` 環境変数で指定された実際のデータベースに接続し、`PREPARE` ステートメントを使って SQL の構文とカラム型を検証する。CI/CD ではオフラインモード（`sqlx prepare` で生成した `.sqlx/` ディレクトリ）を使うことで DB 接続なしでビルド可能。

```bash
# オフライン用のクエリメタデータ生成
cargo sqlx prepare
# → .sqlx/ ディレクトリにメタデータが保存される（Git にコミット）
```

### Q2. diesel と sqlx を同じプロジェクトで併用できる？

**A.** 技術的には可能だが推奨しない。依存関係が増え、接続プールの管理が複雑になる。複雑なクエリだけ生 SQL を使いたい場合、diesel の `sql_query()` や sqlx のように生 SQL を書ける機能で対応する方が良い。

### Q3. Web フレームワーク（Axum/Actix Web）との統合はどうする？

**A.** Axum の場合、`State` でプールを共有するのが一般的。

```rust
use axum::{extract::State, routing::get, Router, Json};
use sqlx::PgPool;

async fn list_users(State(pool): State<PgPool>) -> Json<Vec<User>> {
    let users = sqlx::query_as!(User, "SELECT * FROM users LIMIT 50")
        .fetch_all(&pool)
        .await
        .unwrap();
    Json(users)
}

#[tokio::main]
async fn main() {
    let pool = PgPool::connect("postgres://...").await.unwrap();

    let app = Router::new()
        .route("/users", get(list_users))
        .with_state(pool);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

---

## 8. まとめ

| 項目 | ポイント |
|------|---------|
| **sqlx** | 生 SQL 派向け、コンパイル時検証、非同期ネイティブ |
| **diesel** | 型安全 DSL、コンパイル時保証最強、同期メイン（async 拡張あり） |
| **SeaORM** | ActiveRecord パターン、Rails 的な開発体験、非同期ネイティブ |
| **選定基準** | SQL 制御 → sqlx、型安全 → diesel、生産性 → SeaORM |
| **共通注意点** | N+1 回避、トランザクション活用、接続プールの適切な設定 |

---

## 次に読むべきガイド

- Rust 非同期プログラミングガイド — tokio ランタイムとの統合
- Axum Web フレームワークガイド — DB 層との接続パターン
- SQL パフォーマンスチューニング — インデックス設計とクエリ最適化

---

## 参考文献

1. **sqlx 公式リポジトリ** — https://github.com/launchbadge/sqlx
2. **diesel 公式サイト** — "Getting Started with Diesel" — https://diesel.rs/guides/getting-started
3. **SeaORM 公式ドキュメント** — https://www.sea-ql.org/SeaORM/docs/introduction/
4. **The Rust Programming Language** — "Async/Await" — https://doc.rust-lang.org/book/
