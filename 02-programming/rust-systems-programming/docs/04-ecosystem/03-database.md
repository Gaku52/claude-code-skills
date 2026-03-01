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

### 2.4 トランザクションの高度な使い方

sqlx では `pool.begin()` でトランザクションを開始し、`commit()` または `rollback()` で終了する。`Transaction` は `Drop` 時に自動ロールバックするため、明示的な `rollback()` は必須ではない。

```rust
use sqlx::PgPool;

/// ネストされたトランザクション（SAVEPOINT を使用）
async fn nested_transaction_example(pool: &PgPool) -> Result<(), sqlx::Error> {
    let mut tx = pool.begin().await?;

    // 外側のトランザクションで操作
    sqlx::query!("INSERT INTO audit_log (action) VALUES ('start_batch')")
        .execute(&mut *tx)
        .await?;

    // SAVEPOINT を使ったネストされたトランザクション
    let mut savepoint = tx.begin().await?;  // SAVEPOINT が自動作成される

    let result = sqlx::query!(
        "UPDATE inventory SET quantity = quantity - 1 WHERE product_id = $1 AND quantity > 0",
        product_id
    )
    .execute(&mut *savepoint)
    .await?;

    if result.rows_affected() == 0 {
        // 在庫不足の場合、SAVEPOINT までロールバック
        // savepoint は Drop 時に自動ロールバック
        drop(savepoint);
    } else {
        savepoint.commit().await?;  // SAVEPOINT をリリース
    }

    // 外側のトランザクションはコミット
    sqlx::query!("INSERT INTO audit_log (action) VALUES ('end_batch')")
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;
    Ok(())
}
```

### 2.5 ストリーミングクエリ

大量のデータを扱う場合、全行をメモリにロードするのではなく、ストリーミングで1行ずつ処理できる。

```rust
use sqlx::PgPool;
use futures::TryStreamExt;  // try_next() を使うために必要

/// 100万行のデータをストリーミングで処理
async fn process_large_dataset(pool: &PgPool) -> Result<(), sqlx::Error> {
    let mut stream = sqlx::query_as!(
        User,
        "SELECT id, name, email, created_at FROM users WHERE created_at > $1",
        cutoff_date,
    )
    .fetch(pool);  // fetch() はストリームを返す（fetch_all() ではない）

    let mut count = 0u64;
    while let Some(user) = stream.try_next().await? {
        // 1行ずつ処理 — メモリ使用量は一定
        process_user(&user).await?;
        count += 1;

        if count % 10_000 == 0 {
            tracing::info!("Processed {} users", count);
        }
    }

    tracing::info!("Total processed: {} users", count);
    Ok(())
}
```

### 2.6 動的クエリの構築

`query!` マクロは静的 SQL にしか対応しないが、`QueryBuilder` を使えば動的に SQL を組み立てられる。

```rust
use sqlx::{PgPool, QueryBuilder, Postgres};

/// 動的な検索条件でユーザーを検索
async fn search_users(
    pool: &PgPool,
    name_filter: Option<&str>,
    email_filter: Option<&str>,
    min_created_at: Option<DateTime<Utc>>,
    order_by: &str,
    limit: i64,
) -> Result<Vec<User>, sqlx::Error> {
    let mut builder: QueryBuilder<Postgres> = QueryBuilder::new(
        "SELECT id, name, email, created_at FROM users WHERE 1=1"
    );

    if let Some(name) = name_filter {
        builder.push(" AND name ILIKE ");
        builder.push_bind(format!("%{}%", name));
    }

    if let Some(email) = email_filter {
        builder.push(" AND email ILIKE ");
        builder.push_bind(format!("%{}%", email));
    }

    if let Some(min_date) = min_created_at {
        builder.push(" AND created_at >= ");
        builder.push_bind(min_date);
    }

    // ORDER BY はバインド変数にできないため、ホワイトリストで検証
    let safe_order = match order_by {
        "name" => "name",
        "email" => "email",
        "created_at" => "created_at",
        _ => "created_at",  // デフォルト
    };
    builder.push(format!(" ORDER BY {} DESC", safe_order));

    builder.push(" LIMIT ");
    builder.push_bind(limit);

    builder
        .build_query_as::<User>()
        .fetch_all(pool)
        .await
}
```

### 2.7 バルクインサート

大量のレコードを効率的に挿入する方法。

```rust
use sqlx::{PgPool, QueryBuilder, Postgres};

/// バルクインサート — 1回のクエリで複数行を挿入
async fn bulk_insert_users(
    pool: &PgPool,
    users: &[(String, String)],  // (name, email)
) -> Result<u64, sqlx::Error> {
    // PostgreSQL の場合、VALUES 句に複数行を指定
    let mut builder: QueryBuilder<Postgres> = QueryBuilder::new(
        "INSERT INTO users (id, name, email, created_at) "
    );

    builder.push_values(users.iter(), |mut b, (name, email)| {
        b.push_bind(Uuid::new_v4())
         .push_bind(name)
         .push_bind(email)
         .push("NOW()");
    });

    let result = builder.build().execute(pool).await?;
    Ok(result.rows_affected())
}

/// UNNEST を使ったさらに高効率なバルクインサート
async fn bulk_insert_with_unnest(
    pool: &PgPool,
    names: &[String],
    emails: &[String],
) -> Result<u64, sqlx::Error> {
    let ids: Vec<Uuid> = (0..names.len()).map(|_| Uuid::new_v4()).collect();

    let result = sqlx::query!(
        r#"
        INSERT INTO users (id, name, email, created_at)
        SELECT * FROM UNNEST($1::uuid[], $2::text[], $3::text[])
        AS t(id, name, email),
        LATERAL (SELECT NOW() AS created_at) ts
        "#,
        &ids,
        names,
        emails,
    )
    .execute(pool)
    .await?;

    Ok(result.rows_affected())
}
```

### 2.8 カスタム型のマッピング

sqlx で PostgreSQL のカスタム型（ENUM、複合型）をマッピングする方法。

```rust
use sqlx::Type;

// PostgreSQL ENUM 型のマッピング
// CREATE TYPE user_role AS ENUM ('admin', 'moderator', 'user');
#[derive(Debug, Clone, PartialEq, Type)]
#[sqlx(type_name = "user_role", rename_all = "lowercase")]
pub enum UserRole {
    Admin,
    Moderator,
    User,
}

#[derive(Debug, FromRow)]
struct UserWithRole {
    id: Uuid,
    name: String,
    email: String,
    role: UserRole,
    created_at: DateTime<Utc>,
}

async fn find_admins(pool: &PgPool) -> Result<Vec<UserWithRole>, sqlx::Error> {
    sqlx::query_as!(
        UserWithRole,
        r#"
        SELECT id, name, email, role as "role: UserRole", created_at
        FROM users
        WHERE role = $1
        "#,
        UserRole::Admin as UserRole,
    )
    .fetch_all(pool)
    .await
}

// JSON 型のマッピング
use sqlx::types::Json;
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
struct UserPreferences {
    theme: String,
    language: String,
    notifications_enabled: bool,
}

#[derive(Debug, FromRow)]
struct UserWithPrefs {
    id: Uuid,
    name: String,
    preferences: Json<UserPreferences>,
}

async fn update_preferences(
    pool: &PgPool,
    user_id: Uuid,
    prefs: &UserPreferences,
) -> Result<(), sqlx::Error> {
    sqlx::query!(
        "UPDATE users SET preferences = $1 WHERE id = $2",
        Json(prefs) as _,
        user_id,
    )
    .execute(pool)
    .await?;
    Ok(())
}
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

### 3.4 diesel-async による非同期対応

diesel は本来同期的だが、`diesel-async` クレートを使うと非同期で操作できる。

```toml
# Cargo.toml
[dependencies]
diesel = { version = "2.2", features = ["postgres", "chrono", "uuid"] }
diesel-async = { version = "0.5", features = ["postgres", "deadpool"] }
deadpool = "0.12"
```

```rust
use diesel_async::{AsyncPgConnection, RunQueryDsl, AsyncConnection};
use diesel_async::pooled_connection::deadpool::Pool;
use diesel_async::pooled_connection::AsyncDieselConnectionManager;

type DbPool = Pool<AsyncPgConnection>;

/// 非同期コネクションプールの作成
fn create_pool(database_url: &str) -> DbPool {
    let config = AsyncDieselConnectionManager::<AsyncPgConnection>::new(database_url);
    Pool::builder(config)
        .max_size(16)
        .build()
        .expect("Failed to create pool")
}

/// 非同期での CRUD 操作
async fn create_user_async(
    pool: &DbPool,
    name: &str,
    email: &str,
) -> QueryResult<User> {
    let mut conn = pool.get().await.expect("Failed to get connection");

    let new_user = NewUser { name, email };
    diesel::insert_into(users::table)
        .values(&new_user)
        .returning(User::as_returning())
        .get_result(&mut conn)
        .await
}

/// 非同期でのトランザクション
async fn transfer_with_diesel_async(
    pool: &DbPool,
    from_id: Uuid,
    to_id: Uuid,
    amount: i64,
) -> QueryResult<()> {
    let mut conn = pool.get().await.expect("Failed to get connection");

    conn.transaction::<_, diesel::result::Error, _>(|conn| {
        Box::pin(async move {
            diesel::update(accounts::table.filter(accounts::id.eq(from_id)))
                .set(accounts::balance.eq(accounts::balance - amount))
                .execute(conn)
                .await?;

            diesel::update(accounts::table.filter(accounts::id.eq(to_id)))
                .set(accounts::balance.eq(accounts::balance + amount))
                .execute(conn)
                .await?;

            Ok(())
        })
    })
    .await
}
```

### 3.5 カスタムクエリとページネーション

```rust
use diesel::prelude::*;
use diesel::dsl::count_star;

/// ページネーション付きの検索
fn paginated_users(
    conn: &mut PgConnection,
    page: i64,
    per_page: i64,
    name_filter: Option<&str>,
) -> QueryResult<(Vec<User>, i64)> {
    let mut query = users::table.into_boxed();  // boxed() で動的クエリを有効化

    if let Some(name) = name_filter {
        query = query.filter(users::name.ilike(format!("%{}%", name)));
    }

    // 総件数の取得
    let total = users::table
        .select(count_star())
        .first::<i64>(conn)?;

    // ページネーション
    let results = query
        .order(users::created_at.desc())
        .limit(per_page)
        .offset((page - 1) * per_page)
        .select(User::as_select())
        .load(conn)?;

    Ok((results, total))
}

/// GROUP BY と集計関数
fn user_post_counts(conn: &mut PgConnection) -> QueryResult<Vec<(Uuid, String, i64)>> {
    users::table
        .left_join(posts::table)
        .group_by((users::id, users::name))
        .select((users::id, users::name, diesel::dsl::count(posts::id.nullable())))
        .order(diesel::dsl::count(posts::id.nullable()).desc())
        .load::<(Uuid, String, i64)>(conn)
}

/// サブクエリの使用
fn users_with_recent_posts(conn: &mut PgConnection) -> QueryResult<Vec<User>> {
    let recent_posters = posts::table
        .filter(posts::created_at.gt(chrono::Utc::now() - chrono::Duration::days(7)))
        .select(posts::user_id);

    users::table
        .filter(users::id.eq_any(recent_posters))
        .select(User::as_select())
        .load(conn)
}
```

### 3.6 diesel のマイグレーション管理

```bash
# diesel CLI のインストール
cargo install diesel_cli --no-default-features --features postgres

# プロジェクト初期化（diesel.toml と migrations/ ディレクトリ作成）
diesel setup

# マイグレーション作成
diesel migration generate create_users

# マイグレーション実行
diesel migration run

# マイグレーション巻き戻し
diesel migration revert

# マイグレーション状態確認
diesel migration list

# スキーマの再生成
diesel print-schema > src/schema.rs
```

```sql
-- migrations/2026-01-01-000000_create_users/up.sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- migrations/2026-01-01-000000_create_users/down.sql
DROP TABLE users;
```

diesel のマイグレーションは `up.sql` と `down.sql` のペアで管理される。ロールバック可能なマイグレーションを書くことで、開発時のスキーマ変更が容易になる。

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

### 4.4 SeaORM の高度なクエリ

```rust
use sea_orm::*;
use sea_orm::sea_query::{Expr, Func};

/// 条件付き動的クエリ
async fn search_users_sea(
    db: &DatabaseConnection,
    name_filter: Option<String>,
    email_filter: Option<String>,
    page: u64,
    per_page: u64,
) -> Result<(Vec<user::Model>, u64), DbErr> {
    let mut query = user::Entity::find();

    if let Some(name) = name_filter {
        query = query.filter(user::Column::Name.contains(&name));
    }

    if let Some(email) = email_filter {
        query = query.filter(user::Column::Email.contains(&email));
    }

    let paginator = query
        .order_by_desc(user::Column::CreatedAt)
        .paginate(db, per_page);

    let total = paginator.num_pages().await?;
    let users = paginator.fetch_page(page).await?;

    Ok((users, total))
}

/// カスタム SELECT とグループ化
async fn user_post_stats(
    db: &DatabaseConnection,
) -> Result<Vec<(Uuid, String, i64)>, DbErr> {
    #[derive(Debug, FromQueryResult)]
    struct UserPostCount {
        user_id: Uuid,
        user_name: String,
        post_count: i64,
    }

    let results = user::Entity::find()
        .select_only()
        .column(user::Column::Id)
        .column(user::Column::Name)
        .column_as(post::Column::Id.count(), "post_count")
        .join(JoinType::LeftJoin, user::Relation::Posts.def())
        .group_by(user::Column::Id)
        .group_by(user::Column::Name)
        .order_by_desc(Expr::col(post::Column::Id).count())
        .into_model::<UserPostCount>()
        .all(db)
        .await?;

    Ok(results.into_iter().map(|r| (r.user_id, r.user_name, r.post_count)).collect())
}

/// トランザクション
async fn create_user_with_post(
    db: &DatabaseConnection,
    name: &str,
    email: &str,
    post_title: &str,
    post_body: &str,
) -> Result<(user::Model, post::Model), DbErr> {
    let txn = db.begin().await?;

    let new_user = user::ActiveModel {
        id: Set(Uuid::new_v4()),
        name: Set(name.to_string()),
        email: Set(email.to_string()),
        created_at: Set(chrono::Utc::now()),
        ..Default::default()
    };
    let user = new_user.insert(&txn).await?;

    let new_post = post::ActiveModel {
        id: Set(Uuid::new_v4()),
        user_id: Set(user.id),
        title: Set(post_title.to_string()),
        body: Set(post_body.to_string()),
        published: Set(false),
        created_at: Set(chrono::Utc::now()),
        ..Default::default()
    };
    let post = new_post.insert(&txn).await?;

    txn.commit().await?;
    Ok((user, post))
}
```

### 4.5 SeaORM CLI とコード生成

```bash
# sea-orm-cli のインストール
cargo install sea-orm-cli

# データベースからエンティティを自動生成
sea-orm-cli generate entity \
    --database-url "postgres://user:pass@localhost:5432/mydb" \
    --output-dir src/entities \
    --with-serde both \
    --date-time-crate chrono

# マイグレーション作成
sea-orm-cli migrate generate create_users_table

# マイグレーション実行
sea-orm-cli migrate up

# マイグレーション巻き戻し
sea-orm-cli migrate down
```

エンティティの自動生成は既存のデータベーススキーマから Rust コードを生成するため、特にレガシーデータベースとの統合時に非常に便利である。

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

## 6. 接続プールの設計と最適化

### 6.1 接続プールの基本原則

データベース接続はリソースコストが高いため、接続プールを使って再利用する。プールサイズの設定はアプリケーションのパフォーマンスに直接影響する。

```rust
use sqlx::postgres::PgPoolOptions;

/// プロダクション向けの接続プール設定
async fn create_production_pool(database_url: &str) -> Result<PgPool, sqlx::Error> {
    PgPoolOptions::new()
        // 最大接続数: CPU コア数 * 2 + ディスクスピンドル数が目安
        // 一般的には 10-20 が適切
        .max_connections(20)
        // 最小接続数: アイドル時にも維持する接続数
        .min_connections(5)
        // 接続タイムアウト: プールから接続を取得するまでの最大待ち時間
        .acquire_timeout(std::time::Duration::from_secs(5))
        // アイドルタイムアウト: 使われていない接続を閉じるまでの時間
        .idle_timeout(std::time::Duration::from_secs(600))
        // 接続の最大生存時間: 古い接続を定期的にリフレッシュ
        .max_lifetime(std::time::Duration::from_secs(1800))
        // 接続時に実行される SQL（接続の検証やセッション設定）
        .after_connect(|conn, _meta| {
            Box::pin(async move {
                sqlx::query("SET timezone = 'UTC'")
                    .execute(conn)
                    .await?;
                sqlx::query("SET statement_timeout = '30s'")
                    .execute(conn)
                    .await?;
                Ok(())
            })
        })
        .connect(database_url)
        .await
}
```

### 6.2 接続プールサイズの決定

接続プールサイズは以下の要因を考慮して決定する。

```
推奨プールサイズの計算式:

  pool_size = (CPU コア数 * 2) + 有効ディスクスピンドル数

例:
  4コア CPU + SSD (1 スピンドル相当)
  → pool_size = (4 * 2) + 1 = 9 ≈ 10

注意事項:
  - PostgreSQL のデフォルト最大接続数は 100
  - 複数のアプリケーションインスタンスがある場合は合計を考慮
  - pool_size = max_connections(DB) / app_instances - margin
```

### 6.3 接続プールの監視

```rust
use sqlx::PgPool;
use tracing::info;

/// 接続プールの状態をログに出力
async fn log_pool_status(pool: &PgPool) {
    let size = pool.size();
    let idle = pool.num_idle();
    let acquired = size - idle as u32;

    info!(
        pool.size = size,
        pool.idle = idle,
        pool.acquired = acquired,
        "Connection pool status"
    );

    // プール使用率が 80% を超えたら警告
    if (acquired as f64 / size as f64) > 0.8 {
        tracing::warn!(
            "Connection pool utilization is high: {}/{}",
            acquired,
            size
        );
    }
}

/// ヘルスチェック用のエンドポイント
async fn health_check(pool: &PgPool) -> Result<(), sqlx::Error> {
    sqlx::query("SELECT 1")
        .execute(pool)
        .await?;
    Ok(())
}
```

---

## 7. テスト戦略

### 7.1 テスト用データベースの管理

データベースのテストでは、各テストが独立して実行できるようにすることが重要である。

```rust
/// テスト用のデータベースを作成するヘルパー
async fn create_test_database() -> PgPool {
    let db_name = format!("test_{}", Uuid::new_v4().to_string().replace('-', ""));
    let admin_url = "postgres://user:pass@localhost:5432/postgres";

    // テスト用データベースを作成
    let admin_pool = PgPool::connect(admin_url).await.unwrap();
    sqlx::query(&format!("CREATE DATABASE {}", db_name))
        .execute(&admin_pool)
        .await
        .unwrap();

    // テスト用データベースに接続してマイグレーション実行
    let test_url = format!("postgres://user:pass@localhost:5432/{}", db_name);
    let pool = PgPool::connect(&test_url).await.unwrap();
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .unwrap();

    pool
}

/// テスト終了後にデータベースを削除
async fn cleanup_test_database(pool: PgPool, db_name: &str) {
    pool.close().await;

    let admin_url = "postgres://user:pass@localhost:5432/postgres";
    let admin_pool = PgPool::connect(admin_url).await.unwrap();
    sqlx::query(&format!("DROP DATABASE IF EXISTS {}", db_name))
        .execute(&admin_pool)
        .await
        .unwrap();
}
```

### 7.2 トランザクションベースのテスト

各テストをトランザクション内で実行し、終了後にロールバックすることで高速かつクリーンなテストを実現する。

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::PgPool;

    /// トランザクション内でテストを実行するヘルパー
    /// テスト終了時に自動ロールバックされる
    async fn with_test_tx<F, Fut>(pool: &PgPool, f: F)
    where
        F: FnOnce(sqlx::Transaction<'_, sqlx::Postgres>) -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        let tx = pool.begin().await.unwrap();
        f(tx).await;
        // tx は Drop 時に自動ロールバック
    }

    #[sqlx::test]
    async fn test_create_user(pool: PgPool) {
        // sqlx::test マクロはテスト用DBを自動管理
        let user = create_user(&pool, "Test User", "test@example.com")
            .await
            .unwrap();

        assert_eq!(user.name, "Test User");
        assert_eq!(user.email, "test@example.com");
    }

    #[sqlx::test(fixtures("users"))]
    async fn test_list_users(pool: PgPool) {
        // fixtures ディレクトリから初期データを投入
        let users = list_users(&pool, 10).await.unwrap();
        assert!(!users.is_empty());
    }

    #[sqlx::test]
    async fn test_update_email(pool: PgPool) {
        let user = create_user(&pool, "Test", "old@example.com")
            .await
            .unwrap();

        let updated = update_email(&pool, user.id, "new@example.com")
            .await
            .unwrap();

        assert!(updated);

        let found = sqlx::query_as!(
            User,
            "SELECT id, name, email, created_at FROM users WHERE id = $1",
            user.id
        )
        .fetch_one(&pool)
        .await
        .unwrap();

        assert_eq!(found.email, "new@example.com");
    }
}
```

### 7.3 diesel のテスト

```rust
#[cfg(test)]
mod tests {
    use diesel::prelude::*;
    use diesel::Connection;

    /// diesel でのトランザクションベーステスト
    #[test]
    fn test_create_user_diesel() {
        let mut conn = PgConnection::establish("postgres://...")
            .expect("Failed to connect");

        // test_transaction はトランザクション内でテストを実行し、
        // 終了後に自動ロールバックする
        conn.test_transaction::<_, diesel::result::Error, _>(|conn| {
            let user = create_user(conn, "Test", "test@example.com")?;
            assert_eq!(user.name, "Test");
            assert_eq!(user.email, "test@example.com");

            let found = find_user_by_email(conn, "test@example.com")?;
            assert_eq!(found.id, user.id);

            Ok(())
        });
    }
}
```

### 7.4 テストフィクスチャの管理

```sql
-- fixtures/users.sql（sqlx::test の fixtures で使用）
INSERT INTO users (id, name, email, created_at) VALUES
    ('550e8400-e29b-41d4-a716-446655440001', 'Alice', 'alice@example.com', '2026-01-01 00:00:00+00'),
    ('550e8400-e29b-41d4-a716-446655440002', 'Bob', 'bob@example.com', '2026-01-02 00:00:00+00'),
    ('550e8400-e29b-41d4-a716-446655440003', 'Charlie', 'charlie@example.com', '2026-01-03 00:00:00+00');
```

```rust
/// テストデータビルダーパターン
struct UserBuilder {
    name: String,
    email: String,
}

impl UserBuilder {
    fn new() -> Self {
        Self {
            name: "Default User".to_string(),
            email: format!("user-{}@example.com", Uuid::new_v4()),
        }
    }

    fn name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    fn email(mut self, email: &str) -> Self {
        self.email = email.to_string();
        self
    }

    async fn build(self, pool: &PgPool) -> User {
        create_user(pool, &self.name, &self.email)
            .await
            .expect("Failed to create test user")
    }
}

// 使用例
#[sqlx::test]
async fn test_with_builder(pool: PgPool) {
    let alice = UserBuilder::new()
        .name("Alice")
        .email("alice@test.com")
        .build(&pool)
        .await;

    let bob = UserBuilder::new()
        .name("Bob")
        .build(&pool)  // email は自動生成
        .await;

    assert_ne!(alice.id, bob.id);
}
```

---

## 8. アンチパターン

### 8.1 N+1 クエリ問題

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

### 8.2 トランザクション未使用での複数操作

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

### 8.3 接続プールの枯渇

```rust
// NG: 接続を長時間保持する
async fn bad_long_running(pool: &PgPool) -> Result<()> {
    let mut tx = pool.begin().await?;  // 接続を1つ確保

    // 外部 API の呼び出しなど、DB と無関係な処理
    let response = reqwest::get("https://api.example.com/slow-endpoint")
        .await?;  // ← この間ずっと DB 接続を保持！

    sqlx::query!("INSERT INTO results (data) VALUES ($1)", response.text().await?)
        .execute(&mut *tx).await?;

    tx.commit().await?;
    Ok(())
}

// OK: DB 操作と非 DB 操作を分離
async fn good_minimal_connection(pool: &PgPool) -> Result<()> {
    // 1. まず外部 API を呼び出す（DB 接続不要）
    let response = reqwest::get("https://api.example.com/slow-endpoint")
        .await?;
    let data = response.text().await?;

    // 2. DB 操作は最小限の時間で完了
    sqlx::query!("INSERT INTO results (data) VALUES ($1)", data)
        .execute(pool).await?;

    Ok(())
}
```

### 8.4 不適切なインデックス設計

```rust
// NG: インデックスなしのカラムで頻繁に検索
async fn bad_search(pool: &PgPool, status: &str) -> Result<Vec<Order>> {
    // status カラムにインデックスがない場合、全テーブルスキャンになる
    sqlx::query_as!(Order, "SELECT * FROM orders WHERE status = $1", status)
        .fetch_all(pool).await
}

// OK: 適切なインデックスをマイグレーションで追加
// migrations/xxx_add_orders_status_index.sql:
// CREATE INDEX CONCURRENTLY idx_orders_status ON orders (status);
//
// 複合インデックスの場合はカラム順序が重要:
// CREATE INDEX idx_orders_status_date ON orders (status, created_at DESC);
// ↑ WHERE status = ? ORDER BY created_at DESC のクエリに最適
```

### 8.5 SELECT * の乱用

```rust
// NG: 不要なカラムも含めて全カラム取得
async fn bad_get_names(pool: &PgPool) -> Result<Vec<String>> {
    let users = sqlx::query_as!(User, "SELECT * FROM users")  // 全カラム取得
        .fetch_all(pool).await?;
    Ok(users.into_iter().map(|u| u.name).collect())  // name しか使わない
}

// OK: 必要なカラムだけ取得
async fn good_get_names(pool: &PgPool) -> Result<Vec<String>> {
    let rows = sqlx::query_scalar!("SELECT name FROM users")
        .fetch_all(pool).await?;
    Ok(rows)
}
```

---

## 9. パフォーマンスチューニング

### 9.1 EXPLAIN ANALYZE による実行計画の分析

```rust
/// クエリの実行計画を取得するユーティリティ
async fn explain_query(pool: &PgPool, query: &str) -> Result<String, sqlx::Error> {
    let rows = sqlx::query_scalar::<_, String>(
        &format!("EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {}", query)
    )
    .fetch_all(pool)
    .await?;

    Ok(rows.join("\n"))
}

// 使用例
async fn debug_slow_query(pool: &PgPool) {
    let plan = explain_query(
        pool,
        "SELECT u.*, count(p.id) FROM users u LEFT JOIN posts p ON p.user_id = u.id GROUP BY u.id"
    ).await.unwrap();

    println!("Query Plan:\n{}", plan);
    // Seq Scan が出ている場合はインデックスの追加を検討
    // Nested Loop が出ている場合は JOIN の最適化を検討
}
```

### 9.2 プリペアドステートメントのキャッシュ

```rust
// sqlx の query! マクロはプリペアドステートメントを自動的にキャッシュする。
// 手動でプリペアドステートメントを管理する必要がある場合:

use sqlx::Statement;

async fn cached_query_example(pool: &PgPool) -> Result<Vec<User>, sqlx::Error> {
    // プリペアドステートメントは接続ごとにキャッシュされる
    // query_as! マクロを使う場合は自動管理されるため手動管理は不要

    // query() の場合は persistent() で制御可能
    sqlx::query_as::<_, User>("SELECT id, name, email, created_at FROM users WHERE email = $1")
        .bind("test@example.com")
        .persistent(true)  // プリペアドステートメントをキャッシュ（デフォルト: true）
        .fetch_all(pool)
        .await
}
```

### 9.3 バッチ処理とパイプライン

```rust
/// 大量のデータを効率的にバッチ処理
async fn batch_update_status(
    pool: &PgPool,
    user_ids: &[Uuid],
    new_status: &str,
) -> Result<u64, sqlx::Error> {
    // 一度に更新する数を制限してデッドロックリスクを軽減
    const BATCH_SIZE: usize = 1000;
    let mut total_affected = 0u64;

    for chunk in user_ids.chunks(BATCH_SIZE) {
        let result = sqlx::query!(
            "UPDATE users SET status = $1 WHERE id = ANY($2)",
            new_status,
            chunk,
        )
        .execute(pool)
        .await?;

        total_affected += result.rows_affected();
    }

    Ok(total_affected)
}

/// COPY プロトコルによる超高速バルクロード（PostgreSQL 固有）
async fn bulk_load_with_copy(
    pool: &PgPool,
    csv_data: &str,
) -> Result<u64, sqlx::Error> {
    let mut conn = pool.acquire().await?;

    let copy_result = sqlx::query(
        "COPY users (name, email) FROM STDIN WITH (FORMAT CSV, HEADER true)"
    )
    .execute(&mut *conn)
    .await?;

    Ok(copy_result.rows_affected())
}
```

### 9.4 読み取りレプリカの活用

```rust
use sqlx::PgPool;

/// 読み取り/書き込みの分離パターン
struct DatabasePools {
    writer: PgPool,   // プライマリ（書き込み用）
    reader: PgPool,   // レプリカ（読み取り用）
}

impl DatabasePools {
    async fn new(
        writer_url: &str,
        reader_url: &str,
    ) -> Result<Self, sqlx::Error> {
        let writer = PgPoolOptions::new()
            .max_connections(10)
            .connect(writer_url)
            .await?;

        let reader = PgPoolOptions::new()
            .max_connections(20)  // 読み取りの方が多いため大きく設定
            .connect(reader_url)
            .await?;

        Ok(Self { writer, reader })
    }

    /// 書き込み操作はプライマリに送信
    fn writer(&self) -> &PgPool {
        &self.writer
    }

    /// 読み取り操作はレプリカに送信
    fn reader(&self) -> &PgPool {
        &self.reader
    }
}

// Axum での使用例
async fn list_users_handler(
    State(pools): State<DatabasePools>,
) -> Json<Vec<User>> {
    let users = sqlx::query_as!(User, "SELECT * FROM users LIMIT 50")
        .fetch_all(pools.reader())  // レプリカから読み取り
        .await
        .unwrap();
    Json(users)
}

async fn create_user_handler(
    State(pools): State<DatabasePools>,
    Json(payload): Json<CreateUserRequest>,
) -> Json<User> {
    let user = create_user(pools.writer(), &payload.name, &payload.email)  // プライマリに書き込み
        .await
        .unwrap();
    Json(user)
}
```

---

## 10. エラーハンドリングのベストプラクティス

### 10.1 カスタムエラー型の定義

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Record not found: {entity} with id {id}")]
    NotFound { entity: String, id: String },

    #[error("Duplicate entry: {field} = {value}")]
    DuplicateEntry { field: String, value: String },

    #[error("Constraint violation: {0}")]
    ConstraintViolation(String),

    #[error("Connection pool exhausted")]
    PoolExhausted,
}

impl From<sqlx::Error> for AppError {
    fn from(err: sqlx::Error) -> Self {
        match &err {
            sqlx::Error::RowNotFound => AppError::NotFound {
                entity: "unknown".to_string(),
                id: "unknown".to_string(),
            },
            sqlx::Error::Database(db_err) => {
                // PostgreSQL エラーコードに基づく分類
                if let Some(code) = db_err.code() {
                    match code.as_ref() {
                        "23505" => AppError::DuplicateEntry {
                            field: db_err.constraint()
                                .unwrap_or("unknown")
                                .to_string(),
                            value: "unknown".to_string(),
                        },
                        "23503" => AppError::ConstraintViolation(
                            db_err.message().to_string()
                        ),
                        _ => AppError::Database(err),
                    }
                } else {
                    AppError::Database(err)
                }
            }
            sqlx::Error::PoolTimedOut => AppError::PoolExhausted,
            _ => AppError::Database(err),
        }
    }
}
```

### 10.2 リトライロジック

```rust
use std::time::Duration;
use tokio::time::sleep;

/// デッドロックやタイムアウト時の自動リトライ
async fn with_retry<F, Fut, T>(
    max_retries: u32,
    base_delay: Duration,
    f: F,
) -> Result<T, sqlx::Error>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T, sqlx::Error>>,
{
    let mut retries = 0;

    loop {
        match f().await {
            Ok(result) => return Ok(result),
            Err(err) if is_retryable(&err) && retries < max_retries => {
                retries += 1;
                let delay = base_delay * 2u32.pow(retries - 1);  // 指数バックオフ
                let jitter = Duration::from_millis(rand::random::<u64>() % 100);
                tracing::warn!(
                    retry = retries,
                    max_retries = max_retries,
                    "Retryable database error, waiting {:?}",
                    delay + jitter
                );
                sleep(delay + jitter).await;
            }
            Err(err) => return Err(err),
        }
    }
}

/// リトライ可能なエラーかどうかを判定
fn is_retryable(err: &sqlx::Error) -> bool {
    match err {
        sqlx::Error::PoolTimedOut => true,
        sqlx::Error::Database(db_err) => {
            if let Some(code) = db_err.code() {
                matches!(
                    code.as_ref(),
                    "40001"   // serialization_failure
                    | "40P01" // deadlock_detected
                    | "57P03" // cannot_connect_now
                    | "08006" // connection_failure
                )
            } else {
                false
            }
        }
        sqlx::Error::Io(_) => true,  // ネットワークエラー
        _ => false,
    }
}

// 使用例
async fn reliable_transfer(
    pool: &PgPool,
    from: Uuid,
    to: Uuid,
    amount: i64,
) -> Result<(), sqlx::Error> {
    with_retry(3, Duration::from_millis(100), || async {
        let mut tx = pool.begin().await?;

        sqlx::query!(
            "UPDATE accounts SET balance = balance - $1 WHERE id = $2",
            amount, from
        ).execute(&mut *tx).await?;

        sqlx::query!(
            "UPDATE accounts SET balance = balance + $1 WHERE id = $2",
            amount, to
        ).execute(&mut *tx).await?;

        tx.commit().await
    }).await
}
```

---

## 11. 実践的なリポジトリパターン

### 11.1 リポジトリトレイトの定義

```rust
use async_trait::async_trait;

/// リポジトリパターン — DB 実装の抽象化
#[async_trait]
pub trait UserRepository: Send + Sync {
    async fn find_by_id(&self, id: Uuid) -> Result<Option<User>, AppError>;
    async fn find_by_email(&self, email: &str) -> Result<Option<User>, AppError>;
    async fn list(&self, page: i64, per_page: i64) -> Result<(Vec<User>, i64), AppError>;
    async fn create(&self, name: &str, email: &str) -> Result<User, AppError>;
    async fn update(&self, id: Uuid, name: &str, email: &str) -> Result<User, AppError>;
    async fn delete(&self, id: Uuid) -> Result<bool, AppError>;
}

/// sqlx による実装
pub struct PgUserRepository {
    pool: PgPool,
}

impl PgUserRepository {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl UserRepository for PgUserRepository {
    async fn find_by_id(&self, id: Uuid) -> Result<Option<User>, AppError> {
        let user = sqlx::query_as!(
            User,
            "SELECT id, name, email, created_at FROM users WHERE id = $1",
            id,
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(user)
    }

    async fn find_by_email(&self, email: &str) -> Result<Option<User>, AppError> {
        let user = sqlx::query_as!(
            User,
            "SELECT id, name, email, created_at FROM users WHERE email = $1",
            email,
        )
        .fetch_optional(&self.pool)
        .await?;

        Ok(user)
    }

    async fn list(&self, page: i64, per_page: i64) -> Result<(Vec<User>, i64), AppError> {
        let total = sqlx::query_scalar!("SELECT COUNT(*) FROM users")
            .fetch_one(&self.pool)
            .await?
            .unwrap_or(0);

        let users = sqlx::query_as!(
            User,
            "SELECT id, name, email, created_at FROM users ORDER BY created_at DESC LIMIT $1 OFFSET $2",
            per_page,
            (page - 1) * per_page,
        )
        .fetch_all(&self.pool)
        .await?;

        Ok((users, total))
    }

    async fn create(&self, name: &str, email: &str) -> Result<User, AppError> {
        let user = sqlx::query_as!(
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
        .fetch_one(&self.pool)
        .await?;

        Ok(user)
    }

    async fn update(&self, id: Uuid, name: &str, email: &str) -> Result<User, AppError> {
        let user = sqlx::query_as!(
            User,
            r#"
            UPDATE users SET name = $1, email = $2, updated_at = NOW()
            WHERE id = $3
            RETURNING id, name, email, created_at
            "#,
            name,
            email,
            id,
        )
        .fetch_optional(&self.pool)
        .await?
        .ok_or(AppError::NotFound {
            entity: "User".to_string(),
            id: id.to_string(),
        })?;

        Ok(user)
    }

    async fn delete(&self, id: Uuid) -> Result<bool, AppError> {
        let result = sqlx::query!("DELETE FROM users WHERE id = $1", id)
            .execute(&self.pool)
            .await?;

        Ok(result.rows_affected() > 0)
    }
}
```

### 11.2 テスト用モックリポジトリ

```rust
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

/// テスト用のインメモリリポジトリ
pub struct MockUserRepository {
    users: Arc<Mutex<HashMap<Uuid, User>>>,
}

impl MockUserRepository {
    pub fn new() -> Self {
        Self {
            users: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl UserRepository for MockUserRepository {
    async fn find_by_id(&self, id: Uuid) -> Result<Option<User>, AppError> {
        let users = self.users.lock().unwrap();
        Ok(users.get(&id).cloned())
    }

    async fn find_by_email(&self, email: &str) -> Result<Option<User>, AppError> {
        let users = self.users.lock().unwrap();
        Ok(users.values().find(|u| u.email == email).cloned())
    }

    async fn list(&self, page: i64, per_page: i64) -> Result<(Vec<User>, i64), AppError> {
        let users = self.users.lock().unwrap();
        let total = users.len() as i64;
        let offset = ((page - 1) * per_page) as usize;
        let limit = per_page as usize;

        let mut sorted: Vec<User> = users.values().cloned().collect();
        sorted.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        let page_users = sorted.into_iter().skip(offset).take(limit).collect();
        Ok((page_users, total))
    }

    async fn create(&self, name: &str, email: &str) -> Result<User, AppError> {
        let mut users = self.users.lock().unwrap();
        let user = User {
            id: Uuid::new_v4(),
            name: name.to_string(),
            email: email.to_string(),
            created_at: chrono::Utc::now(),
        };
        users.insert(user.id, user.clone());
        Ok(user)
    }

    async fn update(&self, id: Uuid, name: &str, email: &str) -> Result<User, AppError> {
        let mut users = self.users.lock().unwrap();
        let user = users.get_mut(&id)
            .ok_or(AppError::NotFound {
                entity: "User".to_string(),
                id: id.to_string(),
            })?;

        user.name = name.to_string();
        user.email = email.to_string();
        Ok(user.clone())
    }

    async fn delete(&self, id: Uuid) -> Result<bool, AppError> {
        let mut users = self.users.lock().unwrap();
        Ok(users.remove(&id).is_some())
    }
}

// テストでの使用例
#[tokio::test]
async fn test_user_service_with_mock() {
    let repo = MockUserRepository::new();
    let service = UserService::new(Arc::new(repo));

    let user = service.register("Alice", "alice@example.com").await.unwrap();
    assert_eq!(user.name, "Alice");

    let found = service.find_by_email("alice@example.com").await.unwrap();
    assert!(found.is_some());
}
```

---

## 12. FAQ

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

### Q4. マイグレーションの本番運用はどうする？

**A.** マイグレーションはアプリケーション起動時に自動実行するか、CI/CD パイプラインで実行する。

```rust
// アプリケーション起動時にマイグレーションを実行
#[tokio::main]
async fn main() {
    let pool = PgPool::connect("postgres://...").await.unwrap();

    // マイグレーションの実行
    sqlx::migrate!("./migrations")
        .run(&pool)
        .await
        .expect("Failed to run migrations");

    // アプリケーションの起動
    start_server(pool).await;
}
```

本番環境では以下の点に注意する:
- **ゼロダウンタイム**: `ALTER TABLE` はロックを取得するため、大きなテーブルでは `ALTER TABLE ... ADD COLUMN ... DEFAULT ...` を使う（PostgreSQL 11+ では即座に完了）
- **ロールバック計画**: 各マイグレーションに対応するロールバック SQL を用意する
- **テスト**: ステージング環境で事前にマイグレーションを検証する

### Q5. 大量データのページネーションで OFFSET が遅い場合は？

**A.** OFFSET は指定された行数分をスキャンするため、大きなオフセットでは遅くなる。Cursor-based ページネーションを使う。

```rust
/// Cursor-based ページネーション（OFFSET を使わない）
async fn list_users_cursor(
    pool: &PgPool,
    cursor: Option<DateTime<Utc>>,  // 前ページの最後の created_at
    limit: i64,
) -> Result<Vec<User>, sqlx::Error> {
    match cursor {
        Some(after) => {
            sqlx::query_as!(
                User,
                r#"
                SELECT id, name, email, created_at
                FROM users
                WHERE created_at < $1
                ORDER BY created_at DESC
                LIMIT $2
                "#,
                after,
                limit,
            )
            .fetch_all(pool)
            .await
        }
        None => {
            sqlx::query_as!(
                User,
                r#"
                SELECT id, name, email, created_at
                FROM users
                ORDER BY created_at DESC
                LIMIT $1
                "#,
                limit,
            )
            .fetch_all(pool)
            .await
        }
    }
}
```

### Q6. 複数データベースへの対応（マルチテナント）は？

**A.** マルチテナントの実装方法はいくつかある。

```rust
/// スキーマベースのマルチテナント（PostgreSQL）
async fn with_tenant_schema(
    pool: &PgPool,
    tenant_id: &str,
) -> Result<Vec<User>, sqlx::Error> {
    let mut conn = pool.acquire().await?;

    // テナントのスキーマに切り替え
    sqlx::query(&format!("SET search_path TO tenant_{}, public", tenant_id))
        .execute(&mut *conn)
        .await?;

    // 以降のクエリはテナントのスキーマで実行される
    sqlx::query_as!(User, "SELECT id, name, email, created_at FROM users")
        .fetch_all(&mut *conn)
        .await
}

/// データベース分離型マルチテナント
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

struct TenantPools {
    pools: Arc<RwLock<HashMap<String, PgPool>>>,
}

impl TenantPools {
    async fn get_pool(&self, tenant_id: &str) -> Option<PgPool> {
        let pools = self.pools.read().await;
        pools.get(tenant_id).cloned()
    }

    async fn add_tenant(&self, tenant_id: &str, database_url: &str) -> Result<(), sqlx::Error> {
        let pool = PgPool::connect(database_url).await?;
        let mut pools = self.pools.write().await;
        pools.insert(tenant_id.to_string(), pool);
        Ok(())
    }
}
```

---

## 13. まとめ

| 項目 | ポイント |
|------|---------|
| **sqlx** | 生 SQL 派向け、コンパイル時検証、非同期ネイティブ |
| **diesel** | 型安全 DSL、コンパイル時保証最強、同期メイン（async 拡張あり） |
| **SeaORM** | ActiveRecord パターン、Rails 的な開発体験、非同期ネイティブ |
| **選定基準** | SQL 制御 → sqlx、型安全 → diesel、生産性 → SeaORM |
| **接続プール** | 適切なサイズ設定、監視、タイムアウト設定が重要 |
| **テスト** | トランザクションベースのテスト、リポジトリパターンでモック可能に |
| **エラーハンドリング** | PostgreSQL エラーコードの分類、リトライロジックの実装 |
| **パフォーマンス** | N+1 回避、Cursor-based ページネーション、EXPLAIN ANALYZE 活用 |
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
5. **diesel-async** — https://github.com/weiznich/diesel_async
6. **PostgreSQL 公式ドキュメント** — "Connection Pooling" — https://www.postgresql.org/docs/current/runtime-config-connection.html
7. **Use The Index, Luke** — SQL インデックス設計ガイド — https://use-the-index-luke.com/
