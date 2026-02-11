# データベース -- database/sql, sqlx, GORM

> Goはdatabase/sqlで標準的なDB接続を提供し、sqlx・GORMで生産性を向上させ、マイグレーション・接続プールで本番運用を支える。

---

## この章で学ぶこと

1. **database/sql** -- 標準ライブラリのDB操作
2. **sqlx / GORM** -- 高レベルライブラリの使い分け
3. **接続プールとマイグレーション** -- 本番運用のベストプラクティス

---

### コード例 1: database/sql 基本操作

```go
import (
    "database/sql"
    _ "github.com/lib/pq"
)

func main() {
    db, err := sql.Open("postgres", "postgres://user:pass@localhost/mydb?sslmode=disable")
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    // 接続プール設定
    db.SetMaxOpenConns(25)
    db.SetMaxIdleConns(5)
    db.SetConnMaxLifetime(5 * time.Minute)

    if err := db.Ping(); err != nil {
        log.Fatal(err)
    }
}
```

### コード例 2: CRUD操作 (database/sql)

```go
// SELECT (単一行)
func getUser(ctx context.Context, db *sql.DB, id int) (*User, error) {
    var u User
    err := db.QueryRowContext(ctx,
        "SELECT id, name, email FROM users WHERE id = $1", id,
    ).Scan(&u.ID, &u.Name, &u.Email)
    if errors.Is(err, sql.ErrNoRows) {
        return nil, ErrNotFound
    }
    return &u, err
}

// SELECT (複数行)
func listUsers(ctx context.Context, db *sql.DB) ([]User, error) {
    rows, err := db.QueryContext(ctx, "SELECT id, name, email FROM users")
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    var users []User
    for rows.Next() {
        var u User
        if err := rows.Scan(&u.ID, &u.Name, &u.Email); err != nil {
            return nil, err
        }
        users = append(users, u)
    }
    return users, rows.Err()
}

// INSERT
func createUser(ctx context.Context, db *sql.DB, u *User) error {
    return db.QueryRowContext(ctx,
        "INSERT INTO users (name, email) VALUES ($1, $2) RETURNING id",
        u.Name, u.Email,
    ).Scan(&u.ID)
}
```

### コード例 3: sqlx の活用

```go
import "github.com/jmoiron/sqlx"

type User struct {
    ID    int    `db:"id"`
    Name  string `db:"name"`
    Email string `db:"email"`
}

func getUser(ctx context.Context, db *sqlx.DB, id int) (*User, error) {
    var u User
    err := db.GetContext(ctx, &u, "SELECT * FROM users WHERE id = $1", id)
    return &u, err
}

func listUsers(ctx context.Context, db *sqlx.DB) ([]User, error) {
    var users []User
    err := db.SelectContext(ctx, &users, "SELECT * FROM users ORDER BY id")
    return users, err
}

// Named Query
func createUser(ctx context.Context, db *sqlx.DB, u *User) error {
    query := `INSERT INTO users (name, email) VALUES (:name, :email) RETURNING id`
    rows, err := db.NamedQueryContext(ctx, query, u)
    if err != nil {
        return err
    }
    defer rows.Close()
    rows.Next()
    return rows.Scan(&u.ID)
}
```

### コード例 4: GORM の活用

```go
import "gorm.io/gorm"

type User struct {
    gorm.Model
    Name  string `gorm:"size:100;not null"`
    Email string `gorm:"uniqueIndex;not null"`
}

func main() {
    db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{})
    if err != nil {
        log.Fatal(err)
    }

    db.AutoMigrate(&User{})

    // Create
    user := User{Name: "Tanaka", Email: "tanaka@example.com"}
    db.Create(&user)

    // Read
    var found User
    db.First(&found, user.ID)
    db.Where("email = ?", "tanaka@example.com").First(&found)

    // Update
    db.Model(&found).Update("name", "Yamada")

    // Delete (soft delete)
    db.Delete(&found)
}
```

### コード例 5: トランザクション

```go
// database/sql
func transferFunds(ctx context.Context, db *sql.DB, from, to int, amount float64) error {
    tx, err := db.BeginTx(ctx, nil)
    if err != nil {
        return err
    }
    defer tx.Rollback() // commitされればno-op

    _, err = tx.ExecContext(ctx,
        "UPDATE accounts SET balance = balance - $1 WHERE id = $2", amount, from)
    if err != nil {
        return err
    }

    _, err = tx.ExecContext(ctx,
        "UPDATE accounts SET balance = balance + $1 WHERE id = $2", amount, to)
    if err != nil {
        return err
    }

    return tx.Commit()
}
```

---

## 2. ASCII図解

### 図1: database/sql 接続プール

```
アプリケーション
  │
  ▼
┌──────────────────────────────────┐
│        database/sql              │
│  ┌────────────────────────────┐  │
│  │      Connection Pool       │  │
│  │  ┌────┐ ┌────┐ ┌────┐    │  │
│  │  │Conn│ │Conn│ │Conn│    │  │  MaxOpenConns = 25
│  │  │ 1  │ │ 2  │ │ 3  │... │  │  MaxIdleConns = 5
│  │  └────┘ └────┘ └────┘    │  │
│  └────────────────────────────┘  │
│              │                    │
│  ┌───────────▼───────────────┐   │
│  │      Driver (lib/pq)      │   │
│  └───────────┬───────────────┘   │
└──────────────┼───────────────────┘
               │ TCP
          ┌────▼─────┐
          │PostgreSQL │
          └──────────┘
```

### 図2: ORM vs 生SQL の選択

```
        複雑度
          ▲
          │   ┌─────────────┐
     高   │   │  生SQL +     │  レポート/分析
          │   │  database/sql│  複雑なJOIN
          │   └─────────────┘
          │
     中   │   ┌─────────────┐
          │   │   sqlx       │  型安全な生SQL
          │   └─────────────┘
          │
     低   │   ┌─────────────┐
          │   │   GORM       │  CRUD中心
          │   └─────────────┘  のアプリ
          │
          └──────────────────────> 開発速度
              低              高
```

### 図3: マイグレーションフロー

```
V1_create_users.sql
    │
    ▼
V2_add_email_index.sql
    │
    ▼
V3_create_orders.sql
    │
    ▼
V4_add_status_column.sql

┌──────────────────────────────┐
│ schema_migrations テーブル    │
│                              │
│ version │ applied_at         │
│ --------│------------------- │
│ 1       │ 2024-01-01 00:00  │
│ 2       │ 2024-01-15 00:00  │
│ 3       │ 2024-02-01 00:00  │
│ 4       │ (pending)         │
└──────────────────────────────┘
```

---

## 3. 比較表

### 表1: Go DB ライブラリ比較

| 項目 | database/sql | sqlx | GORM |
|------|-------------|------|------|
| 型安全性 | 手動Scan | structタグ | 完全ORM |
| SQL記述 | 生SQL | 生SQL | メソッドチェーン |
| 学習コスト | 低 | 低 | 中 |
| パフォーマンス | 最高 | 高 | 中 |
| マイグレーション | なし | なし | AutoMigrate |
| N+1検出 | なし | なし | なし (plugin) |
| 推奨場面 | 小規模/高性能 | 中規模 | CRUD中心 |

### 表2: マイグレーションツール比較

| ツール | 言語 | 特徴 |
|--------|------|------|
| golang-migrate | Go | 軽量、SQL/Go両対応 |
| goose | Go | シンプル、Go関数対応 |
| atlas | Go | 宣言的、HCL定義 |
| GORM AutoMigrate | Go | 自動、本番非推奨 |
| Flyway | Java | 企業向け、多DB対応 |

---

## 4. アンチパターン

### アンチパターン 1: rows.Close()忘れ

```go
// BAD: rowsを閉じ忘れると接続リーク
func listUsers(db *sql.DB) ([]User, error) {
    rows, err := db.Query("SELECT * FROM users")
    if err != nil {
        return nil, err
    }
    // rows.Close() がない → 接続プール枯渇

    var users []User
    for rows.Next() { ... }
    return users, nil
}

// GOOD: defer rows.Close()
func listUsers(db *sql.DB) ([]User, error) {
    rows, err := db.Query("SELECT * FROM users")
    if err != nil {
        return nil, err
    }
    defer rows.Close()
    ...
}
```

### アンチパターン 2: GORM AutoMigrate を本番で使う

```go
// BAD: 本番でAutoMigrateは危険
db.AutoMigrate(&User{}, &Order{}, &Product{})
// カラム削除されない、データ損失のリスク

// GOOD: マイグレーションツールを使う
// golang-migrate, goose, atlas などで明示的に管理
// migrate -path ./migrations -database $DB_URL up
```

---

## 5. FAQ

### Q1: 接続プールのサイズはどう決めるか？

目安: `MaxOpenConns = DB最大接続数 / アプリインスタンス数`。PostgreSQLデフォルトは100接続。3インスタンスなら各30程度。`MaxIdleConns`は`MaxOpenConns`の1/5〜1/3程度。ベンチマークで調整する。

### Q2: database/sql と sqlx のどちらを選ぶべきか？

sqlxはdatabase/sqlの上位互換で、structへの自動マッピングを提供する。デメリットはほぼないため、新規プロジェクトではsqlxを推奨する。GORMはCRUD中心のアプリに適するが、複雑なクエリでは生SQLに戻ることが多い。

### Q3: SQLインジェクションはGoではどう防ぐか？

プレースホルダ（`$1`, `?`）を必ず使う。文字列連結でSQLを組み立てない。sqlxのNamedQuery、GORMのWhereも内部でプレースホルダを使う。`database/sql`の標準メソッドを使えば安全。

---

## まとめ

| 概念 | 要点 |
|------|------|
| database/sql | 標準ライブラリ。ドライバ交換可能 |
| 接続プール | MaxOpenConns/MaxIdleConns を設定 |
| sqlx | struct自動マッピング。生SQL |
| GORM | フルORM。CRUD高速開発 |
| トランザクション | BeginTx + defer Rollback + Commit |
| マイグレーション | golang-migrate / goose / atlas |
| Context | 全DB操作に context を渡す |

---

## 次に読むべきガイド

- [03-grpc.md](./03-grpc.md) -- gRPC
- [04-testing.md](./04-testing.md) -- DBテスト
- [../03-tools/03-deployment.md](../03-tools/03-deployment.md) -- デプロイ

---

## 参考文献

1. **Go Standard Library: database/sql** -- https://pkg.go.dev/database/sql
2. **jmoiron/sqlx** -- https://github.com/jmoiron/sqlx
3. **GORM** -- https://gorm.io/docs/
