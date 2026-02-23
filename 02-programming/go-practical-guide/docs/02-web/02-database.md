# データベース -- database/sql, sqlx, GORM

> Goはdatabase/sqlで標準的なDB接続を提供し、sqlx・GORMで生産性を向上させ、マイグレーション・接続プールで本番運用を支える。

---

## この章で学ぶこと

1. **database/sql** -- 標準ライブラリのDB操作と接続プール管理
2. **sqlx / GORM** -- 高レベルライブラリの使い分けと実践テクニック
3. **接続プールとマイグレーション** -- 本番運用のベストプラクティス
4. **トランザクション設計** -- ACID特性の活用とデッドロック回避
5. **パフォーマンス最適化** -- クエリ最適化、N+1問題の検出と対策
6. **テスト戦略** -- DBテストの手法とテストコンテナ

---

## 1. database/sql 基本

### コード例 1: database/sql 接続とプール設定

```go
package main

import (
    "context"
    "database/sql"
    "fmt"
    "log"
    "time"

    _ "github.com/lib/pq"
)

func main() {
    // sql.Open はコネクションプールを初期化するが、実際の接続は行わない
    db, err := sql.Open("postgres", "postgres://user:pass@localhost/mydb?sslmode=disable")
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    // 接続プール設定（本番環境では必ず設定する）
    db.SetMaxOpenConns(25)              // 最大同時接続数
    db.SetMaxIdleConns(5)               // アイドル接続の最大数
    db.SetConnMaxLifetime(5 * time.Minute) // 接続の最大生存時間
    db.SetConnMaxIdleTime(1 * time.Minute) // アイドル接続の最大生存時間

    // Ping で実際の接続を確認
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    if err := db.PingContext(ctx); err != nil {
        log.Fatal("DB接続失敗:", err)
    }

    fmt.Println("DB接続成功")
}
```

### コード例 2: ドライバ登録とDSN構築

```go
package database

import (
    "database/sql"
    "fmt"
    "net/url"

    _ "github.com/go-sql-driver/mysql" // MySQL
    _ "github.com/lib/pq"              // PostgreSQL
    _ "github.com/mattn/go-sqlite3"    // SQLite
    _ "modernc.org/sqlite"             // SQLite (CGO不要)
)

// Config はDB接続設定を表す
type Config struct {
    Driver   string
    Host     string
    Port     int
    User     string
    Password string
    DBName   string
    SSLMode  string
    Params   map[string]string
}

// DSN はドライバごとのDSN文字列を生成する
func (c *Config) DSN() string {
    switch c.Driver {
    case "postgres":
        return fmt.Sprintf(
            "postgres://%s:%s@%s:%d/%s?sslmode=%s",
            url.QueryEscape(c.User),
            url.QueryEscape(c.Password),
            c.Host, c.Port, c.DBName, c.SSLMode,
        )
    case "mysql":
        // user:password@tcp(host:port)/dbname?param=value
        return fmt.Sprintf(
            "%s:%s@tcp(%s:%d)/%s?parseTime=true&loc=Asia%%2FTokyo",
            c.User, c.Password, c.Host, c.Port, c.DBName,
        )
    case "sqlite3", "sqlite":
        return c.DBName // ファイルパスまたは ":memory:"
    default:
        panic("unsupported driver: " + c.Driver)
    }
}

// Open はDB接続プールを初期化する
func Open(cfg Config) (*sql.DB, error) {
    db, err := sql.Open(cfg.Driver, cfg.DSN())
    if err != nil {
        return nil, fmt.Errorf("sql.Open: %w", err)
    }

    // ドライバに応じたプール設定
    switch cfg.Driver {
    case "postgres", "mysql":
        db.SetMaxOpenConns(25)
        db.SetMaxIdleConns(5)
        db.SetConnMaxLifetime(5 * time.Minute)
    case "sqlite3", "sqlite":
        // SQLiteは単一接続推奨
        db.SetMaxOpenConns(1)
    }

    return db, nil
}
```

### コード例 3: CRUD操作 (database/sql)

```go
package repository

import (
    "context"
    "database/sql"
    "errors"
    "fmt"
    "time"
)

// User はユーザーエンティティを表す
type User struct {
    ID        int64
    Name      string
    Email     string
    CreatedAt time.Time
    UpdatedAt time.Time
}

// ErrNotFound はリソースが見つからない場合のエラー
var ErrNotFound = errors.New("resource not found")

// UserRepository はユーザーデータへのアクセスを提供する
type UserRepository struct {
    db *sql.DB
}

// NewUserRepository は新しいUserRepositoryを作成する
func NewUserRepository(db *sql.DB) *UserRepository {
    return &UserRepository{db: db}
}

// GetByID は指定IDのユーザーを取得する
func (r *UserRepository) GetByID(ctx context.Context, id int64) (*User, error) {
    var u User
    err := r.db.QueryRowContext(ctx,
        `SELECT id, name, email, created_at, updated_at
         FROM users WHERE id = $1`, id,
    ).Scan(&u.ID, &u.Name, &u.Email, &u.CreatedAt, &u.UpdatedAt)

    if errors.Is(err, sql.ErrNoRows) {
        return nil, fmt.Errorf("user id=%d: %w", id, ErrNotFound)
    }
    if err != nil {
        return nil, fmt.Errorf("GetByID(%d): %w", id, err)
    }
    return &u, nil
}

// GetByEmail はメールアドレスでユーザーを検索する
func (r *UserRepository) GetByEmail(ctx context.Context, email string) (*User, error) {
    var u User
    err := r.db.QueryRowContext(ctx,
        `SELECT id, name, email, created_at, updated_at
         FROM users WHERE email = $1`, email,
    ).Scan(&u.ID, &u.Name, &u.Email, &u.CreatedAt, &u.UpdatedAt)

    if errors.Is(err, sql.ErrNoRows) {
        return nil, fmt.Errorf("user email=%s: %w", email, ErrNotFound)
    }
    if err != nil {
        return nil, fmt.Errorf("GetByEmail(%s): %w", email, err)
    }
    return &u, nil
}

// List は全ユーザーを取得する（ページネーション付き）
func (r *UserRepository) List(ctx context.Context, limit, offset int) ([]User, error) {
    rows, err := r.db.QueryContext(ctx,
        `SELECT id, name, email, created_at, updated_at
         FROM users
         ORDER BY id
         LIMIT $1 OFFSET $2`, limit, offset,
    )
    if err != nil {
        return nil, fmt.Errorf("List: %w", err)
    }
    defer rows.Close()

    var users []User
    for rows.Next() {
        var u User
        if err := rows.Scan(&u.ID, &u.Name, &u.Email, &u.CreatedAt, &u.UpdatedAt); err != nil {
            return nil, fmt.Errorf("List scan: %w", err)
        }
        users = append(users, u)
    }
    // rows.Err() で反復中のエラーを確認する（重要）
    if err := rows.Err(); err != nil {
        return nil, fmt.Errorf("List rows: %w", err)
    }
    return users, nil
}

// Search はキーワードでユーザーを検索する
func (r *UserRepository) Search(ctx context.Context, keyword string) ([]User, error) {
    // LIKE検索のワイルドカードをエスケープ
    escapedKeyword := "%" + keyword + "%"

    rows, err := r.db.QueryContext(ctx,
        `SELECT id, name, email, created_at, updated_at
         FROM users
         WHERE name ILIKE $1 OR email ILIKE $1
         ORDER BY name`, escapedKeyword,
    )
    if err != nil {
        return nil, fmt.Errorf("Search: %w", err)
    }
    defer rows.Close()

    var users []User
    for rows.Next() {
        var u User
        if err := rows.Scan(&u.ID, &u.Name, &u.Email, &u.CreatedAt, &u.UpdatedAt); err != nil {
            return nil, err
        }
        users = append(users, u)
    }
    return users, rows.Err()
}

// Create は新しいユーザーを作成する
func (r *UserRepository) Create(ctx context.Context, u *User) error {
    err := r.db.QueryRowContext(ctx,
        `INSERT INTO users (name, email, created_at, updated_at)
         VALUES ($1, $2, NOW(), NOW())
         RETURNING id, created_at, updated_at`,
        u.Name, u.Email,
    ).Scan(&u.ID, &u.CreatedAt, &u.UpdatedAt)

    if err != nil {
        return fmt.Errorf("Create user: %w", err)
    }
    return nil
}

// Update は既存ユーザーを更新する
func (r *UserRepository) Update(ctx context.Context, u *User) error {
    result, err := r.db.ExecContext(ctx,
        `UPDATE users SET name = $1, email = $2, updated_at = NOW()
         WHERE id = $3`,
        u.Name, u.Email, u.ID,
    )
    if err != nil {
        return fmt.Errorf("Update user: %w", err)
    }

    rowsAffected, err := result.RowsAffected()
    if err != nil {
        return fmt.Errorf("Update RowsAffected: %w", err)
    }
    if rowsAffected == 0 {
        return fmt.Errorf("user id=%d: %w", u.ID, ErrNotFound)
    }
    return nil
}

// Delete はユーザーを削除する
func (r *UserRepository) Delete(ctx context.Context, id int64) error {
    result, err := r.db.ExecContext(ctx,
        `DELETE FROM users WHERE id = $1`, id,
    )
    if err != nil {
        return fmt.Errorf("Delete user: %w", err)
    }

    rowsAffected, err := result.RowsAffected()
    if err != nil {
        return fmt.Errorf("Delete RowsAffected: %w", err)
    }
    if rowsAffected == 0 {
        return fmt.Errorf("user id=%d: %w", id, ErrNotFound)
    }
    return nil
}

// BulkCreate は複数ユーザーを一括作成する
func (r *UserRepository) BulkCreate(ctx context.Context, users []User) error {
    // 大量データの場合は COPY プロトコルが高速だが、
    // ここでは INSERT ... VALUES の複数行挿入を使う
    tx, err := r.db.BeginTx(ctx, nil)
    if err != nil {
        return fmt.Errorf("BulkCreate begin: %w", err)
    }
    defer tx.Rollback()

    stmt, err := tx.PrepareContext(ctx,
        `INSERT INTO users (name, email, created_at, updated_at)
         VALUES ($1, $2, NOW(), NOW())
         RETURNING id`)
    if err != nil {
        return fmt.Errorf("BulkCreate prepare: %w", err)
    }
    defer stmt.Close()

    for i := range users {
        err := stmt.QueryRowContext(ctx, users[i].Name, users[i].Email).Scan(&users[i].ID)
        if err != nil {
            return fmt.Errorf("BulkCreate insert %d: %w", i, err)
        }
    }

    return tx.Commit()
}
```

### コード例 4: Prepared Statement の活用

```go
package repository

import (
    "context"
    "database/sql"
    "fmt"
)

// PreparedUserRepository はPrepared Statementを使ったリポジトリ
type PreparedUserRepository struct {
    db         *sql.DB
    stmtGetByID *sql.Stmt
    stmtList    *sql.Stmt
    stmtCreate  *sql.Stmt
    stmtUpdate  *sql.Stmt
    stmtDelete  *sql.Stmt
}

// NewPreparedUserRepository は事前にステートメントを準備する
func NewPreparedUserRepository(ctx context.Context, db *sql.DB) (*PreparedUserRepository, error) {
    r := &PreparedUserRepository{db: db}
    var err error

    r.stmtGetByID, err = db.PrepareContext(ctx,
        `SELECT id, name, email, created_at, updated_at FROM users WHERE id = $1`)
    if err != nil {
        return nil, fmt.Errorf("prepare GetByID: %w", err)
    }

    r.stmtList, err = db.PrepareContext(ctx,
        `SELECT id, name, email, created_at, updated_at FROM users ORDER BY id LIMIT $1 OFFSET $2`)
    if err != nil {
        return nil, fmt.Errorf("prepare List: %w", err)
    }

    r.stmtCreate, err = db.PrepareContext(ctx,
        `INSERT INTO users (name, email, created_at, updated_at)
         VALUES ($1, $2, NOW(), NOW()) RETURNING id, created_at, updated_at`)
    if err != nil {
        return nil, fmt.Errorf("prepare Create: %w", err)
    }

    r.stmtUpdate, err = db.PrepareContext(ctx,
        `UPDATE users SET name = $1, email = $2, updated_at = NOW() WHERE id = $3`)
    if err != nil {
        return nil, fmt.Errorf("prepare Update: %w", err)
    }

    r.stmtDelete, err = db.PrepareContext(ctx,
        `DELETE FROM users WHERE id = $1`)
    if err != nil {
        return nil, fmt.Errorf("prepare Delete: %w", err)
    }

    return r, nil
}

// Close はすべてのPrepared Statementを閉じる
func (r *PreparedUserRepository) Close() error {
    stmts := []*sql.Stmt{r.stmtGetByID, r.stmtList, r.stmtCreate, r.stmtUpdate, r.stmtDelete}
    for _, stmt := range stmts {
        if stmt != nil {
            stmt.Close()
        }
    }
    return nil
}

// GetByID はPrepared Statementを使って高速にユーザーを取得する
func (r *PreparedUserRepository) GetByID(ctx context.Context, id int64) (*User, error) {
    var u User
    err := r.stmtGetByID.QueryRowContext(ctx, id).
        Scan(&u.ID, &u.Name, &u.Email, &u.CreatedAt, &u.UpdatedAt)
    if err != nil {
        return nil, err
    }
    return &u, nil
}
```

### コード例 5: トランザクション管理

```go
package service

import (
    "context"
    "database/sql"
    "fmt"
)

// TxFunc はトランザクション内で実行する関数の型
type TxFunc func(tx *sql.Tx) error

// WithTransaction はトランザクション管理のヘルパー関数
// パニックからの回復、ロールバック、コミットを自動処理する
func WithTransaction(ctx context.Context, db *sql.DB, opts *sql.TxOptions, fn TxFunc) error {
    tx, err := db.BeginTx(ctx, opts)
    if err != nil {
        return fmt.Errorf("begin transaction: %w", err)
    }

    defer func() {
        if p := recover(); p != nil {
            // パニック時もロールバック
            _ = tx.Rollback()
            panic(p) // リパニック
        }
    }()

    if err := fn(tx); err != nil {
        if rbErr := tx.Rollback(); rbErr != nil {
            return fmt.Errorf("rollback failed: %v (original: %w)", rbErr, err)
        }
        return err
    }

    if err := tx.Commit(); err != nil {
        return fmt.Errorf("commit: %w", err)
    }
    return nil
}

// 使用例: 送金処理
func (s *AccountService) Transfer(ctx context.Context, fromID, toID int64, amount float64) error {
    return WithTransaction(ctx, s.db, &sql.TxOptions{
        Isolation: sql.LevelSerializable,
    }, func(tx *sql.Tx) error {
        // 送金元の残高確認
        var balance float64
        err := tx.QueryRowContext(ctx,
            "SELECT balance FROM accounts WHERE id = $1 FOR UPDATE", fromID,
        ).Scan(&balance)
        if err != nil {
            return fmt.Errorf("get balance: %w", err)
        }

        if balance < amount {
            return fmt.Errorf("insufficient funds: balance=%.2f, amount=%.2f", balance, amount)
        }

        // 送金元の引き落とし
        _, err = tx.ExecContext(ctx,
            "UPDATE accounts SET balance = balance - $1, updated_at = NOW() WHERE id = $2",
            amount, fromID)
        if err != nil {
            return fmt.Errorf("debit: %w", err)
        }

        // 送金先への入金
        _, err = tx.ExecContext(ctx,
            "UPDATE accounts SET balance = balance + $1, updated_at = NOW() WHERE id = $2",
            amount, toID)
        if err != nil {
            return fmt.Errorf("credit: %w", err)
        }

        // 取引履歴の記録
        _, err = tx.ExecContext(ctx,
            `INSERT INTO transactions (from_account_id, to_account_id, amount, created_at)
             VALUES ($1, $2, $3, NOW())`, fromID, toID, amount)
        if err != nil {
            return fmt.Errorf("record transaction: %w", err)
        }

        return nil
    })
}

// ReadOnlyTransaction は読み取り専用トランザクション
func ReadOnlyTransaction(ctx context.Context, db *sql.DB, fn TxFunc) error {
    return WithTransaction(ctx, db, &sql.TxOptions{
        ReadOnly: true,
    }, fn)
}

// トランザクション分離レベルの使い分け
// sql.LevelDefault          - ドライバのデフォルト（PostgreSQLはReadCommitted）
// sql.LevelReadUncommitted  - ダーティリード可能（ほぼ使わない）
// sql.LevelReadCommitted    - コミット済みデータのみ読取り
// sql.LevelRepeatableRead   - トランザクション内で同じ結果を保証
// sql.LevelSerializable     - 最も厳格（デッドロックリスクあり）
```

### コード例 6: Null値の取り扱い

```go
package model

import (
    "database/sql"
    "encoding/json"
    "time"
)

// NullableUser はNULL可能なフィールドを持つユーザー
type NullableUser struct {
    ID        int64
    Name      string
    Email     sql.NullString  // NULL可能
    Phone     sql.NullString  // NULL可能
    Age       sql.NullInt64   // NULL可能
    Score     sql.NullFloat64 // NULL可能
    IsActive  sql.NullBool    // NULL可能
    DeletedAt sql.NullTime    // NULL可能（ソフトデリート）
}

// UserJSON はJSONレスポンス用の構造体
type UserJSON struct {
    ID        int64   `json:"id"`
    Name      string  `json:"name"`
    Email     *string `json:"email,omitempty"`
    Phone     *string `json:"phone,omitempty"`
    Age       *int64  `json:"age,omitempty"`
    Score     *float64 `json:"score,omitempty"`
    IsActive  *bool   `json:"is_active,omitempty"`
    DeletedAt *time.Time `json:"deleted_at,omitempty"`
}

// ToJSON はNullableUserをJSONフレンドリーな構造体に変換する
func (u *NullableUser) ToJSON() UserJSON {
    j := UserJSON{
        ID:   u.ID,
        Name: u.Name,
    }
    if u.Email.Valid {
        j.Email = &u.Email.String
    }
    if u.Phone.Valid {
        j.Phone = &u.Phone.String
    }
    if u.Age.Valid {
        j.Age = &u.Age.Int64
    }
    if u.Score.Valid {
        j.Score = &u.Score.Float64
    }
    if u.IsActive.Valid {
        j.IsActive = &u.IsActive.Bool
    }
    if u.DeletedAt.Valid {
        j.DeletedAt = &u.DeletedAt.Time
    }
    return j
}

// カスタムNull型（よりGoらしいアプローチ）
// ジェネリクスを使ったNull表現（Go 1.18+）
type Nullable[T any] struct {
    Value T
    Valid bool
}

func NewNullable[T any](v T) Nullable[T] {
    return Nullable[T]{Value: v, Valid: true}
}

func NullValue[T any]() Nullable[T] {
    return Nullable[T]{Valid: false}
}

func (n Nullable[T]) MarshalJSON() ([]byte, error) {
    if !n.Valid {
        return []byte("null"), nil
    }
    return json.Marshal(n.Value)
}
```

---

## 2. sqlx の活用

### コード例 7: sqlx 基本操作

```go
package repository

import (
    "context"
    "fmt"
    "time"

    "github.com/jmoiron/sqlx"
    _ "github.com/lib/pq"
)

// User はsqlxのdbタグを使った構造体
type User struct {
    ID        int64     `db:"id" json:"id"`
    Name      string    `db:"name" json:"name"`
    Email     string    `db:"email" json:"email"`
    Role      string    `db:"role" json:"role"`
    CreatedAt time.Time `db:"created_at" json:"created_at"`
    UpdatedAt time.Time `db:"updated_at" json:"updated_at"`
}

// SqlxUserRepository はsqlxを使ったリポジトリ
type SqlxUserRepository struct {
    db *sqlx.DB
}

// NewSqlxUserRepository はsqlx接続を初期化する
func NewSqlxUserRepository(dsn string) (*SqlxUserRepository, error) {
    // sqlx.Connect は Open + Ping を行う
    db, err := sqlx.Connect("postgres", dsn)
    if err != nil {
        return nil, fmt.Errorf("sqlx.Connect: %w", err)
    }

    db.SetMaxOpenConns(25)
    db.SetMaxIdleConns(5)
    db.SetConnMaxLifetime(5 * time.Minute)

    return &SqlxUserRepository{db: db}, nil
}

// GetByID はsqlx.GetContextで単一行を取得する
func (r *SqlxUserRepository) GetByID(ctx context.Context, id int64) (*User, error) {
    var u User
    err := r.db.GetContext(ctx, &u,
        `SELECT id, name, email, role, created_at, updated_at
         FROM users WHERE id = $1`, id)
    if err != nil {
        return nil, fmt.Errorf("GetByID(%d): %w", id, err)
    }
    return &u, nil
}

// List はsqlx.SelectContextで複数行を取得する
func (r *SqlxUserRepository) List(ctx context.Context, limit, offset int) ([]User, error) {
    var users []User
    err := r.db.SelectContext(ctx, &users,
        `SELECT id, name, email, role, created_at, updated_at
         FROM users ORDER BY id LIMIT $1 OFFSET $2`, limit, offset)
    if err != nil {
        return nil, fmt.Errorf("List: %w", err)
    }
    return users, nil
}

// Create はNamedExecContextで名前付きパラメータを使って挿入する
func (r *SqlxUserRepository) Create(ctx context.Context, u *User) error {
    query := `INSERT INTO users (name, email, role, created_at, updated_at)
              VALUES (:name, :email, :role, NOW(), NOW())
              RETURNING id, created_at, updated_at`
    rows, err := r.db.NamedQueryContext(ctx, query, u)
    if err != nil {
        return fmt.Errorf("Create: %w", err)
    }
    defer rows.Close()
    if rows.Next() {
        if err := rows.Scan(&u.ID, &u.CreatedAt, &u.UpdatedAt); err != nil {
            return fmt.Errorf("Create scan: %w", err)
        }
    }
    return nil
}

// Update はNamedExecContextで名前付きパラメータを使って更新する
func (r *SqlxUserRepository) Update(ctx context.Context, u *User) error {
    result, err := r.db.NamedExecContext(ctx,
        `UPDATE users SET name = :name, email = :email, role = :role, updated_at = NOW()
         WHERE id = :id`, u)
    if err != nil {
        return fmt.Errorf("Update: %w", err)
    }
    rowsAffected, _ := result.RowsAffected()
    if rowsAffected == 0 {
        return ErrNotFound
    }
    return nil
}

// ListByRole は特定ロールのユーザーリストを取得する
func (r *SqlxUserRepository) ListByRole(ctx context.Context, role string) ([]User, error) {
    var users []User
    err := r.db.SelectContext(ctx, &users,
        `SELECT id, name, email, role, created_at, updated_at
         FROM users WHERE role = $1 ORDER BY name`, role)
    return users, err
}

// SearchByNames はIN句を使った複数検索（sqlxのIn関数を使用）
func (r *SqlxUserRepository) SearchByNames(ctx context.Context, names []string) ([]User, error) {
    // sqlx.In はスライスをプレースホルダに展開する
    query, args, err := sqlx.In(
        `SELECT id, name, email, role, created_at, updated_at
         FROM users WHERE name IN (?) ORDER BY name`, names)
    if err != nil {
        return nil, fmt.Errorf("sqlx.In: %w", err)
    }

    // PostgreSQLの場合は$1形式にリバインド
    query = r.db.Rebind(query)

    var users []User
    err = r.db.SelectContext(ctx, &users, query, args...)
    return users, err
}
```

### コード例 8: sqlx の高度な機能

```go
package repository

import (
    "context"
    "database/sql"
    "fmt"

    "github.com/jmoiron/sqlx"
)

// 動的クエリビルダ
type UserFilter struct {
    Name     *string
    Email    *string
    Role     *string
    MinAge   *int
    MaxAge   *int
    IsActive *bool
    OrderBy  string
    Limit    int
    Offset   int
}

// BuildQuery はフィルター条件に応じた動的クエリを構築する
func (f *UserFilter) BuildQuery() (string, map[string]interface{}) {
    query := `SELECT id, name, email, role, created_at, updated_at FROM users WHERE 1=1`
    args := make(map[string]interface{})

    if f.Name != nil {
        query += ` AND name ILIKE :name`
        args["name"] = "%" + *f.Name + "%"
    }
    if f.Email != nil {
        query += ` AND email ILIKE :email`
        args["email"] = "%" + *f.Email + "%"
    }
    if f.Role != nil {
        query += ` AND role = :role`
        args["role"] = *f.Role
    }
    if f.MinAge != nil {
        query += ` AND age >= :min_age`
        args["min_age"] = *f.MinAge
    }
    if f.MaxAge != nil {
        query += ` AND age <= :max_age`
        args["max_age"] = *f.MaxAge
    }
    if f.IsActive != nil {
        query += ` AND is_active = :is_active`
        args["is_active"] = *f.IsActive
    }

    // ORDER BY
    if f.OrderBy != "" {
        // SQLインジェクション防止: ホワイトリスト
        allowed := map[string]bool{
            "name": true, "email": true, "created_at": true, "id": true,
        }
        if allowed[f.OrderBy] {
            query += fmt.Sprintf(` ORDER BY %s`, f.OrderBy)
        }
    } else {
        query += ` ORDER BY id`
    }

    // LIMIT / OFFSET
    if f.Limit > 0 {
        query += ` LIMIT :limit`
        args["limit"] = f.Limit
    }
    if f.Offset > 0 {
        query += ` OFFSET :offset`
        args["offset"] = f.Offset
    }

    return query, args
}

// SearchWithFilter は動的フィルターでユーザーを検索する
func (r *SqlxUserRepository) SearchWithFilter(ctx context.Context, filter UserFilter) ([]User, error) {
    query, args := filter.BuildQuery()

    // sqlx.Named はnamed queryをプレースホルダ付きクエリに変換する
    rows, err := r.db.NamedQueryContext(ctx, query, args)
    if err != nil {
        return nil, fmt.Errorf("SearchWithFilter: %w", err)
    }
    defer rows.Close()

    var users []User
    for rows.Next() {
        var u User
        if err := rows.StructScan(&u); err != nil {
            return nil, err
        }
        users = append(users, u)
    }
    return users, rows.Err()
}

// StructScan vs Scan の比較
func demonstrateScanDifference(db *sqlx.DB) {
    // database/sql: 手動で各カラムをScan
    row := db.QueryRow("SELECT id, name, email FROM users WHERE id = $1", 1)
    var id int64
    var name, email string
    row.Scan(&id, &name, &email) // カラム順に注意が必要

    // sqlx: 構造体に自動マッピング
    var user User
    db.Get(&user, "SELECT id, name, email FROM users WHERE id = $1", 1)
    // db タグに基づいて自動的にマッピング
}
```

---

## 3. GORM の活用

### コード例 9: GORM 基本操作

```go
package repository

import (
    "context"
    "errors"
    "fmt"
    "time"

    "gorm.io/driver/postgres"
    "gorm.io/gorm"
    "gorm.io/gorm/clause"
    "gorm.io/gorm/logger"
)

// GormUser はGORMモデル
type GormUser struct {
    ID        uint           `gorm:"primarykey" json:"id"`
    Name      string         `gorm:"size:100;not null;index" json:"name"`
    Email     string         `gorm:"uniqueIndex;not null;size:255" json:"email"`
    Role      string         `gorm:"size:50;default:'user'" json:"role"`
    Age       int            `gorm:"check:age >= 0" json:"age"`
    Profile   *GormProfile   `gorm:"foreignKey:UserID" json:"profile,omitempty"` // has one
    Orders    []GormOrder    `gorm:"foreignKey:UserID" json:"orders,omitempty"`  // has many
    CreatedAt time.Time      `json:"created_at"`
    UpdatedAt time.Time      `json:"updated_at"`
    DeletedAt gorm.DeletedAt `gorm:"index" json:"deleted_at,omitempty"` // ソフトデリート
}

// TableName はテーブル名をカスタマイズする
func (GormUser) TableName() string {
    return "users"
}

// GormProfile はユーザープロフィール
type GormProfile struct {
    ID     uint   `gorm:"primarykey"`
    UserID uint   `gorm:"uniqueIndex;not null"`
    Bio    string `gorm:"type:text"`
    Avatar string `gorm:"size:500"`
}

// GormOrder は注文
type GormOrder struct {
    ID        uint      `gorm:"primarykey"`
    UserID    uint      `gorm:"index;not null"`
    ProductID uint      `gorm:"index;not null"`
    Quantity  int       `gorm:"not null;check:quantity > 0"`
    Amount    float64   `gorm:"type:decimal(10,2);not null"`
    Status    string    `gorm:"size:20;default:'pending'"`
    CreatedAt time.Time
}

// InitGorm はGORM接続を初期化する
func InitGorm(dsn string) (*gorm.DB, error) {
    db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{
        Logger: logger.Default.LogMode(logger.Info), // SQLログを出力
        NowFunc: func() time.Time {
            return time.Now().UTC()
        },
        // SQL実行前のDry Run モードを無効化
        DryRun: false,
        // PrepareStmt: trueにするとPrepared Statementをキャッシュ
        PrepareStmt: true,
    })
    if err != nil {
        return nil, fmt.Errorf("gorm.Open: %w", err)
    }

    // 接続プール設定
    sqlDB, err := db.DB()
    if err != nil {
        return nil, fmt.Errorf("get sql.DB: %w", err)
    }
    sqlDB.SetMaxOpenConns(25)
    sqlDB.SetMaxIdleConns(5)
    sqlDB.SetConnMaxLifetime(5 * time.Minute)

    return db, nil
}

// GormUserRepository はGORMを使ったリポジトリ
type GormUserRepository struct {
    db *gorm.DB
}

// Create はユーザーを作成する
func (r *GormUserRepository) Create(ctx context.Context, user *GormUser) error {
    result := r.db.WithContext(ctx).Create(user)
    if result.Error != nil {
        return fmt.Errorf("Create: %w", result.Error)
    }
    return nil
}

// GetByID はIDでユーザーを取得する（関連データも含む）
func (r *GormUserRepository) GetByID(ctx context.Context, id uint) (*GormUser, error) {
    var user GormUser
    result := r.db.WithContext(ctx).
        Preload("Profile").                           // プロフィールもロード
        Preload("Orders", "status = ?", "completed"). // 完了済み注文のみ
        First(&user, id)

    if errors.Is(result.Error, gorm.ErrRecordNotFound) {
        return nil, ErrNotFound
    }
    if result.Error != nil {
        return nil, fmt.Errorf("GetByID: %w", result.Error)
    }
    return &user, nil
}

// List はページネーション付きでユーザーリストを取得する
func (r *GormUserRepository) List(ctx context.Context, page, pageSize int) ([]GormUser, int64, error) {
    var users []GormUser
    var total int64

    // カウントクエリ
    r.db.WithContext(ctx).Model(&GormUser{}).Count(&total)

    // データ取得
    result := r.db.WithContext(ctx).
        Order("id ASC").
        Limit(pageSize).
        Offset((page - 1) * pageSize).
        Find(&users)

    return users, total, result.Error
}

// Search は条件検索する
func (r *GormUserRepository) Search(ctx context.Context, filter map[string]interface{}) ([]GormUser, error) {
    var users []GormUser
    query := r.db.WithContext(ctx)

    if name, ok := filter["name"]; ok {
        query = query.Where("name ILIKE ?", "%"+name.(string)+"%")
    }
    if role, ok := filter["role"]; ok {
        query = query.Where("role = ?", role)
    }
    if minAge, ok := filter["min_age"]; ok {
        query = query.Where("age >= ?", minAge)
    }

    result := query.Find(&users)
    return users, result.Error
}

// Update はユーザーを更新する
func (r *GormUserRepository) Update(ctx context.Context, user *GormUser) error {
    // Save は全フィールドを更新する
    result := r.db.WithContext(ctx).Save(user)
    return result.Error
}

// UpdatePartial は指定フィールドのみ更新する
func (r *GormUserRepository) UpdatePartial(ctx context.Context, id uint, updates map[string]interface{}) error {
    result := r.db.WithContext(ctx).
        Model(&GormUser{}).
        Where("id = ?", id).
        Updates(updates)

    if result.RowsAffected == 0 {
        return ErrNotFound
    }
    return result.Error
}

// Delete はソフトデリートする
func (r *GormUserRepository) Delete(ctx context.Context, id uint) error {
    result := r.db.WithContext(ctx).Delete(&GormUser{}, id)
    if result.RowsAffected == 0 {
        return ErrNotFound
    }
    return result.Error
}

// HardDelete は物理削除する
func (r *GormUserRepository) HardDelete(ctx context.Context, id uint) error {
    result := r.db.WithContext(ctx).Unscoped().Delete(&GormUser{}, id)
    return result.Error
}

// Upsert は存在すれば更新、なければ挿入する
func (r *GormUserRepository) Upsert(ctx context.Context, user *GormUser) error {
    result := r.db.WithContext(ctx).
        Clauses(clause.OnConflict{
            Columns:   []clause.Column{{Name: "email"}},
            DoUpdates: clause.AssignmentColumns([]string{"name", "role", "updated_at"}),
        }).
        Create(user)
    return result.Error
}
```

### コード例 10: GORM トランザクションとフック

```go
package service

import (
    "context"
    "fmt"

    "gorm.io/gorm"
)

// GormOrderService は注文サービス
type GormOrderService struct {
    db *gorm.DB
}

// CreateOrder はトランザクション内で注文を作成する
func (s *GormOrderService) CreateOrder(ctx context.Context, order *GormOrder) error {
    return s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
        // 在庫確認
        var product Product
        if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).
            First(&product, order.ProductID).Error; err != nil {
            return fmt.Errorf("product not found: %w", err)
        }

        if product.Stock < order.Quantity {
            return fmt.Errorf("insufficient stock: available=%d, requested=%d",
                product.Stock, order.Quantity)
        }

        // 在庫を減らす
        if err := tx.Model(&product).
            Update("stock", gorm.Expr("stock - ?", order.Quantity)).Error; err != nil {
            return fmt.Errorf("update stock: %w", err)
        }

        // 注文を作成
        order.Amount = product.Price * float64(order.Quantity)
        if err := tx.Create(order).Error; err != nil {
            return fmt.Errorf("create order: %w", err)
        }

        return nil
    })
}

// GORMフック（ライフサイクルコールバック）
func (u *GormUser) BeforeCreate(tx *gorm.DB) error {
    // バリデーション
    if u.Name == "" {
        return fmt.Errorf("name is required")
    }
    if u.Role == "" {
        u.Role = "user" // デフォルト値
    }
    return nil
}

func (u *GormUser) AfterCreate(tx *gorm.DB) error {
    // 監査ログの記録など
    tx.Exec("INSERT INTO audit_logs (action, entity, entity_id) VALUES (?, ?, ?)",
        "CREATE", "users", u.ID)
    return nil
}

func (u *GormUser) BeforeUpdate(tx *gorm.DB) error {
    // 更新時のバリデーション
    if u.Email != "" && !isValidEmail(u.Email) {
        return fmt.Errorf("invalid email format")
    }
    return nil
}
```

---

## 4. マイグレーション

### コード例 11: golang-migrate の使用

```go
package migrations

import (
    "database/sql"
    "fmt"
    "log"

    "github.com/golang-migrate/migrate/v4"
    "github.com/golang-migrate/migrate/v4/database/postgres"
    _ "github.com/golang-migrate/migrate/v4/source/file"
)

// RunMigrations はマイグレーションを実行する
func RunMigrations(db *sql.DB, migrationsPath string) error {
    driver, err := postgres.WithInstance(db, &postgres.Config{})
    if err != nil {
        return fmt.Errorf("create driver: %w", err)
    }

    m, err := migrate.NewWithDatabaseInstance(
        "file://"+migrationsPath,
        "postgres",
        driver,
    )
    if err != nil {
        return fmt.Errorf("create migrate: %w", err)
    }

    // マイグレーション実行
    if err := m.Up(); err != nil && err != migrate.ErrNoChange {
        return fmt.Errorf("migrate up: %w", err)
    }

    version, dirty, err := m.Version()
    if err != nil {
        return fmt.Errorf("get version: %w", err)
    }
    log.Printf("Migration version: %d, dirty: %v", version, dirty)

    return nil
}

// RollbackMigration は1つ前のバージョンに戻す
func RollbackMigration(db *sql.DB, migrationsPath string) error {
    driver, err := postgres.WithInstance(db, &postgres.Config{})
    if err != nil {
        return err
    }

    m, err := migrate.NewWithDatabaseInstance(
        "file://"+migrationsPath,
        "postgres",
        driver,
    )
    if err != nil {
        return err
    }

    return m.Steps(-1)
}
```

マイグレーションファイルの例:

```sql
-- migrations/000001_create_users.up.sql
CREATE TABLE IF NOT EXISTS users (
    id BIGSERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE,
    role VARCHAR(50) NOT NULL DEFAULT 'user',
    age INTEGER CHECK (age >= 0),
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_created_at ON users(created_at);

-- migrations/000001_create_users.down.sql
DROP TABLE IF EXISTS users;

-- migrations/000002_create_orders.up.sql
CREATE TABLE IF NOT EXISTS orders (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    product_id BIGINT NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    amount DECIMAL(10,2) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_created_at ON orders(created_at);

-- migrations/000002_create_orders.down.sql
DROP TABLE IF EXISTS orders;

-- migrations/000003_add_users_phone.up.sql
ALTER TABLE users ADD COLUMN phone VARCHAR(20);
CREATE INDEX idx_users_phone ON users(phone) WHERE phone IS NOT NULL;

-- migrations/000003_add_users_phone.down.sql
DROP INDEX IF EXISTS idx_users_phone;
ALTER TABLE users DROP COLUMN IF EXISTS phone;
```

### コード例 12: goose の使用

```go
package migrations

import (
    "database/sql"
    "fmt"

    "github.com/pressly/goose/v3"
)

// GooseMigrate はgooseを使ったマイグレーション
func GooseMigrate(db *sql.DB, dir string) error {
    goose.SetDialect("postgres")

    if err := goose.Up(db, dir); err != nil {
        return fmt.Errorf("goose up: %w", err)
    }
    return nil
}

// GooseStatus は現在のマイグレーション状態を表示する
func GooseStatus(db *sql.DB, dir string) error {
    goose.SetDialect("postgres")
    return goose.Status(db, dir)
}

// GooseRollback は1つ前に戻す
func GooseRollback(db *sql.DB, dir string) error {
    goose.SetDialect("postgres")
    return goose.Down(db, dir)
}

// Go関数マイグレーション（goose独自機能）
func init() {
    goose.AddMigration(upSeedData, downSeedData)
}

func upSeedData(tx *sql.Tx) error {
    _, err := tx.Exec(`
        INSERT INTO users (name, email, role) VALUES
        ('Admin', 'admin@example.com', 'admin'),
        ('User1', 'user1@example.com', 'user'),
        ('User2', 'user2@example.com', 'user')
    `)
    return err
}

func downSeedData(tx *sql.Tx) error {
    _, err := tx.Exec(`DELETE FROM users WHERE email IN ('admin@example.com', 'user1@example.com', 'user2@example.com')`)
    return err
}
```

---

## 5. テスト戦略

### コード例 13: testcontainers-go を使ったDBテスト

```go
package repository_test

import (
    "context"
    "database/sql"
    "fmt"
    "testing"
    "time"

    _ "github.com/lib/pq"
    "github.com/testcontainers/testcontainers-go"
    "github.com/testcontainers/testcontainers-go/wait"
)

// setupTestDB はテスト用のPostgreSQLコンテナを起動する
func setupTestDB(t *testing.T) (*sql.DB, func()) {
    t.Helper()
    ctx := context.Background()

    // PostgreSQLコンテナの起動
    container, err := testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
        ContainerRequest: testcontainers.ContainerRequest{
            Image:        "postgres:16-alpine",
            ExposedPorts: []string{"5432/tcp"},
            Env: map[string]string{
                "POSTGRES_USER":     "test",
                "POSTGRES_PASSWORD": "test",
                "POSTGRES_DB":       "testdb",
            },
            WaitingFor: wait.ForListeningPort("5432/tcp").
                WithStartupTimeout(60 * time.Second),
        },
        Started: true,
    })
    if err != nil {
        t.Fatalf("コンテナ起動失敗: %v", err)
    }

    // DSN構築
    host, _ := container.Host(ctx)
    port, _ := container.MappedPort(ctx, "5432")
    dsn := fmt.Sprintf("postgres://test:test@%s:%s/testdb?sslmode=disable", host, port.Port())

    // DB接続
    db, err := sql.Open("postgres", dsn)
    if err != nil {
        t.Fatalf("DB接続失敗: %v", err)
    }

    // テーブル作成
    _, err = db.ExecContext(ctx, `
        CREATE TABLE IF NOT EXISTS users (
            id BIGSERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            email VARCHAR(255) NOT NULL UNIQUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    `)
    if err != nil {
        t.Fatalf("テーブル作成失敗: %v", err)
    }

    // クリーンアップ関数
    cleanup := func() {
        db.Close()
        container.Terminate(ctx)
    }

    return db, cleanup
}

func TestUserRepository_Create(t *testing.T) {
    db, cleanup := setupTestDB(t)
    defer cleanup()

    repo := NewUserRepository(db)
    ctx := context.Background()

    user := &User{Name: "Test User", Email: "test@example.com"}
    err := repo.Create(ctx, user)
    if err != nil {
        t.Fatalf("Create失敗: %v", err)
    }

    if user.ID == 0 {
        t.Error("IDが設定されていない")
    }

    // 取得して確認
    got, err := repo.GetByID(ctx, user.ID)
    if err != nil {
        t.Fatalf("GetByID失敗: %v", err)
    }

    if got.Name != "Test User" {
        t.Errorf("Name = %q, want %q", got.Name, "Test User")
    }
    if got.Email != "test@example.com" {
        t.Errorf("Email = %q, want %q", got.Email, "test@example.com")
    }
}

func TestUserRepository_Update(t *testing.T) {
    db, cleanup := setupTestDB(t)
    defer cleanup()

    repo := NewUserRepository(db)
    ctx := context.Background()

    // セットアップ
    user := &User{Name: "Original", Email: "original@example.com"}
    if err := repo.Create(ctx, user); err != nil {
        t.Fatalf("Create失敗: %v", err)
    }

    // 更新
    user.Name = "Updated"
    if err := repo.Update(ctx, user); err != nil {
        t.Fatalf("Update失敗: %v", err)
    }

    // 確認
    got, err := repo.GetByID(ctx, user.ID)
    if err != nil {
        t.Fatalf("GetByID失敗: %v", err)
    }
    if got.Name != "Updated" {
        t.Errorf("Name = %q, want %q", got.Name, "Updated")
    }
}
```

### コード例 14: インターフェースを使ったモック

```go
package service

import (
    "context"
)

// UserRepository はリポジトリのインターフェース
type UserRepository interface {
    GetByID(ctx context.Context, id int64) (*User, error)
    List(ctx context.Context, limit, offset int) ([]User, error)
    Create(ctx context.Context, u *User) error
    Update(ctx context.Context, u *User) error
    Delete(ctx context.Context, id int64) error
}

// UserService はビジネスロジック層
type UserService struct {
    repo UserRepository
}

func NewUserService(repo UserRepository) *UserService {
    return &UserService{repo: repo}
}

func (s *UserService) GetUser(ctx context.Context, id int64) (*User, error) {
    return s.repo.GetByID(ctx, id)
}

// --- テスト側 ---

// mockUserRepository はテスト用のモック
type mockUserRepository struct {
    users map[int64]*User
}

func newMockUserRepository() *mockUserRepository {
    return &mockUserRepository{users: make(map[int64]*User)}
}

func (m *mockUserRepository) GetByID(ctx context.Context, id int64) (*User, error) {
    u, ok := m.users[id]
    if !ok {
        return nil, ErrNotFound
    }
    return u, nil
}

func (m *mockUserRepository) List(ctx context.Context, limit, offset int) ([]User, error) {
    var result []User
    for _, u := range m.users {
        result = append(result, *u)
    }
    return result, nil
}

func (m *mockUserRepository) Create(ctx context.Context, u *User) error {
    u.ID = int64(len(m.users) + 1)
    m.users[u.ID] = u
    return nil
}

func (m *mockUserRepository) Update(ctx context.Context, u *User) error {
    if _, ok := m.users[u.ID]; !ok {
        return ErrNotFound
    }
    m.users[u.ID] = u
    return nil
}

func (m *mockUserRepository) Delete(ctx context.Context, id int64) error {
    if _, ok := m.users[id]; !ok {
        return ErrNotFound
    }
    delete(m.users, id)
    return nil
}
```

---

## 6. ASCII図解

### 図1: database/sql 接続プールアーキテクチャ

```
アプリケーション
  │
  ▼
┌──────────────────────────────────────────────────────┐
│                   database/sql                        │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │              Connection Pool                  │   │
│  │                                              │   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐     ┌──────┐   │   │
│  │  │Active│ │Active│ │Active│ ... │Active│   │   │
│  │  │Conn 1│ │Conn 2│ │Conn 3│     │Conn N│   │   │
│  │  └──────┘ └──────┘ └──────┘     └──────┘   │   │
│  │                                              │   │
│  │  MaxOpenConns = 25                           │   │
│  │                                              │   │
│  │  ┌──────┐ ┌──────┐ ┌──────┐                │   │
│  │  │ Idle │ │ Idle │ │ Idle │                │   │
│  │  │Conn 1│ │Conn 2│ │Conn 3│                │   │
│  │  └──────┘ └──────┘ └──────┘                │   │
│  │                                              │   │
│  │  MaxIdleConns = 5                            │   │
│  │  ConnMaxLifetime = 5m                        │   │
│  │  ConnMaxIdleTime = 1m                        │   │
│  └──────────────────────────────────────────────┘   │
│                                                      │
│  ┌──────────────────────────────────────────────┐   │
│  │           Driver Interface                    │   │
│  │  driver.Driver / driver.Connector             │   │
│  │  ┌─────────┐  ┌─────────┐  ┌──────────┐    │   │
│  │  │ lib/pq  │  │go-mysql │  │go-sqlite3│    │   │
│  │  │(Postgres)│  │(MySQL)  │  │(SQLite)  │    │   │
│  │  └─────────┘  └─────────┘  └──────────┘    │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────┬───────────────────────────────────┘
                   │ TCP / Unix Socket
              ┌────▼──────┐
              │ Database   │
              │ Server     │
              └───────────┘
```

### 図2: ORM vs 生SQL の選択ガイド

```
        クエリの複雑度
          ▲
          │
     高   │   ┌─────────────────┐
          │   │  生SQL +          │  レポート/分析クエリ
          │   │  database/sql    │  複雑なJOIN (5テーブル以上)
          │   │  → 完全な制御     │  Window関数
          │   └─────────────────┘  CTE (WITH句)
          │                         再帰クエリ
          │
     中   │   ┌─────────────────┐
          │   │   sqlx            │  型安全な生SQL
          │   │  → 構造体マッピング│  中程度のJOIN
          │   │  → Named Query   │  動的フィルター
          │   └─────────────────┘
          │
     低   │   ┌─────────────────┐
          │   │   GORM            │  CRUD中心のアプリ
          │   │  → 高速開発       │  基本的なリレーション
          │   │  → マイグレーション│  プロトタイプ
          │   └─────────────────┘
          │
          └──────────────────────────> 開発速度
              低                  高

推奨:
  新規プロジェクト → sqlx（バランスが最も良い）
  CRUD中心        → GORM（開発速度重視）
  高負荷/分析     → database/sql（完全制御）
  ハイブリッド    → GORMのDB().Raw()で生SQLも併用可
```

### 図3: マイグレーションフロー

```
migrations/
├── 000001_create_users.up.sql
├── 000001_create_users.down.sql
├── 000002_create_orders.up.sql
├── 000002_create_orders.down.sql
├── 000003_add_users_phone.up.sql
└── 000003_add_users_phone.down.sql

migrate up                     migrate down
──────────>                    <──────────
V1 → V2 → V3                  V3 → V2 → V1

┌──────────────────────────────────────┐
│ schema_migrations テーブル             │
│                                      │
│ version │ dirty  │ applied_at        │
│ --------│--------│------------------ │
│ 1       │ false  │ 2024-01-01 00:00  │
│ 2       │ false  │ 2024-01-15 00:00  │
│ 3       │ false  │ 2024-02-01 00:00  │
└──────────────────────────────────────┘

dirty = true の場合:
  マイグレーションが途中で失敗した状態
  → migrate force <version> で状態をリセット
  → 失敗したSQLを修正して再実行
```

### 図4: リポジトリパターン

```
┌──────────────────────────────────────────────────┐
│                  HTTP Handler                     │
│  (net/http, gin, echo)                           │
└──────────────────┬───────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────┐
│                  Service Layer                    │
│  ビジネスロジック、トランザクション管理             │
│  → UserService, OrderService                     │
└──────────────────┬───────────────────────────────┘
                   │ interface
                   ▼
┌──────────────────────────────────────────────────┐
│              Repository Interface                 │
│  type UserRepository interface {                 │
│      GetByID(ctx, id) (*User, error)             │
│      List(ctx, limit, offset) ([]User, error)    │
│      Create(ctx, *User) error                    │
│      Update(ctx, *User) error                    │
│      Delete(ctx, id) error                       │
│  }                                               │
└──────────────────┬───────────────────────────────┘
         ┌─────────┼─────────┐
         ▼         ▼         ▼
┌────────────┐ ┌────────┐ ┌──────────┐
│ PostgreSQL │ │ MySQL  │ │  Mock    │
│ Repository │ │ Repo   │ │ (Test)   │
└──────┬─────┘ └───┬────┘ └──────────┘
       │           │
       ▼           ▼
   PostgreSQL    MySQL
```

---

## 7. 比較表

### 表1: Go DB ライブラリ詳細比較

| 項目 | database/sql | sqlx | GORM | ent |
|------|-------------|------|------|-----|
| 型安全性 | 手動Scan | structタグ | 完全ORM | コード生成 |
| SQL記述 | 生SQL | 生SQL | メソッドチェーン | DSL/生SQL |
| 学習コスト | 低 | 低 | 中 | 高 |
| パフォーマンス | 最高 | 高 | 中 | 高 |
| マイグレーション | なし | なし | AutoMigrate | atlas統合 |
| N+1検出 | なし | なし | なし (plugin) | 組み込み |
| トランザクション | 手動 | 手動 | 自動/手動 | 自動/手動 |
| リレーション | 手動JOIN | 手動JOIN | Preload/Joins | Edge |
| Prepared Stmt | 手動 | 手動 | PrepareStmt設定 | 自動 |
| NULL処理 | sql.Null* | sql.Null* | ポインタ型 | Optional field |
| バッチ処理 | 手動 | 手動 | CreateInBatches | Bulk操作 |
| 推奨場面 | 小規模/高性能 | 中規模 | CRUD中心 | 大規模/型安全 |

### 表2: マイグレーションツール詳細比較

| ツール | 言語 | アプローチ | 特徴 | CLI | Go API |
|--------|------|-----------|------|-----|--------|
| golang-migrate | Go | ファイルベース | 軽量、SQL/Go両対応 | あり | あり |
| goose | Go | ファイルベース | シンプル、Go関数対応 | あり | あり |
| atlas | Go | 宣言的 + バージョン | HCL定義、差分検出 | あり | あり |
| GORM AutoMigrate | Go | 自動 | 構造体から推論 | なし | あり |
| Flyway | Java | ファイルベース | 企業向け、多DB対応 | あり | なし |
| dbmate | Go | ファイルベース | Docker対応、軽量 | あり | なし |

### 表3: 接続プールパラメータ設定ガイド

| パラメータ | 小規模 | 中規模 | 大規模 | 説明 |
|-----------|--------|--------|--------|------|
| MaxOpenConns | 5-10 | 15-30 | 50-100 | 同時最大接続数 |
| MaxIdleConns | 2-3 | 5-10 | 10-25 | アイドル接続保持数 |
| ConnMaxLifetime | 10m | 5m | 3m | 接続の最大生存時間 |
| ConnMaxIdleTime | 5m | 2m | 1m | アイドル接続のタイムアウト |

### 表4: トランザクション分離レベル

| 分離レベル | ダーティリード | ノンリピータブルリード | ファントムリード | 用途 |
|-----------|--------------|---------------------|----------------|------|
| Read Uncommitted | あり | あり | あり | ほぼ使用しない |
| Read Committed | なし | あり | あり | PostgreSQL デフォルト |
| Repeatable Read | なし | なし | あり(※) | MySQL InnoDB デフォルト |
| Serializable | なし | なし | なし | 送金、在庫管理 |

※ PostgreSQLのRepeatable Readではファントムリードも防止される

---

## 8. パフォーマンス最適化

### N+1問題の検出と対策

```go
// N+1問題の例
// NG: ユーザーごとに注文を個別クエリ（N+1）
func (r *UserRepository) ListWithOrders_Bad(ctx context.Context) ([]UserWithOrders, error) {
    users, err := r.ListUsers(ctx) // 1回のクエリ
    if err != nil {
        return nil, err
    }

    var result []UserWithOrders
    for _, u := range users {
        // N回のクエリ（ユーザー数分）
        orders, err := r.GetOrdersByUserID(ctx, u.ID)
        if err != nil {
            return nil, err
        }
        result = append(result, UserWithOrders{User: u, Orders: orders})
    }
    return result, nil
}

// GOOD: JOINで1回のクエリにまとめる
func (r *UserRepository) ListWithOrders_Good(ctx context.Context) ([]UserWithOrders, error) {
    rows, err := r.db.QueryContext(ctx, `
        SELECT u.id, u.name, u.email,
               o.id AS order_id, o.product_id, o.quantity, o.amount
        FROM users u
        LEFT JOIN orders o ON o.user_id = u.id
        ORDER BY u.id, o.id
    `)
    if err != nil {
        return nil, err
    }
    defer rows.Close()

    // 結果をグループ化
    userMap := make(map[int64]*UserWithOrders)
    var orderedIDs []int64

    for rows.Next() {
        var (
            userID    int64
            userName  string
            userEmail string
            orderID   sql.NullInt64
            productID sql.NullInt64
            quantity  sql.NullInt32
            amount    sql.NullFloat64
        )
        if err := rows.Scan(&userID, &userName, &userEmail,
            &orderID, &productID, &quantity, &amount); err != nil {
            return nil, err
        }

        if _, ok := userMap[userID]; !ok {
            userMap[userID] = &UserWithOrders{
                User: User{ID: userID, Name: userName, Email: userEmail},
            }
            orderedIDs = append(orderedIDs, userID)
        }

        if orderID.Valid {
            userMap[userID].Orders = append(userMap[userID].Orders, Order{
                ID:        orderID.Int64,
                ProductID: productID.Int64,
                Quantity:  int(quantity.Int32),
                Amount:    amount.Float64,
            })
        }
    }

    var result []UserWithOrders
    for _, id := range orderedIDs {
        result = append(result, *userMap[id])
    }
    return result, rows.Err()
}

// GOOD: IN句で2回のクエリに最適化
func (r *UserRepository) ListWithOrders_Optimized(ctx context.Context) ([]UserWithOrders, error) {
    // クエリ1: ユーザー一覧取得
    users, err := r.ListUsers(ctx)
    if err != nil {
        return nil, err
    }

    // ユーザーIDを収集
    userIDs := make([]int64, len(users))
    for i, u := range users {
        userIDs[i] = u.ID
    }

    // クエリ2: 全注文を一括取得
    orders, err := r.GetOrdersByUserIDs(ctx, userIDs)
    if err != nil {
        return nil, err
    }

    // マッピング
    orderMap := make(map[int64][]Order)
    for _, o := range orders {
        orderMap[o.UserID] = append(orderMap[o.UserID], o)
    }

    result := make([]UserWithOrders, len(users))
    for i, u := range users {
        result[i] = UserWithOrders{
            User:   u,
            Orders: orderMap[u.ID],
        }
    }
    return result, nil
}
```

### クエリパフォーマンスの計測

```go
package middleware

import (
    "context"
    "database/sql"
    "log"
    "time"
)

// QueryLogger はクエリの実行時間を計測するラッパー
type QueryLogger struct {
    db        *sql.DB
    threshold time.Duration // この時間を超えたクエリをログに記録
}

func NewQueryLogger(db *sql.DB, threshold time.Duration) *QueryLogger {
    return &QueryLogger{db: db, threshold: threshold}
}

func (ql *QueryLogger) QueryContext(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
    start := time.Now()
    rows, err := ql.db.QueryContext(ctx, query, args...)
    duration := time.Since(start)

    if duration > ql.threshold {
        log.Printf("SLOW QUERY (%v): %s args=%v", duration, query, args)
    }

    return rows, err
}

// 接続プールの監視
func MonitorPool(db *sql.DB) {
    ticker := time.NewTicker(30 * time.Second)
    defer ticker.Stop()

    for range ticker.C {
        stats := db.Stats()
        log.Printf("DB Pool Stats: "+
            "Open=%d MaxOpen=%d InUse=%d Idle=%d "+
            "WaitCount=%d WaitDuration=%v "+
            "MaxIdleClosed=%d MaxLifetimeClosed=%d",
            stats.OpenConnections,
            stats.MaxOpenConnections,
            stats.InUse,
            stats.Idle,
            stats.WaitCount,
            stats.WaitDuration,
            stats.MaxIdleClosed,
            stats.MaxLifetimeClosed,
        )
    }
}
```

---

## 9. アンチパターン

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
    for rows.Next() {
        var u User
        rows.Scan(&u.ID, &u.Name, &u.Email)
        users = append(users, u)
    }
    return users, nil
}

// GOOD: defer rows.Close()
func listUsers(db *sql.DB) ([]User, error) {
    rows, err := db.Query("SELECT * FROM users")
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
    return users, rows.Err() // rows.Err() の確認も忘れずに
}
```

### アンチパターン 2: GORM AutoMigrate を本番で使う

```go
// BAD: 本番でAutoMigrateは危険
func main() {
    db.AutoMigrate(&User{}, &Order{}, &Product{})
    // 問題点:
    // - カラム削除されない（構造体からフィールド削除しても）
    // - データ損失のリスク
    // - ロールバック不可
    // - チーム開発でスキーマ管理が困難
    // - 本番のテーブルロックによるダウンタイム
}

// GOOD: マイグレーションツールを使う
// 開発時: AutoMigrate OK
// ステージング/本番: golang-migrate, goose, atlas で明示的に管理
// migrate -path ./migrations -database $DB_URL up
```

### アンチパターン 3: SQLインジェクション

```go
// BAD: 文字列連結でSQLを構築
func searchUsers(db *sql.DB, name string) ([]User, error) {
    query := fmt.Sprintf("SELECT * FROM users WHERE name = '%s'", name)
    // name = "'; DROP TABLE users; --" で攻撃可能
    rows, err := db.Query(query)
    // ...
}

// GOOD: プレースホルダを使用
func searchUsers(db *sql.DB, name string) ([]User, error) {
    rows, err := db.Query("SELECT * FROM users WHERE name = $1", name)
    // プレースホルダが自動的にエスケープする
    // ...
}
```

### アンチパターン 4: Context を渡さない

```go
// BAD: Context なしのクエリ
func getUser(db *sql.DB, id int) (*User, error) {
    var u User
    // Query はキャンセル不可 → リクエスト切断時もクエリが実行され続ける
    err := db.QueryRow("SELECT * FROM users WHERE id = $1", id).Scan(&u.ID, &u.Name)
    return &u, err
}

// GOOD: Context 付きのクエリ
func getUser(ctx context.Context, db *sql.DB, id int) (*User, error) {
    var u User
    // QueryRowContext はcontext.Done()でクエリをキャンセルできる
    err := db.QueryRowContext(ctx,
        "SELECT id, name FROM users WHERE id = $1", id).Scan(&u.ID, &u.Name)
    return &u, err
}
```

### アンチパターン 5: グローバル変数にDBを保持

```go
// BAD: グローバル変数
var globalDB *sql.DB

func init() {
    var err error
    globalDB, err = sql.Open("postgres", os.Getenv("DB_URL"))
    if err != nil {
        log.Fatal(err)
    }
}

func GetUser(id int) (*User, error) {
    return globalDB.QueryRow(...) // テスト時にDB差し替えが困難
}

// GOOD: 依存注入
type UserRepository struct {
    db *sql.DB
}

func NewUserRepository(db *sql.DB) *UserRepository {
    return &UserRepository{db: db}
}

func (r *UserRepository) GetUser(ctx context.Context, id int) (*User, error) {
    return r.db.QueryRowContext(ctx, ...) // テスト時にモック差し替え可能
}
```

### アンチパターン 6: Scan のカラム順依存

```go
// BAD: SELECT * を使い、Scan の順序にカラム名を暗黙的に依存
func getUser(db *sql.DB, id int) (*User, error) {
    var u User
    // テーブルにカラムが追加されるとScanが壊れる
    err := db.QueryRow("SELECT * FROM users WHERE id = $1", id).
        Scan(&u.ID, &u.Name, &u.Email)
    return &u, err
}

// GOOD: カラムを明示的に指定
func getUser(ctx context.Context, db *sql.DB, id int) (*User, error) {
    var u User
    err := db.QueryRowContext(ctx,
        "SELECT id, name, email FROM users WHERE id = $1", id).
        Scan(&u.ID, &u.Name, &u.Email)
    return &u, err
}

// BETTER: sqlx を使って構造体タグでマッピング
func getUser(ctx context.Context, db *sqlx.DB, id int) (*User, error) {
    var u User
    err := db.GetContext(ctx, &u,
        "SELECT id, name, email FROM users WHERE id = $1", id)
    return &u, err
}
```

---

## 10. FAQ

### Q1: 接続プールのサイズはどう決めるか？

目安は `MaxOpenConns = DB最大接続数 / アプリインスタンス数`。PostgreSQLデフォルトは100接続。3インスタンスなら各30程度。`MaxIdleConns`は`MaxOpenConns`の1/5〜1/3程度。

考慮事項:
- **WaitCount** が増加していたら `MaxOpenConns` を増やす
- **MaxIdleClosed** が多い場合は `MaxIdleConns` を増やす
- **MaxLifetimeClosed** が多すぎる場合は `ConnMaxLifetime` を延ばす
- PgBouncer 等のコネクションプーラーを使う場合は、アプリ側の設定を控えめにする
- ベンチマーク（`wrk`, `vegeta`等）で最適値を測定する

### Q2: database/sql と sqlx のどちらを選ぶべきか？

sqlxはdatabase/sqlの上位互換で、構造体への自動マッピングを提供する。デメリットはほぼないため、新規プロジェクトではsqlxを推奨する。GORMはCRUD中心のアプリに適するが、複雑なクエリでは生SQLに戻ることが多い。

選択基準:
- **database/sql**: 依存を最小にしたい場合、学習用
- **sqlx**: ほぼ全てのプロジェクトで推奨（標準との互換性 + 便利機能）
- **GORM**: プロトタイプ、CRUD中心、マイグレーション統合が欲しい場合
- **ent**: 大規模プロジェクト、型安全性を最重視する場合

### Q3: SQLインジェクションはGoではどう防ぐか？

プレースホルダ（`$1`, `?`）を必ず使う。文字列連結でSQLを組み立てない。sqlxのNamedQuery、GORMのWhereも内部でプレースホルダを使う。`database/sql`の標準メソッドを使えば安全。

追加対策:
- `ORDER BY` などの動的SQL部分はホワイトリストで検証
- `LIMIT`, `OFFSET` も必ずプレースホルダを使用
- SQLクエリビルダー（squirrel, goqu）を使う
- 入力のバリデーションを多層で行う

### Q4: GORMのN+1問題はどう対処するか？

GORMの`Preload`を使えばEagerローディングで解決できる。ただし`Preload`は内部的にIN句クエリを発行するため、データ量が多い場合は`Joins`の方が効率的な場合がある。

```go
// Preload: 2クエリ（ユーザー + IN句で注文）
db.Preload("Orders").Find(&users)

// Joins: 1クエリ（LEFT JOIN）
db.Joins("Profile").Find(&users)

// 条件付きPreload
db.Preload("Orders", "status = ?", "completed").Find(&users)

// ネストしたPreload
db.Preload("Orders.OrderItems.Product").Find(&users)
```

### Q5: マイグレーションのベストプラクティスは？

1. **バージョン管理**: マイグレーションファイルをGitで管理
2. **ロールバック対応**: 必ずdownファイルも作成
3. **冪等性**: `IF NOT EXISTS`, `IF EXISTS` を活用
4. **小さな変更**: 1ファイル1変更で管理しやすく
5. **データマイグレーション分離**: スキーマ変更とデータ変更は別ファイル
6. **レビュー**: マイグレーションファイルもコードレビュー対象に
7. **テスト**: CI/CDでマイグレーションのup/downをテスト

### Q6: デッドロックを回避するには？

1. **ロック順序の統一**: 常に同じ順序でテーブル/行をロック
2. **トランザクション時間の最小化**: 不要な処理をトランザクション外に
3. **適切な分離レベル**: 必要以上に高い分離レベルを使わない
4. **FOR UPDATE**: 必要な行だけをロック
5. **リトライ**: デッドロック検出時は再試行ロジックを実装

```go
// デッドロック時のリトライパターン
func WithRetry(ctx context.Context, maxRetries int, fn func() error) error {
    for i := 0; i < maxRetries; i++ {
        err := fn()
        if err == nil {
            return nil
        }
        // PostgreSQLのデッドロックエラーコード: 40P01
        if isDeadlockError(err) && i < maxRetries-1 {
            time.Sleep(time.Duration(i+1) * 100 * time.Millisecond) // バックオフ
            continue
        }
        return err
    }
    return fmt.Errorf("max retries exceeded")
}
```

---

## まとめ

| 概念 | 要点 |
|------|------|
| database/sql | 標準ライブラリ。ドライバ交換可能。最も軽量 |
| 接続プール | MaxOpenConns/MaxIdleConns/ConnMaxLifetime を本番では必ず設定 |
| sqlx | struct自動マッピング。生SQL。新規プロジェクトの第一選択 |
| GORM | フルORM。CRUD高速開発。Preload/Joinsでリレーション |
| トランザクション | BeginTx + defer Rollback + Commit。ヘルパー関数で管理 |
| マイグレーション | golang-migrate / goose / atlas。本番ではAutoMigrate禁止 |
| Context | 全DB操作にcontextを渡す。キャンセル・タイムアウトの伝搬 |
| N+1問題 | JOIN / IN句 / Preload で解決。モニタリングで検出 |
| SQLインジェクション | プレースホルダ必須。文字列連結禁止 |
| テスト | testcontainers-go / インターフェースモック |

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
4. **golang-migrate** -- https://github.com/golang-migrate/migrate
5. **goose** -- https://github.com/pressly/goose
6. **atlas** -- https://atlasgo.io/
7. **ent** -- https://entgo.io/
8. **testcontainers-go** -- https://github.com/testcontainers/testcontainers-go
9. **Go Database/SQL Tutorial** -- http://go-database-sql.org/
10. **PgBouncer** -- https://www.pgbouncer.org/
