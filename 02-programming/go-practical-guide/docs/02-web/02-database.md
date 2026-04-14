# Database -- database/sql, sqlx, GORM

> Go provides standard DB connectivity via database/sql, enhances productivity with sqlx and GORM, and supports production operations through migrations and connection pooling.

---

## What You Will Learn in This Chapter

1. **database/sql** -- DB operations and connection pool management with the standard library
2. **sqlx / GORM** -- Choosing between higher-level libraries and practical techniques
3. **Connection Pooling and Migrations** -- Best practices for production operations
4. **Transaction Design** -- Leveraging ACID properties and avoiding deadlocks
5. **Performance Optimization** -- Query optimization, detecting and addressing the N+1 problem
6. **Test Strategies** -- DB testing approaches and test containers


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content in [Gin / Echo -- Go Web Frameworks](./01-gin-echo.md)

---

## 1. database/sql Basics

### Code Example 1: database/sql Connection and Pool Configuration

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
    // sql.Open initializes the connection pool but does not establish an actual connection
    db, err := sql.Open("postgres", "postgres://user:pass@localhost/mydb?sslmode=disable")
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    // Connection pool settings (must be configured in production environments)
    db.SetMaxOpenConns(25)              // Maximum number of concurrent connections
    db.SetMaxIdleConns(5)               // Maximum number of idle connections
    db.SetConnMaxLifetime(5 * time.Minute) // Maximum lifetime of a connection
    db.SetConnMaxIdleTime(1 * time.Minute) // Maximum idle time for a connection

    // Verify the actual connection with Ping
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()

    if err := db.PingContext(ctx); err != nil {
        log.Fatal("DB connection failed:", err)
    }

    fmt.Println("DB connection successful")
}
```

### Code Example 2: Driver Registration and DSN Construction

```go
package database

import (
    "database/sql"
    "fmt"
    "net/url"

    _ "github.com/go-sql-driver/mysql" // MySQL
    _ "github.com/lib/pq"              // PostgreSQL
    _ "github.com/mattn/go-sqlite3"    // SQLite
    _ "modernc.org/sqlite"             // SQLite (no CGO required)
)

// Config represents DB connection settings
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

// DSN generates a DSN string for each driver
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
        return c.DBName // File path or ":memory:"
    default:
        panic("unsupported driver: " + c.Driver)
    }
}

// Open initializes the DB connection pool
func Open(cfg Config) (*sql.DB, error) {
    db, err := sql.Open(cfg.Driver, cfg.DSN())
    if err != nil {
        return nil, fmt.Errorf("sql.Open: %w", err)
    }

    // Pool settings based on the driver
    switch cfg.Driver {
    case "postgres", "mysql":
        db.SetMaxOpenConns(25)
        db.SetMaxIdleConns(5)
        db.SetConnMaxLifetime(5 * time.Minute)
    case "sqlite3", "sqlite":
        // Single connection is recommended for SQLite
        db.SetMaxOpenConns(1)
    }

    return db, nil
}
```

### Code Example 3: CRUD Operations (database/sql)

```go
package repository

import (
    "context"
    "database/sql"
    "errors"
    "fmt"
    "time"
)

// User represents a user entity
type User struct {
    ID        int64
    Name      string
    Email     string
    CreatedAt time.Time
    UpdatedAt time.Time
}

// ErrNotFound is the error returned when a resource is not found
var ErrNotFound = errors.New("resource not found")

// UserRepository provides access to user data
type UserRepository struct {
    db *sql.DB
}

// NewUserRepository creates a new UserRepository
func NewUserRepository(db *sql.DB) *UserRepository {
    return &UserRepository{db: db}
}

// GetByID retrieves a user by the specified ID
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

// GetByEmail searches for a user by email address
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

// List retrieves all users (with pagination)
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
    // Check for errors during iteration with rows.Err() (important)
    if err := rows.Err(); err != nil {
        return nil, fmt.Errorf("List rows: %w", err)
    }
    return users, nil
}

// Search searches for users by keyword
func (r *UserRepository) Search(ctx context.Context, keyword string) ([]User, error) {
    // Escape wildcards for LIKE search
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

// Create creates a new user
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

// Update updates an existing user
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

// Delete deletes a user
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

// BulkCreate creates multiple users in bulk
func (r *UserRepository) BulkCreate(ctx context.Context, users []User) error {
    // For large volumes, the COPY protocol is faster,
    // but here we use multi-row INSERT ... VALUES
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

### Code Example 4: Using Prepared Statements

```go
package repository

import (
    "context"
    "database/sql"
    "fmt"
)

// PreparedUserRepository is a repository that uses Prepared Statements
type PreparedUserRepository struct {
    db         *sql.DB
    stmtGetByID *sql.Stmt
    stmtList    *sql.Stmt
    stmtCreate  *sql.Stmt
    stmtUpdate  *sql.Stmt
    stmtDelete  *sql.Stmt
}

// NewPreparedUserRepository prepares statements in advance
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

// Close closes all Prepared Statements
func (r *PreparedUserRepository) Close() error {
    stmts := []*sql.Stmt{r.stmtGetByID, r.stmtList, r.stmtCreate, r.stmtUpdate, r.stmtDelete}
    for _, stmt := range stmts {
        if stmt != nil {
            stmt.Close()
        }
    }
    return nil
}

// GetByID retrieves a user efficiently using a Prepared Statement
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

### Code Example 5: Transaction Management

```go
package service

import (
    "context"
    "database/sql"
    "fmt"
)

// TxFunc is the function type executed within a transaction
type TxFunc func(tx *sql.Tx) error

// WithTransaction is a helper function for transaction management
// It automatically handles panic recovery, rollback, and commit
func WithTransaction(ctx context.Context, db *sql.DB, opts *sql.TxOptions, fn TxFunc) error {
    tx, err := db.BeginTx(ctx, opts)
    if err != nil {
        return fmt.Errorf("begin transaction: %w", err)
    }

    defer func() {
        if p := recover(); p != nil {
            // Rollback on panic
            _ = tx.Rollback()
            panic(p) // Re-panic
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

// Usage example: Money transfer
func (s *AccountService) Transfer(ctx context.Context, fromID, toID int64, amount float64) error {
    return WithTransaction(ctx, s.db, &sql.TxOptions{
        Isolation: sql.LevelSerializable,
    }, func(tx *sql.Tx) error {
        // Check sender's balance
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

        // Debit from sender
        _, err = tx.ExecContext(ctx,
            "UPDATE accounts SET balance = balance - $1, updated_at = NOW() WHERE id = $2",
            amount, fromID)
        if err != nil {
            return fmt.Errorf("debit: %w", err)
        }

        // Credit to recipient
        _, err = tx.ExecContext(ctx,
            "UPDATE accounts SET balance = balance + $1, updated_at = NOW() WHERE id = $2",
            amount, toID)
        if err != nil {
            return fmt.Errorf("credit: %w", err)
        }

        // Record transaction history
        _, err = tx.ExecContext(ctx,
            `INSERT INTO transactions (from_account_id, to_account_id, amount, created_at)
             VALUES ($1, $2, $3, NOW())`, fromID, toID, amount)
        if err != nil {
            return fmt.Errorf("record transaction: %w", err)
        }

        return nil
    })
}

// ReadOnlyTransaction is a read-only transaction
func ReadOnlyTransaction(ctx context.Context, db *sql.DB, fn TxFunc) error {
    return WithTransaction(ctx, db, &sql.TxOptions{
        ReadOnly: true,
    }, fn)
}

// Transaction isolation level usage guide
// sql.LevelDefault          - Driver default (PostgreSQL uses ReadCommitted)
// sql.LevelReadUncommitted  - Allows dirty reads (rarely used)
// sql.LevelReadCommitted    - Only reads committed data
// sql.LevelRepeatableRead   - Guarantees same results within a transaction
// sql.LevelSerializable     - Most strict (risk of deadlocks)
```

### Code Example 6: Handling Null Values

```go
package model

import (
    "database/sql"
    "encoding/json"
    "time"
)

// NullableUser is a user with nullable fields
type NullableUser struct {
    ID        int64
    Name      string
    Email     sql.NullString  // Nullable
    Phone     sql.NullString  // Nullable
    Age       sql.NullInt64   // Nullable
    Score     sql.NullFloat64 // Nullable
    IsActive  sql.NullBool    // Nullable
    DeletedAt sql.NullTime    // Nullable (soft delete)
}

// UserJSON is a struct for JSON responses
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

// ToJSON converts NullableUser to a JSON-friendly struct
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

// Custom Null type (a more idiomatic Go approach)
// Null representation using generics (Go 1.18+)
type Nullable[T any] struct {
    Value T
    Valid bool
}

func NewNullableT any Nullable[T] {
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

## 2. Using sqlx

### Code Example 7: sqlx Basic Operations

```go
package repository

import (
    "context"
    "fmt"
    "time"

    "github.com/jmoiron/sqlx"
    _ "github.com/lib/pq"
)

// User is a struct using sqlx's db tags
type User struct {
    ID        int64     `db:"id" json:"id"`
    Name      string    `db:"name" json:"name"`
    Email     string    `db:"email" json:"email"`
    Role      string    `db:"role" json:"role"`
    CreatedAt time.Time `db:"created_at" json:"created_at"`
    UpdatedAt time.Time `db:"updated_at" json:"updated_at"`
}

// SqlxUserRepository is a repository using sqlx
type SqlxUserRepository struct {
    db *sqlx.DB
}

// NewSqlxUserRepository initializes a sqlx connection
func NewSqlxUserRepository(dsn string) (*SqlxUserRepository, error) {
    // sqlx.Connect performs Open + Ping
    db, err := sqlx.Connect("postgres", dsn)
    if err != nil {
        return nil, fmt.Errorf("sqlx.Connect: %w", err)
    }

    db.SetMaxOpenConns(25)
    db.SetMaxIdleConns(5)
    db.SetConnMaxLifetime(5 * time.Minute)

    return &SqlxUserRepository{db: db}, nil
}

// GetByID retrieves a single row using sqlx.GetContext
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

// List retrieves multiple rows using sqlx.SelectContext
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

// Create inserts using NamedExecContext with named parameters
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

// Update updates using NamedExecContext with named parameters
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

// ListByRole retrieves a list of users with a specific role
func (r *SqlxUserRepository) ListByRole(ctx context.Context, role string) ([]User, error) {
    var users []User
    err := r.db.SelectContext(ctx, &users,
        `SELECT id, name, email, role, created_at, updated_at
         FROM users WHERE role = $1 ORDER BY name`, role)
    return users, err
}

// SearchByNames performs a multi-value search using an IN clause (using sqlx's In function)
func (r *SqlxUserRepository) SearchByNames(ctx context.Context, names []string) ([]User, error) {
    // sqlx.In expands a slice into placeholders
    query, args, err := sqlx.In(
        `SELECT id, name, email, role, created_at, updated_at
         FROM users WHERE name IN (?) ORDER BY name`, names)
    if err != nil {
        return nil, fmt.Errorf("sqlx.In: %w", err)
    }

    // Rebind to $1-style placeholders for PostgreSQL
    query = r.db.Rebind(query)

    var users []User
    err = r.db.SelectContext(ctx, &users, query, args...)
    return users, err
}
```

### Code Example 8: Advanced sqlx Features

```go
package repository

import (
    "context"
    "database/sql"
    "fmt"

    "github.com/jmoiron/sqlx"
)

// Dynamic query builder
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

// BuildQuery constructs a dynamic query based on filter conditions
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
        // SQL injection prevention: whitelist
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

// SearchWithFilter searches for users with a dynamic filter
func (r *SqlxUserRepository) SearchWithFilter(ctx context.Context, filter UserFilter) ([]User, error) {
    query, args := filter.BuildQuery()

    // sqlx.Named converts a named query into a query with placeholders
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

// StructScan vs Scan comparison
func demonstrateScanDifference(db *sqlx.DB) {
    // database/sql: Manual Scan for each column
    row := db.QueryRow("SELECT id, name, email FROM users WHERE id = $1", 1)
    var id int64
    var name, email string
    row.Scan(&id, &name, &email) // Must match column order

    // sqlx: Automatic mapping to a struct
    var user User
    db.Get(&user, "SELECT id, name, email FROM users WHERE id = $1", 1)
    // Automatically mapped based on db tags
}
```

---

## 3. Using GORM

### Code Example 9: GORM Basic Operations

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

// GormUser is a GORM model
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
    DeletedAt gorm.DeletedAt `gorm:"index" json:"deleted_at,omitempty"` // Soft delete
}

// TableName customizes the table name
func (GormUser) TableName() string {
    return "users"
}

// GormProfile is a user profile
type GormProfile struct {
    ID     uint   `gorm:"primarykey"`
    UserID uint   `gorm:"uniqueIndex;not null"`
    Bio    string `gorm:"type:text"`
    Avatar string `gorm:"size:500"`
}

// GormOrder is an order
type GormOrder struct {
    ID        uint      `gorm:"primarykey"`
    UserID    uint      `gorm:"index;not null"`
    ProductID uint      `gorm:"index;not null"`
    Quantity  int       `gorm:"not null;check:quantity > 0"`
    Amount    float64   `gorm:"type:decimal(10,2);not null"`
    Status    string    `gorm:"size:20;default:'pending'"`
    CreatedAt time.Time
}

// InitGorm initializes a GORM connection
func InitGorm(dsn string) (*gorm.DB, error) {
    db, err := gorm.Open(postgres.Open(dsn), &gorm.Config{
        Logger: logger.Default.LogMode(logger.Info), // Output SQL logs
        NowFunc: func() time.Time {
            return time.Now().UTC()
        },
        // Disable Dry Run mode before SQL execution
        DryRun: false,
        // PrepareStmt: true caches Prepared Statements
        PrepareStmt: true,
    })
    if err != nil {
        return nil, fmt.Errorf("gorm.Open: %w", err)
    }

    // Connection pool settings
    sqlDB, err := db.DB()
    if err != nil {
        return nil, fmt.Errorf("get sql.DB: %w", err)
    }
    sqlDB.SetMaxOpenConns(25)
    sqlDB.SetMaxIdleConns(5)
    sqlDB.SetConnMaxLifetime(5 * time.Minute)

    return db, nil
}

// GormUserRepository is a repository using GORM
type GormUserRepository struct {
    db *gorm.DB
}

// Create creates a user
func (r *GormUserRepository) Create(ctx context.Context, user *GormUser) error {
    result := r.db.WithContext(ctx).Create(user)
    if result.Error != nil {
        return fmt.Errorf("Create: %w", result.Error)
    }
    return nil
}

// GetByID retrieves a user by ID (including related data)
func (r *GormUserRepository) GetByID(ctx context.Context, id uint) (*GormUser, error) {
    var user GormUser
    result := r.db.WithContext(ctx).
        Preload("Profile").                           // Also load profile
        Preload("Orders", "status = ?", "completed"). // Only completed orders
        First(&user, id)

    if errors.Is(result.Error, gorm.ErrRecordNotFound) {
        return nil, ErrNotFound
    }
    if result.Error != nil {
        return nil, fmt.Errorf("GetByID: %w", result.Error)
    }
    return &user, nil
}

// List retrieves a paginated list of users
func (r *GormUserRepository) List(ctx context.Context, page, pageSize int) ([]GormUser, int64, error) {
    var users []GormUser
    var total int64

    // Count query
    r.db.WithContext(ctx).Model(&GormUser{}).Count(&total)

    // Data retrieval
    result := r.db.WithContext(ctx).
        Order("id ASC").
        Limit(pageSize).
        Offset((page - 1) * pageSize).
        Find(&users)

    return users, total, result.Error
}

// Search performs a conditional search
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

// Update updates a user
func (r *GormUserRepository) Update(ctx context.Context, user *GormUser) error {
    // Save updates all fields
    result := r.db.WithContext(ctx).Save(user)
    return result.Error
}

// UpdatePartial updates only the specified fields
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

// Delete performs a soft delete
func (r *GormUserRepository) Delete(ctx context.Context, id uint) error {
    result := r.db.WithContext(ctx).Delete(&GormUser{}, id)
    if result.RowsAffected == 0 {
        return ErrNotFound
    }
    return result.Error
}

// HardDelete performs a permanent (hard) delete
func (r *GormUserRepository) HardDelete(ctx context.Context, id uint) error {
    result := r.db.WithContext(ctx).Unscoped().Delete(&GormUser{}, id)
    return result.Error
}

// Upsert updates if exists, inserts if not
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

### Code Example 10: GORM Transactions and Hooks

```go
package service

import (
    "context"
    "fmt"

    "gorm.io/gorm"
)

// GormOrderService is the order service
type GormOrderService struct {
    db *gorm.DB
}

// CreateOrder creates an order within a transaction
func (s *GormOrderService) CreateOrder(ctx context.Context, order *GormOrder) error {
    return s.db.WithContext(ctx).Transaction(func(tx *gorm.DB) error {
        // Check inventory
        var product Product
        if err := tx.Clauses(clause.Locking{Strength: "UPDATE"}).
            First(&product, order.ProductID).Error; err != nil {
            return fmt.Errorf("product not found: %w", err)
        }

        if product.Stock < order.Quantity {
            return fmt.Errorf("insufficient stock: available=%d, requested=%d",
                product.Stock, order.Quantity)
        }

        // Reduce inventory
        if err := tx.Model(&product).
            Update("stock", gorm.Expr("stock - ?", order.Quantity)).Error; err != nil {
            return fmt.Errorf("update stock: %w", err)
        }

        // Create the order
        order.Amount = product.Price * float64(order.Quantity)
        if err := tx.Create(order).Error; err != nil {
            return fmt.Errorf("create order: %w", err)
        }

        return nil
    })
}

// GORM Hooks (lifecycle callbacks)
func (u *GormUser) BeforeCreate(tx *gorm.DB) error {
    // Validation
    if u.Name == "" {
        return fmt.Errorf("name is required")
    }
    if u.Role == "" {
        u.Role = "user" // Default value
    }
    return nil
}

func (u *GormUser) AfterCreate(tx *gorm.DB) error {
    // Record audit log, etc.
    tx.Exec("INSERT INTO audit_logs (action, entity, entity_id) VALUES (?, ?, ?)",
        "CREATE", "users", u.ID)
    return nil
}

func (u *GormUser) BeforeUpdate(tx *gorm.DB) error {
    // Validation on update
    if u.Email != "" && !isValidEmail(u.Email) {
        return fmt.Errorf("invalid email format")
    }
    return nil
}
```

---

## 4. Migrations

### Code Example 11: Using golang-migrate

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

// RunMigrations executes migrations
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

    // Execute migration
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

// RollbackMigration rolls back to the previous version
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

Migration file examples:

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

### Code Example 12: Using goose

```go
package migrations

import (
    "database/sql"
    "fmt"

    "github.com/pressly/goose/v3"
)

// GooseMigrate performs migration using goose
func GooseMigrate(db *sql.DB, dir string) error {
    goose.SetDialect("postgres")

    if err := goose.Up(db, dir); err != nil {
        return fmt.Errorf("goose up: %w", err)
    }
    return nil
}

// GooseStatus displays the current migration status
func GooseStatus(db *sql.DB, dir string) error {
    goose.SetDialect("postgres")
    return goose.Status(db, dir)
}

// GooseRollback rolls back one step
func GooseRollback(db *sql.DB, dir string) error {
    goose.SetDialect("postgres")
    return goose.Down(db, dir)
}

// Go function migration (goose-specific feature)
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

## 5. Test Strategies

### Code Example 13: DB Testing with testcontainers-go

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

// setupTestDB starts a PostgreSQL container for testing
func setupTestDB(t *testing.T) (*sql.DB, func()) {
    t.Helper()
    ctx := context.Background()

    // Start the PostgreSQL container
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
        t.Fatalf("Failed to start container: %v", err)
    }

    // Build DSN
    host, _ := container.Host(ctx)
    port, _ := container.MappedPort(ctx, "5432")
    dsn := fmt.Sprintf("postgres://test:test@%s:%s/testdb?sslmode=disable", host, port.Port())

    // Connect to DB
    db, err := sql.Open("postgres", dsn)
    if err != nil {
        t.Fatalf("Failed to connect to DB: %v", err)
    }

    // Create tables
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
        t.Fatalf("Failed to create table: %v", err)
    }

    // Cleanup function
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
        t.Fatalf("Create failed: %v", err)
    }

    if user.ID == 0 {
        t.Error("ID was not set")
    }

    // Retrieve and verify
    got, err := repo.GetByID(ctx, user.ID)
    if err != nil {
        t.Fatalf("GetByID failed: %v", err)
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

    // Setup
    user := &User{Name: "Original", Email: "original@example.com"}
    if err := repo.Create(ctx, user); err != nil {
        t.Fatalf("Create failed: %v", err)
    }

    // Update
    user.Name = "Updated"
    if err := repo.Update(ctx, user); err != nil {
        t.Fatalf("Update failed: %v", err)
    }

    // Verify
    got, err := repo.GetByID(ctx, user.ID)
    if err != nil {
        t.Fatalf("GetByID failed: %v", err)
    }
    if got.Name != "Updated" {
        t.Errorf("Name = %q, want %q", got.Name, "Updated")
    }
}
```

### Code Example 14: Mocking with Interfaces

```go
package service

import (
    "context"
)

// UserRepository is the repository interface
type UserRepository interface {
    GetByID(ctx context.Context, id int64) (*User, error)
    List(ctx context.Context, limit, offset int) ([]User, error)
    Create(ctx context.Context, u *User) error
    Update(ctx context.Context, u *User) error
    Delete(ctx context.Context, id int64) error
}

// UserService is the business logic layer
type UserService struct {
    repo UserRepository
}

func NewUserService(repo UserRepository) *UserService {
    return &UserService{repo: repo}
}

func (s *UserService) GetUser(ctx context.Context, id int64) (*User, error) {
    return s.repo.GetByID(ctx, id)
}

// --- Test side ---

// mockUserRepository is a mock for testing
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

## 6. ASCII Diagrams

### Figure 1: database/sql Connection Pool Architecture

```
Application
  |
  v
+------------------------------------------------------+
|                   database/sql                        |
|                                                      |
|  +----------------------------------------------+   |
|  |              Connection Pool                  |   |
|  |                                              |   |
|  |  +------+ +------+ +------+     +------+   |   |
|  |  |Active| |Active| |Active| ... |Active|   |   |
|  |  |Conn 1| |Conn 2| |Conn 3|     |Conn N|   |   |
|  |  +------+ +------+ +------+     +------+   |   |
|  |                                              |   |
|  |  MaxOpenConns = 25                           |   |
|  |                                              |   |
|  |  +------+ +------+ +------+                |   |
|  |  | Idle | | Idle | | Idle |                |   |
|  |  |Conn 1| |Conn 2| |Conn 3|                |   |
|  |  +------+ +------+ +------+                |   |
|  |                                              |   |
|  |  MaxIdleConns = 5                            |   |
|  |  ConnMaxLifetime = 5m                        |   |
|  |  ConnMaxIdleTime = 1m                        |   |
|  +----------------------------------------------+   |
|                                                      |
|  +----------------------------------------------+   |
|  |           Driver Interface                    |   |
|  |  driver.Driver / driver.Connector             |   |
|  |  +---------+  +---------+  +----------+    |   |
|  |  | lib/pq  |  |go-mysql |  |go-sqlite3|    |   |
|  |  |(Postgres)|  |(MySQL)  |  |(SQLite)  |    |   |
|  |  +---------+  +---------+  +----------+    |   |
|  +----------------------------------------------+   |
+------------------+-----------------------------------+
                   | TCP / Unix Socket
              +----v------+
              | Database   |
              | Server     |
              +-----------+
```

### Figure 2: ORM vs Raw SQL Selection Guide

```
        Query Complexity
          ^
          |
     High |   +-----------------+
          |   |  Raw SQL +        |  Report/analytics queries
          |   |  database/sql    |  Complex JOINs (5+ tables)
          |   |  -> Full control  |  Window functions
          |   +-----------------+  CTEs (WITH clause)
          |                         Recursive queries
          |
     Mid  |   +-----------------+
          |   |   sqlx            |  Type-safe raw SQL
          |   |  -> Struct mapping|  Moderate JOINs
          |   |  -> Named Query   |  Dynamic filters
          |   +-----------------+
          |
     Low  |   +-----------------+
          |   |   GORM            |  CRUD-centric apps
          |   |  -> Rapid dev     |  Basic relations
          |   |  -> Migration     |  Prototypes
          |   +-----------------+
          |
          +-----------------------------> Development Speed
              Low                  High

Recommendations:
  New projects     -> sqlx (best balance)
  CRUD-centric     -> GORM (prioritize development speed)
  High-load/analytics -> database/sql (full control)
  Hybrid           -> Use raw SQL via GORM's DB().Raw() as needed
```

### Figure 3: Migration Flow

```
migrations/
├── 000001_create_users.up.sql
├── 000001_create_users.down.sql
├── 000002_create_orders.up.sql
├── 000002_create_orders.down.sql
├── 000003_add_users_phone.up.sql
└── 000003_add_users_phone.down.sql

migrate up                     migrate down
---------->                    <----------
V1 -> V2 -> V3                V3 -> V2 -> V1

+--------------------------------------+
| schema_migrations table              |
|                                      |
| version | dirty  | applied_at        |
| --------|--------|------------------ |
| 1       | false  | 2024-01-01 00:00  |
| 2       | false  | 2024-01-15 00:00  |
| 3       | false  | 2024-02-01 00:00  |
+--------------------------------------+

When dirty = true:
  Migration failed partway through
  -> Reset the state with migrate force <version>
  -> Fix the failed SQL and re-run
```

### Figure 4: Repository Pattern

```
+--------------------------------------------------+
|                  HTTP Handler                     |
|  (net/http, gin, echo)                           |
+------------------+-------------------------------+
                   |
                   v
+--------------------------------------------------+
|                  Service Layer                    |
|  Business logic, transaction management           |
|  -> UserService, OrderService                     |
+------------------+-------------------------------+
                   | interface
                   v
+--------------------------------------------------+
|              Repository Interface                 |
|  type UserRepository interface {                 |
|      GetByID(ctx, id) (*User, error)             |
|      List(ctx, limit, offset) ([]User, error)    |
|      Create(ctx, *User) error                    |
|      Update(ctx, *User) error                    |
|      Delete(ctx, id) error                       |
|  }                                               |
+------------------+-------------------------------+
         +---------+---------+
         v         v         v
+------------+ +--------+ +----------+
| PostgreSQL | | MySQL  | |  Mock    |
| Repository | | Repo   | | (Test)   |
+------+-----+ +---+----+ +----------+
       |           |
       v           v
   PostgreSQL    MySQL
```

---

## 7. Comparison Tables

### Table 1: Detailed Comparison of Go DB Libraries

| Item | database/sql | sqlx | GORM | ent |
|------|-------------|------|------|-----|
| Type safety | Manual Scan | struct tags | Full ORM | Code generation |
| SQL writing | Raw SQL | Raw SQL | Method chaining | DSL/Raw SQL |
| Learning cost | Low | Low | Medium | High |
| Performance | Highest | High | Medium | High |
| Migration | None | None | AutoMigrate | atlas integration |
| N+1 detection | None | None | None (plugin) | Built-in |
| Transactions | Manual | Manual | Auto/Manual | Auto/Manual |
| Relations | Manual JOIN | Manual JOIN | Preload/Joins | Edge |
| Prepared Stmt | Manual | Manual | PrepareStmt config | Automatic |
| NULL handling | sql.Null* | sql.Null* | Pointer types | Optional field |
| Batch processing | Manual | Manual | CreateInBatches | Bulk operations |
| Recommended for | Small-scale/high-perf | Medium-scale | CRUD-centric | Large-scale/type-safe |

### Table 2: Detailed Comparison of Migration Tools

| Tool | Language | Approach | Features | CLI | Go API |
|------|----------|----------|----------|-----|--------|
| golang-migrate | Go | File-based | Lightweight, SQL/Go support | Yes | Yes |
| goose | Go | File-based | Simple, Go function support | Yes | Yes |
| atlas | Go | Declarative + versioned | HCL definitions, diff detection | Yes | Yes |
| GORM AutoMigrate | Go | Automatic | Inferred from structs | No | Yes |
| Flyway | Java | File-based | Enterprise, multi-DB support | Yes | No |
| dbmate | Go | File-based | Docker support, lightweight | Yes | No |

### Table 3: Connection Pool Parameter Configuration Guide

| Parameter | Small-scale | Medium-scale | Large-scale | Description |
|-----------|------------|--------------|-------------|-------------|
| MaxOpenConns | 5-10 | 15-30 | 50-100 | Maximum concurrent connections |
| MaxIdleConns | 2-3 | 5-10 | 10-25 | Number of idle connections to keep |
| ConnMaxLifetime | 10m | 5m | 3m | Maximum lifetime of a connection |
| ConnMaxIdleTime | 5m | 2m | 1m | Timeout for idle connections |

### Table 4: Transaction Isolation Levels

| Isolation Level | Dirty Read | Non-repeatable Read | Phantom Read | Use Case |
|----------------|------------|---------------------|--------------|----------|
| Read Uncommitted | Yes | Yes | Yes | Rarely used |
| Read Committed | No | Yes | Yes | PostgreSQL default |
| Repeatable Read | No | No | Yes(*) | MySQL InnoDB default |
| Serializable | No | No | No | Money transfers, inventory management |

(*) PostgreSQL's Repeatable Read also prevents phantom reads

---

## 8. Performance Optimization

### Detecting and Addressing the N+1 Problem

```go
// N+1 problem example
// BAD: Individual query per user for orders (N+1)
func (r *UserRepository) ListWithOrders_Bad(ctx context.Context) ([]UserWithOrders, error) {
    users, err := r.ListUsers(ctx) // 1 query
    if err != nil {
        return nil, err
    }

    var result []UserWithOrders
    for _, u := range users {
        // N queries (one per user)
        orders, err := r.GetOrdersByUserID(ctx, u.ID)
        if err != nil {
            return nil, err
        }
        result = append(result, UserWithOrders{User: u, Orders: orders})
    }
    return result, nil
}

// GOOD: Consolidate into a single query with JOIN
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

    // Group the results
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

// GOOD: Optimize to 2 queries using an IN clause
func (r *UserRepository) ListWithOrders_Optimized(ctx context.Context) ([]UserWithOrders, error) {
    // Query 1: Retrieve user list
    users, err := r.ListUsers(ctx)
    if err != nil {
        return nil, err
    }

    // Collect user IDs
    userIDs := make([]int64, len(users))
    for i, u := range users {
        userIDs[i] = u.ID
    }

    // Query 2: Bulk retrieve all orders
    orders, err := r.GetOrdersByUserIDs(ctx, userIDs)
    if err != nil {
        return nil, err
    }

    // Mapping
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

### Measuring Query Performance

```go
package middleware

import (
    "context"
    "database/sql"
    "log"
    "time"
)

// QueryLogger is a wrapper that measures query execution time
type QueryLogger struct {
    db        *sql.DB
    threshold time.Duration // Log queries exceeding this duration
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

// Connection pool monitoring
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

## 9. Anti-patterns

### Anti-pattern 1: Forgetting rows.Close()

```go
// BAD: Forgetting to close rows causes connection leaks
func listUsers(db *sql.DB) ([]User, error) {
    rows, err := db.Query("SELECT * FROM users")
    if err != nil {
        return nil, err
    }
    // Missing rows.Close() -> Connection pool exhaustion

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
    return users, rows.Err() // Don't forget to check rows.Err()
}
```

### Anti-pattern 2: Using GORM AutoMigrate in Production

```go
// BAD: AutoMigrate in production is dangerous
func main() {
    db.AutoMigrate(&User{}, &Order{}, &Product{})
    // Problems:
    // - Columns are not dropped (even if fields are removed from the struct)
    // - Risk of data loss
    // - Cannot rollback
    // - Schema management becomes difficult in team development
    // - Table locks in production cause downtime
}

// GOOD: Use migration tools
// Development: AutoMigrate is OK
// Staging/Production: Manage explicitly with golang-migrate, goose, atlas
// migrate -path ./migrations -database $DB_URL up
```

### Anti-pattern 3: SQL Injection

```go
// BAD: Building SQL with string concatenation
func searchUsers(db *sql.DB, name string) ([]User, error) {
    query := fmt.Sprintf("SELECT * FROM users WHERE name = '%s'", name)
    // Exploitable with name = "'; DROP TABLE users; --"
    rows, err := db.Query(query)
    // ...
}

// GOOD: Use placeholders
func searchUsers(db *sql.DB, name string) ([]User, error) {
    rows, err := db.Query("SELECT * FROM users WHERE name = $1", name)
    // Placeholders automatically escape the input
    // ...
}
```

### Anti-pattern 4: Not Passing Context

```go
// BAD: Query without Context
func getUser(db *sql.DB, id int) (*User, error) {
    var u User
    // Query cannot be cancelled -> Continues executing even after request disconnect
    err := db.QueryRow("SELECT * FROM users WHERE id = $1", id).Scan(&u.ID, &u.Name)
    return &u, err
}

// GOOD: Query with Context
func getUser(ctx context.Context, db *sql.DB, id int) (*User, error) {
    var u User
    // QueryRowContext can cancel the query via context.Done()
    err := db.QueryRowContext(ctx,
        "SELECT id, name FROM users WHERE id = $1", id).Scan(&u.ID, &u.Name)
    return &u, err
}
```

### Anti-pattern 5: Holding DB in a Global Variable

```go
// BAD: Global variable
var globalDB *sql.DB

func init() {
    var err error
    globalDB, err = sql.Open("postgres", os.Getenv("DB_URL"))
    if err != nil {
        log.Fatal(err)
    }
}

func GetUser(id int) (*User, error) {
    return globalDB.QueryRow(...) // Difficult to swap DB for testing
}

// GOOD: Dependency injection
type UserRepository struct {
    db *sql.DB
}

func NewUserRepository(db *sql.DB) *UserRepository {
    return &UserRepository{db: db}
}

func (r *UserRepository) GetUser(ctx context.Context, id int) (*User, error) {
    return r.db.QueryRowContext(ctx, ...) // Can swap with a mock for testing
}
```

### Anti-pattern 6: Relying on Scan Column Order

```go
// BAD: Using SELECT * and implicitly relying on column order for Scan
func getUser(db *sql.DB, id int) (*User, error) {
    var u User
    // Scan breaks when columns are added to the table
    err := db.QueryRow("SELECT * FROM users WHERE id = $1", id).
        Scan(&u.ID, &u.Name, &u.Email)
    return &u, err
}

// GOOD: Explicitly specify columns
func getUser(ctx context.Context, db *sql.DB, id int) (*User, error) {
    var u User
    err := db.QueryRowContext(ctx,
        "SELECT id, name, email FROM users WHERE id = $1", id).
        Scan(&u.ID, &u.Name, &u.Email)
    return &u, err
}

// BETTER: Use sqlx to map with struct tags
func getUser(ctx context.Context, db *sqlx.DB, id int) (*User, error) {
    var u User
    err := db.GetContext(ctx, &u,
        "SELECT id, name, email FROM users WHERE id = $1", id)
    return &u, err
}
```

---

## 10. FAQ

### Q1: How should I determine the connection pool size?

A rule of thumb is `MaxOpenConns = DB max connections / number of app instances`. PostgreSQL defaults to 100 connections. With 3 instances, approximately 30 each. `MaxIdleConns` should be about 1/5 to 1/3 of `MaxOpenConns`.

Considerations:
- Increase `MaxOpenConns` if **WaitCount** is increasing
- Increase `MaxIdleConns` if **MaxIdleClosed** is high
- Extend `ConnMaxLifetime` if **MaxLifetimeClosed** is too high
- Use conservative app-side settings when using a connection pooler like PgBouncer
- Measure optimal values with benchmarks (`wrk`, `vegeta`, etc.)

### Q2: Should I choose database/sql or sqlx?

sqlx is a superset of database/sql that provides automatic struct mapping. Since there are virtually no downsides, sqlx is recommended for new projects. GORM is suitable for CRUD-centric apps, but for complex queries, you often end up writing raw SQL anyway.

Selection criteria:
- **database/sql**: When you want to minimize dependencies, for learning purposes
- **sqlx**: Recommended for almost all projects (standard compatibility + convenience features)
- **GORM**: Prototypes, CRUD-centric apps, when you want integrated migration
- **ent**: Large-scale projects, when type safety is the top priority

### Q3: How do you prevent SQL injection in Go?

Always use placeholders (`$1`, `?`). Never build SQL through string concatenation. sqlx's NamedQuery and GORM's Where also use placeholders internally. Using the standard methods of `database/sql` is safe.

Additional measures:
- Validate dynamic SQL parts like `ORDER BY` with a whitelist
- Always use placeholders for `LIMIT` and `OFFSET` as well
- Use SQL query builders (squirrel, goqu)
- Perform input validation at multiple layers

### Q4: How do you address the N+1 problem in GORM?

GORM's `Preload` can solve it with eager loading. However, since `Preload` internally issues IN clause queries, `Joins` may be more efficient when dealing with large volumes of data.

```go
// Preload: 2 queries (users + orders via IN clause)
db.Preload("Orders").Find(&users)

// Joins: 1 query (LEFT JOIN)
db.Joins("Profile").Find(&users)

// Conditional Preload
db.Preload("Orders", "status = ?", "completed").Find(&users)

// Nested Preload
db.Preload("Orders.OrderItems.Product").Find(&users)
```

### Q5: What are the best practices for migrations?

1. **Version control**: Manage migration files in Git
2. **Rollback support**: Always create down files
3. **Idempotency**: Use `IF NOT EXISTS`, `IF EXISTS`
4. **Small changes**: One change per file for easier management
5. **Separate data migrations**: Keep schema changes and data changes in separate files
6. **Review**: Migration files should also be subject to code review
7. **Testing**: Test migration up/down in CI/CD

### Q6: How do you avoid deadlocks?

1. **Consistent lock ordering**: Always lock tables/rows in the same order
2. **Minimize transaction duration**: Move unnecessary processing outside the transaction
3. **Appropriate isolation level**: Don't use a higher isolation level than necessary
4. **FOR UPDATE**: Lock only the required rows
5. **Retry**: Implement retry logic when deadlocks are detected

```go
// Retry pattern for deadlocks
func WithRetry(ctx context.Context, maxRetries int, fn func() error) error {
    for i := 0; i < maxRetries; i++ {
        err := fn()
        if err == nil {
            return nil
        }
        // PostgreSQL deadlock error code: 40P01
        if isDeadlockError(err) && i < maxRetries-1 {
            time.Sleep(time.Duration(i+1) * 100 * time.Millisecond) // Backoff
            continue
        }
        return err
    }
    return fmt.Errorf("max retries exceeded")
}
```

---


## FAQ

### Q1: What is the most important point to keep in mind when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory but by actually writing and running code.

### Q2: What are common mistakes beginners make?

Skipping the fundamentals and jumping ahead to advanced topics. We recommend thoroughly understanding the basic concepts explained in this guide before moving on to the next step.

### Q3: How is this knowledge applied in real-world work?

The knowledge from this topic is frequently used in day-to-day development. It becomes especially important during code reviews and architecture design.

---

## Summary

| Concept | Key Points |
|---------|------------|
| database/sql | Standard library. Swappable drivers. Most lightweight |
| Connection Pool | Must configure MaxOpenConns/MaxIdleConns/ConnMaxLifetime in production |
| sqlx | Automatic struct mapping. Raw SQL. First choice for new projects |
| GORM | Full ORM. Rapid CRUD development. Relations via Preload/Joins |
| Transactions | BeginTx + defer Rollback + Commit. Manage with helper functions |
| Migrations | golang-migrate / goose / atlas. AutoMigrate is prohibited in production |
| Context | Pass context to all DB operations. Propagates cancellation and timeouts |
| N+1 Problem | Solve with JOIN / IN clause / Preload. Detect with monitoring |
| SQL Injection | Placeholders are mandatory. String concatenation is prohibited |
| Testing | testcontainers-go / interface mocking |

---

## Recommended Next Guides

- [03-grpc.md](./03-grpc.md) -- gRPC
- [04-testing.md](./04-testing.md) -- DB Testing
- [../03-tools/03-deployment.md](../03-tools/03-deployment.md) -- Deployment

---

## References

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
