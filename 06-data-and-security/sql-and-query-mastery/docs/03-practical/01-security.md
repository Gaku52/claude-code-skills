# DBセキュリティ — 認証・認可・暗号化・SQLインジェクション・監査

> データベースセキュリティは多層防御（Defense in Depth）の原則に基づき、ネットワーク、認証、認可、暗号化、入力検証、監査の各層で脅威を防ぐ包括的な取り組みである。SQLインジェクションは20年以上にわたりOWASP Top 10に入り続ける最も深刻な攻撃手法であり、パラメータ化クエリによる完全な防御が必須だ。本章ではPostgreSQLを中心に、データベースセキュリティの設計・実装・運用を内部実装レベルまで含めて徹底的に解説する。

---

## この章で学ぶこと

1. **多層防御の設計思想** — ネットワーク層からアプリケーション層まで、なぜ多層で守る必要があるのかを理解する
2. **PostgreSQLの認証・認可モデル** — pg_hba.conf、ロール、権限、Row Level Security（RLS）の内部動作と実装を習得する
3. **SQLインジェクションの完全防御** — 攻撃原理の理解から、パラメータ化クエリ、動的SQLのエスケープ、ORM利用時の注意点まで網羅する
4. **データの暗号化** — 転送中（TLS）、保存時（列レベル暗号化、ディスク暗号化）の実装方法を理解する
5. **監査ログの実装と運用** — トリガーベースの監査、pgAudit、変更履歴の追跡を実装できるようになる

---

## 前提知識

本章を理解するには以下の知識が必要です。

- [SQLの基礎](../00-basics/00-sql-overview.md) — SELECT/INSERT/UPDATE/DELETEの基本操作
- [トランザクション](../01-advanced/02-transactions.md) — ACID特性とトランザクション分離レベル
- [PostgreSQL機能](./00-postgresql-features.md) — JSONB、トリガー、PL/pgSQLの基本
- [セキュリティ基礎](../../security-fundamentals/docs/00-basics/00-security-overview.md) — 情報セキュリティの基本概念
- [暗号化基礎](../../security-fundamentals/docs/02-cryptography/00-crypto-basics.md) — ハッシュ、対称鍵暗号、公開鍵暗号の基礎

---

## 1. 多層防御の設計思想

### 1.1 なぜ多層防御が必要なのか

データベースセキュリティで最も重要な原則は「単一の防御層に依存しない」ことだ。ファイアウォールが突破されても認証がある、認証が突破されても認可で制限される、認可をすり抜けても暗号化でデータが保護される。どの層が破られても次の層で攻撃を阻止する。

```
┌──────────── DBセキュリティの多層防御 ──────────────────┐
│                                                         │
│  Layer 1: ネットワーク                                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ ファイアウォール / VPC / セキュリティグループ     │   │
│  │ pg_hba.conf（接続元IP制限）                      │   │
│  │ ポート変更（5432以外）/ VPN / SSH トンネル       │   │
│  └─────────────────────────────────────────────────┘   │
│  WHY: 不正な接続元を物理的にブロック                     │
│                                                         │
│  Layer 2: 認証 (Authentication) — 「誰であるか」         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ パスワード（SCRAM-SHA-256推奨、MD5非推奨）       │   │
│  │ クライアント証明書（mTLS）                        │   │
│  │ LDAP / Active Directory / Kerberos統合           │   │
│  │ peer認証（ローカルUnixソケット）                  │   │
│  └─────────────────────────────────────────────────┘   │
│  WHY: 正当なユーザーのみが接続できることを保証           │
│                                                         │
│  Layer 3: 認可 (Authorization) — 「何ができるか」        │
│  ┌─────────────────────────────────────────────────┐   │
│  │ GRANT/REVOKE（テーブル・カラム・スキーマ単位）    │   │
│  │ Row Level Security（行単位のアクセス制御）        │   │
│  │ 最小権限の原則（Principle of Least Privilege）    │   │
│  └─────────────────────────────────────────────────┘   │
│  WHY: 認証済みユーザーでも必要最小限の操作のみ許可       │
│                                                         │
│  Layer 4: 入力検証                                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │ パラメータ化クエリ（SQLインジェクション防御）     │   │
│  │ 入力のバリデーション・サニタイズ                  │   │
│  │ ストアドプロシージャによるAPI層の抽象化           │   │
│  └─────────────────────────────────────────────────┘   │
│  WHY: 悪意のある入力によるクエリ改ざんを完全阻止         │
│                                                         │
│  Layer 5: 暗号化                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ TLS通信（転送中暗号化）                          │   │
│  │ 列レベル暗号化（pgcrypto: AES-256, bcrypt）      │   │
│  │ ディスク暗号化（LUKS, AWS EBS Encryption）       │   │
│  │ バックアップ暗号化                                │   │
│  └─────────────────────────────────────────────────┘   │
│  WHY: データ漏洩時でも内容の読み取りを防止               │
│                                                         │
│  Layer 6: 監査                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │ クエリログ / pgAudit                              │   │
│  │ 変更履歴テーブル（audit_log）                     │   │
│  │ 異常検知 / アラート                               │   │
│  └─────────────────────────────────────────────────┘   │
│  WHY: 侵害発生時の原因究明と証跡保全                     │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 認証と認可

### コード例1: pg_hba.confとロール・権限の設計

```sql
-- ===== pg_hba.conf の設定例 =====
-- pg_hba.confはPostgreSQLの認証制御ファイル
-- 接続元、データベース、ユーザーごとに認証方式を指定する
-- 上から順に評価され、最初に一致するルールが適用される

-- TYPE    DATABASE    USER           ADDRESS         METHOD
-- -------------------------------------------------------
-- # ローカル接続: OS認証（peer）
-- local   all         postgres                       peer
-- local   all         all                            peer
--
-- # 同一ホスト: パスワード認証（SCRAM-SHA-256推奨）
-- host    all         all            127.0.0.1/32    scram-sha-256
-- host    all         all            ::1/128         scram-sha-256
--
-- # アプリケーションサーバー: 特定IPからのみ許可
-- host    myapp_db    app_user       10.0.1.0/24     scram-sha-256
-- host    myapp_db    app_readonly   10.0.1.0/24     scram-sha-256
--
-- # 管理者: VPN経由のみ許可（証明書認証）
-- hostssl all         admin_user     10.10.0.0/16    cert
--
-- # その他: 全て拒否（最重要！）
-- host    all         all            0.0.0.0/0       reject
-- host    all         all            ::/0            reject

-- WHY SCRAM-SHA-256？
-- MD5は以下の脆弱性がある:
-- 1. パスワード + ユーザー名をMD5ハッシュするだけ（ソルトが固定）
-- 2. レインボーテーブル攻撃に弱い
-- 3. ハッシュ値が漏洩するとリプレイ攻撃可能
-- SCRAM-SHA-256はチャレンジレスポンス方式で、これらの問題を解決

-- pg_hba.confの変更後はリロードが必要
SELECT pg_reload_conf();

-- ===== ロール設計（最小権限の原則）=====

-- 1. 読み取り専用ロール（分析・レポート用）
CREATE ROLE app_readonly
    LOGIN
    PASSWORD 'readonly_secure_password_2024!'
    CONNECTION LIMIT 10   -- 同時接続数を制限
    VALID UNTIL '2027-12-31';  -- パスワード有効期限

-- 2. アプリケーション用ロール（CRUD操作）
CREATE ROLE app_readwrite
    LOGIN
    PASSWORD 'readwrite_secure_password_2024!'
    CONNECTION LIMIT 30;

-- 3. マイグレーション用ロール（DDL操作）
CREATE ROLE app_migration
    LOGIN
    PASSWORD 'migration_secure_password_2024!'
    CREATEDB
    CONNECTION LIMIT 3;

-- 4. 管理者ロール（最小限の使用に限定）
CREATE ROLE app_admin
    LOGIN
    PASSWORD 'admin_secure_password_2024!'
    CREATEDB CREATEROLE
    CONNECTION LIMIT 2;

-- ===== グループロールによる権限継承 =====

-- 開発チームグループ
CREATE ROLE developers NOLOGIN;  -- ログイン不可のグループロール
GRANT app_readwrite TO developers;

-- 個別の開発者ユーザー
CREATE ROLE dev_tanaka LOGIN PASSWORD '...';
CREATE ROLE dev_suzuki LOGIN PASSWORD '...';
GRANT developers TO dev_tanaka;
GRANT developers TO dev_suzuki;

-- ===== スキーマレベルの権限 =====

-- まずスキーマのUSAGE権限（これがないとスキーマ内を何も参照できない）
GRANT USAGE ON SCHEMA public TO app_readonly;
GRANT USAGE, CREATE ON SCHEMA public TO app_readwrite;

-- ===== テーブルレベルの権限 =====

GRANT SELECT ON ALL TABLES IN SCHEMA public TO app_readonly;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public
    TO app_readwrite;

-- 将来作成されるテーブルにもデフォルト権限を設定（重要！）
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO app_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO app_readwrite;

-- シーケンスの権限（SERIALカラムのINSERTに必要）
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO app_readwrite;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE ON SEQUENCES TO app_readwrite;

-- ===== 列レベルの権限 =====

-- 特定のカラムのみ参照を許可（給与情報を隠す例）
REVOKE SELECT ON users FROM app_readonly;
GRANT SELECT (id, name, email, department, created_at) ON users
    TO app_readonly;
-- → app_readonlyはsalary, password_hashカラムを参照できない

-- ===== 権限の確認 =====

-- テーブル権限の確認
SELECT grantee, table_name, privilege_type
FROM information_schema.table_privileges
WHERE table_schema = 'public'
ORDER BY grantee, table_name;

-- ロールのメンバーシップ確認
SELECT r.rolname AS role, m.rolname AS member
FROM pg_auth_members am
JOIN pg_roles r ON am.roleid = r.oid
JOIN pg_roles m ON am.member = m.oid;
```

### コード例2: Row Level Security（RLS）の実装

```sql
-- ===== RLSの仕組み =====
-- RLSはテーブルの各行に対してアクセスポリシーを設定する機能
-- SQLのWHERE句に自動的に条件が追加される（透過的に動作）

-- テーブル作成
CREATE TABLE documents (
    id          SERIAL PRIMARY KEY,
    title       VARCHAR(200) NOT NULL,
    body        TEXT,
    owner_id    INTEGER NOT NULL,
    department  VARCHAR(50) NOT NULL,
    visibility  VARCHAR(20) NOT NULL DEFAULT 'private'
                CHECK (visibility IN ('private', 'department', 'public')),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- RLSの有効化
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- WHY: RLSはデフォルトで無効。テーブル所有者は常にRLSをバイパスする
-- FORCE ROW LEVEL SECURITY で所有者にも適用可能
ALTER TABLE documents FORCE ROW LEVEL SECURITY;

-- ===== ポリシー設計 =====

-- ポリシー1: ユーザーは自分が所有するドキュメントに対して全操作可能
CREATE POLICY doc_owner_all ON documents
    FOR ALL
    USING (owner_id = current_setting('app.current_user_id')::INTEGER)
    WITH CHECK (owner_id = current_setting('app.current_user_id')::INTEGER);

-- ポリシー2: 同じ部門のドキュメント（visibility='department'）は参照可能
CREATE POLICY doc_department_read ON documents
    FOR SELECT
    USING (
        visibility = 'department'
        AND department = current_setting('app.current_department')
    );

-- ポリシー3: 公開ドキュメントは全員が参照可能
CREATE POLICY doc_public_read ON documents
    FOR SELECT
    USING (visibility = 'public');

-- ポリシー4: 管理者は全ドキュメントに対して全操作可能
CREATE POLICY doc_admin_all ON documents
    FOR ALL
    TO app_admin
    USING (TRUE)
    WITH CHECK (TRUE);

-- ===== アプリケーションからの使用 =====

-- アプリケーションはリクエストごとにセッション変数を設定
-- （通常はミドルウェアやコネクション初期化で行う）
SET app.current_user_id = '42';
SET app.current_department = 'engineering';

-- このSELECTは自動的に以下のいずれかに該当する行のみ返す:
-- 1. owner_id = 42 のドキュメント
-- 2. department = 'engineering' かつ visibility = 'department'
-- 3. visibility = 'public'
SELECT * FROM documents;

-- ===== マルチテナントRLS =====

CREATE TABLE tenant_orders (
    id          SERIAL PRIMARY KEY,
    tenant_id   INTEGER NOT NULL,
    customer_id INTEGER NOT NULL,
    total       DECIMAL(12, 2) NOT NULL,
    status      VARCHAR(20) NOT NULL DEFAULT 'pending',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE tenant_orders ENABLE ROW LEVEL SECURITY;

-- テナント分離ポリシー
CREATE POLICY tenant_isolation ON tenant_orders
    FOR ALL
    USING (tenant_id = current_setting('app.tenant_id')::INTEGER)
    WITH CHECK (tenant_id = current_setting('app.tenant_id')::INTEGER);

-- インデックス（RLSのパフォーマンスに直結）
CREATE INDEX idx_tenant_orders_tenant ON tenant_orders (tenant_id);

-- WHY tenant_idにインデックス？
-- RLSは各クエリにWHERE tenant_id = ... を自動追加する
-- インデックスがないと全行スキャンが必要になり、パフォーマンスが劣化する

-- ===== RLSの動作確認 =====

-- ポリシーの一覧確認
SELECT schemaname, tablename, policyname, permissive, roles, cmd, qual, with_check
FROM pg_policies
WHERE tablename = 'documents';

-- EXPLAIN ANALYZEでポリシーが適用されていることを確認
SET app.current_user_id = '42';
EXPLAIN ANALYZE SELECT * FROM documents;
-- → Filter: (owner_id = 42) が追加されているはず
```

---

## 3. SQLインジェクション

### 3.1 攻撃原理の深い理解

```
┌───────── SQLインジェクションの原理 ──────────────────────┐
│                                                           │
│  ■ 根本原因: SQL文字列の中にユーザー入力を直接結合する    │
│    → 「データ」が「コード」として解釈される                │
│                                                           │
│  ■ 正常なクエリ:                                          │
│  query = "SELECT * FROM users WHERE email = '" + input    │
│  input = "tanaka@example.com"                             │
│  → SELECT * FROM users WHERE email = 'tanaka@example.com' │
│                                                           │
│  ■ 攻撃1: 認証バイパス                                    │
│  input = "' OR '1'='1"                                    │
│  → SELECT * FROM users WHERE email = '' OR '1'='1'        │
│  → 全行が返される（常にTRUE）                             │
│                                                           │
│  ■ 攻撃2: データ破壊                                      │
│  input = "'; DROP TABLE users; --"                        │
│  → SELECT * FROM users WHERE email = '';                  │
│    DROP TABLE users; --'                                   │
│  → テーブルが削除される！                                  │
│                                                           │
│  ■ 攻撃3: データ窃取（UNION-based）                       │
│  input = "' UNION SELECT username, password FROM          │
│           admin_users --"                                  │
│  → 管理者のパスワードが取得される                          │
│                                                           │
│  ■ 攻撃4: ブラインドSQLi（時間ベース）                    │
│  input = "' AND (SELECT pg_sleep(5)) IS NOT NULL --"      │
│  → レスポンスが5秒遅延 → 脆弱性の確認                     │
│                                                           │
│  ■ 攻撃5: 二次SQLインジェクション                         │
│  → ユーザー名に "admin'--" を登録                         │
│  → パスワードリセット処理で:                               │
│    UPDATE users SET pass='...' WHERE name='admin'--'      │
│  → admin のパスワードが変更される                          │
│                                                           │
│  ■ 根本的な対策:                                          │
│  パラメータ化クエリ（プリペアドステートメント）             │
│  → ユーザー入力はデータとしてのみ扱われ、                 │
│    SQL構文として解釈されない                                │
│  → SQL文の構造が固定され、入力値で変化しない              │
└───────────────────────────────────────────────────────────┘
```

### コード例3: SQLインジェクションの完全防御

```sql
-- ===== 各言語でのパラメータ化クエリ =====

-- ■ PostgreSQL プリペアドステートメント（SQL直接）
PREPARE find_user_by_email (TEXT) AS
    SELECT id, name, email FROM users WHERE email = $1;

EXECUTE find_user_by_email('user@example.com');
DEALLOCATE find_user_by_email;

-- WHY プリペアドステートメントで防御できるのか？
-- 1. SQL文の構造（パース木）がPREPARE時に確定する
-- 2. EXECUTE時のパラメータは「値」としてのみバインドされる
-- 3. パラメータ内の ' や ; はSQL構文として解釈されない
-- 4. つまり「データ」と「コード」が完全に分離される
```

```python
# ■ Python (psycopg2) — 正しいパラメータバインディング

import psycopg2

conn = psycopg2.connect("dbname=myapp user=app_user")
cur = conn.cursor()

# NG: 文字列フォーマット（SQLインジェクション脆弱！）
# user_input = "'; DROP TABLE users; --"
# cur.execute(f"SELECT * FROM users WHERE email = '{user_input}'")
# → SELECT * FROM users WHERE email = ''; DROP TABLE users; --'

# OK: パラメータバインディング（%sプレースホルダ）
user_input = "user@example.com"
cur.execute("SELECT * FROM users WHERE email = %s", (user_input,))
# → psycopg2がエスケープ処理を行い、安全にバインド

# OK: 名前付きパラメータ
cur.execute(
    "SELECT * FROM users WHERE email = %(email)s AND status = %(status)s",
    {"email": user_input, "status": "active"}
)

# NG: executeの引数を使わずに自分で組み立てるのもNG
# cur.execute("SELECT * FROM users WHERE email = '%s'" % user_input)

# IN句の安全な処理（psycopg2のタプル対応）
user_ids = [1, 2, 3, 4, 5]
cur.execute("SELECT * FROM users WHERE id = ANY(%s)", (user_ids,))

conn.close()
```

```typescript
// ■ Node.js (pg) — パラメータ化クエリ

import { Pool } from 'pg';

const pool = new Pool({
  connectionString: 'postgres://app_user:pass@localhost:5432/myapp',
});

// NG: テンプレートリテラル（SQLインジェクション脆弱！）
// const result = await pool.query(
//   `SELECT * FROM users WHERE email = '${userInput}'`
// );

// OK: パラメータ化クエリ（$1, $2, ... プレースホルダ）
const userInput = 'user@example.com';
const result = await pool.query(
  'SELECT * FROM users WHERE email = $1 AND status = $2',
  [userInput, 'active']
);

// OK: IN句の安全な処理
const userIds = [1, 2, 3, 4, 5];
const inResult = await pool.query(
  'SELECT * FROM users WHERE id = ANY($1::int[])',
  [userIds]
);

// NG: 動的なテーブル名・カラム名はパラメータ化できない
// → ホワイトリストで検証する
const allowedColumns = ['name', 'email', 'department'];
const sortColumn = allowedColumns.includes(userColumn) ? userColumn : 'name';
const sortResult = await pool.query(
  `SELECT * FROM users ORDER BY ${sortColumn} LIMIT $1`,
  [10]
);
```

```go
// ■ Go (database/sql) — パラメータ化クエリ

package main

import (
    "database/sql"
    "fmt"
    _ "github.com/lib/pq"
)

func findUserByEmail(db *sql.DB, email string) (*User, error) {
    // OK: $1 プレースホルダでパラメータバインド
    var user User
    err := db.QueryRow(
        "SELECT id, name, email FROM users WHERE email = $1",
        email,
    ).Scan(&user.ID, &user.Name, &user.Email)
    if err != nil {
        return nil, fmt.Errorf("findUserByEmail: %w", err)
    }
    return &user, nil
}

// NG: fmt.Sprintfでクエリを組み立てるのはNG
// query := fmt.Sprintf("SELECT * FROM users WHERE email = '%s'", email)
// db.Query(query)
```

```sql
-- ===== 動的SQLが必要な場合のエスケープ（PL/pgSQL）=====

-- テーブル名やカラム名は$1パラメータにできないため、
-- 動的SQLが必要になる場合がある。その場合は以下の関数を使う。

-- quote_ident: 識別子（テーブル名、カラム名）のエスケープ
-- quote_literal: リテラル値のエスケープ
-- format('%I', ...): 識別子のエスケープ
-- format('%L', ...): リテラル値のエスケープ

-- 方法1: quote_ident + quote_literal
CREATE OR REPLACE FUNCTION search_by_column(
    p_table TEXT,
    p_column TEXT,
    p_value TEXT
) RETURNS SETOF RECORD AS $$
BEGIN
    RETURN QUERY EXECUTE
        'SELECT * FROM ' || quote_ident(p_table)
        || ' WHERE ' || quote_ident(p_column)
        || ' = ' || quote_literal(p_value);
END;
$$ LANGUAGE plpgsql;

-- 方法2: format()関数（推奨、より読みやすい）
CREATE OR REPLACE FUNCTION search_by_column_v2(
    p_table TEXT,
    p_column TEXT,
    p_value TEXT
) RETURNS SETOF RECORD AS $$
BEGIN
    -- %I = 識別子（自動的にダブルクォートでエスケープ）
    -- %L = リテラル値（自動的にシングルクォートでエスケープ）
    -- %s = 文字列そのまま（エスケープなし、使用注意）
    RETURN QUERY EXECUTE format(
        'SELECT * FROM %I WHERE %I = %L',
        p_table, p_column, p_value
    );
END;
$$ LANGUAGE plpgsql;

-- 方法3: USING句でパラメータをバインド（最も安全）
CREATE OR REPLACE FUNCTION search_by_column_v3(
    p_table TEXT,
    p_column TEXT,
    p_value TEXT
) RETURNS SETOF RECORD AS $$
BEGIN
    RETURN QUERY EXECUTE format(
        'SELECT * FROM %I WHERE %I = $1',
        p_table, p_column
    ) USING p_value;  -- $1にp_valueをバインド
END;
$$ LANGUAGE plpgsql;

-- ===== テーブル名・カラム名のホワイトリスト検証 =====
CREATE OR REPLACE FUNCTION safe_search(
    p_table TEXT,
    p_column TEXT,
    p_value TEXT
) RETURNS SETOF RECORD AS $$
DECLARE
    allowed_tables TEXT[] := ARRAY['users', 'products', 'orders'];
    allowed_columns TEXT[] := ARRAY['name', 'email', 'status'];
BEGIN
    -- ホワイトリスト検証
    IF NOT (p_table = ANY(allowed_tables)) THEN
        RAISE EXCEPTION 'Invalid table name: %', p_table;
    END IF;
    IF NOT (p_column = ANY(allowed_columns)) THEN
        RAISE EXCEPTION 'Invalid column name: %', p_column;
    END IF;

    RETURN QUERY EXECUTE format(
        'SELECT * FROM %I WHERE %I = $1', p_table, p_column
    ) USING p_value;
END;
$$ LANGUAGE plpgsql;
```

---

## 4. 暗号化

### 4.1 暗号化の3つのレイヤー

```
┌──────── 暗号化の3つのレイヤー ────────────────────────┐
│                                                        │
│  1. 転送中の暗号化 (Encryption in Transit)              │
│  ┌──────────────────────────────────────────────────┐ │
│  │ クライアント ←── TLS 1.3 ──→ PostgreSQL          │ │
│  │ • サーバー証明書 + オプションでクライアント証明書 │ │
│  │ • pg_hba.conf で hostssl のみ許可                │ │
│  │ • sslmode=verify-full で中間者攻撃を防止         │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│  2. 保存時の暗号化 — 列レベル (Column-level)            │
│  ┌──────────────────────────────────────────────────┐ │
│  │ pgcrypto拡張: pgp_sym_encrypt / pgp_sym_decrypt  │ │
│  │ • 機密カラム（SSN、カード番号等）を個別に暗号化   │ │
│  │ • アプリケーション層で暗号化する方がセキュア       │ │
│  │   （DBに鍵を渡さないため）                        │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│  3. 保存時の暗号化 — ディスクレベル (Disk-level)         │
│  ┌──────────────────────────────────────────────────┐ │
│  │ • AWS: EBS Encryption / RDS Encryption            │ │
│  │ • Linux: LUKS (dm-crypt)                          │ │
│  │ • PostgreSQL: pgcrypto + pg_tde（透過的暗号化）   │ │
│  │ • ディスクを盗まれてもデータを読めない             │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│  WHY 3レイヤー全て必要か？                              │
│  • TLSなし → 通信の盗聴でパスワード・データが漏洩      │
│  • 列暗号化なし → DBバックアップ漏洩で機密データ流出   │
│  • ディスク暗号化なし → 物理ディスク盗難でデータ流出   │
└────────────────────────────────────────────────────────┘
```

### コード例4: 暗号化の実装

```sql
-- ===== pgcrypto拡張の有効化 =====
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ===== パスワードのハッシュ化 =====

-- WHY bcrypt？
-- 1. ソルト自動生成（レインボーテーブル攻撃を防止）
-- 2. コストパラメータで計算量を調整可能（ブルートフォース耐性）
-- 3. 意図的に遅い（GPU並列攻撃に対する耐性）
-- コスト12 → 1ハッシュあたり約250ms（現在の推奨値）

-- ユーザーテーブル
CREATE TABLE secure_users (
    id            SERIAL PRIMARY KEY,
    email         VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(60) NOT NULL,  -- bcryptハッシュは60文字固定
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- パスワード登録
INSERT INTO secure_users (email, password_hash) VALUES (
    'user@example.com',
    crypt('user_password_2024!', gen_salt('bf', 12))
    --                          ~~~~~~~~  ~~  ~~
    --                          アルゴリズム|  コスト
    --                          (bf=bcrypt) |
    --                                      12（2^12=4096回反復）
);

-- パスワード検証
SELECT id, email FROM secure_users
WHERE email = 'user@example.com'
  AND password_hash = crypt('user_password_2024!', password_hash);
  --                        ~~~~~~~~~~~~~~~~~~~   ~~~~~~~~~~~~~~
  --                        入力パスワード         保存済みハッシュからソルトを抽出
-- → 一致すれば行が返る、不一致ならNULL

-- ===== 列レベルの暗号化（AES-256） =====

-- 機密データテーブル
CREATE TABLE sensitive_data (
    id               SERIAL PRIMARY KEY,
    user_id          INTEGER NOT NULL REFERENCES secure_users(id),
    -- 暗号化されたデータ（BYTEA型で格納）
    ssn_encrypted    BYTEA,
    card_encrypted   BYTEA,
    -- メタデータ（暗号化不要）
    data_type        VARCHAR(20) NOT NULL,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- 暗号化して格納
-- 鍵は環境変数や鍵管理サービス（AWS KMS等）から取得すべき
INSERT INTO sensitive_data (user_id, ssn_encrypted, data_type) VALUES (
    1,
    pgp_sym_encrypt('123-45-6789', current_setting('app.encryption_key')),
    'ssn'
);

-- 復号して取得
SELECT
    user_id,
    pgp_sym_decrypt(ssn_encrypted, current_setting('app.encryption_key')) AS ssn
FROM sensitive_data
WHERE user_id = 1;

-- WHY アプリケーション層での暗号化がより安全か？
-- DB層暗号化: 暗号化キーがSQL文やセッション変数に含まれる
--   → クエリログに記録される可能性がある
--   → DBの特権ユーザーが復号可能
-- アプリ層暗号化: キーはアプリケーションメモリ内のみ
--   → DBには暗号化済みバイナリのみ格納される
--   → DBが漏洩してもキーなしでは復号できない
```

### コード例5: TLS設定と接続セキュリティ

```sql
-- ===== postgresql.conf でのTLS設定 =====
-- ssl = on
-- ssl_cert_file = '/etc/ssl/certs/server.crt'
-- ssl_key_file = '/etc/ssl/private/server.key'
-- ssl_ca_file = '/etc/ssl/certs/ca.crt'           -- クライアント証明書検証用
-- ssl_min_protocol_version = 'TLSv1.3'            -- TLS 1.3以上を強制
-- ssl_ciphers = 'HIGH:!aNULL:!MD5:!3DES:!RC4'     -- 強い暗号スイートのみ

-- pg_hba.conf でSSL接続を強制
-- hostssl  all  all  0.0.0.0/0  scram-sha-256
-- hostnossl all  all  0.0.0.0/0  reject   -- 非SSL接続を全て拒否

-- ===== TLS接続の確認 =====
SELECT
    datname AS database,
    usename AS user,
    client_addr,
    ssl,
    ssl_version,
    ssl_cipher
FROM pg_stat_ssl
    JOIN pg_stat_activity USING (pid)
WHERE pid != pg_backend_pid();

-- 非SSL接続がないかチェック
SELECT COUNT(*) AS non_ssl_connections
FROM pg_stat_ssl
    JOIN pg_stat_activity USING (pid)
WHERE NOT ssl AND usename != 'postgres';

-- ===== アプリケーション接続文字列での指定 =====
-- Python:
-- postgresql://user:pass@host:5432/db?sslmode=verify-full&sslrootcert=/path/ca.crt
--
-- sslmodeの選択:
-- disable      → SSLなし（NG）
-- allow        → SSLを試みるがなくてもOK（NG）
-- prefer       → SSLを優先するがなくてもOK（不十分）
-- require      → SSLを強制するが証明書は検証しない（中間者攻撃に弱い）
-- verify-ca    → SSL + CA証明書を検証（推奨最低ライン）
-- verify-full  → SSL + CA + ホスト名を検証（推奨）
```

---

## 5. 監査ログ

### コード例6: トリガーベースの監査ログ

```sql
-- ===== 監査ログテーブル =====
CREATE TABLE audit_log (
    id          BIGSERIAL PRIMARY KEY,
    table_name  VARCHAR(100) NOT NULL,
    record_id   TEXT,                    -- 変更されたレコードのID
    operation   VARCHAR(10) NOT NULL     -- INSERT, UPDATE, DELETE
                CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
    old_data    JSONB,                   -- 変更前のデータ（UPDATE, DELETEのみ）
    new_data    JSONB,                   -- 変更後のデータ（INSERT, UPDATEのみ）
    changed_fields TEXT[],               -- UPDATEで変更されたフィールド名
    changed_by  VARCHAR(100) NOT NULL DEFAULT current_user,
    app_user_id INTEGER,                 -- アプリケーションレベルのユーザーID
    client_ip   INET DEFAULT inet_client_addr(),
    session_id  TEXT,
    changed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- パーティション（月次で分割、古い監査ログの管理が容易に）
-- CREATE TABLE audit_log (...) PARTITION BY RANGE (changed_at);
-- CREATE TABLE audit_log_2024_01 PARTITION OF audit_log
--     FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- インデックス
CREATE INDEX idx_audit_log_table_op ON audit_log (table_name, operation);
CREATE INDEX idx_audit_log_changed_at ON audit_log (changed_at);
CREATE INDEX idx_audit_log_record_id ON audit_log (table_name, record_id);
CREATE INDEX idx_audit_log_app_user ON audit_log (app_user_id);

-- ===== 汎用監査トリガー関数 =====
CREATE OR REPLACE FUNCTION audit_trigger_func()
RETURNS TRIGGER AS $$
DECLARE
    record_pk TEXT;
    changed TEXT[] := '{}';
    old_json JSONB;
    new_json JSONB;
    col TEXT;
BEGIN
    -- レコードIDの取得（主キー名が'id'であると仮定）
    IF TG_OP = 'DELETE' THEN
        record_pk := OLD.id::TEXT;
        old_json := to_jsonb(OLD);
    ELSIF TG_OP = 'INSERT' THEN
        record_pk := NEW.id::TEXT;
        new_json := to_jsonb(NEW);
    ELSIF TG_OP = 'UPDATE' THEN
        record_pk := NEW.id::TEXT;
        old_json := to_jsonb(OLD);
        new_json := to_jsonb(NEW);

        -- 変更されたフィールドを特定
        FOR col IN SELECT key FROM jsonb_each(new_json)
        LOOP
            IF old_json->col IS DISTINCT FROM new_json->col THEN
                changed := array_append(changed, col);
            END IF;
        END LOOP;

        -- 変更がない場合はスキップ（updated_atのみの更新など）
        IF array_length(changed, 1) IS NULL OR
           changed = ARRAY['updated_at'] THEN
            RETURN NEW;
        END IF;
    END IF;

    INSERT INTO audit_log (
        table_name, record_id, operation,
        old_data, new_data, changed_fields,
        app_user_id, session_id
    ) VALUES (
        TG_TABLE_NAME, record_pk, TG_OP,
        old_json, new_json, NULLIF(changed, '{}'),
        NULLIF(current_setting('app.current_user_id', true), '')::INTEGER,
        current_setting('app.session_id', true)
    );

    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    ELSE
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- ===== 監査対象テーブルにトリガーを設定 =====
CREATE TRIGGER audit_users
    AFTER INSERT OR UPDATE OR DELETE ON users
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_func();

CREATE TRIGGER audit_orders
    AFTER INSERT OR UPDATE OR DELETE ON orders
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_func();

-- ===== 監査ログの照会 =====

-- 特定ユーザーの全変更履歴
SELECT
    changed_at,
    table_name,
    operation,
    changed_fields,
    old_data,
    new_data
FROM audit_log
WHERE app_user_id = 42
ORDER BY changed_at DESC
LIMIT 20;

-- 特定レコードの変更履歴（タイムライン）
SELECT
    changed_at,
    operation,
    changed_fields,
    CASE operation
        WHEN 'INSERT' THEN new_data
        WHEN 'DELETE' THEN old_data
        WHEN 'UPDATE' THEN jsonb_build_object(
            'before', jsonb_object_agg(f.key, old_data->f.key),
            'after',  jsonb_object_agg(f.key, new_data->f.key)
        )
    END AS changes,
    changed_by
FROM audit_log
LEFT JOIN LATERAL unnest(changed_fields) AS f(key) ON operation = 'UPDATE'
WHERE table_name = 'users' AND record_id = '42'
GROUP BY changed_at, operation, changed_fields, old_data, new_data, changed_by
ORDER BY changed_at;

-- ===== pgAudit（より高度な監査）=====
-- pgAudit拡張はSQLレベルの詳細な監査ログを提供する
-- CREATE EXTENSION IF NOT EXISTS pgaudit;
--
-- -- postgresql.confでの設定
-- pgaudit.log = 'ddl, write, role'  -- DDL, 書き込み, ロール変更を記録
-- pgaudit.log_catalog = off         -- システムカタログへのアクセスは除外
-- pgaudit.role = 'auditor'          -- 監査対象ロール

-- pgAuditの出力例:
-- AUDIT: SESSION,1,1,DDL,CREATE TABLE,,,CREATE TABLE users (...),<not logged>
-- AUDIT: SESSION,2,1,WRITE,INSERT,TABLE,public.users,INSERT INTO users ...
```

### コード例7: セキュリティビューとモニタリング

```sql
-- ===== セキュリティダッシュボード用ビュー =====

-- 1. 現在のアクティブ接続一覧
CREATE VIEW v_active_connections AS
SELECT
    pid,
    usename,
    datname,
    client_addr,
    application_name,
    state,
    backend_start,
    NOW() - backend_start AS connection_duration,
    ssl,
    query_start,
    CASE WHEN state = 'active' THEN query ELSE NULL END AS current_query
FROM pg_stat_activity
WHERE pid != pg_backend_pid()
ORDER BY backend_start;

-- 2. 失敗したログイン試行の監視
-- postgresql.confで log_connections = on, log_disconnections = on を設定
-- ログファイルから FATAL: password authentication failed を監視

-- 3. 長時間実行クエリの検出
CREATE VIEW v_long_running_queries AS
SELECT
    pid,
    usename,
    datname,
    NOW() - query_start AS duration,
    state,
    LEFT(query, 200) AS query_preview
FROM pg_stat_activity
WHERE state = 'active'
  AND NOW() - query_start > interval '30 seconds'
  AND pid != pg_backend_pid()
ORDER BY duration DESC;

-- 4. ロック待ちの検出
CREATE VIEW v_lock_contention AS
SELECT
    blocked.pid AS blocked_pid,
    blocked.usename AS blocked_user,
    LEFT(blocked.query, 100) AS blocked_query,
    blocking.pid AS blocking_pid,
    blocking.usename AS blocking_user,
    LEFT(blocking.query, 100) AS blocking_query,
    NOW() - blocked.query_start AS wait_duration
FROM pg_stat_activity blocked
JOIN pg_locks bl ON bl.pid = blocked.pid AND NOT bl.granted
JOIN pg_locks gl ON gl.pid != blocked.pid
    AND gl.transactionid = bl.transactionid AND gl.granted
JOIN pg_stat_activity blocking ON blocking.pid = gl.pid;

-- 5. スーパーユーザー接続の検出
SELECT usename, client_addr, backend_start
FROM pg_stat_activity
WHERE usename = 'postgres'
  AND client_addr IS NOT NULL;  -- リモートからのpostgres接続は要警戒
```

---

## 認証方式比較表

| 方式 | セキュリティ | 設定容易性 | パスワード送信 | 用途 | 推奨 |
|------|:---:|:---:|:---:|------|:---:|
| trust | 最低（認証なし） | 最も簡単 | なし | 開発環境ローカルのみ | 開発のみ |
| password (平文) | 低 | 簡単 | 平文 | 使用禁止 | NG |
| md5 | 中（非推奨） | 簡単 | MD5ハッシュ | レガシー互換のみ | 非推奨 |
| scram-sha-256 | 高 | 簡単 | チャレンジ応答 | 一般的なパスワード認証 | 推奨 |
| peer / ident | 高 | 中 | なし（OS認証） | ローカルUnix接続 | ローカル推奨 |
| cert (mTLS) | 最高 | 複雑 | なし（証明書） | TLSクライアント証明書 | 最高推奨 |
| ldap | 高 | 複雑 | LDAP経由 | 企業AD/LDAP統合 | 企業環境推奨 |
| gss (Kerberos) | 最高 | 最も複雑 | なし（チケット） | エンタープライズSSO | 大規模推奨 |
| radius | 高 | 複雑 | RADIUS経由 | 二要素認証統合 | 用途次第 |

## 権限レベル比較表

| レベル | 対象 | 設定方法 | 粒度 | 使用例 |
|--------|------|---------|------|--------|
| クラスタ | PostgreSQL全体 | CREATE ROLE + 属性 | 最も粗い | SUPERUSER, CREATEDB, CREATEROLE |
| データベース | 特定DB | GRANT CONNECT | DB単位 | 特定DBへの接続許可 |
| スキーマ | 名前空間 | GRANT USAGE/CREATE | スキーマ単位 | スキーマ内オブジェクトの参照 |
| テーブル | 個別テーブル | GRANT SELECT/INSERT/... | テーブル単位 | CRUD操作の許可 |
| 列 | 個別カラム | GRANT SELECT(col) | カラム単位 | 特定列のみ参照許可 |
| 行 | 個別レコード | CREATE POLICY (RLS) | 行単位 | マルチテナント分離 |

## SQLインジェクション対策方法比較表

| 対策方法 | 防御効果 | 適用場面 | 注意点 |
|---------|:---:|---------|--------|
| パラメータ化クエリ | 完全 | 値のバインド | テーブル名・カラム名には使えない |
| quote_ident / format(%I) | 完全 | 動的テーブル名・カラム名 | ホワイトリスト検証と併用推奨 |
| quote_literal / format(%L) | 完全 | 動的リテラル値 | パラメータ化クエリが使えない場合のみ |
| ホワイトリスト検証 | 完全 | 動的識別子 | 許可リストのメンテナンスが必要 |
| エスケープ関数 | 高い | レガシーコード | 漏れのリスクがあるため非推奨 |
| ORMのAPIのみ使用 | 高い | 一般的なCRUD | Raw SQL使用時は要注意 |
| WAF | 補助的 | 外部防御層 | 単独では不十分、バイパス手法あり |

---

## アンチパターン

### アンチパターン1: アプリケーションにスーパーユーザーで接続

```sql
-- NG: 全アプリケーションがpostgresユーザー（スーパーユーザー）で接続
-- 接続文字列: postgres://postgres:password@db:5432/myapp

-- 問題点:
-- 1. 全テーブルの読み書き削除が可能
-- 2. DROP DATABASE も実行可能
-- 3. システムカタログの変更も可能
-- 4. RLSが無効化される（スーパーユーザーはバイパス）
-- 5. SQLインジェクション時の被害が最大化
-- 6. 監査ログで操作者の特定が困難

-- OK: 最小権限の原則に基づいたロール設計
CREATE ROLE web_app LOGIN PASSWORD 'secure_password_here'
    CONNECTION LIMIT 30     -- 接続数制限
    NOSUPERUSER             -- 明示的にスーパーユーザー権限を拒否
    NOCREATEDB              -- DB作成権限なし
    NOCREATEROLE;           -- ロール作成権限なし

-- 必要最小限の権限のみ付与
GRANT SELECT, INSERT, UPDATE ON customers, orders, order_items TO web_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO web_app;
-- DELETE権限は付与しない → 論理削除（deleted_atカラム）を使用
-- DDL権限（CREATE TABLE等）も付与しない → マイグレーション専用ロールを別に用意
```

### アンチパターン2: 機密データの平文保存

```sql
-- NG (レベル1): パスワードを平文で保存
CREATE TABLE bad_users_v1 (
    id       SERIAL PRIMARY KEY,
    email    VARCHAR(255),
    password VARCHAR(255)  -- 平文のまま格納！
);
-- → データベースにアクセスできる全員がパスワードを読める
-- → バックアップファイルからも読める

-- NG (レベル2): MD5でハッシュ（脆弱）
INSERT INTO users (email, password_hash) VALUES
('user@example.com', md5('password123'));
-- → md5('password123') = '482c811da5d5b4bc6d497ffa98491e38'
-- → レインボーテーブルで瞬時に解読可能
-- → ソルトなしなので同じパスワードは同じハッシュになる

-- NG (レベル3): SHA-256でハッシュ（ソルトなし）
INSERT INTO users (email, password_hash) VALUES
('user@example.com', encode(digest('password123', 'sha256'), 'hex'));
-- → ソルトなしなので同じパスワードは同じハッシュ
-- → GPUで高速に総当たり可能

-- OK: bcryptでハッシュ（ソルト自動生成、意図的に遅い）
INSERT INTO users (email, password_hash) VALUES
('user@example.com', crypt('password123', gen_salt('bf', 12)));
-- → ソルトが自動生成される
-- → コスト12で約250ms/ハッシュ → 総当たり困難
-- → 同じパスワードでも毎回異なるハッシュ値になる
```

### アンチパターン3: RLSポリシーのないマルチテナント

```sql
-- NG: アプリケーション層のみでテナント分離
-- WHERE tenant_id = ? をアプリの全クエリに手動で追加

-- 問題点:
-- 1. 一箇所でもWHERE句の追加を忘れるとデータ漏洩
-- 2. 直接DBアクセスするツール（psql等）からは制御できない
-- 3. ORMのリレーション読み込みでフィルタが漏れる可能性

-- OK: RLSでデータベースレベルのテナント分離を保証
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE orders FORCE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON orders
    FOR ALL
    USING (tenant_id = current_setting('app.tenant_id')::INTEGER)
    WITH CHECK (tenant_id = current_setting('app.tenant_id')::INTEGER);

-- → どんなクエリを実行しても、自テナントのデータしか見えない
-- → psqlから直接SELECTしても、RLSが適用される
-- → WITH CHECKにより、他テナントのデータを挿入することも防止
```

---

## 実践演習

### 演習1（基礎）: ロールと権限の設計

**課題**: 以下の要件を満たすロールと権限設計を実装してください。

- `api_service`: Webアプリ用。customers, orders, productsテーブルのCRUD（DELETEは除く）
- `analytics_user`: 分析用。全テーブルのSELECTのみ。salary列は除外
- `migration_runner`: マイグレーション用。DDL操作が可能
- 各ロールの接続数制限を設定

<details>
<summary>模範解答</summary>

```sql
-- ロール作成
CREATE ROLE api_service
    LOGIN PASSWORD 'api_service_secure_2024!'
    CONNECTION LIMIT 30
    NOSUPERUSER NOCREATEDB NOCREATEROLE;

CREATE ROLE analytics_user
    LOGIN PASSWORD 'analytics_secure_2024!'
    CONNECTION LIMIT 10
    NOSUPERUSER NOCREATEDB NOCREATEROLE;

CREATE ROLE migration_runner
    LOGIN PASSWORD 'migration_secure_2024!'
    CONNECTION LIMIT 3
    NOSUPERUSER CREATEDB NOCREATEROLE;

-- スキーマ権限
GRANT USAGE ON SCHEMA public TO api_service, analytics_user;
GRANT USAGE, CREATE ON SCHEMA public TO migration_runner;

-- api_service: SELECT, INSERT, UPDATE（DELETEなし）
GRANT SELECT, INSERT, UPDATE ON customers, orders, products TO api_service;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO api_service;

-- analytics_user: SELECT のみ（salary列を除外）
GRANT SELECT ON orders, products TO analytics_user;
-- customers テーブルはsalary列を除いて付与
GRANT SELECT (id, name, email, department, created_at) ON customers
    TO analytics_user;

-- migration_runner: DDL操作
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO migration_runner;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO migration_runner;

-- デフォルト権限（将来のテーブルにも適用）
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE ON TABLES TO api_service;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO analytics_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE ON SEQUENCES TO api_service;

-- 確認
SELECT grantee, table_name, privilege_type
FROM information_schema.table_privileges
WHERE table_schema = 'public'
    AND grantee IN ('api_service', 'analytics_user', 'migration_runner')
ORDER BY grantee, table_name, privilege_type;
```

</details>

### 演習2（応用）: マルチテナントRLSの実装

**課題**: SaaS型のプロジェクト管理ツールを想定し、以下の要件を満たすRLSを実装してください。

- テナント分離: 異なるテナントのデータは一切参照できない
- ロール制御: テナント内で admin/member/viewer の3つのロールを持つ
- admin: 全操作可能
- member: 自分が作成したタスクのCRUD + 他のタスクの参照
- viewer: 参照のみ

<details>
<summary>模範解答</summary>

```sql
-- テーブル定義
CREATE TABLE tenant_users (
    id          SERIAL PRIMARY KEY,
    tenant_id   INTEGER NOT NULL,
    name        VARCHAR(100) NOT NULL,
    role        VARCHAR(20) NOT NULL CHECK (role IN ('admin', 'member', 'viewer')),
    UNIQUE (tenant_id, id)
);

CREATE TABLE tasks (
    id          SERIAL PRIMARY KEY,
    tenant_id   INTEGER NOT NULL,
    title       VARCHAR(200) NOT NULL,
    description TEXT,
    status      VARCHAR(20) DEFAULT 'open',
    creator_id  INTEGER NOT NULL,
    assignee_id INTEGER,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- RLS有効化
ALTER TABLE tasks ENABLE ROW LEVEL SECURITY;
ALTER TABLE tasks FORCE ROW LEVEL SECURITY;

-- セッション変数を取得するヘルパー関数
CREATE OR REPLACE FUNCTION get_current_tenant_id() RETURNS INTEGER AS $$
BEGIN
    RETURN current_setting('app.tenant_id')::INTEGER;
EXCEPTION WHEN OTHERS THEN
    RETURN NULL;
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION get_current_user_id() RETURNS INTEGER AS $$
BEGIN
    RETURN current_setting('app.current_user_id')::INTEGER;
EXCEPTION WHEN OTHERS THEN
    RETURN NULL;
END;
$$ LANGUAGE plpgsql STABLE;

CREATE OR REPLACE FUNCTION get_current_user_role() RETURNS TEXT AS $$
BEGIN
    RETURN current_setting('app.user_role');
EXCEPTION WHEN OTHERS THEN
    RETURN NULL;
END;
$$ LANGUAGE plpgsql STABLE;

-- ポリシー1: テナント分離（全ロール共通）
CREATE POLICY tenant_isolation ON tasks
    FOR ALL
    USING (tenant_id = get_current_tenant_id());

-- ポリシー2: admin は全操作可能
CREATE POLICY admin_all ON tasks
    FOR ALL
    USING (
        tenant_id = get_current_tenant_id()
        AND get_current_user_role() = 'admin'
    )
    WITH CHECK (
        tenant_id = get_current_tenant_id()
        AND get_current_user_role() = 'admin'
    );

-- ポリシー3: member は参照は全て可能
CREATE POLICY member_select ON tasks
    FOR SELECT
    USING (
        tenant_id = get_current_tenant_id()
        AND get_current_user_role() = 'member'
    );

-- ポリシー4: member は自分が作成したタスクのみCUD
CREATE POLICY member_modify ON tasks
    FOR ALL
    USING (
        tenant_id = get_current_tenant_id()
        AND get_current_user_role() = 'member'
        AND creator_id = get_current_user_id()
    )
    WITH CHECK (
        tenant_id = get_current_tenant_id()
        AND get_current_user_role() = 'member'
        AND creator_id = get_current_user_id()
    );

-- ポリシー5: viewer は参照のみ
CREATE POLICY viewer_select ON tasks
    FOR SELECT
    USING (
        tenant_id = get_current_tenant_id()
        AND get_current_user_role() = 'viewer'
    );

-- インデックス
CREATE INDEX idx_tasks_tenant ON tasks (tenant_id);
CREATE INDEX idx_tasks_creator ON tasks (tenant_id, creator_id);

-- テスト
SET app.tenant_id = '1';
SET app.current_user_id = '10';
SET app.user_role = 'member';
SELECT * FROM tasks;  -- テナント1のタスクのみ表示
```

</details>

### 演習3（発展）: 包括的セキュリティ監査システム

**課題**: 以下の要件を満たす包括的なセキュリティ監査システムを設計・実装してください。

- 対象テーブルの全CRUD操作を監査ログに記録
- 変更されたフィールドのbefore/afterを記録
- アプリケーションユーザーID、IPアドレスを記録
- 不正な操作パターン（短時間の大量DELETE等）を検出するビュー
- 監査ログの月次パーティション

<details>
<summary>模範解答</summary>

```sql
-- 監査ログテーブル（パーティション対応）
CREATE TABLE security_audit_log (
    id              BIGSERIAL,
    table_name      VARCHAR(100) NOT NULL,
    record_id       TEXT,
    operation       VARCHAR(10) NOT NULL,
    old_data        JSONB,
    new_data        JSONB,
    changed_fields  TEXT[],
    db_user         VARCHAR(100) DEFAULT current_user,
    app_user_id     INTEGER,
    client_ip       INET DEFAULT inet_client_addr(),
    user_agent      TEXT,
    session_id      TEXT,
    risk_level      VARCHAR(10) DEFAULT 'normal'
                    CHECK (risk_level IN ('normal', 'elevated', 'critical')),
    occurred_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, occurred_at)
) PARTITION BY RANGE (occurred_at);

-- 月次パーティション作成
CREATE TABLE security_audit_log_2024_01 PARTITION OF security_audit_log
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE security_audit_log_2024_02 PARTITION OF security_audit_log
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
-- ... 以降も同様

-- インデックス
CREATE INDEX idx_sal_table_op ON security_audit_log (table_name, operation);
CREATE INDEX idx_sal_occurred ON security_audit_log (occurred_at);
CREATE INDEX idx_sal_app_user ON security_audit_log (app_user_id);
CREATE INDEX idx_sal_risk ON security_audit_log (risk_level) WHERE risk_level != 'normal';

-- 監査トリガー関数（リスクレベル自動判定付き）
CREATE OR REPLACE FUNCTION security_audit_func()
RETURNS TRIGGER AS $$
DECLARE
    v_record_pk TEXT;
    v_changed TEXT[] := '{}';
    v_old_json JSONB;
    v_new_json JSONB;
    v_risk VARCHAR(10) := 'normal';
    col TEXT;
BEGIN
    -- リスクレベル判定
    IF TG_OP = 'DELETE' THEN
        v_record_pk := OLD.id::TEXT;
        v_old_json := to_jsonb(OLD);
        v_risk := 'elevated';  -- DELETEは常にelevated
    ELSIF TG_OP = 'INSERT' THEN
        v_record_pk := NEW.id::TEXT;
        v_new_json := to_jsonb(NEW);
    ELSIF TG_OP = 'UPDATE' THEN
        v_record_pk := NEW.id::TEXT;
        v_old_json := to_jsonb(OLD);
        v_new_json := to_jsonb(NEW);

        FOR col IN SELECT key FROM jsonb_each(v_new_json)
        LOOP
            IF v_old_json->col IS DISTINCT FROM v_new_json->col THEN
                v_changed := array_append(v_changed, col);
            END IF;
        END LOOP;

        -- 機密フィールドの変更はelevated
        IF v_changed && ARRAY['password_hash', 'email', 'role', 'status'] THEN
            v_risk := 'elevated';
        END IF;
    END IF;

    INSERT INTO security_audit_log (
        table_name, record_id, operation,
        old_data, new_data, changed_fields,
        app_user_id, session_id, risk_level
    ) VALUES (
        TG_TABLE_NAME, v_record_pk, TG_OP,
        v_old_json, v_new_json, NULLIF(v_changed, '{}'),
        NULLIF(current_setting('app.current_user_id', true), '')::INTEGER,
        current_setting('app.session_id', true),
        v_risk
    );

    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- 不正操作パターン検出ビュー
CREATE VIEW v_suspicious_activities AS
SELECT
    app_user_id,
    table_name,
    operation,
    COUNT(*) AS operation_count,
    MIN(occurred_at) AS first_at,
    MAX(occurred_at) AS last_at,
    'HIGH_FREQUENCY_DELETE' AS alert_type
FROM security_audit_log
WHERE operation = 'DELETE'
  AND occurred_at >= NOW() - INTERVAL '5 minutes'
GROUP BY app_user_id, table_name, operation
HAVING COUNT(*) > 10

UNION ALL

SELECT
    app_user_id,
    table_name,
    operation,
    COUNT(*),
    MIN(occurred_at),
    MAX(occurred_at),
    'ELEVATED_RISK_BURST'
FROM security_audit_log
WHERE risk_level IN ('elevated', 'critical')
  AND occurred_at >= NOW() - INTERVAL '1 hour'
GROUP BY app_user_id, table_name, operation
HAVING COUNT(*) > 5;
```

</details>

---

## FAQ

### Q1: RLS（Row Level Security）はパフォーマンスに影響するか？

RLSポリシーは各クエリにWHERE条件として追加されるため、適切なインデックスがあればオーバーヘッドは最小限（通常1-5%以下）。ただし以下の場合は注意が必要:

- 複雑なポリシー（サブクエリを含む）: JOINやサブクエリを含むポリシーは実行計画を複雑にする
- 多数のポリシー: 同じテーブルに10以上のポリシーがあるとOR条件が増えて遅くなる
- current_setting()の頻繁な呼び出し: ポリシー内でcurrent_setting()を使う場合は STABLE 関数でラップするとプランナが最適化しやすくなる

EXPLAIN ANALYZEで実行計画を確認し、ポリシーのフィルタ条件にインデックスが使われていることを確認すべき。

### Q2: データベースのバックアップも暗号化すべきか？

必須。バックアップファイルには全データが含まれるため、本番環境と同等以上のセキュリティが必要。

- pg_dump出力: GPGで暗号化するか、暗号化ストレージに保存
- WALアーカイブ: aws s3 cpやgsutil cpで送信する場合はServer-Side Encryption（SSE）を有効化
- クラウドバックアップ: AWS RDSは自動的にAES-256で暗号化（デフォルト有効）
- バックアップ転送: 必ずTLS（scp/sftp）を使用。FTPやrsync without SSHは厳禁

### Q3: SQLインジェクション以外のインジェクション攻撃は？

- **OSコマンドインジェクション**: PostgreSQLの `COPY FROM PROGRAM` コマンドの悪用。スーパーユーザー権限で任意のOSコマンドを実行可能
- **LDAPインジェクション**: LDAP認証設定時のフィルタ式への攻撃
- **NoSQLインジェクション**: MongoDBの `$where` 句やJavaScriptインジェクション
- **ORM経由のインジェクション**: Raw SQL機能や動的フィルタの不適切な使用でSQLインジェクションが発生する場合がある
- **二次インジェクション**: 一度DBに保存されたデータが、別のクエリで使用される際にインジェクションが発生

ORMを使っていてもRaw SQLメソッド（Prismaの`$queryRaw`、SQLAlchemyの`text()`等）を使う場合は、必ずパラメータバインドを行うこと。詳細は[ORM比較](./03-orm-comparison.md)を参照。

### Q4: password_hashをSELECTから除外するベストプラクティスは？

1. **列レベル権限**: `REVOKE SELECT(password_hash) ON users FROM app_readonly;`
2. **ビュー**: password_hashを含まないビューを作成し、アプリはビュー経由でアクセス
3. **アプリケーション層**: SELECT時に明示的にカラムを指定（`SELECT *` を禁止）
4. **RLS**: password_hashを必要としない操作ではRLSで行全体を制限

ORMではPrismaの `select` やSQLAlchemyの `defer` でカラムの遅延読み込みを設定できる。

### Q5: スーパーユーザーを使わないと実行できない操作は？

- CREATE EXTENSION（拡張機能のインストール）
- pg_hba.confの変更リロード
- postgresql.confのパラメータ変更
- REPLICATION権限の付与
- シグナル送信（pg_cancel_backend, pg_terminate_backend）
- ファイルシステムアクセス（COPY TO/FROM ファイル）

これらはCI/CDパイプラインやインフラ自動化で行い、日常的なアプリケーション操作では使用しない設計にすべき。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 多層防御 | ネットワーク → 認証 → 認可 → 入力検証 → 暗号化 → 監査の6層 |
| 最小権限の原則 | アプリ用ロールには必要最小限の権限のみ。SUPERUSERは禁止 |
| pg_hba.conf | 最後に`reject`ルールを置く。scram-sha-256を使用 |
| RLS | 行レベルのアクセス制御。マルチテナント分離に必須 |
| SQLインジェクション | パラメータ化クエリで100%防止。動的SQLにはformat(%I, %L)を使用 |
| パスワード | bcrypt (gen_salt('bf', 12))でハッシュ。MD5/SHA-256は不可 |
| 暗号化 | TLS通信(verify-full) + 列レベル暗号化 + ディスク暗号化 + バックアップ暗号化 |
| 監査 | audit_logテーブル + pgAuditで全操作を記録。月次パーティション推奨 |
| モニタリング | アクティブ接続、長時間クエリ、ロック競合を常時監視 |

---

## 次に読むべきガイド

- [00-postgresql-features.md](./00-postgresql-features.md) — pgcrypto、pg_trgm等の活用
- [02-performance-tuning.md](./02-performance-tuning.md) — セキュリティ設定とパフォーマンスの両立
- [03-orm-comparison.md](./03-orm-comparison.md) — ORMのSQLインジェクション対策
- [セキュリティ概要](../../security-fundamentals/docs/00-basics/00-security-overview.md) — 情報セキュリティの全体像
- [OWASP Top 10](../../security-fundamentals/docs/01-web-security/00-owasp-top10.md) — Web脆弱性トップ10
- [インジェクション](../../security-fundamentals/docs/01-web-security/03-injection.md) — SQLi以外のインジェクション攻撃
- [TLS/証明書](../../security-fundamentals/docs/02-cryptography/01-tls-certificates.md) — TLSの仕組みと証明書管理
- [パスワードセキュリティ](../../authentication-and-authorization/docs/00-fundamentals/01-password-security.md) — パスワードハッシュの詳細
- [RBAC](../../authentication-and-authorization/docs/03-authorization/00-rbac.md) — ロールベースアクセス制御

---

## 参考文献

1. PostgreSQL Documentation — "Client Authentication" https://www.postgresql.org/docs/current/client-authentication.html
2. PostgreSQL Documentation — "Row Security Policies" https://www.postgresql.org/docs/current/ddl-rowsecurity.html
3. PostgreSQL Documentation — "pgcrypto" https://www.postgresql.org/docs/current/pgcrypto.html
4. OWASP — "SQL Injection Prevention Cheat Sheet" https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html
5. OWASP — "Database Security Cheat Sheet" https://cheatsheetseries.owasp.org/cheatsheets/Database_Security_Cheat_Sheet.html
6. CIS PostgreSQL Benchmark — https://www.cisecurity.org/benchmark/postgresql
7. Riggs, S. & Ciolli, G. (2022). *PostgreSQL 14 Administration Cookbook*. Packt Publishing.
