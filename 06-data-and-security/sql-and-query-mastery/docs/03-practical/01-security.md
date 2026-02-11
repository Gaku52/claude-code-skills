# DBセキュリティ — 権限・暗号化・SQLインジェクション

> データベースセキュリティは多層防御の原則に基づき、認証・認可・暗号化・入力検証の各層で脅威を防ぐ包括的な取り組みであり、SQLインジェクションは最も一般的かつ深刻な攻撃手法として常に対策が求められる。

## この章で学ぶこと

1. PostgreSQLの認証・認可モデル（ロール、権限、RLS）
2. データの暗号化（転送中・保存時）と機密データの保護
3. SQLインジェクションの原理と完全な防御手法

---

## 1. 認証と認可

```
┌──────────── DBセキュリティの多層防御 ────────────┐
│                                                   │
│  Layer 1: ネットワーク                             │
│  ┌─────────────────────────────────────────────┐ │
│  │ ファイアウォール、VPC、pg_hba.conf          │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  Layer 2: 認証 (Authentication)                   │
│  ┌─────────────────────────────────────────────┐ │
│  │ パスワード(SCRAM-SHA-256)、証明書、LDAP     │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  Layer 3: 認可 (Authorization)                    │
│  ┌─────────────────────────────────────────────┐ │
│  │ GRANT/REVOKE、スキーマ権限、RLS             │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  Layer 4: 暗号化                                  │
│  ┌─────────────────────────────────────────────┐ │
│  │ TLS通信、列レベル暗号化、ディスク暗号化     │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  Layer 5: 監査                                    │
│  ┌─────────────────────────────────────────────┐ │
│  │ ログ、pgAudit、変更履歴                     │ │
│  └─────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────┘
```

### コード例1: ロールと権限の管理

```sql
-- ロール（ユーザー/グループ）の作成
CREATE ROLE app_readonly LOGIN PASSWORD 'secure_password_here'
    VALID UNTIL '2025-12-31';

CREATE ROLE app_readwrite LOGIN PASSWORD 'another_secure_password';

CREATE ROLE app_admin LOGIN PASSWORD 'admin_password'
    CREATEDB CREATEROLE;

-- グループロール
CREATE ROLE developers;
GRANT developers TO app_readwrite;

-- スキーマレベルの権限
GRANT USAGE ON SCHEMA public TO app_readonly;
GRANT CREATE ON SCHEMA public TO app_readwrite;

-- テーブルレベルの権限
GRANT SELECT ON ALL TABLES IN SCHEMA public TO app_readonly;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_readwrite;

-- 将来作成されるテーブルにもデフォルト権限を設定
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO app_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO app_readwrite;

-- シーケンスの権限（INSERTにSERIAL使用時に必要）
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO app_readwrite;

-- 列レベルの権限
GRANT SELECT (id, name, email) ON users TO app_readonly;
-- salary列は除外 → app_readonlyはsalaryを見られない

-- 権限の剥奪
REVOKE DELETE ON users FROM app_readwrite;
```

### コード例2: Row Level Security (RLS)

```sql
-- RLSの有効化
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- ポリシー: ユーザーは自分のドキュメントのみ参照可能
CREATE POLICY documents_owner_policy ON documents
    FOR ALL
    USING (owner_id = current_setting('app.current_user_id')::INTEGER);

-- ポリシー: 管理者は全ドキュメントを参照可能
CREATE POLICY documents_admin_policy ON documents
    FOR ALL
    TO app_admin
    USING (TRUE);

-- ポリシー: SELECTのみ公開ドキュメントを全員に許可
CREATE POLICY documents_public_read ON documents
    FOR SELECT
    USING (visibility = 'public');

-- アプリケーションからの使用
SET app.current_user_id = '42';
SELECT * FROM documents;
-- → owner_id = 42 のドキュメントと公開ドキュメントのみ返される

-- マルチテナントRLS
CREATE POLICY tenant_isolation ON orders
    FOR ALL
    USING (tenant_id = current_setting('app.tenant_id')::INTEGER);
```

---

## 2. SQLインジェクション

### SQLインジェクションの仕組み

```
┌───────── SQLインジェクションの原理 ────────────────┐
│                                                     │
│  正常なクエリ:                                       │
│  "SELECT * FROM users WHERE name = '" + input + "'" │
│  input = "田中"                                     │
│  → SELECT * FROM users WHERE name = '田中'          │
│                                                     │
│  攻撃入力:                                           │
│  input = "' OR '1'='1"                              │
│  → SELECT * FROM users WHERE name = '' OR '1'='1'   │
│  → 全行が返される！                                  │
│                                                     │
│  さらに悪質な攻撃:                                    │
│  input = "'; DROP TABLE users; --"                  │
│  → SELECT * FROM users WHERE name = '';             │
│    DROP TABLE users; --'                             │
│  → テーブルが削除される！                             │
│                                                     │
│  対策: パラメータ化クエリ（プリペアドステートメント）  │
│  → ユーザー入力はデータとして扱われ、                │
│    SQL構文として解釈されない                          │
└─────────────────────────────────────────────────────┘
```

### コード例3: SQLインジェクションの防御

```sql
-- NG: 文字列結合（言語例はPython/Node.js的な擬似コード）

-- Python (NG)
-- query = f"SELECT * FROM users WHERE email = '{user_input}'"
-- cursor.execute(query)

-- Node.js (NG)
-- const query = `SELECT * FROM users WHERE email = '${userInput}'`
-- await pool.query(query)

-- OK: パラメータ化クエリ

-- Python (psycopg2)
-- cursor.execute("SELECT * FROM users WHERE email = %s", (user_input,))

-- Node.js (pg)
-- await pool.query('SELECT * FROM users WHERE email = $1', [userInput])

-- PostgreSQL プリペアドステートメント（SQL直接）
PREPARE user_by_email (TEXT) AS
    SELECT id, name, email FROM users WHERE email = $1;
EXECUTE user_by_email('user@example.com');
DEALLOCATE user_by_email;

-- 動的SQLが必要な場合のエスケープ（PL/pgSQL）
CREATE FUNCTION search_users(p_column TEXT, p_value TEXT)
RETURNS SETOF users AS $$
BEGIN
    -- quote_ident: 識別子のエスケープ
    -- quote_literal: リテラルのエスケープ
    RETURN QUERY EXECUTE
        'SELECT * FROM users WHERE '
        || quote_ident(p_column)
        || ' = '
        || quote_literal(p_value);
END;
$$ LANGUAGE plpgsql;

-- format()関数でより安全に
CREATE FUNCTION search_users_v2(p_column TEXT, p_value TEXT)
RETURNS SETOF users AS $$
BEGIN
    RETURN QUERY EXECUTE format(
        'SELECT * FROM users WHERE %I = %L',  -- %I=識別子, %L=リテラル
        p_column, p_value
    );
END;
$$ LANGUAGE plpgsql;
```

---

## 3. 暗号化

### コード例4: 暗号化の実装

```sql
-- pgcrypto拡張の有効化
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- パスワードのハッシュ化（bcrypt）
INSERT INTO users (email, password_hash) VALUES (
    'user@example.com',
    crypt('user_password', gen_salt('bf', 12))  -- bcrypt, コスト12
);

-- パスワードの検証
SELECT id, email FROM users
WHERE email = 'user@example.com'
  AND password_hash = crypt('user_password', password_hash);

-- 列レベルの暗号化（AES-256）
-- 暗号化して格納
INSERT INTO sensitive_data (user_id, ssn_encrypted) VALUES (
    1,
    pgp_sym_encrypt('123-45-6789', 'encryption_key_here')
);

-- 復号して取得
SELECT user_id,
       pgp_sym_decrypt(ssn_encrypted, 'encryption_key_here') AS ssn
FROM sensitive_data
WHERE user_id = 1;
```

### コード例5: pg_hba.conf と TLS設定

```sql
-- pg_hba.conf の設定例（認証方式の制御）
-- TYPE  DATABASE  USER       ADDRESS        METHOD
-- local all       postgres                  peer
-- host  all       app_user   10.0.0.0/24    scram-sha-256
-- host  all       all        0.0.0.0/0      reject

-- TLS接続の強制
-- postgresql.conf:
-- ssl = on
-- ssl_cert_file = '/etc/ssl/certs/server.crt'
-- ssl_key_file = '/etc/ssl/private/server.key'

-- TLS接続の確認
SELECT
    datname,
    usename,
    client_addr,
    ssl,
    ssl_version,
    ssl_cipher
FROM pg_stat_ssl
    JOIN pg_stat_activity USING (pid);
```

### コード例6: 監査ログ

```sql
-- 監査トリガーの実装
CREATE TABLE audit_log (
    id          BIGSERIAL PRIMARY KEY,
    table_name  VARCHAR(100) NOT NULL,
    operation   VARCHAR(10) NOT NULL,
    old_data    JSONB,
    new_data    JSONB,
    changed_by  VARCHAR(100) DEFAULT current_user,
    changed_at  TIMESTAMPTZ DEFAULT NOW(),
    client_ip   INET DEFAULT inet_client_addr()
);

CREATE OR REPLACE FUNCTION audit_trigger_func()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (table_name, operation, old_data)
        VALUES (TG_TABLE_NAME, 'DELETE', to_jsonb(OLD));
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (table_name, operation, old_data, new_data)
        VALUES (TG_TABLE_NAME, 'UPDATE', to_jsonb(OLD), to_jsonb(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (table_name, operation, new_data)
        VALUES (TG_TABLE_NAME, 'INSERT', to_jsonb(NEW));
        RETURN NEW;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- 監査対象テーブルにトリガーを設定
CREATE TRIGGER audit_users
    AFTER INSERT OR UPDATE OR DELETE ON users
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_func();
```

---

## 認証方式比較表

| 方式 | セキュリティ | 設定容易性 | 用途 |
|------|:---:|:---:|------|
| trust | 低（認証なし） | 最も簡単 | 開発環境のみ |
| password (md5) | 中（非推奨） | 簡単 | レガシー互換 |
| scram-sha-256 | 高 | 簡単 | 推奨 |
| peer / ident | 高 | 中 | ローカルUNIX接続 |
| cert | 最高 | 複雑 | TLSクライアント証明書 |
| ldap | 高 | 複雑 | 企業AD/LDAP統合 |
| gss / kerberos | 最高 | 最も複雑 | エンタープライズSSO |

## 権限レベル比較表

| レベル | 対象 | コマンド | 例 |
|--------|------|---------|-----|
| クラスタ | DB全体 | CREATE ROLE | スーパーユーザー権限 |
| データベース | 特定DB | GRANT CONNECT | 接続許可 |
| スキーマ | 名前空間 | GRANT USAGE | スキーマ内のオブジェクト参照 |
| テーブル | 個別テーブル | GRANT SELECT | CRUD操作の許可 |
| 列 | 個別カラム | GRANT SELECT(col) | 特定列のみ参照許可 |
| 行 | 個別レコード | RLS Policy | マルチテナント分離 |

---

## アンチパターン

### アンチパターン1: アプリケーションにスーパーユーザーで接続

```sql
-- NG: 全アプリケーションがpostgresユーザーで接続
-- → 全テーブルの読み書き削除が可能
-- → DROP DATABASE も実行可能

-- OK: 最小権限の原則に基づいたロール設計
CREATE ROLE web_app LOGIN PASSWORD '...'
    CONNECTION LIMIT 20;
GRANT SELECT, INSERT, UPDATE ON customers, orders, order_items TO web_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO web_app;
-- DELETE権限は付与しない
-- DDL権限も付与しない
```

### アンチパターン2: 機密データの平文保存

```sql
-- NG: パスワードを平文で保存
CREATE TABLE users (
    id       SERIAL PRIMARY KEY,
    email    VARCHAR(255),
    password VARCHAR(255)  -- 平文！
);

-- NG: MD5でハッシュ（脆弱）
INSERT INTO users (email, password) VALUES
('user@example.com', md5('password123'));

-- OK: bcryptでハッシュ（コスト12以上推奨）
INSERT INTO users (email, password_hash) VALUES
('user@example.com', crypt('password123', gen_salt('bf', 12)));
```

---

## FAQ

### Q1: RLS（Row Level Security）はパフォーマンスに影響するか？

RLSポリシーは各クエリにWHERE条件として追加されるため、適切なインデックスがあればオーバーヘッドは最小限。ただし、複雑なポリシー（サブクエリ含む）や大量のポリシーはパフォーマンスに影響する可能性がある。EXPLAIN ANALYZEで確認すべき。

### Q2: データベースのバックアップも暗号化すべきか？

はい。pg_dumpの出力をGPG等で暗号化するか、暗号化ストレージに保存する。バックアップファイルには全データが含まれるため、本番環境と同等以上のセキュリティが必要。特にクラウドストレージへのバックアップはServer-Side Encryption (SSE)を有効にする。

### Q3: SQLインジェクション以外のインジェクション攻撃は？

OSコマンドインジェクション（`COPY FROM PROGRAM`の悪用）、LDAPインジェクション（LDAP認証時）、NoSQLインジェクション（MongoDB等）がある。また、ORMを使っていてもRaw SQL機能やフィルタの不適切な使用でインジェクションが発生する場合がある。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 最小権限の原則 | アプリ用ロールには必要最小限の権限のみ |
| RLS | 行レベルのアクセス制御。マルチテナントに有効 |
| SQLインジェクション | パラメータ化クエリで100%防止 |
| パスワード | bcrypt (gen_salt('bf', 12))でハッシュ |
| 暗号化 | TLS通信 + 列レベル暗号化 + バックアップ暗号化 |
| 監査 | audit_logテーブル + pgAuditで全操作を記録 |

---

## 次に読むべきガイド

- [02-performance-tuning.md](./02-performance-tuning.md) — セキュリティ設定とパフォーマンスの両立
- [00-postgresql-features.md](./00-postgresql-features.md) — pgcrypto等の活用
- [03-orm-comparison.md](./03-orm-comparison.md) — ORMのSQLインジェクション対策

---

## 参考文献

1. PostgreSQL Documentation — "Client Authentication" https://www.postgresql.org/docs/current/client-authentication.html
2. OWASP — "SQL Injection Prevention Cheat Sheet" https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html
3. PostgreSQL Documentation — "Row Security Policies" https://www.postgresql.org/docs/current/ddl-rowsecurity.html
