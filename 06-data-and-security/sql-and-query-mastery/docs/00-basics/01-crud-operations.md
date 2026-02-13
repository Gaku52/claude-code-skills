# CRUD操作 — SELECT / INSERT / UPDATE / DELETE 完全ガイド

> CRUDはCreate（作成）、Read（読取）、Update（更新）、Delete（削除）の頭文字であり、データベース操作の最も基本的な4つの操作を表す。あらゆるアプリケーションのデータ層はこの4操作の組み合わせで構成される。

---

## この章で学ぶこと

1. SELECT文の完全な構文と論理的実行順序を正確に理解する
2. INSERT / UPDATE / DELETE の安全な実行パターンとトランザクション活用法を身につける
3. RETURNING句、UPSERT、論理削除など実務で頻出する応用パターンを習得する
4. 各操作のパフォーマンス特性とアンチパターンを把握する

---

## 前提知識

- [00-sql-overview.md](./00-sql-overview.md) — SQL概要、リレーショナルモデルの基礎、RDBMS選定
- SQLの基本的なデータ型（INTEGER、VARCHAR、DATE、DECIMAL等）の理解
- [../../security-fundamentals/docs/01-web-security/](../../security-fundamentals/docs/01-web-security/) — SQLインジェクション対策（推奨）

---

## 1. SELECT — データの読み取り

SELECTはSQLの中で最も頻繁に使用される文であり、データベースからデータを取得する唯一の手段である。SELECTの理解はSQLマスタリーの基盤となる。

### 1.1 SELECT文の論理的実行順序

SELECT文は「書く順序」と「実行される順序」が異なるという点が、多くの初学者が躓くポイントである。

```
┌──────────────────────────────────────────────────────┐
│              SELECT文の論理的実行順序                  │
│                                                      │
│   書く順序              実行順序                      │
│   ─────────            ─────────                     │
│   SELECT   ──────┐    ① FROM / JOIN                  │
│   FROM     ◀─────┤    ② ON (結合条件)                │
│   WHERE    ──────┤    ③ WHERE (行フィルタ)            │
│   GROUP BY ──────┤    ④ GROUP BY (グループ化)         │
│   HAVING   ──────┤    ⑤ HAVING (グループフィルタ)     │
│   SELECT   ──────┤    ⑥ SELECT (式の評価、エイリアス) │
│   DISTINCT ──────┤    ⑦ DISTINCT (重複排除)           │
│   ORDER BY ──────┤    ⑧ ORDER BY (並べ替え)           │
│   LIMIT    ──────┘    ⑨ LIMIT / OFFSET (件数制限)    │
│                                                      │
│   重要な帰結:                                        │
│   - WHERE句でSELECTのエイリアスは使えない            │
│     (WHEREが先に実行されるため)                       │
│   - ORDER BY句ではSELECTのエイリアスが使える         │
│     (ORDER BYが後に実行されるため)                    │
│   - HAVING句では集約関数が使える                      │
│     (GROUP BYの後に実行されるため)                    │
└──────────────────────────────────────────────────────┘
```

### コード例1: SELECT文の完全な構文と実行順序

```sql
-- テーブル準備
CREATE TABLE departments (
    id   INTEGER PRIMARY KEY,
    name VARCHAR(50) NOT NULL
);

CREATE TABLE employees (
    id            SERIAL PRIMARY KEY,
    name          VARCHAR(100) NOT NULL,
    department_id INTEGER REFERENCES departments(id),
    salary        DECIMAL(10, 2),
    status        VARCHAR(20) DEFAULT 'active',
    hired_date    DATE NOT NULL DEFAULT CURRENT_DATE
);

INSERT INTO departments VALUES (1, '営業'), (2, '開発'), (3, '人事'), (4, '企画');
INSERT INTO employees (name, department_id, salary, status, hired_date) VALUES
    ('田中太郎', 1, 450000, 'active', '2020-04-01'),
    ('鈴木花子', 2, 520000, 'active', '2019-07-15'),
    ('佐藤次郎', 1, 380000, 'active', '2021-01-10'),
    ('高橋三郎', 2, 600000, 'active', '2018-04-01'),
    ('山田四郎', 3, 420000, 'active', '2022-04-01'),
    ('伊藤五郎', 2, 480000, 'inactive', '2017-10-01'),
    ('渡辺六子', 1, 510000, 'active', '2019-01-15');

-- SELECT文の完全な構文例
SELECT DISTINCT
    d.name AS department,              -- ⑥ 列を選択・計算
    COUNT(*) AS employee_count,        -- ⑥ 集約関数の評価
    AVG(e.salary) AS avg_salary        -- ⑥ 集約関数の評価
FROM employees e                       -- ① テーブルを特定
    INNER JOIN departments d           -- ① テーブル結合
        ON e.department_id = d.id      -- ② 結合条件の評価
WHERE e.status = 'active'              -- ③ 行レベルのフィルタ
GROUP BY d.name                        -- ④ グループ化
HAVING COUNT(*) >= 2                   -- ⑤ グループレベルのフィルタ
ORDER BY avg_salary DESC               -- ⑧ 並べ替え
LIMIT 10;                              -- ⑨ 件数制限
```

### コード例2: WHERE句の全条件パターン

```sql
-- === 比較演算子 ===
SELECT * FROM products WHERE price > 1000;               -- より大きい
SELECT * FROM products WHERE price >= 1000;              -- 以上
SELECT * FROM products WHERE price < 5000;               -- より小さい
SELECT * FROM products WHERE price <= 5000;              -- 以下
SELECT * FROM products WHERE price <> 1000;              -- 等しくない（標準SQL）
SELECT * FROM products WHERE price != 1000;              -- 等しくない（多くのRDBMSで使用可）

-- === 範囲 ===
SELECT * FROM products WHERE price BETWEEN 1000 AND 5000;
-- ↑ price >= 1000 AND price <= 5000 と同等（両端を含む）

-- === パターンマッチ（LIKE） ===
SELECT * FROM users WHERE name LIKE '田中%';             -- 前方一致
SELECT * FROM users WHERE email LIKE '%@gmail.com';      -- 後方一致
SELECT * FROM users WHERE name LIKE '%太%';              -- 部分一致
SELECT * FROM users WHERE code LIKE 'A_B';               -- _ = 任意の1文字
-- LIKE のエスケープ
SELECT * FROM products WHERE name LIKE '%25\%%' ESCAPE '\';  -- '%'を含む

-- === NULL判定 ===
-- 重要: = NULL は動作しない（3値論理のため）
SELECT * FROM users WHERE phone IS NULL;                  -- NULLの行
SELECT * FROM users WHERE phone IS NOT NULL;              -- NULLでない行

-- === IN / NOT IN ===
SELECT * FROM orders WHERE status IN ('pending', 'processing', 'shipped');
SELECT * FROM orders WHERE status NOT IN ('cancelled', 'refunded');

-- === 複合条件（AND / OR） ===
SELECT * FROM products
WHERE (category = 'electronics' OR category = 'books')
  AND price < 5000
  AND stock > 0;

-- === EXISTS ===
SELECT * FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o
    WHERE o.customer_id = c.id AND o.total > 10000
);

-- === ANY / ALL ===
SELECT * FROM employees
WHERE salary > ALL (
    SELECT AVG(salary) FROM employees GROUP BY department_id
);

-- === CASE式をWHEREで使用 ===
SELECT * FROM orders
WHERE CASE
    WHEN priority = 'high' THEN total_amount > 0
    WHEN priority = 'low'  THEN total_amount > 1000
    ELSE TRUE
END;
```

### コード例3: SELECT句の応用テクニック

```sql
-- === 計算列 ===
SELECT
    name,
    salary,
    salary * 12 AS annual_salary,                    -- 年収
    salary * 12 * 1.1 AS annual_with_bonus,          -- 年収+賞与
    ROUND(salary / 160, 0) AS hourly_rate            -- 時給換算
FROM employees;

-- === CASE式（条件分岐） ===
SELECT
    name,
    salary,
    CASE
        WHEN salary >= 600000 THEN 'S'
        WHEN salary >= 500000 THEN 'A'
        WHEN salary >= 400000 THEN 'B'
        ELSE 'C'
    END AS grade,
    -- CASE式の簡略形（単純CASE）
    CASE status
        WHEN 'active'   THEN '在籍'
        WHEN 'inactive' THEN '休職'
        WHEN 'retired'  THEN '退職'
        ELSE '不明'
    END AS status_label
FROM employees;

-- === COALESCE（最初の非NULL値） ===
SELECT
    name,
    COALESCE(phone, mobile, email, '連絡先なし') AS primary_contact
FROM employees;

-- === NULLIF（2値が等しければNULL） ===
-- 0除算防止パターン
SELECT
    department_id,
    total_revenue,
    total_cost,
    total_revenue / NULLIF(total_cost, 0) AS cost_ratio  -- 0除算を回避
FROM department_financials;

-- === 型変換（CAST） ===
SELECT
    CAST(price AS INTEGER) AS rounded_price,
    CAST(created_at AS DATE) AS created_date,
    CAST(quantity AS VARCHAR(10)) || '個' AS quantity_text
FROM products;

-- === サブクエリとの組み合わせ ===
SELECT
    e.name,
    e.salary,
    (SELECT AVG(salary) FROM employees) AS company_avg,
    e.salary - (SELECT AVG(salary) FROM employees) AS diff_from_avg,
    ROUND(e.salary / (SELECT AVG(salary) FROM employees) * 100, 1) AS pct_of_avg
FROM employees e
ORDER BY diff_from_avg DESC;
```

### 1.2 ORDER BYの詳細

```sql
-- 基本的なソート
SELECT * FROM employees ORDER BY salary DESC;                -- 降順
SELECT * FROM employees ORDER BY salary ASC;                 -- 昇順（デフォルト）
SELECT * FROM employees ORDER BY department_id ASC, salary DESC;  -- 複合ソート

-- NULLの並び順制御
SELECT * FROM employees ORDER BY department_id NULLS FIRST;  -- NULLを先頭
SELECT * FROM employees ORDER BY department_id NULLS LAST;   -- NULLを末尾

-- 列番号によるソート（可読性が低いため非推奨だが知っておくべき）
SELECT name, salary FROM employees ORDER BY 2 DESC;  -- 2番目の列 = salary

-- CASE式でカスタムソート
SELECT * FROM orders
ORDER BY
    CASE status
        WHEN 'urgent'     THEN 1
        WHEN 'processing' THEN 2
        WHEN 'pending'    THEN 3
        WHEN 'completed'  THEN 4
        ELSE 5
    END,
    created_at DESC;
```

### 1.3 DISTINCT と LIMIT

```sql
-- DISTINCT: 重複行の排除
SELECT DISTINCT department_id FROM employees;

-- DISTINCT ON (PostgreSQL固有): 各グループの最初の行のみ
SELECT DISTINCT ON (department_id) *
FROM employees
ORDER BY department_id, salary DESC;
-- → 各部署で最も給与の高い社員1名ずつ

-- LIMIT / OFFSET: ページネーション
SELECT * FROM products
ORDER BY created_at DESC
LIMIT 20 OFFSET 0;     -- 1ページ目（1-20件）

SELECT * FROM products
ORDER BY created_at DESC
LIMIT 20 OFFSET 20;    -- 2ページ目（21-40件）

-- 標準SQL: FETCH FIRST
SELECT * FROM products
ORDER BY created_at DESC
OFFSET 20 ROWS
FETCH NEXT 20 ROWS ONLY;
```

---

## 2. INSERT — データの作成

### 2.1 INSERTの内部動作

INSERTが実行されると、内部的に以下の処理が行われる:

```
┌──────────────── INSERT の内部処理フロー ────────────────┐
│                                                          │
│  ① 構文解析・権限チェック                                │
│      └→ テーブル存在確認、カラム型チェック                │
│  ② 制約チェック                                         │
│      ├→ NOT NULL 制約                                   │
│      ├→ UNIQUE / PRIMARY KEY 制約                       │
│      ├→ FOREIGN KEY 制約（参照先の存在確認）             │
│      ├→ CHECK 制約                                      │
│      └→ 排他制約（PostgreSQL EXCLUDE）                   │
│  ③ DEFAULT値・GENERATED列の計算                          │
│  ④ トリガー実行（BEFORE INSERT）                         │
│  ⑤ 行の挿入（WALへの書き込み → バッファプール更新）      │
│  ⑥ インデックスの更新（全関連インデックス）               │
│  ⑦ トリガー実行（AFTER INSERT）                          │
│  ⑧ RETURNING句の評価（PostgreSQL）                       │
│                                                          │
│  ※ これら全体がトランザクション内で原子的に実行される      │
└──────────────────────────────────────────────────────────┘
```

### コード例4: 各種INSERTパターン

```sql
-- === 基本的なINSERT（単一行） ===
INSERT INTO employees (name, department_id, salary, hired_date)
VALUES ('山田太郎', 10, 400000, '2024-04-01');

-- === 複数行の一括INSERT ===
-- パフォーマンス: 個別INSERTの10〜100倍高速
INSERT INTO employees (name, department_id, salary, hired_date)
VALUES
    ('鈴木花子', 20, 450000, '2024-04-01'),
    ('佐藤次郎', 10, 380000, '2024-04-01'),
    ('高橋三郎', 30, 420000, '2024-04-01');

-- === SELECTの結果をINSERT ===
-- データ移行・アーカイブに最適
INSERT INTO employee_archive (name, department_id, salary, archived_at)
SELECT name, department_id, salary, CURRENT_TIMESTAMP
FROM employees
WHERE status = 'retired';

-- === RETURNING句で挿入結果を取得（PostgreSQL） ===
-- アプリケーション側で自動生成されたIDを即座に取得可能
INSERT INTO employees (name, department_id, salary)
VALUES ('新人一号', 10, 350000)
RETURNING id, name, created_at;
-- → id=42, name='新人一号', created_at='2024-04-01 10:30:00'

-- === DEFAULT VALUES ===
-- 全列がDEFAULT値またはNULL許可の場合
INSERT INTO audit_log DEFAULT VALUES;

-- === INSERT with CTE ===
-- 複雑な変換を伴う挿入
WITH source_data AS (
    SELECT
        name,
        department_id,
        salary * 1.05 AS adjusted_salary  -- 5%昇給した金額で挿入
    FROM employees
    WHERE status = 'active' AND hired_date < '2020-01-01'
)
INSERT INTO salary_adjustments (employee_name, dept_id, new_salary, adjusted_at)
SELECT name, department_id, adjusted_salary, CURRENT_TIMESTAMP
FROM source_data;
```

### コード例5: UPSERT（存在すれば更新、なければ挿入）

```sql
-- === PostgreSQL: ON CONFLICT ===
-- user_settingsテーブル: (user_id, key) がUNIQUE制約
CREATE TABLE user_settings (
    user_id    INTEGER NOT NULL,
    key        VARCHAR(100) NOT NULL,
    value      TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, key)
);

-- 存在すればvalueを更新、なければ新規挿入
INSERT INTO user_settings (user_id, key, value)
VALUES (1, 'theme', 'dark')
ON CONFLICT (user_id, key)
DO UPDATE SET
    value = EXCLUDED.value,          -- EXCLUDED = 挿入しようとした値
    updated_at = CURRENT_TIMESTAMP;

-- ON CONFLICT DO NOTHING: 重複を無視（エラーを出さない）
INSERT INTO user_settings (user_id, key, value)
VALUES (1, 'theme', 'dark')
ON CONFLICT (user_id, key) DO NOTHING;

-- 条件付きUPSERT: 既存値より新しい場合のみ更新
INSERT INTO cache_entries (key, value, version)
VALUES ('user:1', '{"name":"田中"}', 5)
ON CONFLICT (key)
DO UPDATE SET
    value = EXCLUDED.value,
    version = EXCLUDED.version
WHERE cache_entries.version < EXCLUDED.version;  -- バージョンが新しい場合のみ

-- === MySQL: ON DUPLICATE KEY UPDATE ===
INSERT INTO user_settings (user_id, setting_key, setting_value)
VALUES (1, 'theme', 'dark')
ON DUPLICATE KEY UPDATE
    setting_value = VALUES(setting_value),
    updated_at = NOW();

-- === SQL Server: MERGE ===
MERGE INTO user_settings AS target
USING (VALUES (1, 'theme', 'dark')) AS source (user_id, key, value)
ON target.user_id = source.user_id AND target.key = source.key
WHEN MATCHED THEN
    UPDATE SET value = source.value, updated_at = GETDATE()
WHEN NOT MATCHED THEN
    INSERT (user_id, key, value) VALUES (source.user_id, source.key, source.value);
```

---

## 3. UPDATE — データの更新

### 3.1 UPDATEの安全な実行手順

本番環境でUPDATEを実行する際は、以下のプロトコルに従うことが推奨される:

```
┌──────────── 安全なUPDATE実行プロトコル ────────────┐
│                                                      │
│  Step 1: 対象行の確認                                │
│  ┌──────────────────────────────────────────┐      │
│  │ SELECT * FROM employees                   │      │
│  │ WHERE department_id = 10                  │      │
│  │   AND performance_rating >= 4;            │      │
│  │ -- → 5行ヒット（期待通りか確認）          │      │
│  └──────────────────────────────────────────┘      │
│                                                      │
│  Step 2: トランザクション開始                         │
│  ┌──────────────────────────────────────────┐      │
│  │ BEGIN;                                    │      │
│  └──────────────────────────────────────────┘      │
│                                                      │
│  Step 3: UPDATE実行                                  │
│  ┌──────────────────────────────────────────┐      │
│  │ UPDATE employees                          │      │
│  │ SET salary = salary * 1.05                │      │
│  │ WHERE department_id = 10                  │      │
│  │   AND performance_rating >= 4;            │      │
│  │ -- → UPDATE 5 (件数が一致するか確認)      │      │
│  └──────────────────────────────────────────┘      │
│                                                      │
│  Step 4: 結果確認                                    │
│  ┌──────────────────────────────────────────┐      │
│  │ SELECT * FROM employees                   │      │
│  │ WHERE department_id = 10                  │      │
│  │   AND performance_rating >= 4;            │      │
│  │ -- → 変更後の値が期待通りか確認           │      │
│  └──────────────────────────────────────────┘      │
│                                                      │
│  Step 5: 確定 or 取消                                │
│  ┌──────────────────────────────────────────┐      │
│  │ COMMIT;    -- 問題なければ確定            │      │
│  │ -- ROLLBACK;  -- 問題あれば取り消し       │      │
│  └──────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────┘
```

### コード例6: 各種UPDATEパターン

```sql
-- === 基本的なUPDATE ===
UPDATE employees
SET salary = 500000
WHERE employee_id = 42;

-- === 複数列の同時更新 ===
UPDATE employees
SET salary = salary * 1.05,
    updated_at = CURRENT_TIMESTAMP
WHERE department_id = 10
  AND performance_rating >= 4;

-- === 計算式を使った更新 ===
-- 全社員の給与を等級に応じて昇給
UPDATE employees
SET salary = CASE
    WHEN grade = 'S' THEN salary * 1.10   -- S等級: 10%昇給
    WHEN grade = 'A' THEN salary * 1.07   -- A等級: 7%昇給
    WHEN grade = 'B' THEN salary * 1.05   -- B等級: 5%昇給
    ELSE salary * 1.03                     -- その他: 3%昇給
END,
updated_at = CURRENT_TIMESTAMP
WHERE status = 'active';

-- === JOINを使ったUPDATE（PostgreSQL） ===
-- 禁止顧客の未処理注文をキャンセル
UPDATE orders o
SET status = 'cancelled',
    cancelled_at = NOW(),
    cancel_reason = 'customer_banned'
FROM customers c
WHERE o.customer_id = c.id
  AND c.is_banned = TRUE
  AND o.status = 'pending';

-- === JOINを使ったUPDATE（MySQL） ===
-- UPDATE orders o
-- INNER JOIN customers c ON o.customer_id = c.id
-- SET o.status = 'cancelled',
--     o.cancelled_at = NOW()
-- WHERE c.is_banned = TRUE
--   AND o.status = 'pending';

-- === サブクエリを使ったUPDATE ===
UPDATE products
SET price = price * 0.9
WHERE category_id IN (
    SELECT id FROM categories WHERE name = 'セール対象'
);

-- === RETURNING句で更新結果を確認（PostgreSQL） ===
UPDATE employees
SET salary = salary * 1.10
WHERE department_id = 20
RETURNING id, name, salary AS new_salary;

-- === CTEを使った複雑なUPDATE ===
-- 部署平均を下回る社員を最低ラインまで引き上げ
WITH dept_min AS (
    SELECT department_id, AVG(salary) * 0.8 AS min_salary
    FROM employees
    WHERE status = 'active'
    GROUP BY department_id
)
UPDATE employees e
SET salary = dm.min_salary
FROM dept_min dm
WHERE e.department_id = dm.department_id
  AND e.salary < dm.min_salary
  AND e.status = 'active'
RETURNING e.id, e.name, e.salary AS adjusted_salary;
```

---

## 4. DELETE — データの削除

### 4.1 物理削除 vs 論理削除

```
┌──────────── 物理削除 vs 論理削除 ──────────────┐
│                                                  │
│  物理削除（Hard Delete）                         │
│  ┌─────────────────────────────────────┐       │
│  │ DELETE FROM users WHERE id = 42;     │       │
│  │                                      │       │
│  │ メリット:                             │       │
│  │ - テーブルがシンプルに保たれる         │       │
│  │ - ストレージを即座に解放              │       │
│  │ - クエリにフィルタ条件不要            │       │
│  │                                      │       │
│  │ デメリット:                            │       │
│  │ - 復元が困難（バックアップのみ）       │       │
│  │ - 監査証跡が残らない                  │       │
│  │ - 外部キー制約の連鎖削除リスク        │       │
│  └─────────────────────────────────────┘       │
│                                                  │
│  論理削除（Soft Delete）                         │
│  ┌─────────────────────────────────────┐       │
│  │ UPDATE users                         │       │
│  │ SET deleted_at = CURRENT_TIMESTAMP,  │       │
│  │     status = 'deleted'               │       │
│  │ WHERE id = 42;                       │       │
│  │                                      │       │
│  │ メリット:                             │       │
│  │ - 復元が容易（フラグを戻すだけ）      │       │
│  │ - 監査証跡が残る                      │       │
│  │ - 参照整合性が維持される              │       │
│  │                                      │       │
│  │ デメリット:                            │       │
│  │ - 全クエリにWHERE deleted_at IS NULLが必要│   │
│  │ - ストレージが増大し続ける            │       │
│  │ - UNIQUE制約の設計が複雑化            │       │
│  └─────────────────────────────────────┘       │
│                                                  │
│  推奨: 論理削除 + 定期的なアーカイブ/パージ      │
└──────────────────────────────────────────────────┘
```

### コード例7: 安全なDELETEパターン

```sql
-- === 基本的なDELETE ===
DELETE FROM sessions
WHERE expires_at < CURRENT_TIMESTAMP;

-- === JOINを使ったDELETE（PostgreSQL: USING句） ===
DELETE FROM order_items oi
USING orders o
WHERE oi.order_id = o.id
  AND o.status = 'cancelled';

-- === RETURNING句で削除した行を取得 ===
DELETE FROM notifications
WHERE user_id = 42 AND read_at IS NOT NULL
RETURNING id, message, created_at;

-- === 論理削除（ソフトデリート）の実装パターン ===
-- 基本パターン
UPDATE users
SET deleted_at = CURRENT_TIMESTAMP,
    status = 'deleted',
    -- 個人情報の匿名化（GDPR対応）
    email = 'deleted_' || id || '@deleted.example.com',
    name = '削除済みユーザー'
WHERE id = 42;

-- 論理削除対応のビュー
CREATE VIEW active_users AS
SELECT * FROM users WHERE deleted_at IS NULL;

-- 論理削除 + UNIQUE制約の問題と解決
-- 問題: deleted_atがNULLの行同士でのみUNIQUEを保証したい
-- PostgreSQL: 部分インデックスで解決
CREATE UNIQUE INDEX idx_users_email_active
ON users (email)
WHERE deleted_at IS NULL;

-- === TRUNCATE: テーブル全行の高速削除（DDL操作） ===
TRUNCATE TABLE temp_import_data;
-- ※ WHERE句使用不可
-- ※ TRUNCATEはロールバック不可（MySQL）、PostgreSQLでは可能
-- ※ トリガーが発火しない
-- ※ 自動連番がリセットされる

-- === CASCADE付きTRUNCATE ===
-- 外部キーで参照されているテーブルも同時にTRUNCATE
TRUNCATE TABLE orders CASCADE;  -- order_itemsも同時にTRUNCATE

-- === DELETEの安全な実行（トランザクション） ===
BEGIN;
    -- Step 1: 削除対象の確認
    SELECT COUNT(*) FROM logs WHERE created_at < '2023-01-01';
    -- → 15,000行

    -- Step 2: 削除実行
    DELETE FROM logs WHERE created_at < '2023-01-01';
    -- → DELETE 15000

    -- Step 3: 確認して確定
    SELECT COUNT(*) FROM logs WHERE created_at < '2023-01-01';
    -- → 0行（期待通り）
COMMIT;

-- === 大量データの分割削除（ロック時間を短くする） ===
-- 一度に全件削除するとロック時間が長くなるため、バッチで実行
DO $$
DECLARE
    deleted_count INTEGER;
BEGIN
    LOOP
        DELETE FROM logs
        WHERE id IN (
            SELECT id FROM logs
            WHERE created_at < '2023-01-01'
            LIMIT 10000  -- 1回あたり10,000件ずつ
        );
        GET DIAGNOSTICS deleted_count = ROW_COUNT;
        EXIT WHEN deleted_count = 0;

        -- 他のトランザクションに実行機会を与える
        PERFORM pg_sleep(0.1);
    END LOOP;
END $$;
```

---

## 5. CRUD操作のデータフロー全体像

```
┌─────────────────── CRUD操作のライフサイクル ──────────────────┐
│                                                               │
│  アプリケーション層                                            │
│  ┌───────────────────────────────────────────┐               │
│  │  ORM / クエリビルダー / 生SQL              │               │
│  │  パラメータバインド（SQLインジェクション防止）│               │
│  └──────────────────┬────────────────────────┘               │
│                     │                                         │
│  データベース接続層   ▼                                        │
│  ┌───────────────────────────────────────────┐               │
│  │  接続プーリング（PgBouncer等）             │               │
│  │  プリペアドステートメント                   │               │
│  └──────────────────┬────────────────────────┘               │
│                     │                                         │
│  データベースエンジン ▼                                        │
│  ┌───────────────────────────────────────────┐               │
│  │                                           │               │
│  │  INSERT INTO → 制約チェック → 行追加      │               │
│  │                → インデックス更新          │               │
│  │                                           │               │
│  │  SELECT FROM → 実行計画生成 → データ取得   │               │
│  │              → フィルタ → ソート → 返却    │               │
│  │                                           │               │
│  │  UPDATE SET  → 行ロック → 旧バージョン保持 │               │
│  │             → 新値書き込み → ロック解放     │               │
│  │                                           │               │
│  │  DELETE FROM → 行ロック → 論理/物理削除    │               │
│  │             → 領域管理 → ロック解放        │               │
│  │                                           │               │
│  └───────────────────────────────────────────┘               │
│                                                               │
│  WAL（Write-Ahead Log） → ディスクへの永続化                  │
└───────────────────────────────────────────────────────────────┘
```

---

## 比較表

### DELETE vs TRUNCATE vs DROP 比較表

| 特徴 | DELETE | TRUNCATE | DROP |
|------|--------|----------|------|
| 操作種別 | DML | DDL | DDL |
| WHERE句 | 使用可能 | 使用不可 | 使用不可 |
| ロールバック | 可能 | DB依存（PG可、MySQL不可） | DB依存 |
| 速度 | 遅い（行単位ログ） | 高速（ページ単位） | 高速 |
| トリガー発火 | する | しない | しない |
| 自動連番リセット | しない | する | N/A |
| 領域解放 | しない（VACUUM必要） | する | する |
| テーブル構造 | 残る | 残る | 消える |
| 権限 | DML権限 | DDL権限 | DDL権限 |
| 外部キー制約 | 制約チェックあり | CASCADEが必要な場合あり | CASCADE指定可 |

### INSERT方式の比較表

| 方式 | 用途 | 速度 | 安全性 | RDBMS |
|------|------|------|--------|-------|
| 単一行INSERT | 個別レコード追加 | 低 | 高 | 全DB |
| 複数行INSERT | バッチ挿入（〜1000行） | 中 | 高 | 全DB |
| INSERT...SELECT | データ移行/複製 | 高 | 中 | 全DB |
| COPY / LOAD DATA | 大量データ投入（万〜億行） | 最速 | 低 | PG / MySQL |
| UPSERT | 冪等な挿入/更新 | 中 | 高 | 全DB（構文異なる） |
| INSERT...RETURNING | IDの即時取得 | 中 | 高 | PostgreSQL |
| バルクINSERT (ORM) | アプリ層での最適化 | 中〜高 | 高 | 全DB |

### UPDATE/DELETE方式の比較表

| 方式 | 用途 | ロック影響 | 推奨場面 |
|------|------|----------|---------|
| 単純WHERE | 少量行の更新/削除 | 小 | 日常操作 |
| JOIN/FROM付き | 関連テーブルに基づく更新 | 中 | データ連携 |
| サブクエリ付き | 複雑な条件の更新 | 中〜大 | 条件が複合的 |
| CTE付き | 計算結果に基づく更新 | 中 | 集約値での更新 |
| バッチ分割 | 大量行の更新/削除 | 小（分割） | 本番大量更新 |
| RETURNING付き | 更新結果の即時確認 | 小 | 監査・ログ |

---

## アンチパターン

### アンチパターン1: WHERE句なしのUPDATE/DELETE

```sql
-- NG: 全行が更新されてしまう！
UPDATE employees SET salary = 0;
-- → 全社員の給与が0になる

-- NG: 全行が削除されてしまう！
DELETE FROM employees;
-- → 全社員のデータが消える

-- OK: 必ずWHERE句で対象を限定
UPDATE employees SET salary = 500000 WHERE employee_id = 42;

-- 安全策1: 先にSELECTで確認
SELECT COUNT(*), MIN(salary), MAX(salary) FROM employees
WHERE department_id = 99;
-- → 5行、320000〜480000（妥当か確認）

DELETE FROM employees WHERE department_id = 99;

-- 安全策2: トランザクションで囲む
BEGIN;
    DELETE FROM employees WHERE department_id = 99;
    -- DELETE 5 ← 件数が期待通りか確認
    -- 問題なければ → COMMIT;
    -- 問題あれば → ROLLBACK;
COMMIT;

-- 安全策3: 本番DBではsafe_updateモード有効化（MySQL）
-- SET sql_safe_updates = 1;
-- → WHERE句のないUPDATE/DELETEがエラーになる
```

**WHY**: WHERE句のないUPDATEやDELETEは、データベースの全行に影響する。本番環境での「うっかり全件削除」は、企業にとって致命的な損失になりうる。この問題は「プログラマーの注意力」ではなく「仕組み」で防ぐべきである。

### アンチパターン2: SELECT * の濫用

```sql
-- NG: 不要な列まで全て取得
SELECT * FROM orders;

-- 問題点:
-- 1. ネットワーク帯域の浪費（BLOB/TEXT列がある場合は特に深刻）
--    例: 100万行 × 不要なBLOB列 = 数GBのネットワーク転送
-- 2. テーブル構造変更時にアプリケーションが壊れる
--    例: 列の追加/削除/順序変更でORMのマッピングが崩れる
-- 3. インデックスオンリースキャン（カバリングインデックス）が効かない
--    例: CREATE INDEX ON orders(status, customer_id)があっても
--         SELECT * では全行ヒープアクセスが必要
-- 4. 実行計画が非効率になる
--    不要な列のデシリアライズ、メモリ消費、ソートコスト増大

-- OK: 必要な列だけ明示的に指定
SELECT order_id, customer_id, total_amount, status
FROM orders
WHERE status = 'pending';

-- 例外: 対話的な探索やデバッグ時にはSELECT *は許容される
-- ただしLIMITを付ける
SELECT * FROM orders LIMIT 10;
```

**WHY**: SELECT *は「必要十分なデータだけを取得する」というデータベース操作の基本原則に反する。特に大規模テーブルでは、不要な列の転送コストが積算されてパフォーマンスに深刻な影響を与える。

### アンチパターン3: 文字列連結でSQLを組み立てる

```sql
-- NG: SQLインジェクションの脆弱性
-- Python疑似コード
-- query = f"SELECT * FROM users WHERE name = '{user_input}'"
-- user_input = "'; DROP TABLE users; --"
-- → SELECT * FROM users WHERE name = ''; DROP TABLE users; --'

-- OK: パラメータバインド（プリペアドステートメント）
-- Python (psycopg2)
-- cursor.execute("SELECT * FROM users WHERE name = %s", (user_input,))
-- Java (JDBC)
-- PreparedStatement ps = conn.prepareStatement("SELECT * FROM users WHERE name = ?");
-- ps.setString(1, userInput);
-- Node.js (pg)
-- client.query('SELECT * FROM users WHERE name = $1', [userInput])

-- PostgreSQLでのプリペアドステートメント
PREPARE get_user (TEXT) AS
SELECT * FROM users WHERE name = $1;

EXECUTE get_user('田中太郎');

DEALLOCATE get_user;
```

**WHY**: SQLインジェクションは最も古くから知られ、最も被害の大きいWebアプリケーションの脆弱性の一つ。パラメータバインドを使えば、ユーザー入力がSQL構文として解釈されることを防止できる。

---

## 実践演習

### 演習1（基礎）: 従業員データの基本CRUD

以下のテーブル定義に対して、指定されたCRUD操作をSQLで記述せよ。

```sql
CREATE TABLE departments (
    id   SERIAL PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE
);

CREATE TABLE employees (
    id            SERIAL PRIMARY KEY,
    name          VARCHAR(100) NOT NULL,
    email         VARCHAR(255) NOT NULL UNIQUE,
    department_id INTEGER REFERENCES departments(id),
    salary        DECIMAL(10, 2) NOT NULL CHECK (salary >= 0),
    status        VARCHAR(20) DEFAULT 'active',
    hired_date    DATE NOT NULL DEFAULT CURRENT_DATE
);
```

1. departments テーブルに「営業」「開発」「人事」を挿入せよ
2. employees テーブルに5名の社員を挿入せよ（複数行INSERT使用）
3. 開発部門の社員のみを給与の高い順に取得せよ
4. 営業部門の全社員の給与を5%昇給させよ
5. 「退職」status の社員を論理削除（status='deleted', deleted_atを設定）せよ

<details>
<summary>模範解答</summary>

```sql
-- 1. departments テーブルへの挿入
INSERT INTO departments (name) VALUES ('営業'), ('開発'), ('人事');

-- 2. employees テーブルへの5名挿入
INSERT INTO employees (name, email, department_id, salary, status, hired_date) VALUES
    ('田中太郎', 'tanaka@example.com', 1, 450000, 'active', '2020-04-01'),
    ('鈴木花子', 'suzuki@example.com', 2, 520000, 'active', '2019-07-15'),
    ('佐藤次郎', 'sato@example.com',   1, 380000, 'active', '2021-01-10'),
    ('高橋三郎', 'takahashi@example.com', 2, 600000, 'active', '2018-04-01'),
    ('山田四郎', 'yamada@example.com', 3, 420000, 'active', '2022-04-01');

-- 3. 開発部門の社員を給与順で取得
SELECT e.id, e.name, e.salary
FROM employees e
    INNER JOIN departments d ON e.department_id = d.id
WHERE d.name = '開発'
ORDER BY e.salary DESC;

-- 4. 営業部門の全社員を5%昇給
UPDATE employees
SET salary = salary * 1.05
WHERE department_id = (SELECT id FROM departments WHERE name = '営業');
-- または
UPDATE employees e
SET salary = salary * 1.05
FROM departments d
WHERE e.department_id = d.id AND d.name = '営業';

-- 5. 退職者の論理削除
-- まずdeleted_at列を追加（既存テーブルの変更）
ALTER TABLE employees ADD COLUMN deleted_at TIMESTAMP;

UPDATE employees
SET status = 'deleted',
    deleted_at = CURRENT_TIMESTAMP
WHERE status = 'retired';
```

</details>

### 演習2（応用）: 安全な大量データ操作

以下のシナリオでSQLを記述せよ。全ての操作はトランザクション内で安全に実行すること。

1. `products`テーブルで、在庫数が0の商品の価格を10%値下げし、結果をRETURNINGで確認せよ
2. `orders`テーブルで、2023年より前に作成され、statusが'completed'の注文をアーカイブテーブルに移動（INSERT...SELECT → DELETE）せよ
3. `user_sessions`テーブルで、24時間以上前に期限切れのセッションを削除する「バッチ削除」を記述せよ（1回の削除は5000件以内）

```sql
-- テーブル定義
CREATE TABLE products (
    id    SERIAL PRIMARY KEY,
    name  VARCHAR(100) NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    stock INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE orders (
    id         SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    total      DECIMAL(10, 2),
    status     VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE orders_archive (LIKE orders INCLUDING ALL);

CREATE TABLE user_sessions (
    id         UUID PRIMARY KEY,
    user_id    INTEGER NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

<details>
<summary>模範解答</summary>

```sql
-- 1. 在庫切れ商品の10%値下げ（RETURNING付き）
BEGIN;
    -- 対象確認
    SELECT id, name, price, stock FROM products WHERE stock = 0;

    -- 値下げ実行
    UPDATE products
    SET price = ROUND(price * 0.9, 2)
    WHERE stock = 0
    RETURNING id, name, price AS new_price;
COMMIT;

-- 2. 古い完了注文のアーカイブ移動
BEGIN;
    -- Step 1: アーカイブテーブルへコピー
    INSERT INTO orders_archive (id, customer_id, total, status, created_at)
    SELECT id, customer_id, total, status, created_at
    FROM orders
    WHERE created_at < '2023-01-01'
      AND status = 'completed';
    -- → INSERT 0 12345

    -- Step 2: コピー完了を確認してから元テーブルから削除
    DELETE FROM orders
    WHERE created_at < '2023-01-01'
      AND status = 'completed';
    -- → DELETE 12345（件数がINSERTと一致することを確認）

    -- Step 3: 確認
    SELECT COUNT(*) FROM orders_archive WHERE created_at < '2023-01-01';
    -- → 12345
COMMIT;

-- 3. 期限切れセッションのバッチ削除
DO $$
DECLARE
    batch_size CONSTANT INTEGER := 5000;
    deleted_count INTEGER;
    total_deleted INTEGER := 0;
BEGIN
    LOOP
        DELETE FROM user_sessions
        WHERE id IN (
            SELECT id FROM user_sessions
            WHERE expires_at < CURRENT_TIMESTAMP - INTERVAL '24 hours'
            LIMIT batch_size
            FOR UPDATE SKIP LOCKED  -- ロック競合を回避
        );
        GET DIAGNOSTICS deleted_count = ROW_COUNT;
        total_deleted := total_deleted + deleted_count;

        -- 進捗ログ（RAISE NOTICEはPostgreSQL固有）
        RAISE NOTICE 'Deleted % sessions (total: %)', deleted_count, total_deleted;

        EXIT WHEN deleted_count < batch_size;  -- 残りがバッチサイズ未満なら終了

        -- 他のトランザクションに実行機会を与える
        PERFORM pg_sleep(0.1);
    END LOOP;

    RAISE NOTICE 'Total deleted: % sessions', total_deleted;
END $$;
```

**ポイント:**
- `FOR UPDATE SKIP LOCKED`で他トランザクションがロック中の行をスキップ（デッドロック防止）
- バッチサイズを制限することでロック時間を短縮
- `PERFORM pg_sleep(0.1)`で他トランザクションに実行機会を与える

</details>

### 演習3（発展）: UPSERTとRETURNINGの組み合わせ

ECサイトのカート機能を実装せよ。以下の要件を満たすSQLを記述すること。

要件:
1. カートに商品を追加する際、既に同じ商品があれば数量を加算する（UPSERT）
2. カートの内容を商品名・価格付きで取得する
3. カートの合計金額を計算する
4. カートから特定商品を削除し、削除した内容をRETURNINGで返す
5. カートの有効期限チェック（30日以上前のカートを自動削除）

```sql
CREATE TABLE cart_items (
    id         SERIAL PRIMARY KEY,
    user_id    INTEGER NOT NULL,
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL CHECK (quantity > 0),
    added_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, product_id)
);
```

<details>
<summary>模範解答</summary>

```sql
-- 1. カートに商品を追加（UPSERT: 既存なら数量加算）
INSERT INTO cart_items (user_id, product_id, quantity)
VALUES (42, 101, 2)
ON CONFLICT (user_id, product_id)
DO UPDATE SET
    quantity = cart_items.quantity + EXCLUDED.quantity,
    updated_at = CURRENT_TIMESTAMP
RETURNING id, product_id, quantity AS total_quantity;
-- → 既にquantity=3だった場合: total_quantity=5

-- 2. カートの内容を商品情報付きで取得
SELECT
    ci.id AS cart_item_id,
    p.name AS product_name,
    p.price AS unit_price,
    ci.quantity,
    p.price * ci.quantity AS subtotal,
    ci.added_at
FROM cart_items ci
    INNER JOIN products p ON ci.product_id = p.id
WHERE ci.user_id = 42
ORDER BY ci.added_at;

-- 3. カートの合計金額
SELECT
    ci.user_id,
    COUNT(*) AS item_count,
    SUM(ci.quantity) AS total_quantity,
    SUM(p.price * ci.quantity) AS total_amount
FROM cart_items ci
    INNER JOIN products p ON ci.product_id = p.id
WHERE ci.user_id = 42
GROUP BY ci.user_id;

-- 4. カートから特定商品を削除（RETURNING付き）
DELETE FROM cart_items
WHERE user_id = 42 AND product_id = 101
RETURNING id, product_id, quantity;
-- → id=7, product_id=101, quantity=5

-- 5. 期限切れカートの自動削除
WITH expired_carts AS (
    DELETE FROM cart_items
    WHERE updated_at < CURRENT_TIMESTAMP - INTERVAL '30 days'
    RETURNING user_id, product_id, quantity
)
SELECT
    user_id,
    COUNT(*) AS items_removed,
    SUM(quantity) AS total_quantity_removed
FROM expired_carts
GROUP BY user_id;
-- → user_id=15, items_removed=3, total_quantity_removed=8
-- → user_id=23, items_removed=1, total_quantity_removed=2

-- ボーナス: カート内容の一括更新（数量変更）
UPDATE cart_items
SET quantity = new_data.quantity,
    updated_at = CURRENT_TIMESTAMP
FROM (VALUES
    (42, 101, 3),  -- user_id=42, product_id=101, new_quantity=3
    (42, 102, 1),  -- user_id=42, product_id=102, new_quantity=1
    (42, 103, 5)   -- user_id=42, product_id=103, new_quantity=5
) AS new_data (user_id, product_id, quantity)
WHERE cart_items.user_id = new_data.user_id
  AND cart_items.product_id = new_data.product_id
RETURNING cart_items.product_id, cart_items.quantity;
```

**設計のポイント:**
- UPSERTにより「カートに追加」が冪等操作になる（同じリクエストを何度実行しても安全）
- RETURNING句により追加のSELECTクエリが不要（ラウンドトリップ削減）
- CTEとDELETE...RETURNINGの組み合わせで削除と集計を1クエリで実行

</details>

---

## FAQ

### Q1: UPDATE文で「変更前の値」を参照できるか？

PostgreSQLではRETURNING句で更新後の値を取得できるが、更新前の値は直接取得できない。更新前の値が必要な場合はCTEを使う:

```sql
WITH old AS (
    SELECT id, salary FROM employees WHERE id = 42
)
UPDATE employees SET salary = salary * 1.1 WHERE id = 42
RETURNING id,
    (SELECT salary FROM old) AS old_salary,
    salary AS new_salary;
-- → id=42, old_salary=450000, new_salary=495000
```

別の方法として、監査トリガーを設定する:

```sql
CREATE TABLE salary_audit (
    id          SERIAL PRIMARY KEY,
    employee_id INTEGER,
    old_salary  DECIMAL(10, 2),
    new_salary  DECIMAL(10, 2),
    changed_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE OR REPLACE FUNCTION log_salary_change()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.salary <> NEW.salary THEN
        INSERT INTO salary_audit (employee_id, old_salary, new_salary)
        VALUES (OLD.id, OLD.salary, NEW.salary);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_salary_audit
    AFTER UPDATE ON employees
    FOR EACH ROW
    EXECUTE FUNCTION log_salary_change();
```

### Q2: 大量データのINSERTを高速化するには？

速度目安と推奨方法:

| 方法 | 速度目安（100万行） | 推奨場面 |
|------|-------------------|---------|
| 個別INSERT | 数十分 | 使用しない |
| 複数行INSERT（1000行/バッチ） | 数分 | 小〜中規模 |
| COPY文（PostgreSQL） | 数十秒 | 大規模データ投入 |
| LOAD DATA（MySQL） | 数十秒 | 大規模データ投入 |
| インデックス無効化 + COPY + 再作成 | 数秒〜十数秒 | 初期データ投入 |

```sql
-- PostgreSQL: COPY文（最速）
COPY employees (name, department_id, salary, hired_date)
FROM '/path/to/data.csv'
WITH (FORMAT CSV, HEADER TRUE, DELIMITER ',');

-- さらに高速化する場合
BEGIN;
    -- 1. インデックスを一時的に無効化
    ALTER TABLE employees DISABLE TRIGGER ALL;
    DROP INDEX IF EXISTS idx_employees_dept;

    -- 2. COPY実行
    COPY employees FROM '/path/to/data.csv' WITH (FORMAT CSV, HEADER);

    -- 3. インデックスを再作成
    CREATE INDEX idx_employees_dept ON employees(department_id);
    ALTER TABLE employees ENABLE TRIGGER ALL;

    -- 4. 統計情報の更新
    ANALYZE employees;
COMMIT;
```

### Q3: 論理削除と物理削除のどちらを使うべきか？

判断基準:

| 条件 | 推奨 |
|------|------|
| 法規制で保存義務がある | 論理削除 |
| ユーザーに「元に戻す」機能を提供 | 論理削除 |
| 監査証跡が必要 | 論理削除 |
| データ量が多くストレージ制約がある | 物理削除 + アーカイブ |
| GDPR等で「忘れられる権利」に対応 | 物理削除（匿名化後） |
| テーブル設計をシンプルに保ちたい | 物理削除 |

実務では**論理削除 + 定期的なアーカイブ/パージ**が最もバランスの良い選択である。論理削除したデータは一定期間後にアーカイブテーブルに移動し、さらに一定期間後に物理削除する。

### Q4: RETURNING句はどのRDBMSで使えるか？

| RDBMS | RETURNING対応 | 代替手段 |
|-------|-------------|---------|
| PostgreSQL | INSERT/UPDATE/DELETE全対応 | - |
| SQLite | INSERT/UPDATE/DELETE全対応 (3.35.0+) | - |
| MySQL | なし | `LAST_INSERT_ID()` / `SELECT` |
| SQL Server | `OUTPUT`句（同等機能） | `SCOPE_IDENTITY()` |
| Oracle | `RETURNING INTO`（PL/SQL内で使用） | シーケンス.CURRVAL |

### Q5: 1つのINSERT文で挿入できる最大行数は？

| RDBMS | 最大行数 | 推奨バッチサイズ |
|-------|---------|----------------|
| PostgreSQL | 制限なし（メモリ依存） | 1,000〜10,000行 |
| MySQL | 制限なし（max_allowed_packet依存） | 1,000行程度 |
| SQLite | 500行（コンパイル時設定） | 500行 |
| SQL Server | 1,000行（リテラル値の場合） | 1,000行 |

---

## まとめ

| 操作 | SQL文 | 要注意点 |
|------|-------|----------|
| CREATE | `INSERT INTO ... VALUES` | 制約違反、重複キー、バッチサイズ |
| READ | `SELECT ... FROM ... WHERE` | 実行順序理解、SELECT *回避、インデックス活用 |
| UPDATE | `UPDATE ... SET ... WHERE` | WHERE句忘れ防止、トランザクション使用 |
| DELETE | `DELETE FROM ... WHERE` | WHERE句忘れ防止、論理/物理削除の選択 |
| UPSERT | `ON CONFLICT DO UPDATE` | 方言差が大きい、冪等性の設計 |
| RETURNING | `INSERT/UPDATE/DELETE ... RETURNING` | PostgreSQL/SQLiteで使用可能 |
| 安全策 | `BEGIN` → 確認 → `COMMIT/ROLLBACK` | 本番操作は必ずトランザクション |
| パラメータバインド | `$1` / `?` / `%s` | SQLインジェクション防止の必須手段 |

---

## 次に読むべきガイド

- [02-joins.md](./02-joins.md) — 複数テーブルを結合するJOINの全種類
- [03-aggregation.md](./03-aggregation.md) — GROUP BYと集約関数によるデータ分析
- [04-subqueries.md](./04-subqueries.md) — サブクエリの活用パターン
- [../01-advanced/02-transactions.md](../01-advanced/02-transactions.md) — トランザクション管理の詳細
- [../../security-fundamentals/docs/01-web-security/](../../security-fundamentals/docs/01-web-security/) — SQLインジェクション対策

---

## 参考文献

1. PostgreSQL Documentation — "Data Manipulation" https://www.postgresql.org/docs/current/dml.html
2. PostgreSQL Documentation — "INSERT" https://www.postgresql.org/docs/current/sql-insert.html
3. MySQL Reference Manual — "Data Manipulation Statements" https://dev.mysql.com/doc/refman/8.0/en/sql-data-manipulation-statements.html
4. Karwin, B. (2010). *SQL Antipatterns: Avoiding the Pitfalls of Database Programming*. Pragmatic Bookshelf.
5. Winand, M. (2012). *SQL Performance Explained*. Markus Winand. https://use-the-index-luke.com/
6. OWASP — "SQL Injection Prevention Cheat Sheet" https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html
