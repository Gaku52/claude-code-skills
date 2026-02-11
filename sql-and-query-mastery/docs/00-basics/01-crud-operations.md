# CRUD操作 — SELECT / INSERT / UPDATE / DELETE

> CRUDはCreate（作成）、Read（読取）、Update（更新）、Delete（削除）の頭文字であり、データベース操作の最も基本的な4つの操作を表す。

## この章で学ぶこと

1. SELECT文の構文と実行順序を正確に理解する
2. INSERT / UPDATE / DELETE の安全な実行パターンを身につける
3. RETURNING句やUPSERTなど実務で頻出する応用パターンを習得する

---

## 1. SELECT — データの読み取り

### コード例1: SELECT文の完全な構文と実行順序

```sql
-- SELECT文の論理的な実行順序（書く順序とは異なる）
--
-- 1. FROM     → テーブルを特定
-- 2. WHERE    → 行をフィルタ
-- 3. GROUP BY → グループ化
-- 4. HAVING   → グループをフィルタ
-- 5. SELECT   → 列を選択・計算
-- 6. DISTINCT → 重複排除
-- 7. ORDER BY → 並べ替え
-- 8. LIMIT    → 件数制限

SELECT DISTINCT
    d.name AS department,
    COUNT(*) AS employee_count,
    AVG(e.salary) AS avg_salary
FROM employees e
    INNER JOIN departments d ON e.department_id = d.id
WHERE e.status = 'active'
GROUP BY d.name
HAVING COUNT(*) >= 3
ORDER BY avg_salary DESC
LIMIT 10;
```

### SELECT文の実行順序

```
┌──────────────────────────────────────────────────────┐
│              SELECT文の論理的実行順序                  │
│                                                      │
│   書く順序              実行順序                      │
│   ─────────            ─────────                     │
│   SELECT   ──────┐    ① FROM / JOIN                  │
│   FROM     ◀─────┤    ② WHERE                        │
│   WHERE    ──────┤    ③ GROUP BY                      │
│   GROUP BY ──────┤    ④ HAVING                        │
│   HAVING   ──────┤    ⑤ SELECT (式の評価)             │
│   ORDER BY ──────┤    ⑥ DISTINCT                      │
│   LIMIT    ──────┘    ⑦ ORDER BY                      │
│                       ⑧ LIMIT / OFFSET                │
│                                                      │
│   ※ WHEREでエイリアスが使えないのは                    │
│     SELECTより先にWHEREが実行されるため               │
└──────────────────────────────────────────────────────┘
```

### コード例2: WHERE句の条件パターン

```sql
-- 比較演算子
SELECT * FROM products WHERE price > 1000;
SELECT * FROM products WHERE price BETWEEN 1000 AND 5000;

-- パターンマッチ
SELECT * FROM users WHERE name LIKE '田中%';          -- 前方一致
SELECT * FROM users WHERE email LIKE '%@gmail.com';   -- 後方一致

-- NULL判定（= NULLは不可！）
SELECT * FROM users WHERE phone IS NULL;
SELECT * FROM users WHERE phone IS NOT NULL;

-- IN / NOT IN
SELECT * FROM orders WHERE status IN ('pending', 'processing');

-- 複合条件
SELECT * FROM products
WHERE (category = 'electronics' OR category = 'books')
  AND price < 5000
  AND stock > 0;
```

---

## 2. INSERT — データの作成

### コード例3: 各種INSERTパターン

```sql
-- 基本的なINSERT（単一行）
INSERT INTO employees (name, department_id, salary, hired_date)
VALUES ('山田太郎', 10, 400000, '2024-04-01');

-- 複数行の一括INSERT
INSERT INTO employees (name, department_id, salary, hired_date)
VALUES
    ('鈴木花子', 20, 450000, '2024-04-01'),
    ('佐藤次郎', 10, 380000, '2024-04-01'),
    ('高橋三郎', 30, 420000, '2024-04-01');

-- SELECTの結果をINSERT
INSERT INTO employee_archive (name, department_id, salary)
SELECT name, department_id, salary
FROM employees
WHERE status = 'retired';

-- RETURNING句で挿入結果を取得（PostgreSQL）
INSERT INTO employees (name, department_id, salary)
VALUES ('新人一号', 10, 350000)
RETURNING id, name, created_at;

-- UPSERT（存在すれば更新、なければ挿入）
-- PostgreSQL
INSERT INTO user_settings (user_id, key, value)
VALUES (1, 'theme', 'dark')
ON CONFLICT (user_id, key)
DO UPDATE SET value = EXCLUDED.value, updated_at = NOW();

-- MySQL
INSERT INTO user_settings (user_id, `key`, value)
VALUES (1, 'theme', 'dark')
ON DUPLICATE KEY UPDATE value = VALUES(value), updated_at = NOW();
```

---

## 3. UPDATE — データの更新

### コード例4: 安全なUPDATEパターン

```sql
-- 基本的なUPDATE
UPDATE employees
SET salary = 500000
WHERE employee_id = 42;

-- 複数列の同時更新
UPDATE employees
SET salary = salary * 1.05,
    updated_at = CURRENT_TIMESTAMP
WHERE department_id = 10
  AND performance_rating >= 4;

-- JOINを使ったUPDATE（PostgreSQL）
UPDATE orders o
SET status = 'cancelled',
    cancelled_at = NOW()
FROM customers c
WHERE o.customer_id = c.id
  AND c.is_banned = TRUE
  AND o.status = 'pending';

-- サブクエリを使ったUPDATE
UPDATE products
SET price = price * 0.9
WHERE category_id IN (
    SELECT id FROM categories WHERE name = 'セール対象'
);

-- RETURNING句で更新結果を確認（PostgreSQL）
UPDATE employees
SET salary = salary * 1.10
WHERE department_id = 20
RETURNING id, name, salary AS new_salary;
```

---

## 4. DELETE — データの削除

### コード例5: 安全なDELETEパターン

```sql
-- 基本的なDELETE
DELETE FROM sessions
WHERE expires_at < CURRENT_TIMESTAMP;

-- JOINを使ったDELETE（PostgreSQL）
DELETE FROM order_items oi
USING orders o
WHERE oi.order_id = o.id
  AND o.status = 'cancelled';

-- RETURNING句で削除した行を取得
DELETE FROM notifications
WHERE user_id = 42 AND read_at IS NOT NULL
RETURNING id, message;

-- 論理削除（ソフトデリート）パターン
-- 物理削除の代わりにフラグを立てる
UPDATE users
SET deleted_at = CURRENT_TIMESTAMP,
    status = 'deleted'
WHERE id = 42;

-- TRUNCATE: テーブル全行の高速削除（DDL操作）
TRUNCATE TABLE temp_import_data;
-- ※ TRUNCATE はロールバック不可（MySQL）、WHERE句使用不可
```

---

## 5. CRUD操作のデータフロー

```
┌─────────────────── CRUD操作のライフサイクル ──────────────────┐
│                                                               │
│  クライアント                    データベース                  │
│                                                               │
│  ┌──────────┐   INSERT INTO    ┌──────────────┐              │
│  │ CREATE   │ ───────────────► │  新規行追加   │              │
│  └──────────┘                  └──────────────┘              │
│                                       │                      │
│  ┌──────────┐   SELECT         ┌──────────────┐              │
│  │ READ     │ ◄─────────────── │  行の読み取り │              │
│  └──────────┘                  └──────────────┘              │
│                                       │                      │
│  ┌──────────┐   UPDATE SET     ┌──────────────┐              │
│  │ UPDATE   │ ───────────────► │  既存行変更   │              │
│  └──────────┘                  └──────────────┘              │
│                                       │                      │
│  ┌──────────┐   DELETE FROM    ┌──────────────┐              │
│  │ DELETE   │ ───────────────► │  行の削除     │              │
│  └──────────┘                  └──────────────┘              │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## DELETE vs TRUNCATE vs DROP 比較表

| 特徴 | DELETE | TRUNCATE | DROP |
|------|--------|----------|------|
| 操作種別 | DML | DDL | DDL |
| WHERE句 | 使用可能 | 使用不可 | 使用不可 |
| ロールバック | 可能 | DB依存 | DB依存 |
| 速度 | 遅い（行単位） | 高速（ページ単位） | 高速 |
| トリガー発火 | する | しない | しない |
| 自動連番リセット | しない | する | N/A |
| 領域解放 | しない（VACUUM必要） | する | する |
| テーブル構造 | 残る | 残る | 消える |

---

## INSERT方式の比較表

| 方式 | 用途 | 速度 | 安全性 |
|------|------|------|--------|
| 単一行INSERT | 個別レコード追加 | 低 | 高 |
| 複数行INSERT | バッチ挿入 | 中 | 高 |
| INSERT...SELECT | データ移行/複製 | 高 | 中 |
| COPY / LOAD DATA | 大量データ投入 | 最速 | 低 |
| UPSERT | 冪等な挿入/更新 | 中 | 高 |

---

## アンチパターン

### アンチパターン1: WHERE句なしのUPDATE/DELETE

```sql
-- NG: 全行が更新されてしまう！
UPDATE employees SET salary = 0;

-- NG: 全行が削除されてしまう！
DELETE FROM employees;

-- OK: 必ずWHERE句で対象を限定
UPDATE employees SET salary = 500000 WHERE employee_id = 42;

-- 安全策: 先にSELECTで確認
-- 1. まず対象を確認
SELECT * FROM employees WHERE department_id = 99;
-- 2. 件数が妥当であることを確認してから実行
DELETE FROM employees WHERE department_id = 99;

-- さらに安全策: トランザクションで囲む
BEGIN;
DELETE FROM employees WHERE department_id = 99;
-- 結果を確認して問題なければ
COMMIT;
-- 問題があれば
-- ROLLBACK;
```

### アンチパターン2: SELECT * の濫用

```sql
-- NG: 不要な列まで全て取得
SELECT * FROM orders;

-- 問題点:
-- 1. ネットワーク帯域の浪費（BLOB/TEXT列含む場合は特に深刻）
-- 2. テーブル構造変更時にアプリケーションが壊れる
-- 3. インデックスオンリースキャンが効かない

-- OK: 必要な列だけ明示的に指定
SELECT order_id, customer_id, total_amount, status
FROM orders
WHERE status = 'pending';
```

---

## FAQ

### Q1: UPDATE文で「変更前の値」を参照できるか？

PostgreSQLではRETURNING句を使えば更新後の値を取得できるが、更新前の値は直接取得できない。更新前の値が必要な場合はCTEを使う:

```sql
WITH old AS (
    SELECT id, salary FROM employees WHERE id = 42
)
UPDATE employees SET salary = salary * 1.1 WHERE id = 42
RETURNING id, (SELECT salary FROM old) AS old_salary, salary AS new_salary;
```

### Q2: 大量データのINSERTを高速化するには？

1. **COPY文（PostgreSQL）/ LOAD DATA（MySQL）** を使う
2. **トランザクションでまとめる**（AUTO COMMITを無効化）
3. **インデックスを一時的に無効化**してからINSERT、完了後に再作成
4. **バッチサイズ**を調整する（1000〜10000行/バッチが目安）

### Q3: 論理削除と物理削除のどちらを使うべきか？

論理削除（ソフトデリート）は復元可能性と監査証跡の面で優れるが、全クエリにWHERE句が必要になり複雑化する。物理削除はシンプルだがデータ復元が困難。多くの実務では論理削除+定期的なアーカイブ/パージが推奨される。

---

## まとめ

| 操作 | SQL文 | 要注意点 |
|------|-------|----------|
| CREATE | `INSERT INTO ... VALUES` | 制約違反、重複キー |
| READ | `SELECT ... FROM ... WHERE` | 実行順序、SELECT *回避 |
| UPDATE | `UPDATE ... SET ... WHERE` | WHERE句忘れ防止 |
| DELETE | `DELETE FROM ... WHERE` | WHERE句忘れ防止、論理/物理削除の選択 |
| UPSERT | `ON CONFLICT DO UPDATE` | 方言差が大きい |
| 安全策 | `BEGIN` → 確認 → `COMMIT/ROLLBACK` | 本番操作は必ずトランザクション |

---

## 次に読むべきガイド

- [02-joins.md](./02-joins.md) — 複数テーブルを結合するJOIN
- [03-aggregation.md](./03-aggregation.md) — GROUP BYと集約関数
- [02-transactions.md](../01-advanced/02-transactions.md) — トランザクション管理

---

## 参考文献

1. PostgreSQL Documentation — "Data Manipulation" https://www.postgresql.org/docs/current/dml.html
2. MySQL Reference Manual — "Data Manipulation Statements" https://dev.mysql.com/doc/refman/8.0/en/sql-data-manipulation-statements.html
3. Karwin, B. (2010). *SQL Antipatterns: Avoiding the Pitfalls of Database Programming*. Pragmatic Bookshelf.
