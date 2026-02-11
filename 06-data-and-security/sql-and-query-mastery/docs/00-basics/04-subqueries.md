# サブクエリ — 相関 / 非相関・EXISTS・IN

> サブクエリはクエリの中にネストされたクエリであり、複雑な条件指定やデータ変換を単一のSQL文で表現する強力な手段である。

## この章で学ぶこと

1. 非相関サブクエリと相関サブクエリの違いと実行モデル
2. EXISTS、IN、スカラーサブクエリの使い分け
3. サブクエリのパフォーマンス特性とJOINへの書き換え判断

---

## 1. サブクエリの分類

```
┌──────────────────── サブクエリの分類 ────────────────────┐
│                                                          │
│  ┌──────────────────────────────────────────┐           │
│  │ 返す結果の形状による分類                   │           │
│  ├──────────────────┬───────────────────────┤           │
│  │ スカラーサブクエリ │ 1行1列（単一値）      │           │
│  │ 行サブクエリ      │ 1行N列               │           │
│  │ テーブルサブクエリ │ N行N列               │           │
│  └──────────────────┴───────────────────────┘           │
│                                                          │
│  ┌──────────────────────────────────────────┐           │
│  │ 外部クエリとの依存関係による分類           │           │
│  ├──────────────────┬───────────────────────┤           │
│  │ 非相関サブクエリ  │ 独立実行可能           │           │
│  │ 相関サブクエリ    │ 外部の行に依存         │           │
│  └──────────────────┴───────────────────────┘           │
│                                                          │
│  ┌──────────────────────────────────────────┐           │
│  │ 使用場所による分類                        │           │
│  ├──────────────────┬───────────────────────┤           │
│  │ WHERE句          │ 条件式として           │           │
│  │ FROM句           │ 派生テーブルとして     │           │
│  │ SELECT句         │ スカラー値として       │           │
│  │ HAVING句         │ グループ条件として     │           │
│  └──────────────────┴───────────────────────┘           │
└──────────────────────────────────────────────────────────┘
```

---

## 2. 非相関サブクエリ

### コード例1: WHERE句での非相関サブクエリ

```sql
-- IN: サブクエリ結果のリストに含まれるか
SELECT name, salary
FROM employees
WHERE department_id IN (
    SELECT id FROM departments WHERE location = '東京'
);

-- 比較演算子: スカラーサブクエリ
SELECT name, salary
FROM employees
WHERE salary > (
    SELECT AVG(salary) FROM employees  -- 全社平均を返す
);

-- ALL / ANY: 集合との比較
-- 全部署の平均給与より高い社員
SELECT name, salary
FROM employees
WHERE salary > ALL (
    SELECT AVG(salary) FROM employees GROUP BY department_id
);

-- いずれかの部署の平均給与より高い社員
SELECT name, salary
FROM employees
WHERE salary > ANY (
    SELECT AVG(salary) FROM employees GROUP BY department_id
);
```

### コード例2: FROM句での非相関サブクエリ（派生テーブル）

```sql
-- 派生テーブル: サブクエリの結果をテーブルとして使用
SELECT
    dept_stats.department_name,
    dept_stats.avg_salary,
    dept_stats.employee_count
FROM (
    SELECT
        d.name AS department_name,
        AVG(e.salary) AS avg_salary,
        COUNT(*) AS employee_count
    FROM employees e
        INNER JOIN departments d ON e.department_id = d.id
    GROUP BY d.name
) AS dept_stats
WHERE dept_stats.avg_salary > 500000
ORDER BY dept_stats.avg_salary DESC;

-- SELECT句でのスカラーサブクエリ
SELECT
    e.name,
    e.salary,
    (SELECT AVG(salary) FROM employees) AS company_avg,
    e.salary - (SELECT AVG(salary) FROM employees) AS diff_from_avg
FROM employees e
ORDER BY diff_from_avg DESC;
```

---

## 3. 相関サブクエリ

### 相関サブクエリの実行モデル

```
┌────────── 相関サブクエリの実行フロー ──────────┐
│                                                 │
│  外部クエリ: SELECT * FROM employees e          │
│  WHERE salary > (...)                           │
│                                                 │
│  外部の各行について:                             │
│  ┌─────────────────────────────────────────┐   │
│  │ 行1: 田中 (dept=10)                     │   │
│  │  → サブクエリ実行: AVG WHERE dept=10    │   │
│  │  → 結果: 450000                         │   │
│  │  → 田中の給与 > 450000 ? → 判定        │   │
│  ├─────────────────────────────────────────┤   │
│  │ 行2: 鈴木 (dept=20)                     │   │
│  │  → サブクエリ実行: AVG WHERE dept=20    │   │
│  │  → 結果: 520000                         │   │
│  │  → 鈴木の給与 > 520000 ? → 判定        │   │
│  ├─────────────────────────────────────────┤   │
│  │ 行3: 佐藤 (dept=10)                     │   │
│  │  → サブクエリ実行: AVG WHERE dept=10    │   │
│  │  → 結果: 450000                         │   │
│  │  → 佐藤の給与 > 450000 ? → 判定        │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  ※ 外部の行数分だけサブクエリが実行される        │
└─────────────────────────────────────────────────┘
```

### コード例3: 相関サブクエリ

```sql
-- 自部署の平均給与より高い社員を取得
SELECT e.name, e.salary, e.department_id
FROM employees e
WHERE e.salary > (
    SELECT AVG(e2.salary)
    FROM employees e2
    WHERE e2.department_id = e.department_id  -- 外部行を参照
);

-- 各カテゴリで最新の商品を取得
SELECT p.name, p.category_id, p.created_at
FROM products p
WHERE p.created_at = (
    SELECT MAX(p2.created_at)
    FROM products p2
    WHERE p2.category_id = p.category_id
);
```

---

## 4. EXISTS / NOT EXISTS

### コード例4: EXISTSの活用

```sql
-- 注文実績のある顧客を取得
SELECT c.name, c.email
FROM customers c
WHERE EXISTS (
    SELECT 1
    FROM orders o
    WHERE o.customer_id = c.id
      AND o.order_date >= '2024-01-01'
);

-- 注文実績のない顧客を取得
SELECT c.name, c.email
FROM customers c
WHERE NOT EXISTS (
    SELECT 1
    FROM orders o
    WHERE o.customer_id = c.id
);

-- EXISTS vs IN の書き換え
-- 以下は論理的に同等（パフォーマンスは異なる場合あり）

-- INバージョン
SELECT * FROM customers
WHERE id IN (SELECT customer_id FROM orders);

-- EXISTSバージョン（大規模データではこちらが有利な場合が多い）
SELECT * FROM customers c
WHERE EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.id);
```

### コード例5: LATERAL JOIN（相関サブクエリのFROM句版）

```sql
-- LATERAL JOIN: 各行に対してサブクエリを実行しFROM句で使用
-- 各部署の給与TOP3を取得
SELECT d.name AS department, top3.name, top3.salary
FROM departments d
    CROSS JOIN LATERAL (
        SELECT e.name, e.salary
        FROM employees e
        WHERE e.department_id = d.id
        ORDER BY e.salary DESC
        LIMIT 3
    ) AS top3;

-- 従来の相関サブクエリでは困難な「各グループのN件」を簡潔に表現
```

### コード例6: 実践的なサブクエリパターン

```sql
-- 全社員を4分位（四分位）に分類
SELECT
    name,
    salary,
    CASE
        WHEN salary >= (SELECT PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY salary) FROM employees)
            THEN 'Q4 (上位25%)'
        WHEN salary >= (SELECT PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY salary) FROM employees)
            THEN 'Q3'
        WHEN salary >= (SELECT PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY salary) FROM employees)
            THEN 'Q2'
        ELSE 'Q1 (下位25%)'
    END AS quartile
FROM employees
ORDER BY salary DESC;

-- 前月比の売上比較
SELECT
    current_month.month,
    current_month.total,
    prev_month.total AS prev_total,
    ROUND((current_month.total - prev_month.total) / prev_month.total * 100, 1) AS growth_pct
FROM (
    SELECT DATE_TRUNC('month', sale_date) AS month, SUM(amount) AS total
    FROM sales GROUP BY 1
) current_month
LEFT JOIN (
    SELECT DATE_TRUNC('month', sale_date) AS month, SUM(amount) AS total
    FROM sales GROUP BY 1
) prev_month ON current_month.month = prev_month.month + INTERVAL '1 month'
ORDER BY current_month.month;
```

---

## IN vs EXISTS vs JOIN 比較表

| 手法 | 最適な場面 | NULLの扱い | パフォーマンス |
|------|-----------|-----------|--------------|
| IN | サブクエリの結果が少量 | NULLがあると問題 | 小テーブル向き |
| NOT IN | — | NULL含むと全行除外の危険 | 非推奨 |
| EXISTS | サブクエリの結果が大量 | NULL問題なし | 大テーブル向き |
| NOT EXISTS | 存在しない行の検索 | NULL安全 | 推奨 |
| JOIN | 結合データが必要 | 明示的に制御可能 | 最も柔軟 |

## サブクエリの使用場所比較表

| 使用場所 | 返す形状 | 用途 | 例 |
|---------|---------|------|-----|
| WHERE句 | スカラー/リスト | フィルタ条件 | `WHERE x IN (SELECT ...)` |
| FROM句 | テーブル | 派生テーブル | `FROM (SELECT ...) AS t` |
| SELECT句 | スカラー | 計算列 | `SELECT (SELECT AVG(...))` |
| HAVING句 | スカラー | グループフィルタ | `HAVING COUNT(*) > (SELECT ...)` |
| INSERT INTO | テーブル | データ移行 | `INSERT INTO ... SELECT` |

---

## アンチパターン

### アンチパターン1: NOT INとNULLの落とし穴

```sql
-- NG: NULLが含まれるとNOT INは全行を除外する
SELECT * FROM employees
WHERE department_id NOT IN (
    SELECT department_id FROM temp_exclusions
    -- temp_exclusionsにNULLが1行でもあると、結果は0行！
);

-- 理由: NULL との比較は常にUNKNOWN
-- x NOT IN (1, 2, NULL) → x<>1 AND x<>2 AND x<>NULL
-- → ... AND UNKNOWN → UNKNOWN → フィルタされない

-- OK: NOT EXISTS を使う
SELECT * FROM employees e
WHERE NOT EXISTS (
    SELECT 1 FROM temp_exclusions t
    WHERE t.department_id = e.department_id
);
```

### アンチパターン2: SELECT句の相関サブクエリの濫用

```sql
-- NG: 行ごとにサブクエリが実行される（N+1問題と同等）
SELECT
    e.name,
    (SELECT d.name FROM departments d WHERE d.id = e.department_id) AS dept_name,
    (SELECT COUNT(*) FROM projects p WHERE p.lead_id = e.id) AS project_count,
    (SELECT MAX(r.rating) FROM reviews r WHERE r.employee_id = e.id) AS best_rating
FROM employees e;

-- OK: JOINと集約で1クエリに
SELECT
    e.name,
    d.name AS dept_name,
    COUNT(DISTINCT p.id) AS project_count,
    MAX(r.rating) AS best_rating
FROM employees e
    LEFT JOIN departments d ON d.id = e.department_id
    LEFT JOIN projects p ON p.lead_id = e.id
    LEFT JOIN reviews r ON r.employee_id = e.id
GROUP BY e.id, e.name, d.name;
```

---

## FAQ

### Q1: サブクエリとCTE（WITH句）のどちらを使うべきか？

同じサブクエリを複数回参照する場合はCTEが適している（DRY原則）。1回だけ使用するならサブクエリでも問題ない。可読性の面ではCTEが優れることが多い。パフォーマンスはRDBMS依存だが、PostgreSQL 12以降はCTEのインライン展開が行われるため差は小さい。

### Q2: 相関サブクエリは常に遅いのか？

必ずしも遅くない。オプティマイザが内部的にJOINに変換することがある。ただし、外部テーブルの行数が多い場合はJOINやウィンドウ関数への書き換えを検討すべき。EXPLAINで実行計画を確認することが重要。

### Q3: サブクエリのネストは何段まで許容されるか？

技術的な制限はRDBMSによるが、可読性の観点から2段以内を推奨。3段以上のネストはCTEやビューに分解して可読性を確保する。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 非相関サブクエリ | 外部クエリと独立。1回だけ実行される |
| 相関サブクエリ | 外部の各行に依存。行数分実行される可能性 |
| EXISTS | 存在確認に最適。NULL安全 |
| NOT IN | NULLの罠がある。NOT EXISTSを推奨 |
| LATERAL JOIN | FROM句での相関サブクエリ。Top-N問題に有効 |
| パフォーマンス | EXPLAINで実行計画を確認。JOINへの書き換えを検討 |

---

## 次に読むべきガイド

- [00-window-functions.md](../01-advanced/00-window-functions.md) — ウィンドウ関数でサブクエリを置換
- [01-cte-recursive.md](../01-advanced/01-cte-recursive.md) — CTEで複雑なサブクエリを整理
- [04-query-optimization.md](../01-advanced/04-query-optimization.md) — サブクエリの最適化

---

## 参考文献

1. PostgreSQL Documentation — "Subquery Expressions" https://www.postgresql.org/docs/current/functions-subquery.html
2. Celko, J. (2010). *Joe Celko's SQL for Smarties*. Morgan Kaufmann.
3. Winand, M. (2012). *SQL Performance Explained*. Markus Winand. https://use-the-index-luke.com/
