# サブクエリ — 相関 / 非相関・EXISTS・IN

> サブクエリはクエリの中にネストされたクエリであり、複雑な条件指定やデータ変換を単一のSQL文で表現する強力な手段である。

## この章で学ぶこと

1. 非相関サブクエリと相関サブクエリの違いと実行モデル
2. EXISTS、IN、スカラーサブクエリの使い分け
3. サブクエリのパフォーマンス特性とJOINへの書き換え判断
4. LATERAL JOINとサブクエリの関係
5. オプティマイザによるサブクエリの内部変換メカニズム

## 前提知識

- SQLの基本構文（SELECT、WHERE、JOIN）
- 集約関数（COUNT、SUM、AVG）の基礎
- [03-joins.md](./03-joins.md) でJOINの種類を理解していること

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
│  │ INSERT INTO ... SELECT │ データ移行として │           │
│  │ UPDATE ... SET   │ 値の算出として         │           │
│  │ DELETE ... WHERE │ 削除条件として         │           │
│  └──────────────────┴───────────────────────┘           │
└──────────────────────────────────────────────────────────┘
```

### 1.1 サブクエリの実行モデル概要

```
┌──────── サブクエリの実行モデル ────────────────┐
│                                                  │
│  非相関サブクエリの実行:                         │
│  ┌──────────────────────────────┐               │
│  │ Step 1: サブクエリを1回実行   │               │
│  │ Step 2: 結果をメモリに保持   │               │
│  │ Step 3: 外部クエリで結果を参照│               │
│  │                                │               │
│  │ 計算量: O(M) + O(N)           │               │
│  │ M = サブクエリの処理行数       │               │
│  │ N = 外部クエリの処理行数       │               │
│  └──────────────────────────────┘               │
│                                                  │
│  相関サブクエリの実行（ナイーブな場合）:          │
│  ┌──────────────────────────────┐               │
│  │ 外部の各行について:           │               │
│  │   → サブクエリを実行          │               │
│  │   → 結果で外部行を評価        │               │
│  │                                │               │
│  │ 計算量: O(N * M)（最悪ケース） │               │
│  │ ※ オプティマイザが最適化する    │               │
│  │   場合は O(N + M) に改善       │               │
│  └──────────────────────────────┘               │
└──────────────────────────────────────────────────┘
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

-- BETWEEN とサブクエリの組み合わせ
SELECT name, salary
FROM employees
WHERE salary BETWEEN
    (SELECT AVG(salary) - STDDEV(salary) FROM employees)
    AND
    (SELECT AVG(salary) + STDDEV(salary) FROM employees);
-- → 標準偏差の範囲内にいる社員
```

#### ALL / ANY の内部動作

```
┌──────── ALL / ANY の論理的展開 ──────────────┐
│                                                │
│  salary > ALL (10, 20, 30)                     │
│  ≡ salary > 10 AND salary > 20 AND salary > 30│
│  ≡ salary > MAX(10, 20, 30)                    │
│  ≡ salary > 30                                 │
│                                                │
│  salary > ANY (10, 20, 30)                     │
│  ≡ salary > 10 OR salary > 20 OR salary > 30   │
│  ≡ salary > MIN(10, 20, 30)                    │
│  ≡ salary > 10                                 │
│                                                │
│  注意: 空集合の場合                             │
│  salary > ALL (空) → TRUE （全ての要素が条件を   │
│                      満たす = 要素なし = 真）    │
│  salary > ANY (空) → FALSE                     │
│                                                │
│  注意: NULLが含まれる場合                       │
│  salary > ALL (10, NULL, 30)                    │
│  → salary > 10 AND salary > NULL AND salary > 30│
│  → ... AND UNKNOWN AND ...                     │
│  → 結果がUNKNOWN → フィルタされない            │
└────────────────────────────────────────────────┘
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

-- 派生テーブルの中でウィンドウ関数を使用
SELECT
    ranked.name,
    ranked.salary,
    ranked.department_name,
    ranked.salary_rank
FROM (
    SELECT
        e.name,
        e.salary,
        d.name AS department_name,
        RANK() OVER (PARTITION BY d.id ORDER BY e.salary DESC) AS salary_rank
    FROM employees e
        JOIN departments d ON e.department_id = d.id
) AS ranked
WHERE ranked.salary_rank <= 3;
-- → 各部署の給与TOP3
```

#### 派生テーブルとCTEの比較

```sql
-- 派生テーブルで書いた場合
SELECT dept_name, avg_salary
FROM (
    SELECT d.name AS dept_name, AVG(e.salary) AS avg_salary
    FROM employees e JOIN departments d ON e.department_id = d.id
    GROUP BY d.name
) AS dept_stats
WHERE avg_salary > 500000;

-- CTEで書いた場合（同等の結果）
WITH dept_stats AS (
    SELECT d.name AS dept_name, AVG(e.salary) AS avg_salary
    FROM employees e JOIN departments d ON e.department_id = d.id
    GROUP BY d.name
)
SELECT dept_name, avg_salary
FROM dept_stats
WHERE avg_salary > 500000;

-- CTEの利点:
-- 1. 同じサブクエリを複数回参照できる
-- 2. 可読性が高い（上から下に読める）
-- 3. 再帰クエリが書ける

-- パフォーマンスの違い（PostgreSQL 12以降）:
-- PostgreSQL 12以降ではCTEがインライン展開される（NOT MATERIALIZED）
-- → パフォーマンスは派生テーブルとほぼ同等
-- 強制的にマテリアライズしたい場合: WITH x AS MATERIALIZED (...)
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
│  │  → 結果: 450000（キャッシュ可能）       │   │
│  │  → 佐藤の給与 > 450000 ? → 判定        │   │
│  └─────────────────────────────────────────┘   │
│                                                 │
│  ※ 外部の行数分だけサブクエリが実行される        │
│  ※ ただしオプティマイザが最適化する場合がある    │
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

-- 同等のウィンドウ関数版（通常こちらが推奨）
SELECT name, category_id, created_at
FROM (
    SELECT
        name, category_id, created_at,
        ROW_NUMBER() OVER (PARTITION BY category_id ORDER BY created_at DESC) AS rn
    FROM products
) sub
WHERE rn = 1;

-- 相関サブクエリでの更新
UPDATE employees e
SET salary = salary * 1.1
WHERE salary < (
    SELECT AVG(e2.salary)
    FROM employees e2
    WHERE e2.department_id = e.department_id
);
-- → 自部署の平均以下の社員に10%昇給

-- 相関サブクエリでの削除
DELETE FROM order_items oi
WHERE NOT EXISTS (
    SELECT 1 FROM orders o
    WHERE o.id = oi.order_id
    AND o.status != 'cancelled'
);
-- → キャンセル済み注文の明細を削除
```

### 3.1 オプティマイザによる相関サブクエリの変換

```
┌──── オプティマイザのサブクエリ最適化戦略 ─────┐
│                                                 │
│  1. サブクエリの非ネスト化（Unnesting）          │
│     相関サブクエリ → JOINに変換                  │
│     例:                                         │
│     SELECT * FROM t1                            │
│     WHERE t1.x IN (SELECT t2.x FROM t2          │
│                     WHERE t2.y = t1.y)          │
│     →                                           │
│     SELECT t1.* FROM t1                         │
│     SEMI JOIN t2 ON t1.x = t2.x AND t1.y = t2.y│
│                                                 │
│  2. サブクエリのマテリアライズ                    │
│     非相関サブクエリの結果をハッシュテーブル化    │
│     → 外部クエリの各行でO(1)参照                │
│                                                 │
│  3. EXISTS → SEMI JOINへの変換                  │
│     PostgreSQLではEXISTSを内部的にSEMI JOINに    │
│     変換してHash Semi Joinを使用                │
│                                                 │
│  4. スカラーサブクエリのキャッシュ               │
│     同じパラメータの相関サブクエリ結果を          │
│     再利用（dept_id=10が2回出たらキャッシュ使用）│
│                                                 │
│  確認方法: EXPLAIN ANALYZE で実行計画を参照      │
└─────────────────────────────────────────────────┘
```

```sql
-- オプティマイザの変換を確認する例
EXPLAIN ANALYZE
SELECT e.name
FROM employees e
WHERE e.department_id IN (
    SELECT d.id FROM departments d WHERE d.location = '東京'
);

-- PostgreSQLの実行計画（変換後）:
-- Hash Semi Join  (actual time=0.050..1.200 rows=500 loops=1)
--   Hash Cond: (e.department_id = d.id)
--   -> Seq Scan on employees e
--   -> Hash
--     -> Seq Scan on departments d
--       Filter: (location = '東京')
-- → INサブクエリがSEMI JOINに自動変換されている
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

-- 複数条件のEXISTS
-- 「未出荷の注文がある」かつ「VIP顧客」
SELECT c.name, c.email
FROM customers c
WHERE c.tier = 'VIP'
  AND EXISTS (
    SELECT 1 FROM orders o
    WHERE o.customer_id = c.id
      AND o.status = 'pending'
      AND o.total_amount > 10000
  );
```

#### EXISTSの内部動作

```
┌──────── EXISTSの実行メカニズム ─────────────┐
│                                                │
│  EXISTS (SELECT 1 FROM orders WHERE ...)       │
│                                                │
│  動作:                                         │
│  1. サブクエリを実行開始                        │
│  2. 1行でも見つかった時点で TRUE を返す        │
│  3. 残りの行は評価しない（短絡評価）           │
│                                                │
│  → SELECT 1 でも SELECT * でも結果は同じ       │
│    （列の内容は評価されない）                   │
│                                                │
│  NOT EXISTS:                                   │
│  1. サブクエリを実行                            │
│  2. 1行も見つからなければ TRUE を返す          │
│  3. 1行でも見つかった時点で FALSE を返す       │
│                                                │
│  パフォーマンスのポイント:                      │
│  ┌──────────────────────────────────┐         │
│  │ サブクエリ側のWHERE条件に         │         │
│  │ インデックスがあれば非常に高速    │         │
│  │ Index Scan → 1行見つけて即終了   │         │
│  └──────────────────────────────────┘         │
└────────────────────────────────────────────────┘
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

-- LATERAL JOINの別例: 最新の注文情報を横に並べる
SELECT
    c.name,
    c.email,
    latest.order_date,
    latest.total_amount
FROM customers c
    LEFT JOIN LATERAL (
        SELECT o.order_date, o.total_amount
        FROM orders o
        WHERE o.customer_id = c.id
        ORDER BY o.order_date DESC
        LIMIT 1
    ) AS latest ON TRUE;
-- LEFT JOIN LATERAL ... ON TRUE で顧客に注文がない場合もNULLで返す

-- LATERALで時系列データの前の行を参照
SELECT
    m.month,
    m.revenue,
    prev.revenue AS prev_revenue,
    ROUND((m.revenue - prev.revenue) / prev.revenue * 100, 1) AS growth_pct
FROM monthly_sales m
    LEFT JOIN LATERAL (
        SELECT revenue
        FROM monthly_sales
        WHERE month = m.month - INTERVAL '1 month'
    ) prev ON TRUE
ORDER BY m.month;
```

#### LATERAL JOINの実行計画

```
┌──── LATERAL JOIN vs 相関サブクエリの実行計画比較 ───┐
│                                                      │
│  LATERAL JOIN:                                       │
│  Nested Loop                                         │
│    -> Seq Scan on departments d                      │
│    -> Limit                                          │
│      -> Index Scan Backward on employees e           │
│           Index Cond: (department_id = d.id)         │
│           Sort: salary DESC                          │
│  → 部署ごとにインデックスでTOP3を取得               │
│  → LIMITが効くので部署あたり最大3行のみ読む         │
│                                                      │
│  相関サブクエリ（SELECT句）で同等のことをする場合:    │
│  → 列ごとにサブクエリが必要 → 非常に非効率          │
│  → TOP-Nパターンには LATERAL が最適                  │
└──────────────────────────────────────────────────────┘
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

-- ウィンドウ関数版（推奨: サブクエリが1回で済む）
SELECT
    name,
    salary,
    NTILE(4) OVER (ORDER BY salary) AS quartile_num,
    CASE NTILE(4) OVER (ORDER BY salary)
        WHEN 4 THEN 'Q4 (上位25%)'
        WHEN 3 THEN 'Q3'
        WHEN 2 THEN 'Q2'
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

-- CTEで書き直した版（DRY原則、推奨）
WITH monthly AS (
    SELECT DATE_TRUNC('month', sale_date) AS month, SUM(amount) AS total
    FROM sales GROUP BY 1
)
SELECT
    c.month,
    c.total,
    p.total AS prev_total,
    ROUND((c.total - p.total) / p.total * 100, 1) AS growth_pct
FROM monthly c
LEFT JOIN monthly p ON c.month = p.month + INTERVAL '1 month'
ORDER BY c.month;
```

### コード例7: 高度なサブクエリパターン

```sql
-- パターン1: 存在確認と条件付き集約
-- 「直近30日に注文があり、かつ返品がない優良顧客」
SELECT c.id, c.name, c.email,
    (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.id
     AND o.order_date >= CURRENT_DATE - INTERVAL '30 days') AS recent_orders,
    (SELECT SUM(o.total_amount) FROM orders o WHERE o.customer_id = c.id
     AND o.order_date >= CURRENT_DATE - INTERVAL '30 days') AS recent_total
FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o
    WHERE o.customer_id = c.id
    AND o.order_date >= CURRENT_DATE - INTERVAL '30 days'
)
AND NOT EXISTS (
    SELECT 1 FROM returns r
    JOIN orders o ON r.order_id = o.id
    WHERE o.customer_id = c.id
    AND r.return_date >= CURRENT_DATE - INTERVAL '30 days'
);

-- パターン2: 行値コンストラクタとサブクエリ
-- 複数列の同時比較
SELECT * FROM employees
WHERE (department_id, salary) IN (
    SELECT department_id, MAX(salary)
    FROM employees
    GROUP BY department_id
);
-- → 各部署の最高給与者を取得

-- パターン3: INSERT ... SELECT サブクエリ
-- アーカイブテーブルへの移行
INSERT INTO orders_archive (id, customer_id, order_date, total_amount)
SELECT id, customer_id, order_date, total_amount
FROM orders
WHERE order_date < CURRENT_DATE - INTERVAL '2 years'
  AND status = 'delivered';

-- パターン4: UPDATE with サブクエリ
-- 各商品の平均評価を更新
UPDATE products p
SET avg_rating = sub.avg_rating,
    review_count = sub.review_count
FROM (
    SELECT
        product_id,
        AVG(rating)::DECIMAL(3,2) AS avg_rating,
        COUNT(*) AS review_count
    FROM reviews
    GROUP BY product_id
) sub
WHERE p.id = sub.product_id;

-- パターン5: DELETE with サブクエリ
-- 重複行の削除（最小IDを残す）
DELETE FROM employees
WHERE id NOT IN (
    SELECT MIN(id)
    FROM employees
    GROUP BY name, department_id, salary
);

-- より安全な書き方（NOT EXISTS版）
DELETE FROM employees e1
WHERE EXISTS (
    SELECT 1 FROM employees e2
    WHERE e2.name = e1.name
      AND e2.department_id = e1.department_id
      AND e2.salary = e1.salary
      AND e2.id < e1.id
);
```

---

## 5. サブクエリのパフォーマンス分析

### 5.1 実行計画の比較

```sql
-- テストデータの前提: employees 10万行, departments 50行

-- パターン1: INサブクエリ
EXPLAIN ANALYZE
SELECT * FROM employees
WHERE department_id IN (
    SELECT id FROM departments WHERE location = '東京'
);
-- 実行計画:
-- Hash Semi Join  (cost=1.63..2500.00 rows=20000)
--   (actual time=0.050..25.000 rows=20000 loops=1)
--   Hash Cond: (employees.department_id = departments.id)
--   -> Seq Scan on employees
--   -> Hash
--     -> Seq Scan on departments
--       Filter: (location = '東京')
-- Execution Time: 25.500 ms

-- パターン2: EXISTS
EXPLAIN ANALYZE
SELECT * FROM employees e
WHERE EXISTS (
    SELECT 1 FROM departments d
    WHERE d.id = e.department_id AND d.location = '東京'
);
-- 実行計画（PostgreSQLでは同じ計画になることが多い）:
-- Hash Semi Join  (cost=1.63..2500.00 rows=20000)
--   (actual time=0.050..25.000 rows=20000 loops=1)
-- → オプティマイザが同じ実行計画に変換

-- パターン3: JOIN
EXPLAIN ANALYZE
SELECT e.* FROM employees e
JOIN departments d ON e.department_id = d.id
WHERE d.location = '東京';
-- 実行計画:
-- Hash Join  (cost=1.63..2500.00 rows=20000)
--   (actual time=0.050..24.000 rows=20000 loops=1)
-- → ほぼ同じ実行計画（JOINは重複の可能性がある点に注意）
```

### 5.2 RDBMS間のサブクエリ最適化の差異

```
┌──── RDBMS間のサブクエリ最適化比較 ────────────┐
│                                                  │
│  PostgreSQL:                                     │
│  - IN → Semi Join に自動変換                     │
│  - EXISTS → Semi Join に自動変換                 │
│  - 相関サブクエリ → 可能ならJOINに変換           │
│  - CTE 12+: デフォルトでインライン展開           │
│  - LATERAL: Nested Loop で効率的に実行           │
│                                                  │
│  MySQL:                                          │
│  - 5.6以前: INサブクエリの最適化が弱い           │
│  - 5.6+: Semi Join最適化を導入                   │
│  - 8.0+: 派生テーブルのマージ最適化              │
│  - 8.0+: CTE のサポート追加                      │
│  - LATERAL: 8.0.14+でサポート                    │
│                                                  │
│  Oracle:                                         │
│  - 非常に高度なサブクエリ非ネスト化              │
│  - UNNEST / NO_UNNEST ヒントで制御可能           │
│  - スカラーサブクエリのキャッシュが強力           │
│                                                  │
│  SQL Server:                                     │
│  - Apply演算子でLATERAL相当を実装                │
│  - サブクエリの非ネスト化が自動的                │
│  - OPTION (RECOMPILE) で再最適化                 │
└──────────────────────────────────────────────────┘
```

---

## IN vs EXISTS vs JOIN 比較表

| 手法 | 最適な場面 | NULLの扱い | パフォーマンス | オプティマイザ変換 |
|------|-----------|-----------|--------------|------------------|
| IN | サブクエリの結果が少量 | NULLがあると問題 | 小テーブル向き | Semi Join化 |
| NOT IN | — | NULL含むと全行除外の危険 | 非推奨 | Anti Join化（不完全） |
| EXISTS | サブクエリの結果が大量 | NULL問題なし | 大テーブル向き | Semi Join化 |
| NOT EXISTS | 存在しない行の検索 | NULL安全 | 推奨 | Anti Join化 |
| JOIN | 結合データが必要 | 明示的に制御可能 | 最も柔軟 | そのまま実行 |
| LATERAL | 各行でTOP-N | N/A | TOP-Nに最適 | Nested Loop |

## サブクエリの使用場所比較表

| 使用場所 | 返す形状 | 用途 | 例 | パフォーマンス注意点 |
|---------|---------|------|-----|------------------|
| WHERE句 | スカラー/リスト | フィルタ条件 | `WHERE x IN (SELECT ...)` | インデックスの有無が重要 |
| FROM句 | テーブル | 派生テーブル | `FROM (SELECT ...) AS t` | マテリアライズのコスト |
| SELECT句 | スカラー | 計算列 | `SELECT (SELECT AVG(...))` | N+1問題の原因になりうる |
| HAVING句 | スカラー | グループフィルタ | `HAVING COUNT(*) > (SELECT ...)` | 集約後のフィルタ |
| INSERT INTO | テーブル | データ移行 | `INSERT INTO ... SELECT` | バルク操作の性能 |
| UPDATE SET | スカラー | 値の更新 | `SET x = (SELECT ...)` | 相関の場合行数に注意 |
| DELETE WHERE | ブーリアン | 削除条件 | `WHERE EXISTS (SELECT ...)` | インデックスの有無が重要 |

## サブクエリ vs 代替手法 比較表

| 要件 | サブクエリ | JOIN | ウィンドウ関数 | CTE | 推奨 |
|------|-----------|------|-------------|-----|------|
| フィルタ条件 | WHERE IN/EXISTS | JOIN + DISTINCT | - | WITH | EXISTS推奨 |
| 各グループのTOP-N | 相関 + LIMIT | - | ROW_NUMBER | WITH | LATERAL推奨 |
| 集約値との比較 | スカラーサブクエリ | 自己JOIN | ウィンドウ集約 | WITH | ウィンドウ関数推奨 |
| 前月比較 | 自己JOIN | LAG対応 | LAG() | WITH | ウィンドウ関数推奨 |
| 存在確認 | EXISTS | LEFT JOIN IS NULL | - | WITH + EXISTS | EXISTS推奨 |
| データ移行 | INSERT SELECT | - | - | INSERT WITH | サブクエリ推奨 |

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

-- OK: NOT IN でもNULLを除外すれば安全
SELECT * FROM employees
WHERE department_id NOT IN (
    SELECT department_id FROM temp_exclusions
    WHERE department_id IS NOT NULL
);
```

```
┌──── NOT IN のNULL問題の図解 ──────────────────┐
│                                                 │
│  employees:                                     │
│  dept_id: 10, 20, 30                            │
│                                                 │
│  temp_exclusions:                                │
│  dept_id: 10, NULL                               │
│                                                 │
│  NOT IN の展開:                                  │
│  dept_id NOT IN (10, NULL)                       │
│  = dept_id <> 10 AND dept_id <> NULL            │
│                                                 │
│  dept_id=20 の場合:                              │
│  20 <> 10 → TRUE                                │
│  20 <> NULL → UNKNOWN                           │
│  TRUE AND UNKNOWN → UNKNOWN                     │
│  → WHERE の結果は UNKNOWN → 行はフィルタされる  │
│                                                 │
│  dept_id=30 の場合:                              │
│  30 <> 10 → TRUE                                │
│  30 <> NULL → UNKNOWN                           │
│  TRUE AND UNKNOWN → UNKNOWN                     │
│  → 同様にフィルタされる                          │
│                                                 │
│  結果: 0行が返る（意図しない動作）               │
└─────────────────────────────────────────────────┘
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
-- → 社員1000人なら 1000 * 3 = 3000回のサブクエリ実行

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

-- OK: LATERAL JOINで集約を分離（JOINによる行爆発を防ぐ）
SELECT
    e.name,
    d.name AS dept_name,
    pc.project_count,
    br.best_rating
FROM employees e
    LEFT JOIN departments d ON d.id = e.department_id
    LEFT JOIN LATERAL (
        SELECT COUNT(*) AS project_count
        FROM projects p WHERE p.lead_id = e.id
    ) pc ON TRUE
    LEFT JOIN LATERAL (
        SELECT MAX(rating) AS best_rating
        FROM reviews r WHERE r.employee_id = e.id
    ) br ON TRUE;
```

### アンチパターン3: 不要なサブクエリのネスト

```sql
-- NG: 不必要にネストされたサブクエリ
SELECT * FROM (
    SELECT * FROM (
        SELECT * FROM (
            SELECT id, name, salary, department_id
            FROM employees
            WHERE salary > 500000
        ) AS step1
        WHERE department_id IN (10, 20, 30)
    ) AS step2
    ORDER BY salary DESC
) AS step3
LIMIT 10;

-- OK: 1つのクエリに統合
SELECT id, name, salary, department_id
FROM employees
WHERE salary > 500000
  AND department_id IN (10, 20, 30)
ORDER BY salary DESC
LIMIT 10;
```

---

## エッジケース

### エッジケース1: 空の結果セットとの比較

```sql
-- ALL で空集合との比較
-- salary > ALL (空集合) → TRUE（全ての要素を満たす = 空虚な真）
SELECT * FROM employees
WHERE salary > ALL (
    SELECT salary FROM employees WHERE department_id = 999
    -- department_id=999が存在しない場合、空集合
);
-- → 全社員が返される

-- ANY で空集合との比較
-- salary > ANY (空集合) → FALSE
SELECT * FROM employees
WHERE salary > ANY (
    SELECT salary FROM employees WHERE department_id = 999
);
-- → 0行が返される

-- IN で空集合
-- department_id IN (空集合) → FALSE
SELECT * FROM employees
WHERE department_id IN (
    SELECT id FROM departments WHERE location = '火星'
);
-- → 0行が返される
```

### エッジケース2: 複数行を返すスカラーサブクエリ

```sql
-- スカラーサブクエリが複数行を返すとエラー
SELECT name, (
    SELECT salary FROM employees WHERE department_id = 10
    -- 部署10に複数社員がいるとエラー！
) AS salary
FROM departments;
-- → ERROR: more than one row returned by a subquery

-- 対策1: 集約関数で1行に確定
SELECT name, (
    SELECT AVG(salary) FROM employees e WHERE e.department_id = d.id
) AS avg_salary
FROM departments d;

-- 対策2: LIMIT 1 で強制的に1行
SELECT name, (
    SELECT salary FROM employees e
    WHERE e.department_id = d.id
    ORDER BY salary DESC LIMIT 1
) AS max_salary
FROM departments d;
```

### エッジケース3: 自己参照サブクエリ

```sql
-- 自分自身のテーブルを参照するサブクエリ
-- 「同じ部署の全員より給与が高い社員」= 各部署の最高給与者
SELECT e.name, e.salary, e.department_id
FROM employees e
WHERE e.salary >= ALL (
    SELECT e2.salary FROM employees e2
    WHERE e2.department_id = e.department_id
);

-- 注意: 部署に1人しかいない場合も正しく動作する
-- （自分自身との比較: salary >= salary → TRUE）

-- 「自分より給与が高い同僚がいない社員」（同等だが NOT EXISTS版）
SELECT e.name, e.salary, e.department_id
FROM employees e
WHERE NOT EXISTS (
    SELECT 1 FROM employees e2
    WHERE e2.department_id = e.department_id
      AND e2.salary > e.salary
);
```

---

## 演習

### 演習1（基礎）: サブクエリの書き換え

以下のINサブクエリをEXISTS、JOIN、CTEの3パターンに書き換えよ。

```sql
-- 元のクエリ
SELECT * FROM products
WHERE category_id IN (
    SELECT id FROM categories WHERE is_active = TRUE
);
```

<details>
<summary>解答例</summary>

```sql
-- EXISTS版
SELECT * FROM products p
WHERE EXISTS (
    SELECT 1 FROM categories c
    WHERE c.id = p.category_id AND c.is_active = TRUE
);

-- JOIN版
SELECT p.* FROM products p
JOIN categories c ON p.category_id = c.id
WHERE c.is_active = TRUE;

-- CTE版
WITH active_categories AS (
    SELECT id FROM categories WHERE is_active = TRUE
)
SELECT p.* FROM products p
JOIN active_categories ac ON p.category_id = ac.id;
```
</details>

### 演習2（応用）: 複合条件のサブクエリ

以下の要件を1つのSQLで実現せよ。

**要件**: 各部署で「給与が部署平均以上」かつ「勤続年数が5年以上」の社員を取得し、部署平均給与と社員の給与の差分も表示する。

<details>
<summary>解答例</summary>

```sql
-- 方法1: 相関サブクエリ
SELECT
    e.name,
    e.salary,
    e.department_id,
    e.salary - (
        SELECT AVG(e2.salary)
        FROM employees e2
        WHERE e2.department_id = e.department_id
    ) AS diff_from_dept_avg
FROM employees e
WHERE e.salary >= (
    SELECT AVG(e2.salary)
    FROM employees e2
    WHERE e2.department_id = e.department_id
)
AND e.hired_date <= CURRENT_DATE - INTERVAL '5 years';

-- 方法2: ウィンドウ関数（推奨）
SELECT name, salary, department_id, diff_from_dept_avg
FROM (
    SELECT
        e.name,
        e.salary,
        e.department_id,
        e.hired_date,
        e.salary - AVG(e.salary) OVER (PARTITION BY e.department_id) AS diff_from_dept_avg,
        AVG(e.salary) OVER (PARTITION BY e.department_id) AS dept_avg
    FROM employees e
) sub
WHERE salary >= dept_avg
  AND hired_date <= CURRENT_DATE - INTERVAL '5 years';

-- 方法3: CTEで段階的に
WITH dept_avg AS (
    SELECT department_id, AVG(salary) AS avg_salary
    FROM employees
    GROUP BY department_id
)
SELECT
    e.name,
    e.salary,
    e.department_id,
    e.salary - da.avg_salary AS diff_from_dept_avg
FROM employees e
    JOIN dept_avg da ON e.department_id = da.department_id
WHERE e.salary >= da.avg_salary
  AND e.hired_date <= CURRENT_DATE - INTERVAL '5 years';
```
</details>

### 演習3（発展）: パフォーマンス最適化

以下のクエリを、EXPLAIN ANALYZEの出力を参考に最適化せよ。

```sql
-- 最適化前のクエリ（遅い）
SELECT
    c.name,
    (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.id) AS order_count,
    (SELECT SUM(oi.quantity * oi.unit_price)
     FROM order_items oi
     JOIN orders o ON oi.order_id = o.id
     WHERE o.customer_id = c.id) AS lifetime_value,
    (SELECT MAX(o.order_date) FROM orders o WHERE o.customer_id = c.id) AS last_order
FROM customers c
WHERE (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.id) > 5;
-- → SELECT句で4つの相関サブクエリ + WHERE句で1つ = 5回のサブクエリ
```

**ヒント**: SELECT句のサブクエリをLATERAL JOINまたはCTEに統合する。

<details>
<summary>解答例</summary>

```sql
-- 最適化版: CTEとJOINで統合
WITH customer_stats AS (
    SELECT
        o.customer_id,
        COUNT(*) AS order_count,
        MAX(o.order_date) AS last_order,
        SUM(oi.quantity * oi.unit_price) AS lifetime_value
    FROM orders o
        JOIN order_items oi ON o.id = oi.order_id
    GROUP BY o.customer_id
    HAVING COUNT(*) > 5  -- フィルタ条件をここに移動
)
SELECT
    c.name,
    cs.order_count,
    cs.lifetime_value,
    cs.last_order
FROM customers c
    JOIN customer_stats cs ON c.id = cs.customer_id;

-- 改善効果:
-- Before: 顧客数 × 5回のサブクエリ = O(N * 5M)
-- After:  orders/order_itemsを1回スキャン + 1回JOIN = O(M + N)
-- 顧客10万人、注文100万件の場合: 数十秒 → 数百ミリ秒
```
</details>

---

## FAQ

### Q1: サブクエリとCTE（WITH句）のどちらを使うべきか？

同じサブクエリを複数回参照する場合はCTEが適している（DRY原則）。1回だけ使用するならサブクエリでも問題ない。可読性の面ではCTEが優れることが多い。パフォーマンスはRDBMS依存だが、PostgreSQL 12以降はCTEのインライン展開が行われるため差は小さい。

### Q2: 相関サブクエリは常に遅いのか？

必ずしも遅くない。オプティマイザが内部的にJOINに変換することがある。ただし、外部テーブルの行数が多い場合はJOINやウィンドウ関数への書き換えを検討すべき。EXPLAINで実行計画を確認することが重要。PostgreSQLでは相関サブクエリの結果をキャッシュする機能もある。

### Q3: サブクエリのネストは何段まで許容されるか？

技術的な制限はRDBMSによるが、可読性の観点から2段以内を推奨。3段以上のネストはCTEやビューに分解して可読性を確保する。

### Q4: LATERALはいつ使うべきか？

LATERAL JOINは以下の場面で特に有効である: (1) 各グループのTOP-Nを取得する場合、(2) FROM句で外部行を参照する必要がある場合、(3) 複数の集約結果を1行に並べたい場合。通常のJOINでは表現できない「行ごとの計算」を可能にする。

### Q5: サブクエリで SELECT * は避けるべきか？

EXISTS内の `SELECT *` は問題ない（オプティマイザが無視する）。ただし、IN句のサブクエリやFROM句の派生テーブルでは、必要なカラムのみを指定することで実行計画の最適化が促進される。

### Q6: サブクエリの結果がNULLを含む場合の注意点は？

IN/NOT INはNULLに対して特殊な挙動を示す。NOT INは1つでもNULLがあると全行を除外する。EXISTS/NOT EXISTSはNULLの影響を受けない。NULL安全性の観点からNOT EXISTSの使用を推奨する。

---

## トラブルシューティング

### サブクエリに関する一般的な問題と対処法

| 問題 | 原因 | 対処法 |
|------|------|--------|
| NOT INが結果を返さない | サブクエリ内にNULLがある | NOT EXISTSに書き換え |
| スカラーサブクエリでエラー | 複数行を返している | 集約関数かLIMIT 1を追加 |
| サブクエリが遅い | 相関サブクエリのN+1問題 | JOINまたはLATERALに書き換え |
| 結果が重複する | INをJOINに書き換えた際の問題 | DISTINCT追加またはEXISTSに変更 |
| メモリ不足 | 大きな非相関サブクエリのマテリアライズ | work_memの調整、分割処理 |
| 実行計画が不安定 | 統計情報の不足 | ANALYZEで統計更新 |

### パフォーマンスデバッグのフロー

```
┌──── サブクエリのパフォーマンスデバッグ ────────┐
│                                                  │
│  Step 1: EXPLAIN ANALYZE で実行計画を確認        │
│  │                                              │
│  Step 2: Nested Loop + SubPlan があるか？        │
│  │  ├── Yes → 相関サブクエリのN+1問題          │
│  │  │         → JOINまたはLATERALに書き換え     │
│  │  └── No → Step 3                            │
│  │                                              │
│  Step 3: Seq Scan があるか？                     │
│  │  ├── Yes → インデックスの追加を検討          │
│  │  └── No → Step 4                            │
│  │                                              │
│  Step 4: 推定行数と実際の行数が乖離しているか？  │
│  │  ├── Yes → ANALYZE でテーブル統計を更新      │
│  │  └── No → Step 5                            │
│  │                                              │
│  Step 5: Hash Join / Merge Join のコスト確認     │
│       → work_mem の調整を検討                    │
└──────────────────────────────────────────────────┘
```

---

## セキュリティに関する注意事項

### SQLインジェクションとサブクエリ

```sql
-- NG: ユーザー入力をサブクエリに直接埋め込む
-- （これは擬似コード: 実際のプログラミング言語で起こる問題）
-- query = "SELECT * FROM products WHERE id IN (" + user_input + ")"
-- user_input = "1); DROP TABLE products; --"

-- OK: プリペアドステートメントを使用
-- Python (psycopg2)
-- cursor.execute(
--     "SELECT * FROM products WHERE id IN (SELECT id FROM categories WHERE name = %s)",
--     (user_input,)
-- )

-- OK: INリストにはANY + 配列パラメータを使用（PostgreSQL）
-- cursor.execute(
--     "SELECT * FROM products WHERE id = ANY(%s)",
--     ([1, 2, 3],)
-- )
```

---

## まとめ

| 項目 | 要点 |
|------|------|
| 非相関サブクエリ | 外部クエリと独立。1回だけ実行される |
| 相関サブクエリ | 外部の各行に依存。行数分実行される可能性。オプティマイザが最適化する場合あり |
| EXISTS | 存在確認に最適。NULL安全。短絡評価で効率的 |
| NOT IN | NULLの罠がある。NOT EXISTSを推奨 |
| LATERAL JOIN | FROM句での相関サブクエリ。Top-N問題に有効 |
| パフォーマンス | EXPLAINで実行計画を確認。JOINへの書き換えを検討 |
| オプティマイザ | IN/EXISTSはSemi Joinに自動変換されることが多い |
| CTE | 複数回参照するサブクエリはCTEで可読性向上 |

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
4. PostgreSQL Documentation — "LATERAL Subqueries" https://www.postgresql.org/docs/current/queries-table-expressions.html#QUERIES-LATERAL
5. MySQL Documentation — "Optimizing Subqueries" https://dev.mysql.com/doc/refman/8.0/en/subquery-optimization.html
6. Date, C.J. (2015). *SQL and Relational Theory*. O'Reilly Media. Chapter 12: Subqueries.
