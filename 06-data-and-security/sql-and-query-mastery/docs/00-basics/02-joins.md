# JOIN — INNER / LEFT / RIGHT / FULL / CROSS

> JOINはリレーショナルデータベースの核心機能であり、複数テーブルに分散したデータを結合条件に基づいて一つの結果セットにまとめる操作である。E.F. Coddが1970年に提唱した関係代数において、JOINは「直積（Cartesian Product）」と「選択（Selection）」の合成操作として定義されている。

## 前提知識

- [01-select.md](./01-select.md) — SELECT文の基本構文
- [00-sql-overview.md](./00-sql-overview.md) — SQL全体像の理解
- テーブル設計の基礎（PRIMARY KEY, FOREIGN KEY）

## この章で学ぶこと

1. 全JOIN種別（INNER, LEFT, RIGHT, FULL, CROSS）の動作原理と使い分け
2. 結合条件の設計とパフォーマンスへの影響
3. 自己結合、複数テーブル結合など実践的なJOINパターン
4. JOIN処理の内部実装（Nested Loop, Hash Join, Merge Join）
5. RDBMS間のJOIN構文・動作の違い
6. LATERAL JOIN, SEMI JOIN, ANTI JOINなどの高度なJOINパターン

---

## 1. JOINの全体像

```
┌──────────────────── JOIN種別の分類 ─────────────────────┐
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              結合（JOIN）                         │   │
│  ├──────────────┬──────────────────────────────────┤   │
│  │  内部結合     │  外部結合                         │   │
│  │  INNER JOIN  │  LEFT / RIGHT / FULL OUTER JOIN  │   │
│  ├──────────────┼──────────────────────────────────┤   │
│  │  交差結合     │  その他                           │   │
│  │  CROSS JOIN  │  NATURAL JOIN / LATERAL JOIN     │   │
│  └──────────────┴──────────────────────────────────┘   │
│                                                         │
│  INNER JOIN  : 両方に存在する行のみ                      │
│  LEFT JOIN   : 左テーブルの全行 + 右の一致行             │
│  RIGHT JOIN  : 右テーブルの全行 + 左の一致行             │
│  FULL JOIN   : 両テーブルの全行                          │
│  CROSS JOIN  : 全組み合わせ（直積）                      │
│  LATERAL JOIN: 左テーブルの各行を参照するサブクエリ結合   │
│  SEMI JOIN   : 存在判定のみ（EXISTS相当）               │
│  ANTI JOIN   : 非存在判定（NOT EXISTS相当）             │
└─────────────────────────────────────────────────────────┘
```

### 関係代数とJOINの数学的基礎

JOINの操作は関係代数（Relational Algebra）で厳密に定義されている。

```
関係代数におけるJOIN演算
==========================

1. 直積（Cartesian Product）: R × S
   結果 = |R| × |S| 行
   → CROSS JOINに対応

2. シータ結合（Theta Join）: R ⋈θ S
   R × S のうち、条件θを満たす行のみ
   → ON句付きJOINに対応

3. 等値結合（Equi Join）: R ⋈(R.a = S.b) S
   シータ結合の特殊ケース（等号条件のみ）
   → 最も一般的なJOIN

4. 自然結合（Natural Join）: R ⋈ S
   同名列で自動等値結合 + 重複列の除去
   → NATURAL JOINに対応（非推奨）

5. 半結合（Semi Join）: R ⋉ S
   Sに一致するRの行のみ（Sの列は返さない）
   → EXISTS/INサブクエリに対応

6. 反結合（Anti Join）: R ▷ S
   Sに一致しないRの行のみ
   → NOT EXISTS/NOT INサブクエリに対応
```

### サンプルデータ

```sql
-- 以降の例で使用するテーブル
CREATE TABLE departments (
    id   INTEGER PRIMARY KEY,
    name VARCHAR(50)
);

CREATE TABLE employees (
    id            INTEGER PRIMARY KEY,
    name          VARCHAR(100),
    department_id INTEGER REFERENCES departments(id),
    salary        DECIMAL(10, 2),
    hire_date     DATE
);

INSERT INTO departments VALUES (1, '営業'), (2, '開発'), (3, '人事');
INSERT INTO employees VALUES
    (101, '田中', 1, 450000, '2020-04-01'),
    (102, '鈴木', 2, 520000, '2019-07-15'),
    (103, '佐藤', 1, 380000, '2021-01-10'),
    (104, '高橋', NULL, 400000, '2022-03-01');  -- 部署未所属
```

---

## 2. INNER JOIN

INNER JOINは最も基本的な結合操作であり、両テーブルの結合条件に一致する行のみを返す。結合キーにNULLを持つ行は常に除外される（NULLは何とも等しくならないため）。

### コード例1: INNER JOIN

```sql
-- 両方のテーブルに一致する行のみ返す
SELECT e.name AS employee, d.name AS department
FROM employees e
    INNER JOIN departments d ON e.department_id = d.id;

-- 結果:
-- employee | department
-- ---------+-----------
-- 田中     | 営業
-- 鈴木     | 開発
-- 佐藤     | 営業
--
-- ※ 高橋（department_id=NULL）は除外
-- ※ 人事（社員なし）も除外
```

### INNER JOINの内部動作

```
INNER JOIN の処理フロー
========================

employees テーブル          departments テーブル
+-----+------+--------+    +----+------+
| id  | name | dep_id |    | id | name |
+-----+------+--------+    +----+------+
| 101 | 田中 |   1    |    |  1 | 営業 |
| 102 | 鈴木 |   2    |    |  2 | 開発 |
| 103 | 佐藤 |   1    |    |  3 | 人事 |
| 104 | 高橋 |  NULL  |    +----+------+
+-----+------+--------+

結合処理:
  101: dep_id=1 → departments(1)=営業  ✓ 一致
  102: dep_id=2 → departments(2)=開発  ✓ 一致
  103: dep_id=1 → departments(1)=営業  ✓ 一致
  104: dep_id=NULL → NULL≠1, NULL≠2, NULL≠3  ✗ 全て不一致

結果: 3行（一致した組み合わせのみ）
```

### コード例2: 複合条件のINNER JOIN

```sql
-- 複数の結合条件を使用
CREATE TABLE project_assignments (
    employee_id   INTEGER,
    department_id INTEGER,
    project_id    INTEGER,
    role          VARCHAR(50),
    start_date    DATE,
    PRIMARY KEY (employee_id, project_id)
);

-- 同じ部署に所属し、かつプロジェクトに配属された社員
SELECT
    e.name AS employee,
    d.name AS department,
    pa.role
FROM employees e
    INNER JOIN departments d
        ON e.department_id = d.id
    INNER JOIN project_assignments pa
        ON e.id = pa.employee_id
        AND e.department_id = pa.department_id  -- 複合結合条件
ORDER BY d.name, e.name;

-- 非等値結合（Non-Equi Join）の例
-- 給与が同じ部署の平均以上の社員を取得
SELECT e.name, e.salary, dept_avg.avg_salary
FROM employees e
    INNER JOIN (
        SELECT department_id, AVG(salary) AS avg_salary
        FROM employees
        GROUP BY department_id
    ) dept_avg
        ON e.department_id = dept_avg.department_id
        AND e.salary >= dept_avg.avg_salary;
```

---

## 3. LEFT JOIN

LEFT JOIN（LEFT OUTER JOIN）は左テーブルの全行を保持し、右テーブルに一致する行がない場合はNULLで埋める。「オプショナルな関連」を表現する際に最も頻繁に使われるJOIN種別である。

### コード例3: LEFT JOIN (LEFT OUTER JOIN)

```sql
-- 左テーブル（employees）の全行を保持
SELECT e.name AS employee, d.name AS department
FROM employees e
    LEFT JOIN departments d ON e.department_id = d.id;

-- 結果:
-- employee | department
-- ---------+-----------
-- 田中     | 営業
-- 鈴木     | 開発
-- 佐藤     | 営業
-- 高橋     | NULL        ← 一致なしでもNULLで表示

-- LEFT JOINで「一致しない行」だけ取得（ANTI JOINパターン）
SELECT e.name
FROM employees e
    LEFT JOIN departments d ON e.department_id = d.id
WHERE d.id IS NULL;
-- → 高橋（部署未所属の社員）
```

### LEFT JOINの動作原理

```
LEFT JOIN の処理フロー
========================

LEFT テーブル (employees)    RIGHT テーブル (departments)
+------+--------+            +----+------+
| name | dep_id |            | id | name |
+------+--------+            +----+------+
| 田中 |   1    | ──────┐    |  1 | 営業 | ← 一致
| 鈴木 |   2    | ──────┤    |  2 | 開発 | ← 一致
| 佐藤 |   1    | ──────┤    |  3 | 人事 | ← 一致なし（結果に出ない）
| 高橋 |  NULL  | ─── ✗     +----+------+
+------+--------+

結果:
+------+------+--------+
| 田中 | 営業 |  一致  |
| 鈴木 | 開発 |  一致  |
| 佐藤 | 営業 |  一致  |
| 高橋 | NULL | 不一致 | ← 左テーブルの行は必ず保持
+------+------+--------+

ポイント:
  - 左テーブルの全行が結果に含まれることが保証される
  - 右テーブルに一致がない場合、右側の列はすべてNULLになる
  - WHERE d.id IS NULL で「一致しない行」だけを抽出できる
```

### コード例4: LEFT JOINの実践パターン

```sql
-- 実践パターン1: オプショナルなプロフィール情報の取得
SELECT
    u.id,
    u.name,
    u.email,
    p.avatar_url,
    p.bio,
    COALESCE(p.display_name, u.name) AS display_name
FROM users u
    LEFT JOIN user_profiles p ON u.id = p.user_id;

-- 実践パターン2: 集約と組み合わせた部署別社員数
SELECT
    d.name AS department,
    COUNT(e.id) AS employee_count,
    COALESCE(SUM(e.salary), 0) AS total_salary
FROM departments d
    LEFT JOIN employees e ON d.id = e.department_id
GROUP BY d.id, d.name
ORDER BY employee_count DESC;
-- 結果: 人事部も0人として表示される

-- 実践パターン3: 直近の注文情報の結合
SELECT
    c.name AS customer,
    c.email,
    lo.last_order_date,
    lo.last_order_total
FROM customers c
    LEFT JOIN LATERAL (
        SELECT
            order_date AS last_order_date,
            total AS last_order_total
        FROM orders
        WHERE customer_id = c.id
        ORDER BY order_date DESC
        LIMIT 1
    ) lo ON TRUE;
```

---

## 4. RIGHT JOIN / FULL JOIN

### コード例5: RIGHT JOINとFULL JOIN

```sql
-- RIGHT JOIN: 右テーブル（departments）の全行を保持
SELECT e.name AS employee, d.name AS department
FROM employees e
    RIGHT JOIN departments d ON e.department_id = d.id;

-- 結果:
-- employee | department
-- ---------+-----------
-- 田中     | 営業
-- 佐藤     | 営業
-- 鈴木     | 開発
-- NULL     | 人事        ← 社員がいない部署もNULLで表示

-- FULL OUTER JOIN: 両方の全行を保持
SELECT e.name AS employee, d.name AS department
FROM employees e
    FULL OUTER JOIN departments d ON e.department_id = d.id;

-- 結果:
-- employee | department
-- ---------+-----------
-- 田中     | 営業
-- 佐藤     | 営業
-- 鈴木     | 開発
-- 高橋     | NULL        ← 左のみ
-- NULL     | 人事        ← 右のみ
```

### FULL OUTER JOINの実践的な使い方

```sql
-- 差分検出: 2つのデータソースの比較
-- テーブルAにあってBにない、Bにあって Aにない行を検出

SELECT
    COALESCE(a.product_id, b.product_id) AS product_id,
    a.stock AS warehouse_a_stock,
    b.stock AS warehouse_b_stock,
    CASE
        WHEN a.product_id IS NULL THEN 'Bにのみ存在'
        WHEN b.product_id IS NULL THEN 'Aにのみ存在'
        WHEN a.stock <> b.stock  THEN '数量不一致'
        ELSE '一致'
    END AS status
FROM warehouse_a a
    FULL OUTER JOIN warehouse_b b ON a.product_id = b.product_id
WHERE a.product_id IS NULL
   OR b.product_id IS NULL
   OR a.stock <> b.stock;

-- データ同期の差分チェック
SELECT
    COALESCE(src.id, dst.id) AS id,
    src.updated_at AS source_updated,
    dst.updated_at AS dest_updated,
    CASE
        WHEN dst.id IS NULL THEN 'INSERT'
        WHEN src.id IS NULL THEN 'DELETE'
        WHEN src.updated_at > dst.updated_at THEN 'UPDATE'
        ELSE 'SYNC'
    END AS action_needed
FROM source_table src
    FULL OUTER JOIN destination_table dst ON src.id = dst.id
WHERE dst.id IS NULL
   OR src.id IS NULL
   OR src.updated_at > dst.updated_at;
```

### MySQLでFULL OUTER JOINをエミュレート

MySQLはFULL OUTER JOINを直接サポートしていない。LEFT JOINとRIGHT JOINのUNIONで代替する。

```sql
-- MySQL: FULL OUTER JOINのエミュレーション
SELECT e.name AS employee, d.name AS department
FROM employees e
    LEFT JOIN departments d ON e.department_id = d.id

UNION

SELECT e.name AS employee, d.name AS department
FROM employees e
    RIGHT JOIN departments d ON e.department_id = d.id;

-- UNION ALL + 重複排除の最適化版
SELECT e.name AS employee, d.name AS department
FROM employees e
    LEFT JOIN departments d ON e.department_id = d.id

UNION ALL

SELECT e.name AS employee, d.name AS department
FROM employees e
    RIGHT JOIN departments d ON e.department_id = d.id
WHERE e.id IS NULL;  -- LEFT JOINと重複しない行のみ
```

---

## 5. CROSS JOIN

CROSS JOIN（交差結合）は2つのテーブルの全組み合わせ（直積）を生成する。結合条件を指定しないため、左テーブルがM行、右テーブルがN行の場合、結果はM×N行になる。

### コード例6: CROSS JOINと実用例

```sql
-- CROSS JOIN: 全組み合わせ（直積）
-- 4社員 × 3部署 = 12行
SELECT e.name, d.name
FROM employees e
    CROSS JOIN departments d;

-- 実用例1: カレンダーテーブルの生成
SELECT
    y.year,
    m.month
FROM generate_series(2020, 2025) AS y(year)
    CROSS JOIN generate_series(1, 12) AS m(month)
ORDER BY y.year, m.month;

-- 実用例2: 全商品×全店舗の在庫マトリクス
SELECT
    p.name AS product,
    s.name AS store,
    COALESCE(i.quantity, 0) AS stock
FROM products p
    CROSS JOIN stores s
    LEFT JOIN inventory i ON i.product_id = p.id AND i.store_id = s.id;

-- 実用例3: 時間帯ごとの売上マトリクス
WITH hours AS (
    SELECT generate_series(0, 23) AS hour
),
days AS (
    SELECT generate_series(0, 6) AS dow,
           CASE generate_series(0, 6)
               WHEN 0 THEN '日' WHEN 1 THEN '月'
               WHEN 2 THEN '火' WHEN 3 THEN '水'
               WHEN 4 THEN '木' WHEN 5 THEN '金'
               WHEN 6 THEN '土'
           END AS day_name
)
SELECT
    d.day_name,
    h.hour,
    COALESCE(s.sale_count, 0) AS sales
FROM days d
    CROSS JOIN hours h
    LEFT JOIN (
        SELECT
            EXTRACT(DOW FROM sale_time) AS dow,
            EXTRACT(HOUR FROM sale_time) AS hour,
            COUNT(*) AS sale_count
        FROM sales
        GROUP BY 1, 2
    ) s ON d.dow = s.dow AND h.hour = s.hour
ORDER BY d.dow, h.hour;
```

### JOIN種別のベン図

```
  INNER JOIN          LEFT JOIN           RIGHT JOIN          FULL JOIN
  (共通部分)           (左全体+共通)       (共通+右全体)        (全体)

  ┌───┐ ┌───┐       ┌───┐ ┌───┐       ┌───┐ ┌───┐       ┌───┐ ┌───┐
  │   │█│   │       │███│█│   │       │   │█│███│       │███│█│███│
  │ A │█│ B │       │█A█│█│ B │       │ A │█│█B█│       │█A█│█│█B█│
  │   │█│   │       │███│█│   │       │   │█│███│       │███│█│███│
  └───┘ └───┘       └───┘ └───┘       └───┘ └───┘       └───┘ └───┘
  █ = 結果に含む    █ = 結果に含む    █ = 結果に含む    █ = 結果に含む


  LEFT ANTI JOIN      RIGHT ANTI JOIN     CROSS JOIN
  (左のみ)            (右のみ)            (直積)

  ┌───┐ ┌───┐       ┌───┐ ┌───┐       ┌───────────────┐
  │███│ │   │       │   │ │███│       │ A × B         │
  │█A█│ │ B │       │ A │ │█B█│       │ 全組み合わせ   │
  │███│ │   │       │   │ │███│       │ M行 × N行     │
  └───┘ └───┘       └───┘ └───┘       └───────────────┘
  WHERE b.id        WHERE a.id
  IS NULL           IS NULL
```

---

## 6. LATERAL JOIN

LATERAL JOINは左テーブルの各行を参照しながらサブクエリを実行する高度なJOINパターンである。PostgreSQL 9.3+、MySQL 8.0.14+で使用可能。

### コード例7: LATERAL JOIN

```sql
-- 各部署の給与上位3名を取得
SELECT
    d.name AS department,
    top3.name AS employee,
    top3.salary
FROM departments d
    CROSS JOIN LATERAL (
        SELECT name, salary
        FROM employees
        WHERE department_id = d.id  -- 外側テーブルの列を参照
        ORDER BY salary DESC
        LIMIT 3
    ) top3;

-- LATERAL vs 相関サブクエリ: LATERAL の方がSELECT句で複数列を返せる
-- 相関サブクエリはSELECT句で1値のみ
SELECT
    d.name AS department,
    latest.order_date,
    latest.total_amount,
    latest.item_count
FROM departments d
    LEFT JOIN LATERAL (
        SELECT
            o.order_date,
            o.total_amount,
            COUNT(oi.id) AS item_count
        FROM orders o
            LEFT JOIN order_items oi ON o.id = oi.order_id
        WHERE o.department_id = d.id
        ORDER BY o.order_date DESC
        LIMIT 1
    ) latest ON TRUE;

-- 時系列データの直近N件取得
SELECT
    s.sensor_id,
    s.location,
    readings.reading_time,
    readings.value
FROM sensors s
    CROSS JOIN LATERAL (
        SELECT reading_time, value
        FROM sensor_readings
        WHERE sensor_id = s.sensor_id
        ORDER BY reading_time DESC
        LIMIT 5
    ) readings
ORDER BY s.sensor_id, readings.reading_time DESC;
```

---

## 7. SEMI JOIN / ANTI JOIN

SEMI JOINとANTI JOINはSQL構文上は独立したJOIN種別として記述しないが、EXISTS/NOT EXISTSやIN/NOT INとして記述され、オプティマイザ内部でSEMI JOIN/ANTI JOINとして最適化される。

### コード例8: SEMI JOINとANTI JOIN

```sql
-- SEMI JOIN: 注文がある顧客のみ取得（EXISTS使用）
SELECT c.id, c.name
FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o WHERE o.customer_id = c.id
);

-- 同等のSEMI JOIN（IN使用）
SELECT c.id, c.name
FROM customers c
WHERE c.id IN (SELECT customer_id FROM orders);

-- ANTI JOIN: 注文がない顧客のみ取得（NOT EXISTS使用）
SELECT c.id, c.name
FROM customers c
WHERE NOT EXISTS (
    SELECT 1 FROM orders o WHERE o.customer_id = c.id
);

-- ANTI JOIN: LEFT JOIN + IS NULL パターン
SELECT c.id, c.name
FROM customers c
    LEFT JOIN orders o ON c.id = o.customer_id
WHERE o.id IS NULL;

-- ※ パフォーマンス比較:
-- NOT EXISTS vs LEFT JOIN + IS NULL vs NOT IN
-- 一般的に NOT EXISTS が最も安定した性能
-- NOT IN は NULLがあると意図しない結果になる危険がある
```

### SEMI JOIN vs ANTI JOIN vs通常JOIN の違い

```
SEMI JOIN / ANTI JOIN の動作
==============================

テーブル: customers        テーブル: orders
+----+------+              +----+--------+
| id | name |              | id | cust_id|
+----+------+              +----+--------+
|  1 | 田中 |              | 10 |   1    |
|  2 | 鈴木 |              | 11 |   1    |
|  3 | 佐藤 |              | 12 |   3    |
+----+------+              +----+--------+

INNER JOIN (customers JOIN orders ON id = cust_id):
→ 田中(10), 田中(11), 佐藤(12) = 3行（田中が2行！）

SEMI JOIN (EXISTS):
→ 田中, 佐藤 = 2行（注文が1件でもあれば1行）

ANTI JOIN (NOT EXISTS):
→ 鈴木 = 1行（注文がない顧客のみ）

ポイント:
  SEMI JOIN = 「少なくとも1件一致する行」
  ANTI JOIN = 「1件も一致しない行」
  → INNER JOINと異なり重複が発生しない
```

---

## 8. 実践的なJOINパターン

### コード例9: 自己結合（Self Join）

```sql
-- 社員テーブルで上司と部下の関係を表現
CREATE TABLE staff (
    id         INTEGER PRIMARY KEY,
    name       VARCHAR(100),
    manager_id INTEGER REFERENCES staff(id)
);

-- 自己結合で上司名を取得
SELECT
    s.name AS employee,
    m.name AS manager
FROM staff s
    LEFT JOIN staff m ON s.manager_id = m.id;

-- 同じ部署の社員ペアを列挙（自己結合）
SELECT
    e1.name AS employee_1,
    e2.name AS employee_2
FROM employees e1
    INNER JOIN employees e2
        ON e1.department_id = e2.department_id
        AND e1.id < e2.id;  -- 重複ペア防止

-- 同期入社（同じ年に入社）の社員ペア
SELECT
    e1.name AS emp1,
    e2.name AS emp2,
    EXTRACT(YEAR FROM e1.hire_date) AS hire_year
FROM employees e1
    INNER JOIN employees e2
        ON EXTRACT(YEAR FROM e1.hire_date) = EXTRACT(YEAR FROM e2.hire_date)
        AND e1.id < e2.id
ORDER BY hire_year;
```

### コード例10: 複数テーブルの結合

```sql
-- 3テーブル結合: 注文 → 注文明細 → 商品
SELECT
    o.id AS order_id,
    o.order_date,
    c.name AS customer_name,
    p.name AS product_name,
    oi.quantity,
    oi.unit_price,
    oi.quantity * oi.unit_price AS subtotal
FROM orders o
    INNER JOIN customers c ON o.customer_id = c.id
    INNER JOIN order_items oi ON o.id = oi.order_id
    INNER JOIN products p ON oi.product_id = p.id
WHERE o.customer_id = 42
ORDER BY o.order_date DESC, p.name;

-- 5テーブル結合: 完全な注文情報の取得
SELECT
    o.id AS order_id,
    c.name AS customer,
    c.email,
    p.name AS product,
    cat.name AS category,
    oi.quantity,
    oi.unit_price,
    (oi.quantity * oi.unit_price) AS line_total,
    s.company AS shipping_company,
    s.tracking_number,
    o.status
FROM orders o
    INNER JOIN customers c ON o.customer_id = c.id
    INNER JOIN order_items oi ON o.id = oi.order_id
    INNER JOIN products p ON oi.product_id = p.id
    LEFT JOIN categories cat ON p.category_id = cat.id
    LEFT JOIN shipments s ON o.id = s.order_id
WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY o.order_date DESC;
```

### コード例11: 条件付きJOIN

```sql
-- JOINのON句にフィルタ条件を含める vs WHERE句
-- 挙動の違いに注意

-- パターン1: ON句にフィルタ → LEFT JOINで全部署が表示される
SELECT d.name, e.name, e.salary
FROM departments d
    LEFT JOIN employees e
        ON d.id = e.department_id
        AND e.salary > 400000;  -- JOINの条件として
-- → 人事部もNULLで表示、営業部の佐藤（380000）はNULLで表示

-- パターン2: WHERE句にフィルタ → 該当部署のみ
SELECT d.name, e.name, e.salary
FROM departments d
    LEFT JOIN employees e ON d.id = e.department_id
WHERE e.salary > 400000;  -- 結合後のフィルタ
-- → 人事部は表示されない（WHERE句でNULL > 400000がFALSE）

-- この違いはOUTER JOINで特に重要
-- INNER JOINでは結果は同じになる
```

```
ON句 vs WHERE句 でのフィルタ条件（LEFT JOIN）
================================================

ON句の場合:
  departments     LEFT JOIN employees
  +------+        ON dep_id = d.id AND salary > 400000
  | 営業 | ←───── 田中(450000) ✓一致
  | 開発 | ←───── 鈴木(520000) ✓一致
  | 人事 | ←───── (一致なし → NULL)  ← 表示される
  +------+

  佐藤(380000): ON条件の salary > 400000 を満たさない
                → 結合されないが、営業部自体はNULLで表示

WHERE句の場合:
  departments     LEFT JOIN employees
  +------+        ON dep_id = d.id
  | 営業 | ←───── 田中(450000), 佐藤(380000)
  | 開発 | ←───── 鈴木(520000)
  | 人事 | ←───── (NULL)
  +------+

  WHERE salary > 400000 で全結果をフィルタ:
    田中(450000) ✓  → 表示
    佐藤(380000) ✗  → 非表示
    鈴木(520000) ✓  → 表示
    人事(NULL)   ✗  → 非表示（NULL > 400000 = UNKNOWN）
```

---

## 9. JOINの内部実装アルゴリズム

クエリオプティマイザは結合の実行時に以下の3つのアルゴリズムから最適なものを選択する。

### 結合アルゴリズムの比較

```
┌────────────── JOIN アルゴリズム ──────────────┐
│                                                │
│  1. Nested Loop Join (ネステッドループ)         │
│     外側テーブルの各行に対して内側テーブルを走査 │
│                                                │
│     for each row r in outer_table:             │
│         for each row s in inner_table:         │
│             if r.key == s.key:                 │
│                 emit(r, s)                     │
│                                                │
│     計算量: O(M × N)                           │
│     ※ 内側にインデックスがあれば O(M × log N)  │
│     最適: 小テーブル × 大テーブル（索引あり）   │
│                                                │
│  2. Hash Join (ハッシュ結合)                    │
│     小テーブルでハッシュ表を構築し、大テーブルを │
│     走査して一致を検索                          │
│                                                │
│     build hash_table from smaller_table        │
│     for each row r in larger_table:            │
│         probe hash_table with r.key            │
│         if found: emit(r, match)               │
│                                                │
│     計算量: O(M + N)                            │
│     メモリ: O(min(M, N))                        │
│     最適: 大テーブル同士、等値結合              │
│                                                │
│  3. Merge Join (マージ結合/ソートマージ)         │
│     両テーブルを結合キーでソートし、              │
│     マージしながら一致を検出                    │
│                                                │
│     sort outer_table by key                    │
│     sort inner_table by key                    │
│     merge both sorted streams                  │
│                                                │
│     計算量: O(M log M + N log N + M + N)        │
│     ※ 既にソート済みなら O(M + N)              │
│     最適: 大テーブル同士、ソート済み、範囲結合  │
└────────────────────────────────────────────────┘
```

### JOINアルゴリズムの選択基準

| 条件 | 選択されるアルゴリズム | 理由 |
|------|----------------------|------|
| 小テーブル × 大テーブル（インデックスあり） | Nested Loop + Index Scan | インデックスで高速ルックアップ |
| 大テーブル × 大テーブル（等値結合） | Hash Join | ハッシュ構築+プローブが効率的 |
| 両テーブルがソート済み | Merge Join | ソート不要でマージのみ |
| メモリが少ない + 大テーブル | Merge Join | ディスク上でソート可能 |
| 非等値結合（<, >, BETWEEN） | Nested Loop or Merge Join | Hash Joinは等値のみ |
| CROSS JOIN | Nested Loop | 全組み合わせなので他の選択肢なし |

### コード例12: 実行計画でJOINアルゴリズムを確認

```sql
-- PostgreSQL: EXPLAIN ANALYZEで実行計画を確認
EXPLAIN ANALYZE
SELECT e.name, d.name
FROM employees e
    INNER JOIN departments d ON e.department_id = d.id;

-- 出力例（Nested Loop の場合）:
-- Nested Loop  (cost=0.28..16.34 rows=3 width=64)
--   -> Seq Scan on employees e  (cost=0.00..1.04 rows=4 width=36)
--   -> Index Scan using departments_pkey on departments d
--        (cost=0.14..0.16 rows=1 width=36)
--        Index Cond: (id = e.department_id)

-- Hash Join を強制（テスト用、本番では非推奨）
SET enable_nestloop = off;
SET enable_mergejoin = off;

EXPLAIN ANALYZE
SELECT e.name, d.name
FROM employees e
    INNER JOIN departments d ON e.department_id = d.id;

-- 出力例（Hash Join の場合）:
-- Hash Join  (cost=1.07..2.15 rows=3 width=64)
--   Hash Cond: (e.department_id = d.id)
--   -> Seq Scan on employees e  (cost=0.00..1.04 rows=4 width=36)
--   -> Hash  (cost=1.03..1.03 rows=3 width=36)
--         -> Seq Scan on departments d  (cost=0.00..1.03 rows=3 width=36)

-- 設定を元に戻す
RESET enable_nestloop;
RESET enable_mergejoin;
```

---

## 10. RDBMS間のJOIN構文の違い

### RDBMS別JOIN対応表

| 機能 | PostgreSQL | MySQL | SQL Server | Oracle | SQLite |
|------|-----------|-------|------------|--------|--------|
| INNER JOIN | ○ | ○ | ○ | ○ | ○ |
| LEFT JOIN | ○ | ○ | ○ | ○ | ○ |
| RIGHT JOIN | ○ | ○ | ○ | ○ | ○ |
| FULL OUTER JOIN | ○ | ×（エミュ要） | ○ | ○ | ×（エミュ要） |
| CROSS JOIN | ○ | ○ | ○ | ○ | ○ |
| LATERAL JOIN | ○(9.3+) | ○(8.0.14+) | CROSS/OUTER APPLY | ○(12c+) | × |
| NATURAL JOIN | ○ | ○ | × | ○ | ○ |
| USING句 | ○ | ○ | × | ○ | ○ |

### 各RDBMS固有の構文

```sql
-- Oracle: 旧式のOUTER JOIN構文（(+)記法）
-- 非推奨だがレガシーコードで頻出
SELECT e.name, d.name
FROM employees e, departments d
WHERE e.department_id = d.id(+);  -- LEFT JOIN相当
-- → 標準のLEFT JOIN構文を使うべき

-- SQL Server: CROSS APPLY / OUTER APPLY
-- LATERALに相当
SELECT d.name, top3.name, top3.salary
FROM departments d
CROSS APPLY (
    SELECT TOP 3 name, salary
    FROM employees
    WHERE department_id = d.id
    ORDER BY salary DESC
) top3;

-- MySQL: STRAIGHT_JOIN（結合順序の強制）
SELECT STRAIGHT_JOIN e.name, d.name
FROM employees e
    INNER JOIN departments d ON e.department_id = d.id;
-- → オプティマイザの結合順序選択を無効化
```

---

## JOIN種別比較表

| JOIN種別 | 結果行数 | NULL行 | 主な用途 | 計算コスト |
|---------|---------|--------|---------|-----------|
| INNER JOIN | 両方に一致する行のみ | なし | 関連データの結合 | 低〜中 |
| LEFT JOIN | 左テーブル全行 | 右側にNULL | オプショナルな関連 | 中 |
| RIGHT JOIN | 右テーブル全行 | 左側にNULL | LEFT JOINの逆（稀） | 中 |
| FULL OUTER JOIN | 両テーブル全行 | 両側にNULL | 差分検出 | 高 |
| CROSS JOIN | 左×右の全組み合わせ | なし | マトリクス生成 | 非常に高 |
| LATERAL JOIN | 左テーブル全行 × サブクエリ | 設定による | Top-N per group | 中〜高 |
| NATURAL JOIN | 同名列で自動結合 | なし | 非推奨（暗黙結合） | INNER JOINと同等 |

## ON句 vs USING句 vs WHERE句 比較表

| 方式 | 構文例 | 柔軟性 | 可読性 | 注意点 |
|------|--------|--------|--------|--------|
| ON句 | `ON a.id = b.a_id` | 高い（複合条件可） | 明示的 | 最も推奨 |
| USING句 | `USING (id)` | 低い（同名列のみ） | 簡潔 | SQL Serverは非対応 |
| WHERE句 | `WHERE a.id = b.a_id` | 高い | 結合条件と混在 | OUTER JOINで不可 |
| NATURAL | 暗黙 | 最低（制御不能） | 危険 | 列追加で動作変化 |

---

## パフォーマンス最適化

### JOINのパフォーマンスに影響する要因

```
JOIN パフォーマンス最適化チェックリスト
========================================

1. インデックス
   [✓] 結合キー列にインデックスがあるか？
   [✓] 外部キー制約にインデックスが自動作成されているか？
       （PostgreSQLでは自動作成されない！明示的に作成が必要）
   [✓] 複合インデックスの列順序は適切か？

2. 結合順序
   [✓] 小テーブルから大テーブルへの結合になっているか？
   [✓] オプティマイザの統計情報は最新か？（ANALYZE実行）
   [✓] 必要に応じてヒント句で結合順序を制御

3. データ量の削減
   [✓] JOINの前にWHEREで行数を減らせないか？
   [✓] 不要な列をSELECTしていないか？
   [✓] サブクエリで事前に集約できないか？

4. JOIN種別の選択
   [✓] 不要なOUTER JOINを使っていないか？
   [✓] EXISTS/INで代替できるJOINはないか？（SEMI JOIN最適化）
   [✓] CROSS JOINの結果が爆発的に大きくないか？
```

### コード例13: JOINパフォーマンスの改善

```sql
-- [NG] 全データをJOINしてからフィルタ
SELECT e.name, d.name, o.total
FROM employees e
    INNER JOIN departments d ON e.department_id = d.id
    INNER JOIN orders o ON e.id = o.employee_id
WHERE o.order_date >= '2024-01-01'
  AND d.name = '営業';

-- [OK] サブクエリで先に絞り込み
SELECT e.name, d.name, o.total
FROM (
    SELECT * FROM employees WHERE department_id = 1
) e
    INNER JOIN departments d ON e.department_id = d.id
    INNER JOIN (
        SELECT * FROM orders WHERE order_date >= '2024-01-01'
    ) o ON e.id = o.employee_id;
-- ※ 実際にはオプティマイザが同等の最適化を行う場合が多い
-- ※ EXPLAIN ANALYZEで確認して判断すること

-- インデックスの作成
CREATE INDEX idx_employees_department_id ON employees(department_id);
CREATE INDEX idx_orders_employee_id ON orders(employee_id);
CREATE INDEX idx_orders_date ON orders(order_date);

-- 統計情報の更新
ANALYZE employees;
ANALYZE orders;
ANALYZE departments;
```

---

## エッジケース

### エッジケース1: NULLとJOIN

```sql
-- NULL同士は等しくならない（NULL = NULL → UNKNOWN → FALSE扱い）
SELECT * FROM table_a a
    INNER JOIN table_b b ON a.nullable_col = b.nullable_col;
-- → 両方がNULLの行は結合されない

-- NULLも含めて結合したい場合
SELECT * FROM table_a a
    INNER JOIN table_b b
        ON a.nullable_col = b.nullable_col
        OR (a.nullable_col IS NULL AND b.nullable_col IS NULL);

-- もしくは COALESCE を使用
SELECT * FROM table_a a
    INNER JOIN table_b b
        ON COALESCE(a.nullable_col, -1) = COALESCE(b.nullable_col, -1);
-- ※ センチネル値（-1）が実データに存在しないことを確認
```

### エッジケース2: JOINによる行の増幅

```sql
-- 1:Nの関係でJOINすると行が増幅する
-- orders(1) : order_items(N) で注文テーブルの行が増える

-- [NG] 集約とJOINの組み合わせで二重カウント
SELECT
    d.name,
    SUM(o.total) AS dept_total  -- 重複して加算される！
FROM departments d
    INNER JOIN employees e ON d.id = e.department_id
    INNER JOIN orders o ON e.id = o.employee_id
    INNER JOIN order_items oi ON o.id = oi.order_id
GROUP BY d.name;

-- [OK] サブクエリで先に集約してからJOIN
SELECT
    d.name,
    order_totals.dept_total
FROM departments d
    INNER JOIN (
        SELECT e.department_id, SUM(o.total) AS dept_total
        FROM employees e
            INNER JOIN orders o ON e.id = o.employee_id
        GROUP BY e.department_id
    ) order_totals ON d.id = order_totals.department_id;
```

### エッジケース3: 多対多関係のJOIN

```sql
-- 多対多関係: 学生 ←→ 中間テーブル ←→ コース
CREATE TABLE students (id INTEGER PRIMARY KEY, name VARCHAR(100));
CREATE TABLE courses (id INTEGER PRIMARY KEY, title VARCHAR(100));
CREATE TABLE enrollments (
    student_id INTEGER REFERENCES students(id),
    course_id  INTEGER REFERENCES courses(id),
    enrolled_at DATE,
    PRIMARY KEY (student_id, course_id)
);

-- 全学生の履修コース一覧
SELECT
    s.name AS student,
    STRING_AGG(c.title, ', ' ORDER BY c.title) AS courses,
    COUNT(c.id) AS course_count
FROM students s
    LEFT JOIN enrollments e ON s.id = e.student_id
    LEFT JOIN courses c ON e.course_id = c.id
GROUP BY s.id, s.name
ORDER BY s.name;

-- 同じコースを履修している学生ペア
SELECT DISTINCT
    s1.name AS student1,
    s2.name AS student2,
    c.title AS shared_course
FROM enrollments e1
    INNER JOIN enrollments e2
        ON e1.course_id = e2.course_id
        AND e1.student_id < e2.student_id
    INNER JOIN students s1 ON e1.student_id = s1.id
    INNER JOIN students s2 ON e2.student_id = s2.id
    INNER JOIN courses c ON e1.course_id = c.id;
```

### エッジケース4: 日付範囲による結合

```sql
-- 非等値結合: 日付範囲で結合する
-- 為替レートテーブル（日次レートが不定期に更新される）
CREATE TABLE exchange_rates (
    currency VARCHAR(3),
    rate DECIMAL(10, 4),
    effective_date DATE
);

-- 各注文に対して、注文日時点の為替レートを適用
SELECT
    o.id AS order_id,
    o.order_date,
    o.amount_usd,
    er.rate,
    o.amount_usd * er.rate AS amount_jpy
FROM orders o
    INNER JOIN LATERAL (
        SELECT rate
        FROM exchange_rates
        WHERE currency = 'JPY'
          AND effective_date <= o.order_date
        ORDER BY effective_date DESC
        LIMIT 1
    ) er ON TRUE;
```

---

## セキュリティに関する注意事項

### 1. JOINによる権限の越境

```sql
-- リスク: JOINで本来見えないデータが見えてしまう
-- ユーザーAは自分の注文のみ閲覧可能だが、
-- JOINで他のユーザーの情報が漏洩する可能性

-- [NG] テナント分離が不十分
SELECT o.*, c.name, c.email
FROM orders o
    INNER JOIN customers c ON o.customer_id = c.id;
-- → 他テナントの顧客情報も取得可能

-- [OK] 行レベルセキュリティ（RLS）またはWHERE句で制限
SELECT o.*, c.name, c.email
FROM orders o
    INNER JOIN customers c ON o.customer_id = c.id
WHERE o.tenant_id = current_setting('app.current_tenant')::INTEGER;

-- PostgreSQL: 行レベルセキュリティ
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON orders
    USING (tenant_id = current_setting('app.current_tenant')::INTEGER);
```

### 2. SQLインジェクションとJOIN

```sql
-- [NG] 動的JOINのテーブル名を文字列連結で構築
-- ユーザー入力: "employees; DROP TABLE users; --"

-- [OK] ホワイトリストでテーブル名を検証
-- アプリケーション側:
-- allowed_tables = {'employees', 'departments', 'projects'}
-- if table_name not in allowed_tables:
--     raise ValueError("Invalid table name")
```

---

## アンチパターン

### アンチパターン1: 暗黙的結合（カンマ結合）

```sql
-- NG: 旧式のカンマ結合（暗黙的CROSS JOIN + WHERE）
SELECT e.name, d.name
FROM employees e, departments d
WHERE e.department_id = d.id;

-- 問題点:
-- 1. WHERE句を忘れるとCROSS JOINになる
-- 2. 結合条件とフィルタ条件が混在して読みにくい
-- 3. OUTER JOINが表現できない
-- 4. テーブルが増えると結合条件の漏れが検出困難

-- OK: 明示的なJOIN構文
SELECT e.name, d.name
FROM employees e
    INNER JOIN departments d ON e.department_id = d.id;
```

### アンチパターン2: N+1クエリ問題

```sql
-- NG: ループ内でJOINすべきクエリを個別実行
-- アプリケーション側の疑似コード:
-- for dept in get_all_departments():
--     employees = query("SELECT * FROM employees WHERE dept_id = ?", dept.id)
--     # 部署数 N に対して N+1 回のクエリ

-- OK: JOINで1回のクエリに
SELECT d.name AS department, e.name AS employee
FROM departments d
    LEFT JOIN employees e ON d.id = e.department_id
ORDER BY d.name, e.name;
```

### アンチパターン3: 不要なJOINによるパフォーマンス劣化

```sql
-- NG: 使わないテーブルもJOINしている
SELECT e.name, e.salary
FROM employees e
    INNER JOIN departments d ON e.department_id = d.id
    INNER JOIN locations l ON d.location_id = l.id  -- 使ってない！
WHERE d.name = '営業';

-- OK: 必要なテーブルのみJOIN
SELECT e.name, e.salary
FROM employees e
    INNER JOIN departments d ON e.department_id = d.id
WHERE d.name = '営業';

-- もしくはEXISTSで軽量化
SELECT e.name, e.salary
FROM employees e
WHERE EXISTS (
    SELECT 1 FROM departments d
    WHERE d.id = e.department_id AND d.name = '営業'
);
```

### アンチパターン4: DISTINCT でJOINの重複を隠蔽

```sql
-- NG: JOINで行が増えたのをDISTINCTで無理やり解消
SELECT DISTINCT e.name, e.department_id
FROM employees e
    INNER JOIN orders o ON e.id = o.employee_id;
-- → JOINの設計が間違っている可能性が高い

-- OK: EXISTSを使用してSEMI JOINに
SELECT e.name, e.department_id
FROM employees e
WHERE EXISTS (
    SELECT 1 FROM orders o WHERE o.employee_id = e.id
);
```

---

## 演習問題

### 演習1（基礎）: 基本的なJOIN

以下のテーブル構造で、各問いに答えるSQLを書きなさい。

```sql
-- テーブル定義
CREATE TABLE authors (id INT PRIMARY KEY, name VARCHAR(100), country VARCHAR(50));
CREATE TABLE books (id INT PRIMARY KEY, title VARCHAR(200), author_id INT, published_year INT);
CREATE TABLE reviews (id INT PRIMARY KEY, book_id INT, rating INT, reviewer_name VARCHAR(100));
```

1. 全ての著者とその著書を表示せよ（著書がない著者も含む）
2. レビューが1件もない書籍を列挙せよ
3. 日本の著者が書いた書籍のレビュー平均点を著者別に表示せよ

<details>
<summary>解答例</summary>

```sql
-- 1. 全ての著者とその著書（LEFT JOIN）
SELECT a.name AS author, b.title AS book
FROM authors a
    LEFT JOIN books b ON a.id = b.author_id
ORDER BY a.name, b.title;

-- 2. レビューがない書籍（ANTI JOIN）
SELECT b.title
FROM books b
    LEFT JOIN reviews r ON b.id = r.book_id
WHERE r.id IS NULL;

-- 3. 日本の著者の書籍レビュー平均点
SELECT
    a.name AS author,
    ROUND(AVG(r.rating), 1) AS avg_rating,
    COUNT(r.id) AS review_count
FROM authors a
    INNER JOIN books b ON a.id = b.author_id
    INNER JOIN reviews r ON b.id = r.book_id
WHERE a.country = '日本'
GROUP BY a.id, a.name
ORDER BY avg_rating DESC;
```

</details>

### 演習2（応用）: 複数テーブルJOINと集約

以下のECサイトのテーブルで、各問いに答えるSQLを書きなさい。

```sql
CREATE TABLE customers (id INT PRIMARY KEY, name VARCHAR(100), registered_at DATE);
CREATE TABLE orders (id INT PRIMARY KEY, customer_id INT, order_date DATE, status VARCHAR(20));
CREATE TABLE order_items (id INT PRIMARY KEY, order_id INT, product_id INT, quantity INT, unit_price DECIMAL);
CREATE TABLE products (id INT PRIMARY KEY, name VARCHAR(100), category VARCHAR(50));
```

1. 月別のカテゴリ別売上合計を算出せよ（売上がない月×カテゴリもゼロで表示）
2. 過去90日間に注文がない顧客のリストを取得せよ
3. 各顧客の「最も購入金額が大きい商品カテゴリ」を表示せよ

<details>
<summary>解答例</summary>

```sql
-- 1. 月別カテゴリ別売上（CROSS JOIN + LEFT JOIN）
WITH months AS (
    SELECT generate_series(
        DATE_TRUNC('month', MIN(order_date)),
        DATE_TRUNC('month', MAX(order_date)),
        '1 month'
    )::DATE AS month
    FROM orders
),
categories AS (
    SELECT DISTINCT category FROM products
)
SELECT
    m.month,
    c.category,
    COALESCE(SUM(oi.quantity * oi.unit_price), 0) AS total_sales
FROM months m
    CROSS JOIN categories c
    LEFT JOIN orders o
        ON DATE_TRUNC('month', o.order_date) = m.month
    LEFT JOIN order_items oi ON o.id = oi.order_id
    LEFT JOIN products p ON oi.product_id = p.id AND p.category = c.category
GROUP BY m.month, c.category
ORDER BY m.month, c.category;

-- 2. 過去90日間に注文がない顧客（ANTI JOIN）
SELECT c.id, c.name, c.registered_at
FROM customers c
    LEFT JOIN orders o
        ON c.id = o.customer_id
        AND o.order_date >= CURRENT_DATE - INTERVAL '90 days'
WHERE o.id IS NULL
ORDER BY c.name;

-- 3. 各顧客の最大購入カテゴリ（LATERAL JOIN）
SELECT
    c.name AS customer,
    top_cat.category,
    top_cat.total_spent
FROM customers c
    CROSS JOIN LATERAL (
        SELECT p.category, SUM(oi.quantity * oi.unit_price) AS total_spent
        FROM orders o
            INNER JOIN order_items oi ON o.id = oi.order_id
            INNER JOIN products p ON oi.product_id = p.id
        WHERE o.customer_id = c.id
        GROUP BY p.category
        ORDER BY total_spent DESC
        LIMIT 1
    ) top_cat
ORDER BY c.name;
```

</details>

### 演習3（発展）: 自己結合とグラフ探索

```sql
CREATE TABLE employees_v2 (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    manager_id INT REFERENCES employees_v2(id),
    department VARCHAR(50),
    salary DECIMAL(10, 2)
);
```

1. 各社員について「直属の上司の給与との差額」を計算せよ
2. 全ての上司-部下ペアのうち、部下の方が給与が高いケースを列挙せよ
3. 3階層以上の管理チェーン（社員→上司→上司の上司）を再帰なしで取得せよ

<details>
<summary>解答例</summary>

```sql
-- 1. 上司との給与差額（自己結合）
SELECT
    e.name AS employee,
    e.salary AS emp_salary,
    m.name AS manager,
    m.salary AS mgr_salary,
    e.salary - m.salary AS salary_diff
FROM employees_v2 e
    LEFT JOIN employees_v2 m ON e.manager_id = m.id
ORDER BY salary_diff DESC NULLS LAST;

-- 2. 部下の方が給与が高いケース
SELECT
    m.name AS manager,
    m.salary AS mgr_salary,
    e.name AS subordinate,
    e.salary AS sub_salary,
    e.salary - m.salary AS overpay
FROM employees_v2 e
    INNER JOIN employees_v2 m ON e.manager_id = m.id
WHERE e.salary > m.salary
ORDER BY overpay DESC;

-- 3. 3階層の管理チェーン（自己結合3回）
SELECT
    e.name AS employee,
    m1.name AS direct_manager,
    m2.name AS skip_level_manager
FROM employees_v2 e
    INNER JOIN employees_v2 m1 ON e.manager_id = m1.id
    INNER JOIN employees_v2 m2 ON m1.manager_id = m2.id
ORDER BY m2.name, m1.name, e.name;
```

</details>

---

## FAQ

### Q1: LEFT JOINとINNER JOINのどちらを使うべきか？

「結合先にデータが存在しない場合も結果に含めたいか？」が判断基準。例えば「部署未所属の社員も表示したい」ならLEFT JOIN、「部署に所属する社員だけ表示したい」ならINNER JOINを使う。迷ったらLEFT JOINを使い、不要なNULL行がないか確認するのも一つの方法。

### Q2: JOINの順序はパフォーマンスに影響するか？

理論的にはクエリオプティマイザが最適な結合順序を選択するため、書く順序は影響しない。ただし、複雑なクエリやオプティマイザの限界がある場合、ヒント句（`/*+ LEADING(...) */`等）で制御することがある。PostgreSQLの`join_collapse_limit`（デフォルト8）を超えるテーブル数の場合、書いた順序で結合されるため注意が必要。

### Q3: NATURAL JOINはなぜ非推奨か？

NATURAL JOINは同名の全列で自動結合するため、テーブルに列を追加しただけで結合条件が変わり、予期しない結果を返す危険がある。例えば、両テーブルに`created_at`列が追加されると、意図しない結合条件に含まれてしまう。常にON句で明示的に結合条件を指定すべき。

### Q4: EXISTSとINとJOINはどう使い分けるか？

- **INNER JOIN**: 結合先の列も結果に必要な場合
- **EXISTS**: 存在確認のみで結合先の列が不要な場合（SEMI JOIN）
- **IN**: サブクエリの結果が小さく、NULLが含まれない場合

パフォーマンスは多くのRDBMSで同等だが、NOT INはNULLの問題があるためNOT EXISTSを推奨する。

### Q5: JOINで「最新の1件」を取得するベストプラクティスは？

```sql
-- 方法1: LATERAL JOIN（PostgreSQL 9.3+, MySQL 8.0.14+）
SELECT c.*, latest_order.*
FROM customers c
    LEFT JOIN LATERAL (
        SELECT order_date, total
        FROM orders WHERE customer_id = c.id
        ORDER BY order_date DESC LIMIT 1
    ) latest_order ON TRUE;

-- 方法2: ROW_NUMBER() + CTE
WITH ranked AS (
    SELECT o.*, ROW_NUMBER() OVER (
        PARTITION BY customer_id ORDER BY order_date DESC
    ) AS rn
    FROM orders o
)
SELECT c.*, r.order_date, r.total
FROM customers c
    LEFT JOIN ranked r ON c.id = r.customer_id AND r.rn = 1;

-- 方法3: 相関サブクエリ（非推奨、大量データで遅い）
SELECT c.*,
    (SELECT MAX(order_date) FROM orders WHERE customer_id = c.id)
FROM customers c;
```

### Q6: 多数のテーブルをJOINするときの注意点は？

- PostgreSQLの`join_collapse_limit`（デフォルト8）を超えると結合順序の最適化が制限される
- 必要なJOINのみ記述し、不要なJOINは削除する
- 中間結果をCTEやサブクエリで事前に集約することで結合対象を小さくする
- EXPLAIN ANALYZEで実行計画を確認し、ボトルネックを特定する

---

## トラブルシューティング

### 問題1: JOINが予想より多くの行を返す

**原因**: 1:N関係で行が増幅されている。結合キーにユニーク制約がない。

**対処**:
1. `SELECT COUNT(*) FROM result` で行数を確認
2. 結合キーのカーディナリティを確認: `SELECT column, COUNT(*) FROM table GROUP BY column HAVING COUNT(*) > 1`
3. DISTINCTやGROUP BYで重複を排除、またはEXISTS（SEMI JOIN）に書き換え

### 問題2: JOINが遅い

**原因**: インデックスがない、統計情報が古い、結合順序が最適でない。

**対処**:
1. `EXPLAIN ANALYZE`で実行計画を確認
2. 結合キー列にインデックスを作成
3. `ANALYZE`で統計情報を更新
4. work_memを増やしてHash Joinのスピルを防ぐ

### 問題3: FULL OUTER JOINがMySQLで使えない

**対処**: LEFT JOIN UNION ALL RIGHT JOIN（WHERE左.id IS NULL）で代替する。上記のコード例を参照。

---

## まとめ

| 項目 | 要点 |
|------|------|
| INNER JOIN | 両テーブルの一致行のみ。最も基本的 |
| LEFT JOIN | 左テーブル全行保持。実務で最頻出 |
| RIGHT JOIN | LEFT JOINの逆。可読性のためLEFTに書き換え推奨 |
| FULL OUTER JOIN | 両テーブル全行保持。差分検出に有用 |
| CROSS JOIN | 直積。マトリクス生成用。結果行数に注意 |
| LATERAL JOIN | 外側テーブルの各行を参照するサブクエリ結合 |
| SEMI/ANTI JOIN | EXISTS/NOT EXISTSで記述。重複が発生しない |
| 結合条件 | ON句で明示指定。NATURAL JOINは避ける |
| ON vs WHERE | OUTER JOINでは結果が異なる。用途を理解して使い分け |
| パフォーマンス | 結合列にインデックスを設定。EXPLAIN ANALYZEで検証 |
| 内部アルゴリズム | Nested Loop, Hash Join, Merge Joinの3種類 |

---

## 次に読むべきガイド

- [03-aggregation.md](./03-aggregation.md) — GROUP BYと集約関数
- [04-subqueries.md](./04-subqueries.md) — サブクエリの活用
- [03-indexing.md](../01-advanced/03-indexing.md) — JOIN性能を左右するインデックス
- [00-window-functions.md](../01-advanced/00-window-functions.md) — ウィンドウ関数との組み合わせ
- [04-query-optimization.md](../01-advanced/04-query-optimization.md) — クエリ最適化の全体像

---

## 参考文献

1. PostgreSQL Documentation — "Joins Between Tables" https://www.postgresql.org/docs/current/tutorial-join.html
2. PostgreSQL Documentation — "EXPLAIN" https://www.postgresql.org/docs/current/sql-explain.html
3. Garcia-Molina, H., Ullman, J.D., & Widom, J. (2008). *Database Systems: The Complete Book*. Pearson.
4. Molinaro, A. (2005). *SQL Cookbook*. O'Reilly Media.
5. Winand, M. — "SQL Performance Explained" https://sql-performance-explained.com/
6. Karwin, B. (2010). *SQL Antipatterns*. Pragmatic Bookshelf.
7. Date, C.J. (2003). *An Introduction to Database Systems*. Addison Wesley. — 関係代数の理論的基礎
