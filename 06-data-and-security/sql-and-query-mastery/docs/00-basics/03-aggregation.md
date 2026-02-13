# 集約 -- GROUP BY・HAVING・集約関数

> 集約操作は複数行のデータを要約して単一の値に変換する処理であり、レポーティングや分析クエリの基盤となる。集約関数はSQL標準で定義された基本操作であり、GROUP BYやHAVINGと組み合わせることで、あらゆるビジネスレポートの土台を構築できる。

## 前提知識

- SELECT / WHERE / FROM の基本構文（[01-select.md](./01-select.md)）
- データ型（数値型、文字列型、日付型）の基礎知識
- NULL の概念と三値論理の理解

## この章で学ぶこと

1. 主要な集約関数（COUNT, SUM, AVG, MIN, MAX）の動作と内部実装の理解
2. GROUP BYとHAVINGの正しい使い方と実行順序の完全理解
3. GROUPING SETS、ROLLUP、CUBEによる多次元集約
4. 統計関数（STDDEV, VARIANCE, PERCENTILE）と文字列・配列集約
5. 集約クエリのパフォーマンス最適化と実行計画の読み方
6. RDBMS間の集約関数の互換性と移植時の注意点

---

## 1. 集約関数の基本

### 集約関数の内部動作

```
┌──────── 集約関数の内部処理フロー ──────────────────────────┐
│                                                             │
│  入力行: [100, NULL, 200, NULL, 300, 150, 200]             │
│                                                             │
│  ■ COUNT(*):                                                │
│    全行をカウント → 7                                       │
│    NULLを含む全行が対象                                     │
│                                                             │
│  ■ COUNT(col):                                              │
│    NULLを除外してカウント → 5                               │
│    [100, 200, 300, 150, 200]                                │
│                                                             │
│  ■ COUNT(DISTINCT col):                                     │
│    ユニーク値をカウント → 4                                 │
│    {100, 150, 200, 300}                                     │
│                                                             │
│  ■ SUM(col):                                                │
│    NULLを除外して合計 → 950                                 │
│    100 + 200 + 300 + 150 + 200                              │
│                                                             │
│  ■ AVG(col):                                                │
│    NULLを除外して平均 → 190 （= 950 / 5）                  │
│    ※ 7件の平均ではなく5件の平均であることに注意！           │
│                                                             │
│  ■ MIN(col) / MAX(col):                                     │
│    NULLを除外して最小/最大 → 100 / 300                     │
│                                                             │
│  ★ 重要: 全行がNULLの場合                                  │
│    COUNT(*) → 行数（0以上）                                 │
│    COUNT(col) → 0                                           │
│    SUM/AVG/MIN/MAX → NULL（0ではない！）                    │
└─────────────────────────────────────────────────────────────┘
```

### コード例1: 基本的な集約関数

```sql
-- サンプルテーブル
CREATE TABLE sales (
    id          SERIAL PRIMARY KEY,
    product     VARCHAR(50),
    category    VARCHAR(30),
    amount      DECIMAL(10, 2),
    quantity    INTEGER,
    sale_date   DATE,
    region      VARCHAR(20),
    salesperson VARCHAR(50)
);

-- サンプルデータ挿入
INSERT INTO sales (product, category, amount, quantity, sale_date, region, salesperson) VALUES
    ('ノートPC', '家電', 89000, 2, '2024-01-15', '東京', '田中'),
    ('マウス', '周辺機器', 3500, 10, '2024-01-20', '東京', '佐藤'),
    ('モニター', '家電', 45000, 3, '2024-02-01', '大阪', '鈴木'),
    ('キーボード', '周辺機器', 8000, 5, '2024-02-15', '大阪', '田中'),
    ('プリンター', '家電', 32000, 1, '2024-03-01', '東京', '佐藤'),
    ('USBメモリ', '周辺機器', 1500, 20, '2024-03-10', '福岡', '高橋'),
    ('タブレット', '家電', 55000, 2, '2024-03-20', '東京', '田中'),
    ('ヘッドセット', '周辺機器', 12000, 4, '2024-04-01', '大阪', '鈴木'),
    ('デスクトップPC', '家電', 120000, 1, '2024-04-15', '福岡', '高橋'),
    ('Webカメラ', '周辺機器', 5000, 8, '2024-04-20', '東京', '佐藤');

-- 基本集約関数
SELECT
    COUNT(*)          AS total_rows,        -- 全行数: 10
    COUNT(amount)     AS non_null_count,    -- NULL以外の行数: 10
    COUNT(DISTINCT category) AS categories, -- ユニーク値数: 2
    SUM(amount)       AS total_sales,       -- 合計
    AVG(amount)       AS avg_sale,          -- 平均
    MIN(amount)       AS min_sale,          -- 最小
    MAX(amount)       AS max_sale,          -- 最大
    MIN(sale_date)    AS first_sale,        -- 最古日
    MAX(sale_date)    AS last_sale          -- 最新日
FROM sales;

-- 条件付き集約（CASE式を使用）
SELECT
    COUNT(*) AS total,
    SUM(CASE WHEN category = '家電' THEN amount ELSE 0 END) AS electronics_total,
    SUM(CASE WHEN category = '周辺機器' THEN amount ELSE 0 END) AS peripheral_total,
    AVG(CASE WHEN region = '東京' THEN amount END) AS tokyo_avg,
    -- NULLが返る行はAVGの分母に含まれない
    COUNT(CASE WHEN amount > 50000 THEN 1 END) AS high_value_count
FROM sales;
```

### コード例2: COUNT の3つの使い方と注意点

```sql
-- COUNT(*): NULLを含む全行を数える
SELECT COUNT(*) FROM employees;           -- → 100

-- COUNT(column): NULLを除いた行を数える
SELECT COUNT(phone) FROM employees;       -- → 85 (15人はNULL)

-- COUNT(DISTINCT column): ユニーク値を数える
SELECT COUNT(DISTINCT department_id) FROM employees;  -- → 8

-- ★ COUNT(DISTINCT) の複数列版（PostgreSQL）
SELECT COUNT(DISTINCT (department_id, status)) FROM employees;
-- → (department_id, status) の組み合わせのユニーク数

-- ★ 注意: 全行NULLの場合の挙動
CREATE TABLE empty_test (val INTEGER);
INSERT INTO empty_test VALUES (NULL), (NULL), (NULL);

SELECT
    COUNT(*)    AS star,   -- 3（行数は数える）
    COUNT(val)  AS col,    -- 0（NULLは除外）
    SUM(val)    AS total,  -- NULL（0ではない！）
    AVG(val)    AS average -- NULL
FROM empty_test;

-- SUM/AVGがNULLのとき0にしたい場合
SELECT COALESCE(SUM(val), 0) AS safe_sum FROM empty_test;
```

### コード例3: 集約関数とNULLの詳細

```sql
-- NULLの影響を実演
CREATE TABLE scores (
    student_id INTEGER,
    subject    VARCHAR(20),
    score      INTEGER  -- NULLは未受験を表す
);

INSERT INTO scores VALUES
    (1, '数学', 80), (1, '英語', 90), (1, '国語', NULL),
    (2, '数学', 70), (2, '英語', NULL), (2, '国語', 60),
    (3, '数学', NULL), (3, '英語', NULL), (3, '国語', NULL);

-- 生徒別の集計
SELECT
    student_id,
    COUNT(*) AS total_subjects,           -- 全科目数
    COUNT(score) AS taken_subjects,       -- 受験科目数
    SUM(score) AS total_score,            -- 合計点（NULLは無視）
    AVG(score) AS avg_score,              -- 平均点（受験科目のみ）
    AVG(COALESCE(score, 0)) AS avg_with_zero  -- 未受験を0点として計算
FROM scores
GROUP BY student_id
ORDER BY student_id;

-- 結果:
-- student_id | total | taken | sum | avg  | avg_with_zero
-- 1          | 3     | 2     | 170 | 85.0 | 56.67
-- 2          | 3     | 2     | 130 | 65.0 | 43.33
-- 3          | 3     | 0     | NULL| NULL | 0.00
```

---

## 2. GROUP BY

### SQL論理実行順序

```
┌──────── SQL論理実行順序（集約を含む場合）────────────────────┐
│                                                              │
│  ① FROM / JOIN     テーブルの結合・直積の生成               │
│  ② WHERE           行レベルのフィルタ（集約前）             │
│  ③ GROUP BY        行のグループ化                           │
│  ④ 集約関数の計算   SUM, COUNT, AVG等の計算                 │
│  ⑤ HAVING          グループレベルのフィルタ（集約後）       │
│  ⑥ SELECT          列の選択・式の計算                       │
│  ⑦ DISTINCT        重複の排除                               │
│  ⑧ ORDER BY        ソート                                   │
│  ⑨ LIMIT/OFFSET    結果の制限                               │
│                                                              │
│  ★ 重要ルール:                                              │
│  - WHERE句では集約関数は使用不可（③の前に実行されるため）    │
│  - SELECT句の非集約列は全てGROUP BYに含める必要がある        │
│  - ORDER BY句では集約関数のエイリアスを使用可能              │
│  - HAVING句では集約関数を使用可能                            │
└──────────────────────────────────────────────────────────────┘
```

### コード例4: GROUP BYの基本と応用

```sql
-- 基本: カテゴリ別の売上集計
SELECT
    category,
    COUNT(*) AS sales_count,
    SUM(amount) AS total_amount,
    ROUND(AVG(amount), 2) AS avg_amount,
    MIN(amount) AS min_amount,
    MAX(amount) AS max_amount
FROM sales
GROUP BY category
ORDER BY total_amount DESC;

-- 複数列でグループ化
SELECT
    category,
    region,
    COUNT(*) AS sales_count,
    SUM(amount) AS total_amount
FROM sales
GROUP BY category, region
ORDER BY category, total_amount DESC;

-- 式でグループ化（月別集計）
SELECT
    DATE_TRUNC('month', sale_date) AS month,
    SUM(amount) AS monthly_total,
    COUNT(*) AS transaction_count,
    ROUND(AVG(amount), 2) AS avg_per_transaction
FROM sales
GROUP BY DATE_TRUNC('month', sale_date)
ORDER BY month;

-- 年月 + カテゴリのクロス集計
SELECT
    TO_CHAR(sale_date, 'YYYY-MM') AS year_month,
    category,
    SUM(amount) AS total,
    COUNT(*) AS cnt
FROM sales
GROUP BY TO_CHAR(sale_date, 'YYYY-MM'), category
ORDER BY year_month, category;

-- CASE式でグループ化（金額帯別）
SELECT
    CASE
        WHEN amount < 5000 THEN '低額（5千未満）'
        WHEN amount < 50000 THEN '中額（5千-5万）'
        ELSE '高額（5万以上）'
    END AS price_range,
    COUNT(*) AS count,
    SUM(amount) AS total,
    ROUND(AVG(amount), 2) AS avg
FROM sales
GROUP BY
    CASE
        WHEN amount < 5000 THEN '低額（5千未満）'
        WHEN amount < 50000 THEN '中額（5千-5万）'
        ELSE '高額（5万以上）'
    END
ORDER BY avg;
```

### GROUP BY の実行フロー

```
┌─────────────────── GROUP BY 処理の流れ ──────────────────────────┐
│                                                                   │
│  元テーブル (sales)                                                │
│  ┌──────────┬──────────┬──────────┐                              │
│  │ category │ region   │ amount   │                              │
│  ├──────────┼──────────┼──────────┤                              │
│  │ 食品     │ 東京     │ 1000     │                              │
│  │ 食品     │ 大阪     │ 1500     │                              │
│  │ 家電     │ 東京     │ 5000     │                              │
│  │ 食品     │ 東京     │ 2000     │                              │
│  │ 家電     │ 大阪     │ 3000     │                              │
│  └──────────┴──────────┴──────────┘                              │
│       │                                                           │
│       ▼ GROUP BY category                                         │
│  ┌──────────┬────────────────────────────┐                       │
│  │ category │ グループ内の行              │                       │
│  ├──────────┼────────────────────────────┤                       │
│  │ 食品     │ {1000, 1500, 2000}         │ → SUM=4500, AVG=1500 │
│  │ 家電     │ {5000, 3000}               │ → SUM=8000, AVG=4000 │
│  └──────────┴────────────────────────────┘                       │
│       │                                                           │
│       ▼ 集約関数の適用                                             │
│  ┌──────────┬───────┬───────┬───────┐                            │
│  │ category │ COUNT │ SUM   │ AVG   │                            │
│  ├──────────┼───────┼───────┼───────┤                            │
│  │ 食品     │ 3     │ 4500  │ 1500  │                            │
│  │ 家電     │ 2     │ 8000  │ 4000  │                            │
│  └──────────┴───────┴───────┴───────┘                            │
│                                                                   │
│  ★ GROUP BYの内部実装（PostgreSQL）:                              │
│  ┌──────────────────────────────────────────────┐                │
│  │ HashAggregate: ハッシュテーブルでグループ化   │                │
│  │ - work_mem内に収まる場合に使用                │                │
│  │ - O(N) の計算量                               │                │
│  │                                               │                │
│  │ GroupAggregate: ソート済みデータをスキャン     │                │
│  │ - インデックスが利用可能な場合に高速           │                │
│  │ - ソート + 1パスの計算量                       │                │
│  │                                               │                │
│  │ Mixed: PostgreSQL 13+ ではハッシュが溢れたら  │                │
│  │ ディスクにスピルする機能が追加                 │                │
│  └──────────────────────────────────────────────┘                │
└───────────────────────────────────────────────────────────────────┘
```

---

## 3. HAVING

### コード例5: HAVINGによるグループのフィルタリング

```sql
-- WHEREは行をフィルタ、HAVINGはグループをフィルタ
SELECT
    category,
    COUNT(*) AS sales_count,
    SUM(amount) AS total_amount
FROM sales
WHERE sale_date >= '2024-01-01'     -- ① 行レベルのフィルタ（集約前）
GROUP BY category
HAVING SUM(amount) >= 10000         -- ② グループレベルのフィルタ（集約後）
ORDER BY total_amount DESC;

-- 実用例: 注文回数が5回以上の顧客
SELECT
    customer_id,
    COUNT(*) AS order_count,
    SUM(total_amount) AS lifetime_value
FROM orders
GROUP BY customer_id
HAVING COUNT(*) >= 5
ORDER BY lifetime_value DESC;

-- 実用例: 重複データの検出
SELECT
    email,
    COUNT(*) AS duplicate_count
FROM users
GROUP BY email
HAVING COUNT(*) > 1;

-- 実用例: 集約結果の複数条件フィルタ
SELECT
    category,
    region,
    COUNT(*) AS cnt,
    SUM(amount) AS total,
    AVG(amount) AS avg
FROM sales
GROUP BY category, region
HAVING COUNT(*) >= 2
   AND SUM(amount) > 10000
   AND AVG(amount) > 3000
ORDER BY total DESC;

-- 実用例: HAVINGでサブクエリを使用
SELECT
    category,
    AVG(amount) AS avg_amount
FROM sales
GROUP BY category
HAVING AVG(amount) > (SELECT AVG(amount) FROM sales);
-- → 全体平均より高い平均金額のカテゴリのみ
```

### WHERE vs HAVING の実行タイミング

```
┌──────────── WHERE vs HAVING の違い ─────────────────────┐
│                                                          │
│  FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY  │
│           │                   │                          │
│    ┌──────┴──────┐    ┌──────┴──────┐                   │
│    │  WHERE      │    │  HAVING     │                   │
│    │  行をフィルタ│    │グループを    │                   │
│    │  集約前      │    │フィルタ      │                   │
│    │  集約関数    │    │  集約後      │                   │
│    │  使用不可    │    │  集約関数    │                   │
│    │  インデックス│    │  使用可能    │                   │
│    │  利用可能    │    │  インデックス│                   │
│    │             │    │  利用不可    │                   │
│    └─────────────┘    └─────────────┘                   │
│                                                          │
│  ★ パフォーマンスの鉄則:                                │
│  「WHEREで書ける条件はWHEREに書く」                      │
│                                                          │
│  例: WHERE amount > 100     ← 各行で判定（高速）        │
│      HAVING SUM(amount) > 10000 ← グループで判定        │
│                                                          │
│  NG: HAVING category = '家電'                            │
│  OK: WHERE category = '家電'                             │
│  → 同じ結果だがWHEREの方が先にフィルタされ高速          │
└──────────────────────────────────────────────────────────┘
```

---

## 4. 高度な集約

### コード例6: GROUPING SETS / ROLLUP / CUBE

```sql
-- GROUPING SETS: 複数の集約レベルを一度に取得
SELECT
    category,
    region,
    SUM(amount) AS total
FROM sales
GROUP BY GROUPING SETS (
    (category, region),  -- カテゴリ × 地域
    (category),          -- カテゴリ別小計
    (region),            -- 地域別小計
    ()                   -- 総計
)
ORDER BY category NULLS LAST, region NULLS LAST;

-- ROLLUP: 階層的な小計 + 総計
-- ROLLUP(A, B) = GROUPING SETS((A,B), (A), ())
SELECT
    COALESCE(category, '【総計】') AS category,
    COALESCE(region, '【小計】') AS region,
    SUM(amount) AS total,
    COUNT(*) AS cnt
FROM sales
GROUP BY ROLLUP (category, region);

-- CUBE: 全組み合わせの集約
-- CUBE(A, B) = GROUPING SETS((A,B), (A), (B), ())
SELECT
    category,
    region,
    SUM(amount) AS total
FROM sales
GROUP BY CUBE (category, region);

-- GROUPING関数: NULLが「集約によるもの」か「データのNULL」かを区別
SELECT
    CASE WHEN GROUPING(category) = 1 THEN '【全カテゴリ】'
         ELSE category END AS category,
    CASE WHEN GROUPING(region) = 1 THEN '【全地域】'
         ELSE region END AS region,
    SUM(amount) AS total,
    GROUPING(category) AS is_cat_total,
    GROUPING(region) AS is_reg_total,
    GROUPING(category, region) AS grouping_id
FROM sales
GROUP BY CUBE (category, region)
ORDER BY GROUPING(category, region), category, region;

-- 部分ROLLUP: 一部の列のみROLLUP
SELECT
    category,
    region,
    TO_CHAR(sale_date, 'YYYY-MM') AS month,
    SUM(amount) AS total
FROM sales
GROUP BY category, ROLLUP(region, TO_CHAR(sale_date, 'YYYY-MM'));
-- → categoryは常に存在し、region→monthの階層でROLLUP
```

### ROLLUP / CUBE / GROUPING SETS の展開図

```
┌──────── ROLLUP vs CUBE vs GROUPING SETS ───────────────────────┐
│                                                                 │
│  GROUP BY ROLLUP(A, B, C) は以下と等価:                         │
│  GROUP BY GROUPING SETS(                                        │
│      (A, B, C),   -- 詳細                                      │
│      (A, B),      -- Cの小計                                   │
│      (A),         -- B,Cの小計                                  │
│      ()           -- 総計                                       │
│  )                                                              │
│  → N+1 = 4グループ（列数+1）                                   │
│                                                                 │
│  GROUP BY CUBE(A, B, C) は以下と等価:                           │
│  GROUP BY GROUPING SETS(                                        │
│      (A, B, C),   (A, B),   (A, C),   (B, C),                  │
│      (A),         (B),      (C),                                │
│      ()                                                         │
│  )                                                              │
│  → 2^N = 8グループ（2の列数乗）                                │
│                                                                 │
│  ★ パフォーマンスへの影響:                                     │
│  ┌───────────────────────────────────────┐                     │
│  │ CUBE(A,B,C,D) → 2^4 = 16グループ     │                     │
│  │ CUBE(A,B,C,D,E) → 2^5 = 32グループ   │                     │
│  │ 列数が増えると指数的にグループ数が増加 │                     │
│  │ → CUBEは3列以下で使用を推奨          │                     │
│  └───────────────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

### コード例7: FILTER句（PostgreSQL / SQLite）

```sql
-- FILTER句: 条件付き集約をCASE式より簡潔に書ける
SELECT
    category,
    COUNT(*) AS total_count,
    COUNT(*) FILTER (WHERE region = '東京') AS tokyo_count,
    COUNT(*) FILTER (WHERE region = '大阪') AS osaka_count,
    COUNT(*) FILTER (WHERE region = '福岡') AS fukuoka_count,
    SUM(amount) FILTER (WHERE sale_date >= '2024-01-01'
                        AND sale_date < '2024-04-01') AS q1_amount,
    SUM(amount) FILTER (WHERE sale_date >= '2024-04-01'
                        AND sale_date < '2024-07-01') AS q2_amount,
    AVG(amount) FILTER (WHERE quantity >= 5) AS avg_bulk_amount
FROM sales
GROUP BY category;

-- FILTER句とCASE式の比較
-- FILTER（PostgreSQL推奨）:
SELECT COUNT(*) FILTER (WHERE status = 'active') AS active_count FROM users;

-- CASE（全RDBMS対応）:
SELECT COUNT(CASE WHEN status = 'active' THEN 1 END) AS active_count FROM users;
-- または
SELECT SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END) AS active_count FROM users;
```

### コード例8: 統計関数

```sql
-- 分散と標準偏差
SELECT
    department_id,
    COUNT(*) AS emp_count,
    AVG(salary) AS avg_salary,
    STDDEV_SAMP(salary) AS salary_stddev,      -- 標本標準偏差（N-1）
    STDDEV_POP(salary) AS salary_stddev_pop,   -- 母標準偏差（N）
    VAR_SAMP(salary) AS salary_variance,       -- 標本分散
    VAR_POP(salary) AS salary_variance_pop     -- 母分散
FROM employees
GROUP BY department_id
HAVING COUNT(*) >= 3;  -- 統計値は3件以上で意味がある

-- 中央値（PERCENTILE_CONT）
SELECT
    department_id,
    AVG(salary) AS mean_salary,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary,
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY salary) AS q1_salary,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY salary) AS q3_salary,
    -- 四分位範囲（IQR）
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY salary)
    - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY salary) AS iqr,
    -- 最頻値（モード）はMODE()で取得（PostgreSQL）
    MODE() WITHIN GROUP (ORDER BY salary) AS mode_salary
FROM employees
GROUP BY department_id;

-- PERCENTILE_DISC: 離散値（実際に存在する値を返す）
SELECT
    department_id,
    PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY salary) AS median_exact
FROM employees
GROUP BY department_id;
-- PERCENTILE_CONTは補間値を返すが、PERCENTILE_DISCは実在する値を返す

-- 相関係数と回帰分析
SELECT
    CORR(age, salary) AS correlation,           -- 相関係数
    REGR_SLOPE(salary, age) AS slope,           -- 回帰直線の傾き
    REGR_INTERCEPT(salary, age) AS intercept,   -- 回帰直線の切片
    REGR_R2(salary, age) AS r_squared,          -- 決定係数
    REGR_COUNT(salary, age) AS valid_pairs      -- 有効なペア数
FROM employees;
```

### コード例9: 文字列集約と配列集約

```sql
-- 文字列集約（STRING_AGG / GROUP_CONCAT）
-- PostgreSQL / SQL Server
SELECT
    department_id,
    STRING_AGG(name, ', ' ORDER BY name) AS member_list,
    STRING_AGG(DISTINCT title, ' / ' ORDER BY title) AS unique_titles
FROM employees
GROUP BY department_id;

-- MySQL
-- SELECT department_id, GROUP_CONCAT(name ORDER BY name SEPARATOR ', ')
-- FROM employees GROUP BY department_id;

-- 配列集約（PostgreSQL固有）
SELECT
    department_id,
    ARRAY_AGG(name ORDER BY hired_date) AS members_by_tenure,
    ARRAY_AGG(DISTINCT department_id) AS depts,  -- 重複排除
    ARRAY_AGG(salary ORDER BY salary DESC) AS salaries_desc
FROM employees
GROUP BY department_id;

-- JSON集約（PostgreSQL 9.5+ / MySQL 5.7+）
-- PostgreSQL
SELECT
    department_id,
    JSON_AGG(
        JSON_BUILD_OBJECT(
            'name', name,
            'salary', salary,
            'hired', hired_date
        ) ORDER BY salary DESC
    ) AS members_json
FROM employees
GROUP BY department_id;

-- BOOL集約（PostgreSQL）
SELECT
    department_id,
    BOOL_AND(is_active) AS all_active,   -- 全員active？
    BOOL_OR(is_manager) AS has_manager,  -- マネージャーがいる？
    EVERY(salary > 300000) AS all_above_300k  -- 全員30万超？
FROM employees
GROUP BY department_id;

-- BIT集約
SELECT
    department_id,
    BIT_AND(permissions) AS common_perms,  -- 共通権限（AND）
    BIT_OR(permissions) AS union_perms     -- 権限の和（OR）
FROM employees
GROUP BY department_id;
```

---

## 5. パフォーマンス最適化

### 集約クエリの実行計画

```
┌──────── 集約クエリの実行計画の読み方 ──────────────────────────┐
│                                                                │
│  ■ HashAggregate                                               │
│  ┌────────────────────────────────────────────┐               │
│  │ 動作: ハッシュテーブルでグループを管理       │               │
│  │ 特徴:                                       │               │
│  │ - 入力の事前ソートが不要                     │               │
│  │ - work_mem内に収まる場合に高速               │               │
│  │ - メモリ使用量 = グループ数 × 行サイズ       │               │
│  │ 使われる場面:                                │               │
│  │ - グループ数が少ない〜中程度                 │               │
│  │ - ORDER BYがない                             │               │
│  │ - 適切なインデックスがない                   │               │
│  └────────────────────────────────────────────┘               │
│                                                                │
│  ■ GroupAggregate                                              │
│  ┌────────────────────────────────────────────┐               │
│  │ 動作: ソート済みデータを1パスでグループ化    │               │
│  │ 特徴:                                       │               │
│  │ - 入力がソート済み（インデックス or Sort）   │               │
│  │ - メモリ使用量が少ない（1グループ分のみ）    │               │
│  │ - ORDER BY + GROUP BYが同一列なら有利        │               │
│  │ 使われる場面:                                │               │
│  │ - GROUP BY列にインデックスがある             │               │
│  │ - ORDER BY + GROUP BYを兼ねられる            │               │
│  │ - work_memが不足している                     │               │
│  └────────────────────────────────────────────┘               │
│                                                                │
│  ■ PostgreSQL 13+ の改善                                       │
│  ┌────────────────────────────────────────────┐               │
│  │ - HashAggregateのディスクスピル対応           │               │
│  │ - 複数GROUP BYセットの効率的な処理            │               │
│  │ - Incremental Sort との組み合わせ             │               │
│  └────────────────────────────────────────────┘               │
└────────────────────────────────────────────────────────────────┘
```

### コード例10: パフォーマンス最適化の実践

```sql
-- ■ インデックスによる最適化
-- GROUP BY列にインデックスがあるとGroupAggregateが使われやすい
CREATE INDEX idx_sales_category ON sales (category);
CREATE INDEX idx_sales_cat_region ON sales (category, region);

-- 実行計画の確認
EXPLAIN (ANALYZE, BUFFERS)
SELECT category, SUM(amount)
FROM sales
GROUP BY category;

-- ■ WHERE句で先にフィルタ（最も効果的な最適化）
-- NG: 全行を読んでからHAVINGでフィルタ
SELECT region, SUM(amount)
FROM sales
GROUP BY region
HAVING region IN ('東京', '大阪');  -- 全行をGROUP BY後にフィルタ

-- OK: WHEREで先に絞り込み
SELECT region, SUM(amount)
FROM sales
WHERE region IN ('東京', '大阪')     -- 先に行を絞る（インデックス利用可）
GROUP BY region;

-- ■ 部分インデックスの活用
CREATE INDEX idx_sales_active ON sales (category, amount)
WHERE sale_date >= '2024-01-01';

-- ■ マテリアライズドビューによるキャッシュ（PostgreSQL）
CREATE MATERIALIZED VIEW mv_monthly_sales AS
SELECT
    DATE_TRUNC('month', sale_date) AS month,
    category,
    region,
    SUM(amount) AS total_amount,
    COUNT(*) AS transaction_count,
    AVG(amount) AS avg_amount
FROM sales
GROUP BY DATE_TRUNC('month', sale_date), category, region;

CREATE UNIQUE INDEX ON mv_monthly_sales (month, category, region);

-- 更新
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_monthly_sales;

-- ■ 近似集約（大規模データ向け）
-- PostgreSQL拡張: HyperLogLog（概算COUNT DISTINCT）
-- HLLを使えば数十億行のCOUNT DISTINCTを定数メモリで計算可能
-- CREATE EXTENSION hll;
-- SELECT hll_cardinality(hll_add_agg(hll_hash_text(user_id)))
-- FROM huge_events;

-- ■ work_memの調整
SET work_mem = '256MB';  -- HashAggregateのメモリ上限を引き上げ
EXPLAIN (ANALYZE, BUFFERS)
SELECT category, region, COUNT(*)
FROM large_table
GROUP BY category, region;
```

---

## 6. RDBMS間の互換性

### 集約関数サポート比較表

| 関数 | PostgreSQL | MySQL 8.0+ | SQL Server | Oracle | SQLite |
|------|-----------|------------|------------|--------|--------|
| COUNT/SUM/AVG/MIN/MAX | 対応 | 対応 | 対応 | 対応 | 対応 |
| COUNT(DISTINCT) | 対応 | 対応 | 対応 | 対応 | 対応 |
| STRING_AGG | 対応(9.0+) | GROUP_CONCAT | 対応(2017+) | LISTAGG | GROUP_CONCAT |
| ARRAY_AGG | 対応 | 非対応 | 非対応 | 非対応 | 非対応 |
| JSON_AGG | 対応(9.5+) | JSON_ARRAYAGG | 非対応 | JSON_ARRAYAGG | 非対応 |
| FILTER句 | 対応(9.4+) | 非対応 | 非対応 | 非対応 | 対応(3.30+) |
| BOOL_AND/BOOL_OR | 対応 | BIT_AND/BIT_OR | 非対応 | 非対応 | 非対応 |
| PERCENTILE_CONT | 対応 | 非対応 | PERCENTILE_CONT | 対応 | 非対応 |
| MODE() | 対応 | 非対応 | 非対応 | STATS_MODE | 非対応 |
| STDDEV_SAMP/POP | 対応 | 対応 | STDEV/STDEVP | 対応 | 非対応 |
| CORR/REGR_* | 対応 | 非対応 | 非対応 | 対応 | 非対応 |
| GROUPING SETS | 対応(9.5+) | 非対応 | 対応 | 対応 | 非対応 |
| ROLLUP | 対応(9.5+) | 対応 | 対応 | 対応 | 非対応 |
| CUBE | 対応(9.5+) | 非対応 | 対応 | 対応 | 非対応 |
| GROUPING() | 対応 | 対応(8.0+) | 対応 | 対応 | 非対応 |

### RDBMS固有の構文

```sql
-- ■ MySQL: GROUP_CONCATの注意
-- デフォルト最大長は1024バイト（切り詰められる！）
SET GROUP_CONCAT_MAX_LEN = 1000000;
SELECT department_id,
       GROUP_CONCAT(name ORDER BY name SEPARATOR ', ') AS members
FROM employees
GROUP BY department_id;

-- ■ MySQL: WITH ROLLUP構文
SELECT category, region, SUM(amount)
FROM sales
GROUP BY category, region WITH ROLLUP;
-- ※ CUBE, GROUPING SETSは非対応

-- ■ SQL Server: STRING_AGGの注意
-- SQL Server 2017+で対応。それ以前はFOR XML PATHを使用
SELECT
    department_id,
    STRING_AGG(name, ', ') WITHIN GROUP (ORDER BY name) AS members
FROM employees
GROUP BY department_id;

-- ■ Oracle: LISTAGG（文字列集約）
SELECT
    department_id,
    LISTAGG(name, ', ') WITHIN GROUP (ORDER BY name) AS members
FROM employees
GROUP BY department_id;

-- Oracle 19c+: LISTAGG DISTINCT
SELECT
    department_id,
    LISTAGG(DISTINCT title, ', ') WITHIN GROUP (ORDER BY title) AS titles
FROM employees
GROUP BY department_id;
```

---

## 集約関数一覧表

| 関数 | 説明 | NULL扱い | 使用例 | 注意点 |
|------|------|---------|--------|--------|
| COUNT(*) | 全行数 | 含む | `COUNT(*)` | NULLも数える |
| COUNT(col) | 非NULL行数 | 除外 | `COUNT(email)` | NULL行は除外 |
| COUNT(DISTINCT col) | ユニーク値数 | 除外 | `COUNT(DISTINCT category)` | メモリ消費大 |
| SUM(col) | 合計 | 除外 | `SUM(amount)` | 全NULLでNULL返却 |
| AVG(col) | 平均 | 除外 | `AVG(salary)` | 分母はNULL除外後の件数 |
| MIN(col) | 最小値 | 除外 | `MIN(price)` | 文字列・日付にも使用可 |
| MAX(col) | 最大値 | 除外 | `MAX(created_at)` | 文字列・日付にも使用可 |
| STRING_AGG | 文字列結合 | 除外 | `STRING_AGG(name, ',')` | MySQL:GROUP_CONCAT |
| ARRAY_AGG | 配列化 | 含む | `ARRAY_AGG(tag)` | PostgreSQL固有 |
| JSON_AGG | JSON配列化 | 含む | `JSON_AGG(col)` | PostgreSQL 9.5+ |
| BOOL_AND/OR | 論理積/和 | 除外 | `BOOL_AND(is_active)` | PostgreSQL固有 |
| EVERY | 全件TRUE判定 | 除外 | `EVERY(score > 60)` | BOOL_ANDのエイリアス |
| STDDEV_SAMP | 標本標準偏差 | 除外 | `STDDEV_SAMP(salary)` | N-1で除算 |
| STDDEV_POP | 母標準偏差 | 除外 | `STDDEV_POP(salary)` | Nで除算 |
| VAR_SAMP | 標本分散 | 除外 | `VAR_SAMP(salary)` | N-1で除算 |
| VAR_POP | 母分散 | 除外 | `VAR_POP(salary)` | Nで除算 |
| PERCENTILE_CONT | 連続パーセンタイル | 除外 | `PERCENTILE_CONT(0.5)` | 補間値を返す |
| PERCENTILE_DISC | 離散パーセンタイル | 除外 | `PERCENTILE_DISC(0.5)` | 実在値を返す |
| MODE | 最頻値 | 除外 | `MODE() WITHIN GROUP(...)` | PostgreSQL固有 |
| CORR | 相関係数 | 除外 | `CORR(x, y)` | -1.0 〜 1.0 |

## ROLLUP vs CUBE vs GROUPING SETS 比較表

| 機能 | 生成される集約 | 行数目安 | 用途 | 対応RDBMS |
|------|---------------|---------|------|-----------|
| GROUP BY A, B | (A, B) | グループ数 | 基本集約 | 全RDBMS |
| ROLLUP(A, B) | (A,B), (A), () | N+1 レベル | 帳票の小計行 | PG, SS, Oracle |
| CUBE(A, B) | (A,B), (A), (B), () | 2^N 組み合わせ | 多次元分析 | PG, SS, Oracle |
| GROUPING SETS | 明示指定 | 指定数分 | 柔軟な集約 | PG, SS, Oracle |

---

## エッジケース

### エッジケース1: 空テーブルに対する集約

```sql
-- 空テーブル（行が0件）に対する集約
CREATE TABLE empty_table (amount INTEGER);

SELECT
    COUNT(*)   AS cnt,      -- 0（空でも1行返る）
    SUM(amount) AS total,   -- NULL（0ではない！）
    AVG(amount) AS avg,     -- NULL
    MIN(amount) AS mn,      -- NULL
    MAX(amount) AS mx       -- NULL
FROM empty_table;
-- → 結果: 1行が返る（0行ではない）

-- GROUP BYを付けた場合
SELECT amount, COUNT(*) FROM empty_table GROUP BY amount;
-- → 結果: 0行（グループ自体が存在しない）

-- ★ 安全な書き方
SELECT COALESCE(SUM(amount), 0) AS safe_total FROM empty_table;
```

### エッジケース2: GROUP BYとNULL

```sql
-- NULLは1つのグループとして扱われる
INSERT INTO sales (product, category, amount, quantity, sale_date, region) VALUES
    ('不明商品', NULL, 1000, 1, '2024-05-01', NULL);

SELECT category, COUNT(*), SUM(amount)
FROM sales
GROUP BY category;
-- → NULLも1つのグループとして集約される
-- → (NULL, 1, 1000) が結果に含まれる

-- COALESCE で明示的に扱う
SELECT COALESCE(category, '未分類') AS category, COUNT(*)
FROM sales
GROUP BY COALESCE(category, '未分類');
```

### エッジケース3: DISTINCT + 集約の組み合わせ

```sql
-- COUNT(DISTINCT)のメモリ消費
-- 大テーブルでCOUNT(DISTINCT)を使うとハッシュテーブルのメモリ消費が大きい
SELECT COUNT(DISTINCT user_id) FROM huge_events;  -- 数百万ユニーク値

-- 近似でよい場合の代替手段
-- PostgreSQL: HyperLogLog拡張
-- BigQuery: APPROX_COUNT_DISTINCT(user_id)
-- Redshift: APPROXIMATE COUNT(DISTINCT user_id)

-- 複数DISTINCT集約は非効率
-- NG: 2つのDISTINCT（2つのハッシュテーブルが必要）
SELECT
    COUNT(DISTINCT user_id) AS unique_users,
    COUNT(DISTINCT product_id) AS unique_products
FROM orders;

-- OK: サブクエリで分割（大テーブルの場合）
SELECT
    (SELECT COUNT(DISTINCT user_id) FROM orders) AS unique_users,
    (SELECT COUNT(DISTINCT product_id) FROM orders) AS unique_products;
```

### エッジケース4: 浮動小数点の集約精度

```sql
-- DECIMAL型とFLOAT型のSUM精度の違い
CREATE TABLE float_test (
    val_decimal DECIMAL(10,2),
    val_float   FLOAT
);

INSERT INTO float_test VALUES (0.1, 0.1), (0.2, 0.2), (0.3, 0.3);

SELECT
    SUM(val_decimal) AS decimal_sum,  -- 0.60（正確）
    SUM(val_float) AS float_sum       -- 0.6000000000000001（誤差あり）
FROM float_test;

-- 金額計算では必ずDECIMAL/NUMERICを使用すること
```

---

## アンチパターン

### アンチパターン1: GROUP BYに含めない列をSELECTする

```sql
-- NG: nameはGROUP BYに含まれていない
SELECT department_id, name, AVG(salary)
FROM employees
GROUP BY department_id;
-- → PostgreSQLではエラー
-- → MySQLでは不定値が返る（ONLY_FULL_GROUP_BY無効時）

-- OK: GROUP BYに含めるか、集約関数で包む
SELECT department_id, MIN(name) AS first_name, AVG(salary)
FROM employees
GROUP BY department_id;

-- OK: ウィンドウ関数を使う
SELECT DISTINCT
    department_id,
    FIRST_VALUE(name) OVER (PARTITION BY department_id ORDER BY salary DESC),
    AVG(salary) OVER (PARTITION BY department_id)
FROM employees;

-- OK: 最高給与者の名前を取得（サブクエリ）
SELECT
    e.department_id,
    e.name AS top_earner,
    d.avg_salary
FROM employees e
INNER JOIN (
    SELECT department_id, AVG(salary) AS avg_salary, MAX(salary) AS max_salary
    FROM employees
    GROUP BY department_id
) d ON e.department_id = d.department_id
   AND e.salary = d.max_salary;
```

### アンチパターン2: AVGのNULL問題を無視する

```sql
-- NGパターン: NULLの影響を考慮しない
-- データ: [100, NULL, 200, NULL, 300]
SELECT AVG(score) FROM tests;
-- → 200（3件の平均）、NULLを含む5件の平均ではない！

-- OK: NULLを0として扱いたい場合は明示する
SELECT AVG(COALESCE(score, 0)) FROM tests;
-- → 120（5件の平均）

-- OK: NULL行を除外した上での平均が正しい場合
SELECT AVG(score) FROM tests WHERE score IS NOT NULL;
-- → 200（明示的にNULL除外）
```

### アンチパターン3: HAVINGでWHEREの仕事をさせる

```sql
-- NG: 行レベルの条件をHAVINGに書く
SELECT region, SUM(amount) AS total
FROM sales
GROUP BY region
HAVING region = '東京';  -- WHERE で書くべき

-- OK: WHEREで先にフィルタ
SELECT region, SUM(amount) AS total
FROM sales
WHERE region = '東京'
GROUP BY region;
-- → WHEREの方がインデックスを利用でき高速
```

### アンチパターン4: 不要なDISTINCTとGROUP BYの併用

```sql
-- NG: GROUP BYの結果はすでにユニーク
SELECT DISTINCT category, COUNT(*)
FROM sales
GROUP BY category;
-- DISTINCTは無意味（GROUP BYで既にユニーク化されている）

-- OK: DISTINCTを削除
SELECT category, COUNT(*)
FROM sales
GROUP BY category;
```

---

## セキュリティに関する注意

```
┌──────── セキュリティ上の考慮点 ────────────────────────────┐
│                                                            │
│  ■ 集約によるデータ漏洩                                    │
│  ┌──────────────────────────────────────────────┐         │
│  │ - COUNT(*)の結果から行の存在を推測可能         │         │
│  │ - MIN/MAXから個別の値が特定されうる            │         │
│  │ - GROUP BYの結果が1件のグループは個人特定可能  │         │
│  │                                               │         │
│  │ 対策:                                         │         │
│  │ - HAVING COUNT(*) >= k で少数グループを隠す    │         │
│  │   （k-匿名性の確保）                          │         │
│  │ - ビューで集約結果のみ公開                     │         │
│  │ - 行レベルセキュリティ（RLS）の活用            │         │
│  └──────────────────────────────────────────────┘         │
│                                                            │
│  ■ SQLインジェクションとGROUP BY                           │
│  ┌──────────────────────────────────────────────┐         │
│  │ - GROUP BY列を動的に生成する場合:              │         │
│  │   NG: "GROUP BY " + user_input                │         │
│  │   OK: ホワイトリストで許可列を制限             │         │
│  │                                               │         │
│  │ - 例（Python）:                                │         │
│  │   allowed = {'category', 'region', 'month'}   │         │
│  │   if col not in allowed: raise ValueError     │         │
│  └──────────────────────────────────────────────┘         │
│                                                            │
│  ■ パフォーマンスDoS                                       │
│  ┌──────────────────────────────────────────────┐         │
│  │ - CUBE(A,B,C,D,E)は2^5=32グループを生成       │         │
│  │ - 大テーブルへの無制限集約はリソースを大量消費  │         │
│  │ - statement_timeoutを設定して保護              │         │
│  │ - ユーザー向けクエリにはLIMITを必ず付与        │         │
│  └──────────────────────────────────────────────┘         │
└────────────────────────────────────────────────────────────┘
```

---

## 演習問題

### 演習1: 基礎 -- 集約関数とGROUP BY

```sql
-- 問題: salesテーブルに対して以下のクエリを書きなさい。
-- Q1: カテゴリ別の売上件数・合計金額・平均金額・最大金額を求めよ
-- Q2: 地域別で、売上件数が2件以上のものだけを合計金額降順で表示せよ
-- Q3: 月別（YYYY-MM形式）の売上合計と前月からの増減を表示せよ
```

**模範解答:**

```sql
-- Q1: カテゴリ別集計
SELECT
    category,
    COUNT(*) AS sales_count,
    SUM(amount) AS total_amount,
    ROUND(AVG(amount), 2) AS avg_amount,
    MAX(amount) AS max_amount
FROM sales
GROUP BY category
ORDER BY total_amount DESC;

-- Q2: 地域別（2件以上）
SELECT
    region,
    COUNT(*) AS sales_count,
    SUM(amount) AS total_amount
FROM sales
GROUP BY region
HAVING COUNT(*) >= 2
ORDER BY total_amount DESC;

-- Q3: 月別推移（ウィンドウ関数との組み合わせ）
WITH monthly AS (
    SELECT
        TO_CHAR(sale_date, 'YYYY-MM') AS month,
        SUM(amount) AS total
    FROM sales
    GROUP BY TO_CHAR(sale_date, 'YYYY-MM')
)
SELECT
    month,
    total,
    LAG(total) OVER (ORDER BY month) AS prev_month,
    total - LAG(total) OVER (ORDER BY month) AS change
FROM monthly
ORDER BY month;
```

### 演習2: 応用 -- ROLLUP/CUBEと条件付き集約

```sql
-- 問題:
-- Q1: カテゴリ・地域のROLLUPで階層的小計を含む帳票を作成せよ
--     （小計行と総計行にはラベルを付けること）
-- Q2: 四半期別（Q1-Q4）のカテゴリ別売上をクロス集計（ピボット）せよ
-- Q3: 担当者別に、1万円以上の取引回数と1万円未満の取引回数を横持ちで表示せよ
```

**模範解答:**

```sql
-- Q1: ROLLUP + GROUPING関数
SELECT
    CASE WHEN GROUPING(category) = 1 THEN '★総計'
         ELSE category END AS category,
    CASE WHEN GROUPING(region) = 1 THEN
        CASE WHEN GROUPING(category) = 1 THEN '' ELSE '☆小計' END
    ELSE region END AS region,
    SUM(amount) AS total_amount,
    COUNT(*) AS cnt
FROM sales
GROUP BY ROLLUP(category, region)
ORDER BY GROUPING(category, region), category, region;

-- Q2: クロス集計（ピボットテーブル）
SELECT
    category,
    SUM(amount) FILTER (WHERE EXTRACT(QUARTER FROM sale_date) = 1) AS q1,
    SUM(amount) FILTER (WHERE EXTRACT(QUARTER FROM sale_date) = 2) AS q2,
    SUM(amount) FILTER (WHERE EXTRACT(QUARTER FROM sale_date) = 3) AS q3,
    SUM(amount) FILTER (WHERE EXTRACT(QUARTER FROM sale_date) = 4) AS q4,
    SUM(amount) AS annual_total
FROM sales
GROUP BY category
ORDER BY annual_total DESC;

-- FILTER句が使えないRDBMSの場合
SELECT
    category,
    SUM(CASE WHEN EXTRACT(QUARTER FROM sale_date) = 1 THEN amount ELSE 0 END) AS q1,
    SUM(CASE WHEN EXTRACT(QUARTER FROM sale_date) = 2 THEN amount ELSE 0 END) AS q2,
    SUM(amount) AS annual_total
FROM sales
GROUP BY category;

-- Q3: 条件別カウント
SELECT
    salesperson,
    COUNT(*) AS total_transactions,
    COUNT(*) FILTER (WHERE amount >= 10000) AS high_value_count,
    COUNT(*) FILTER (WHERE amount < 10000) AS low_value_count,
    ROUND(
        COUNT(*) FILTER (WHERE amount >= 10000)::NUMERIC / COUNT(*) * 100, 1
    ) AS high_value_pct
FROM sales
GROUP BY salesperson
ORDER BY total_transactions DESC;
```

### 演習3: 上級 -- 統計分析と実践的な集約

```sql
-- 問題:
-- Q1: 各カテゴリの売上金額の標準偏差と変動係数（CV = 標準偏差/平均）を求めよ
-- Q2: 売上金額の分布を10個のバケット（デシル）に分割し、各バケットの件数と
--     金額範囲を表示せよ
-- Q3: 担当者別の売上推移で、3回移動平均を計算し、トレンド（上昇/下降）を判定せよ
```

**模範解答:**

```sql
-- Q1: 変動係数
SELECT
    category,
    COUNT(*) AS cnt,
    ROUND(AVG(amount), 2) AS avg_amount,
    ROUND(STDDEV_SAMP(amount), 2) AS stddev_amount,
    ROUND(STDDEV_SAMP(amount) / NULLIF(AVG(amount), 0) * 100, 2) AS cv_pct
FROM sales
GROUP BY category
HAVING COUNT(*) >= 2;

-- Q2: デシル分析（NTILE使用）
WITH deciles AS (
    SELECT
        amount,
        NTILE(10) OVER (ORDER BY amount) AS decile
    FROM sales
)
SELECT
    decile,
    COUNT(*) AS cnt,
    MIN(amount) AS min_amount,
    MAX(amount) AS max_amount,
    ROUND(AVG(amount), 2) AS avg_amount,
    SUM(amount) AS total_amount
FROM deciles
GROUP BY decile
ORDER BY decile;

-- Q3: 移動平均とトレンド判定
WITH sales_ordered AS (
    SELECT
        salesperson,
        sale_date,
        amount,
        ROW_NUMBER() OVER (PARTITION BY salesperson ORDER BY sale_date) AS rn,
        AVG(amount) OVER (
            PARTITION BY salesperson ORDER BY sale_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS ma3
    FROM sales
),
with_prev AS (
    SELECT
        *,
        LAG(ma3) OVER (PARTITION BY salesperson ORDER BY sale_date) AS prev_ma3
    FROM sales_ordered
)
SELECT
    salesperson,
    sale_date,
    amount,
    ROUND(ma3, 2) AS moving_avg_3,
    CASE
        WHEN ma3 > prev_ma3 THEN '上昇↑'
        WHEN ma3 < prev_ma3 THEN '下降↓'
        ELSE '横ばい→'
    END AS trend
FROM with_prev
ORDER BY salesperson, sale_date;
```

---

## FAQ

### Q1: WHEREとHAVINGはどう使い分けるか？

WHEREは個々の行に対するフィルタ（集約前）、HAVINGはグループに対するフィルタ（集約後）。集約関数を使わない条件は必ずWHEREに書く。WHEREで先にフィルタすることで処理対象行数が減り、パフォーマンスが向上する。HAVINGは `SUM(amount) > 10000` のように集約結果を条件にする場合にのみ使用する。

### Q2: COUNT(*)とCOUNT(1)に性能差はあるか？

現代のRDBMS（PostgreSQL, MySQL, SQL Server, Oracle）では差はない。オプティマイザが同じ実行計画を生成する。`COUNT(*)`がSQL標準で意図が明確なため推奨。`COUNT(column)`は意味が異なり（NULL除外）、意図せずに使うとバグの原因になる。

### Q3: GROUP BYで日付を扱う場合の注意点は？

TIMESTAMP型でGROUP BYすると秒単位やミリ秒単位でグループ化されるため、意図通りに集約されない。`DATE_TRUNC('day', timestamp_col)`や`CAST(... AS DATE)`で丸める必要がある。タイムゾーンにも注意が必要で、`DATE_TRUNC('day', timestamp_col AT TIME ZONE 'Asia/Tokyo')`のように明示する。

### Q4: GROUP BYの列番号指定（GROUP BY 1, 2）は安全か？

SQL標準外だが多くのRDBMSで対応。列の追加や順序変更でバグが生じるため、本番コードでは列名を明示的に指定することを推奨。ただし、アドホッククエリでは便利。

### Q5: 集約関数をネストできるか？

直接のネストは不可。`SUM(COUNT(*))` はエラー。ただし、ウィンドウ関数を介せば `SUM(COUNT(*)) OVER ()` のように書ける。これはGROUP BY後の結果に対してウィンドウ集約を行う。

### Q6: SUM()がオーバーフローする場合の対策は？

INTEGER型の列をSUMするとBIGINT型で計算される（PostgreSQL）。それでも不足する場合は `SUM(amount::NUMERIC)` でNUMERIC型にキャストする。NUMERIC型は任意精度のため事実上オーバーフローしない。

---

## トラブルシューティング

### 症状1: GROUP BYの結果が期待と異なる

**原因:** TIMESTAMP列でGROUP BYしており、秒やミリ秒の違いで別グループになっている。

**解決:** `DATE_TRUNC('day', col)` で丸める。`EXPLAIN` でグループ数を確認する。

### 症状2: AVGの結果が予想より高い/低い

**原因:** NULL行がAVGの分母から除外されている。5件中2件がNULLの場合、分母は3になる。

**解決:** `AVG(COALESCE(col, 0))` でNULLを0として計算するか、意図通りの動作か確認する。

### 症状3: GROUP BYクエリが極端に遅い

**原因:** (1) HashAggregateのwork_mem不足でディスクスピル発生、(2) GROUP BY列にインデックスがない、(3) WHERE句のフィルタが不十分。

**解決:** (1) `SET work_mem = '256MB'`、(2) GROUP BY列の複合インデックス追加、(3) WHERE句で先にフィルタ、(4) マテリアライズドビューの検討。

### 症状4: STRING_AGG/GROUP_CONCATの結果が切り詰められる

**原因:** MySQL の `GROUP_CONCAT_MAX_LEN`（デフォルト1024）の制限。

**解決:** `SET GROUP_CONCAT_MAX_LEN = 1000000;` で上限を引き上げる。PostgreSQLのSTRING_AGGにはこの制限はない。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 集約関数 | COUNT, SUM, AVG, MIN, MAX が基本5関数。NULLの扱いを理解必須 |
| NULLの扱い | COUNT(*)以外はNULLを除外して計算。全NULL時はSUM/AVG=NULL |
| GROUP BY | SELECT句の非集約列は全てGROUP BYに含める。式でのグループ化も可 |
| HAVING | GROUP BY後のフィルタ。集約関数が使用可能。WHEREと使い分ける |
| ROLLUP/CUBE | 階層的/多次元的な小計を一度のクエリで取得。CUBEは列数に注意 |
| FILTER句 | 条件別集約をCASE式より簡潔に書ける（PostgreSQL / SQLite） |
| 統計関数 | STDDEV, VARIANCE, PERCENTILE_CONT, CORR等で高度な分析可能 |
| パフォーマンス | WHERE先行フィルタ、インデックス、work_mem調整が基本戦略 |

---

## 次に読むべきガイド

- [04-subqueries.md](./04-subqueries.md) -- 集約結果をサブクエリで活用
- [00-window-functions.md](../01-advanced/00-window-functions.md) -- ウィンドウ関数で行ごとの集約
- [04-query-optimization.md](../01-advanced/04-query-optimization.md) -- 集約クエリの最適化

---

## 参考文献

1. PostgreSQL Documentation -- "Aggregate Functions" https://www.postgresql.org/docs/current/functions-aggregate.html
2. PostgreSQL Documentation -- "Querying a Table: GROUP BY and HAVING" https://www.postgresql.org/docs/current/queries-table-expressions.html#QUERIES-GROUP
3. Celko, J. (2010). *Joe Celko's SQL for Smarties: Advanced SQL Programming*. Morgan Kaufmann.
4. Kline, K., Kline, D., & Hunt, B. (2008). *SQL in a Nutshell*. O'Reilly Media.
5. Winand, M. -- "GROUP BY, HAVING, and Aggregate Functions" https://use-the-index-luke.com/sql/partial-results/distinct
6. ISO/IEC 9075-2:2023 -- SQL Part 2: Foundation (Aggregate Functions)
