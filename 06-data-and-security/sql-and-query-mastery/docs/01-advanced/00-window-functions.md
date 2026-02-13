# ウィンドウ関数 — ROW_NUMBER・RANK・LAG/LEAD

> ウィンドウ関数はGROUP BYのように行を折りたたまず、各行に対して「窓」を定義し、その範囲内での集約・順位付け・前後行参照を行う機能である。SQL:2003で標準化され、分析クエリにおいてサブクエリや自己結合を排除する最も強力なツールの一つである。

## 前提知識

- SELECT / WHERE / GROUP BY の基本構文（[01-select.md](../00-basics/01-select.md)）
- 集約関数（COUNT, SUM, AVG）の動作（[03-aggregation.md](../00-basics/03-aggregation.md)）
- ORDER BY とソート処理の理解

## この章で学ぶこと

1. ウィンドウ関数の構文（OVER句、PARTITION BY、ORDER BY、フレーム）を完全理解する
2. ROW_NUMBER、RANK、DENSE_RANK、NTILE の使い分けと内部実装
3. LAG/LEAD、FIRST_VALUE/LAST_VALUE、NTH_VALUE の実践パターン
4. フレーム指定（ROWS / RANGE / GROUPS）の詳細と累積集約
5. ウィンドウ関数のクエリオプティマイザ処理と実行計画の読み方
6. RDBMS間の互換性と移植時の注意点

---

## 1. ウィンドウ関数の構文

### ウィンドウ関数の全体構造

```
┌─────────── ウィンドウ関数の構文構造 ──────────────────────────┐
│                                                                │
│  関数名(...) OVER (                                            │
│      [PARTITION BY 列1, 列2, ...]   -- グループ分割（省略可）  │
│      [ORDER BY 列 ASC|DESC, ...]    -- 並び順（省略可）        │
│      [frame_clause]                  -- フレーム定義（省略可）  │
│  )                                                             │
│                                                                │
│  frame_clause:                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ {ROWS | RANGE | GROUPS}                                  │ │
│  │ BETWEEN                                                  │ │
│  │   { UNBOUNDED PRECEDING                                  │ │
│  │   | <N> PRECEDING                                        │ │
│  │   | CURRENT ROW                                          │ │
│  │   | <N> FOLLOWING                                        │ │
│  │   | UNBOUNDED FOLLOWING }                                │ │
│  │ AND                                                      │ │
│  │   { UNBOUNDED PRECEDING                                  │ │
│  │   | <N> PRECEDING                                        │ │
│  │   | CURRENT ROW                                          │ │
│  │   | <N> FOLLOWING                                        │ │
│  │   | UNBOUNDED FOLLOWING }                                │ │
│  │                                                          │ │
│  │ [EXCLUDE { CURRENT ROW | GROUP | TIES | NO OTHERS }]     │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  デフォルトフレーム:                                            │
│  - ORDER BY 指定時: RANGE BETWEEN UNBOUNDED PRECEDING          │
│                     AND CURRENT ROW                             │
│  - ORDER BY 省略時: RANGE BETWEEN UNBOUNDED PRECEDING          │
│                     AND UNBOUNDED FOLLOWING                     │
└────────────────────────────────────────────────────────────────┘
```

### SQL実行順序におけるウィンドウ関数の位置

```
┌────────── SQL論理的実行順序 ──────────────────────┐
│                                                    │
│  ① FROM / JOIN       テーブルの結合                │
│  ② WHERE             行のフィルタ                  │
│  ③ GROUP BY          グループ化                    │
│  ④ HAVING            グループのフィルタ            │
│  ⑤ SELECT            列の計算                      │
│     ├─ 式の評価      通常の式                      │
│     └─ WINDOW        ★ ウィンドウ関数はここ ★     │
│  ⑥ DISTINCT          重複排除                      │
│  ⑦ ORDER BY          ソート                        │
│  ⑧ LIMIT/OFFSET      結果の制限                   │
│                                                    │
│  重要:                                             │
│  - ウィンドウ関数はWHERE/HAVINGで使用不可          │
│  - GROUP BY後の結果に対して動作する                 │
│  - ORDER BY句では使用可能                          │
│  - 複数のウィンドウ関数は同じ実行フェーズで並行評価│
└────────────────────────────────────────────────────┘
```

### コード例1: ウィンドウ関数 vs GROUP BY

```sql
-- サンプルデータ
CREATE TABLE employees (
    id            SERIAL PRIMARY KEY,
    name          VARCHAR(100) NOT NULL,
    department_id INTEGER NOT NULL,
    salary        INTEGER NOT NULL,
    hired_date    DATE NOT NULL,
    status        VARCHAR(20) DEFAULT 'active'
);

INSERT INTO employees (name, department_id, salary, hired_date) VALUES
    ('田中太郎', 10, 450000, '2018-04-01'),
    ('佐藤花子', 10, 380000, '2020-07-15'),
    ('鈴木一郎', 20, 520000, '2015-01-10'),
    ('高橋美咲', 20, 450000, '2019-06-01'),
    ('渡辺健太', 30, 600000, '2012-04-01'),
    ('伊藤恵子', 30, 480000, '2017-09-15'),
    ('山本大輔', 30, 420000, '2021-04-01');

-- GROUP BY: 行が折りたたまれる（部署ごとに1行）
SELECT department_id, AVG(salary) AS avg_salary
FROM employees
GROUP BY department_id;
-- 結果: 3行（部署10, 20, 30）

-- ウィンドウ関数: 全行が保持される（各行に部署平均が付与）
SELECT
    name,
    department_id,
    salary,
    AVG(salary) OVER (PARTITION BY department_id) AS dept_avg_salary,
    salary - AVG(salary) OVER (PARTITION BY department_id) AS diff_from_avg,
    ROUND(
        salary::NUMERIC / SUM(salary) OVER (PARTITION BY department_id) * 100, 1
    ) AS pct_of_dept
FROM employees
ORDER BY department_id, salary DESC;

-- 結果イメージ（7行すべて保持）:
-- name     | dept_id | salary  | dept_avg  | diff     | pct
-- 田中太郎 |    10   | 450000  | 415000.0  | +35000.0 | 54.2
-- 佐藤花子 |    10   | 380000  | 415000.0  | -35000.0 | 45.8
-- 鈴木一郎 |    20   | 520000  | 485000.0  | +35000.0 | 53.6
-- 高橋美咲 |    20   | 450000  | 485000.0  | -35000.0 | 46.4
-- 渡辺健太 |    30   | 600000  | 500000.0  | +100000  | 40.0
-- 伊藤恵子 |    30   | 480000  | 500000.0  | -20000.0 | 32.0
-- 山本大輔 |    30   | 420000  | 500000.0  | -80000.0 | 28.0
```

### コード例2: ウィンドウ関数とGROUP BYの併用

```sql
-- GROUP BY後の結果にウィンドウ関数を適用する
-- ウィンドウ関数はGROUP BY実行後に評価される
SELECT
    department_id,
    COUNT(*) AS emp_count,
    SUM(salary) AS dept_total,
    SUM(COUNT(*)) OVER () AS company_total_count,
    ROUND(
        SUM(salary)::NUMERIC / SUM(SUM(salary)) OVER () * 100, 1
    ) AS pct_of_company,
    RANK() OVER (ORDER BY SUM(salary) DESC) AS salary_rank
FROM employees
GROUP BY department_id;

-- 結果:
-- dept_id | emp_count | dept_total | company_total | pct     | rank
-- 30      | 3         | 1500000    | 7             | 45.5    | 1
-- 20      | 2         | 970000     | 7             | 29.4    | 2
-- 10      | 2         | 830000     | 7             | 25.2    | 3
```

---

## 2. 順位付け関数

### コード例3: ROW_NUMBER / RANK / DENSE_RANK / NTILE の完全比較

```sql
-- 4種類の順位付けの違い
SELECT
    name,
    department_id,
    salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC) AS row_num,
    RANK()       OVER (ORDER BY salary DESC) AS rnk,
    DENSE_RANK() OVER (ORDER BY salary DESC) AS dense_rnk,
    NTILE(3)     OVER (ORDER BY salary DESC) AS tertile,
    PERCENT_RANK() OVER (ORDER BY salary DESC) AS pct_rank,
    CUME_DIST()    OVER (ORDER BY salary DESC) AS cume_dist
FROM employees;

-- 結果:
-- name     | dept | salary | row_num | rnk | dense | tertile | pct_rank | cume_dist
-- 渡辺健太 |  30  | 600000 |    1    |  1  |   1   |    1    | 0.0000   | 0.1429
-- 鈴木一郎 |  20  | 520000 |    2    |  2  |   2   |    1    | 0.1667   | 0.2857
-- 田中太郎 |  10  | 450000 |    3    |  3  |   3   |    1    | 0.3333   | 0.5714
-- 高橋美咲 |  20  | 450000 |    4    |  3  |   3   |    2    | 0.3333   | 0.5714
-- 伊藤恵子 |  30  | 480000 |    5    |  5  |   4   |    2    | 0.6667   | 0.7143
-- 山本大輔 |  30  | 420000 |    6    |  6  |   5   |    2    | 0.8333   | 0.8571
-- 佐藤花子 |  10  | 380000 |    7    |  7  |   6   |    3    | 1.0000   | 1.0000
```

### 順位付けの違い図解

```
┌────────── ROW_NUMBER vs RANK vs DENSE_RANK ──────────────────┐
│                                                               │
│  データ: 100, 90, 90, 80, 70, 70, 70, 60                     │
│                                                               │
│  値    ROW_NUMBER   RANK    DENSE_RANK   PERCENT_RANK         │
│  100       1          1         1         0.0000              │
│  90        2          2         2         0.1429              │
│  90        3          2         2         0.1429  ← 同値      │
│  80        4          4         3         0.4286  ← RANKは飛ぶ│
│  70        5          5         4         0.5714              │
│  70        6          5         4         0.5714  ← 同値      │
│  70        7          5         4         0.5714  ← 同値      │
│  60        8          8         5         1.0000  ← RANKは飛ぶ│
│                                                               │
│  ROW_NUMBER  : 一意の連番。同値でも異なる番号（非決定的）     │
│  RANK        : 同値は同順。次の順位は飛ぶ（1,2,2,4）         │
│  DENSE_RANK  : 同値は同順。次の順位は連番（1,2,2,3）         │
│  PERCENT_RANK: (RANK - 1) / (行数 - 1) で正規化              │
│  CUME_DIST   : 累積分布。現在値以下の行数/全行数              │
│                                                               │
│  NTILE(n):                                                    │
│  ┌────────────────────────────┐                              │
│  │ 8行を3分割: [3, 3, 2]     │                              │
│  │ 値100,90,90 → グループ1   │                              │
│  │ 値80,70,70  → グループ2   │                              │
│  │ 値70,60     → グループ3   │                              │
│  │ ※余りは先頭グループに分配 │                              │
│  └────────────────────────────┘                              │
└───────────────────────────────────────────────────────────────┘
```

### コード例4: PARTITION BY + 順位付け（Top-N問題）

```sql
-- 部署ごとの給与Top3を取得（最も頻出のパターン）
WITH ranked AS (
    SELECT
        name,
        department_id,
        salary,
        ROW_NUMBER() OVER (
            PARTITION BY department_id
            ORDER BY salary DESC
        ) AS rn
    FROM employees
)
SELECT name, department_id, salary
FROM ranked
WHERE rn <= 3;

-- 同点を含むTop-N（DENSE_RANKを使用）
WITH ranked AS (
    SELECT
        name,
        department_id,
        salary,
        DENSE_RANK() OVER (
            PARTITION BY department_id
            ORDER BY salary DESC
        ) AS dr
    FROM employees
)
SELECT name, department_id, salary, dr
FROM ranked
WHERE dr <= 3;
-- → 同点3位が複数人いる場合、全員が返される

-- NTILE: N等分に分割（四分位数の計算）
SELECT
    name,
    salary,
    NTILE(4) OVER (ORDER BY salary) AS quartile,
    CASE NTILE(4) OVER (ORDER BY salary)
        WHEN 1 THEN '下位25%'
        WHEN 2 THEN '25-50%'
        WHEN 3 THEN '50-75%'
        WHEN 4 THEN '上位25%'
    END AS quartile_label
FROM employees;
```

### コード例5: 重複排除（Deduplication）

```sql
-- メールアドレスの重複データから最新レコードのみ残す
WITH dedup AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY email
            ORDER BY updated_at DESC
        ) AS rn
    FROM users
)
-- 確認
SELECT * FROM dedup WHERE rn > 1;

-- 削除実行
DELETE FROM users
WHERE id IN (
    SELECT id FROM (
        SELECT
            id,
            ROW_NUMBER() OVER (
                PARTITION BY email
                ORDER BY updated_at DESC
            ) AS rn
        FROM users
    ) ranked
    WHERE rn > 1
);

-- PostgreSQL固有: CTEを使ったDELETE
WITH dedup AS (
    SELECT id,
           ROW_NUMBER() OVER (
               PARTITION BY email
               ORDER BY updated_at DESC
           ) AS rn
    FROM users
)
DELETE FROM users
WHERE id IN (SELECT id FROM dedup WHERE rn > 1);
```

---

## 3. 前後行参照: LAG / LEAD

### コード例6: LAG / LEAD の基本と応用

```sql
-- 月次売上テーブル
CREATE TABLE monthly_sales (
    month    DATE PRIMARY KEY,
    revenue  DECIMAL(12, 2)
);

-- 月次売上の前月比・前年同月比を計算
SELECT
    month,
    revenue,
    LAG(revenue, 1) OVER (ORDER BY month)  AS prev_month,
    LEAD(revenue, 1) OVER (ORDER BY month) AS next_month,
    LAG(revenue, 12) OVER (ORDER BY month) AS prev_year_same_month,
    -- 前月比（金額差）
    revenue - LAG(revenue, 1) OVER (ORDER BY month) AS mom_change,
    -- 前月比（パーセント）
    ROUND(
        (revenue - LAG(revenue, 1) OVER (ORDER BY month))::NUMERIC
        / NULLIF(LAG(revenue, 1) OVER (ORDER BY month), 0) * 100, 1
    ) AS mom_pct,
    -- 前年同月比
    ROUND(
        (revenue - LAG(revenue, 12) OVER (ORDER BY month))::NUMERIC
        / NULLIF(LAG(revenue, 12) OVER (ORDER BY month), 0) * 100, 1
    ) AS yoy_pct
FROM monthly_sales
ORDER BY month;

-- LAGの第3引数: デフォルト値（NULLの代わり）
SELECT
    month,
    revenue,
    LAG(revenue, 1, 0) OVER (ORDER BY month) AS prev_or_zero,
    -- 先頭行でもNULLにならない
    revenue - LAG(revenue, 1, revenue) OVER (ORDER BY month) AS change_safe
FROM monthly_sales;
```

### コード例7: FIRST_VALUE / LAST_VALUE / NTH_VALUE

```sql
-- 各部署の最高給与者と最低給与者の名前
SELECT
    name,
    department_id,
    salary,
    FIRST_VALUE(name) OVER w AS highest_paid,
    LAST_VALUE(name) OVER (
        PARTITION BY department_id ORDER BY salary DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS lowest_paid,
    NTH_VALUE(name, 2) OVER w AS second_highest,
    -- 最高給与との差
    FIRST_VALUE(salary) OVER w - salary AS gap_from_top
FROM employees
WINDOW w AS (PARTITION BY department_id ORDER BY salary DESC);

-- セッション内の最初と最後のページ
SELECT
    session_id,
    page_url,
    viewed_at,
    FIRST_VALUE(page_url) OVER (
        PARTITION BY session_id ORDER BY viewed_at
    ) AS landing_page,
    LAST_VALUE(page_url) OVER (
        PARTITION BY session_id ORDER BY viewed_at
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS exit_page,
    FIRST_VALUE(viewed_at) OVER (
        PARTITION BY session_id ORDER BY viewed_at
    ) AS session_start,
    LAST_VALUE(viewed_at) OVER (
        PARTITION BY session_id ORDER BY viewed_at
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS session_end
FROM page_views;
```

### コード例8: 連続データの差分計算（IoTセンサーデータ）

```sql
-- センサーデータの変化検出
SELECT
    sensor_id,
    measured_at,
    temperature,
    LAG(temperature) OVER w AS prev_temp,
    temperature - LAG(temperature) OVER w AS temp_change,
    -- 急激な変化の検出（前回から5度以上変化）
    CASE
        WHEN ABS(temperature - LAG(temperature) OVER w) > 5.0
        THEN 'ALERT'
        ELSE 'NORMAL'
    END AS status,
    -- 計測間隔の確認
    measured_at - LAG(measured_at) OVER w AS time_gap,
    -- 欠損検出（通常間隔の2倍以上）
    CASE
        WHEN measured_at - LAG(measured_at) OVER w > INTERVAL '10 minutes'
        THEN 'GAP_DETECTED'
        ELSE 'OK'
    END AS continuity
FROM sensor_data
WINDOW w AS (PARTITION BY sensor_id ORDER BY measured_at);
```

---

## 4. フレーム指定と累積集約

### ROWS vs RANGE vs GROUPS の詳細

```
┌──────── ROWS vs RANGE vs GROUPS の違い ──────────────────────────┐
│                                                                   │
│  データ: ORDER BY salary で以下の行がある場合                      │
│  行A: salary=300000                                               │
│  行B: salary=400000                                               │
│  行C: salary=400000  ← Bと同値（ピア）                           │
│  行D: salary=500000                                               │
│  行E: salary=600000                                               │
│                                                                   │
│  ■ ROWS: 物理的な行数で範囲を決定                                 │
│    ROWS BETWEEN 1 PRECEDING AND CURRENT ROW                       │
│    行Cの場合: [行B, 行C]（物理的に前1行 + 現在行）               │
│                                                                   │
│  ■ RANGE: 論理的な値の範囲で決定                                  │
│    RANGE BETWEEN CURRENT ROW AND CURRENT ROW                      │
│    行Cの場合: [行B, 行C]（salary=400000のピアグループ全体）       │
│                                                                   │
│    RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW (デフォルト) │
│    行Cの場合: [行A, 行B, 行C]（先頭〜同値含む現在行まで）        │
│                                                                   │
│  ■ GROUPS (SQL:2011 / PostgreSQL 11+):                            │
│    ピアグループ単位で範囲を決定                                    │
│    GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW                     │
│    行Cの場合: [行A, 行B, 行C]                                    │
│    （前1グループ{300000} + 現在グループ{400000, 400000}）         │
│                                                                   │
│  デフォルトフレーム（ORDER BY指定時）:                             │
│  RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW                │
│  → 同値の行がすべて含まれるため、ROWSと異なる結果になりうる       │
└───────────────────────────────────────────────────────────────────┘
```

### コード例9: 累積合計と移動平均

```sql
-- 日次売上テーブル
CREATE TABLE daily_sales (
    sale_date DATE PRIMARY KEY,
    amount    DECIMAL(10, 2)
);

-- 累積合計（Running Total）
SELECT
    sale_date,
    amount,
    SUM(amount) OVER (
        ORDER BY sale_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total
FROM daily_sales;

-- 7日間移動平均（直近7日の平均）
SELECT
    sale_date,
    amount,
    ROUND(
        AVG(amount) OVER (
            ORDER BY sale_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ), 2
    ) AS moving_avg_7d,
    COUNT(*) OVER (
        ORDER BY sale_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS window_size  -- 先頭付近ではデータが7日分ない
FROM daily_sales;

-- 月次累積（月ごとにリセット）
SELECT
    sale_date,
    amount,
    SUM(amount) OVER (
        PARTITION BY DATE_TRUNC('month', sale_date)
        ORDER BY sale_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS mtd_total  -- Month-to-Date
FROM daily_sales;

-- 前後3日間の中央値的な平滑化（外れ値対策）
SELECT
    sale_date,
    amount,
    AVG(amount) OVER (
        ORDER BY sale_date
        ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
    ) AS smoothed_7d
FROM daily_sales;
```

### コード例10: フレームを活用した高度な分析

```sql
-- パーティション内の割合と累積割合
SELECT
    name,
    department_id,
    salary,
    -- 部署内での割合
    ROUND(
        salary::NUMERIC / SUM(salary) OVER (PARTITION BY department_id) * 100, 1
    ) AS pct_of_dept,
    -- 全社内での割合
    ROUND(
        salary::NUMERIC / SUM(salary) OVER () * 100, 1
    ) AS pct_of_total,
    -- 累積割合（部署内、給与降順）
    ROUND(
        SUM(salary) OVER (
            PARTITION BY department_id
            ORDER BY salary DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )::NUMERIC / SUM(salary) OVER (PARTITION BY department_id) * 100, 1
    ) AS cumulative_pct
FROM employees
ORDER BY department_id, salary DESC;

-- パレート分析（上位20%が売上の80%を占めるか）
WITH product_sales AS (
    SELECT
        product_id,
        SUM(amount) AS total_sales,
        RANK() OVER (ORDER BY SUM(amount) DESC) AS sales_rank,
        COUNT(*) OVER () AS total_products
    FROM orders
    GROUP BY product_id
)
SELECT
    product_id,
    total_sales,
    sales_rank,
    ROUND(sales_rank::NUMERIC / total_products * 100, 1) AS rank_pct,
    ROUND(
        SUM(total_sales) OVER (ORDER BY total_sales DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        / SUM(total_sales) OVER () * 100, 1
    ) AS cumulative_revenue_pct
FROM product_sales
ORDER BY sales_rank;
```

### コード例11: EXCLUDE句（SQL:2011 / PostgreSQL 11+）

```sql
-- EXCLUDE句による同値行（ピア）の除外
SELECT
    name,
    salary,
    -- デフォルト: 同値のピアを含む
    AVG(salary) OVER (
        ORDER BY salary
        ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
    ) AS avg_with_current,
    -- EXCLUDE CURRENT ROW: 現在行を除外
    AVG(salary) OVER (
        ORDER BY salary
        ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
        EXCLUDE CURRENT ROW
    ) AS avg_without_current,
    -- EXCLUDE TIES: 同値の他の行を除外（現在行は含む）
    AVG(salary) OVER (
        ORDER BY salary
        ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
        EXCLUDE TIES
    ) AS avg_exclude_ties,
    -- EXCLUDE GROUP: 同値グループ全体を除外
    AVG(salary) OVER (
        ORDER BY salary
        ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING
        EXCLUDE GROUP
    ) AS avg_exclude_group
FROM employees;
```

### フレーム指定の動作の視覚化

```
┌──────── フレーム指定の視覚化（ROWS BETWEEN 2 PRECEDING AND 1 FOLLOWING）──┐
│                                                                            │
│  行1: 100  ─┐                                                             │
│  行2: 200  ─┤─ 行2のフレーム [行1,行2,行3]                               │
│  行3: 150  ─┘                                                             │
│  行4: 300  ─┐                                                             │
│  行5: 250  ─┤─ 行5のフレーム [行3,行4,行5,行6]                           │
│  行6: 180  ─┘                                                             │
│  行7: 220      行7のフレーム [行5,行6,行7] ← 末尾近くはフレームが縮小     │
│                                                                            │
│  ┌─────── よく使うフレーム指定一覧 ───────────────────┐                   │
│  │                                                     │                   │
│  │ 累積合計:                                           │                   │
│  │   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW  │                   │
│  │   [先頭 ... 現在行]                                 │                   │
│  │                                                     │                   │
│  │ 移動平均（7日間）:                                   │                   │
│  │   ROWS BETWEEN 6 PRECEDING AND CURRENT ROW          │                   │
│  │   [6行前 ... 現在行]                                │                   │
│  │                                                     │                   │
│  │ 中心移動平均:                                        │                   │
│  │   ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING          │                   │
│  │   [3行前 ... 現在行 ... 3行後]                      │                   │
│  │                                                     │                   │
│  │ パーティション全体:                                  │                   │
│  │   ROWS BETWEEN UNBOUNDED PRECEDING                  │                   │
│  │                AND UNBOUNDED FOLLOWING               │                   │
│  │   [先頭 ... 末尾]                                   │                   │
│  └─────────────────────────────────────────────────────┘                   │
└────────────────────────────────────────────────────────────────────────────┘
```

### コード例12: 名前付きウィンドウ（WINDOW句）

```sql
-- 同じウィンドウ定義を使い回す
SELECT
    name,
    department_id,
    salary,
    ROW_NUMBER() OVER w_dept AS rn,
    RANK() OVER w_dept AS rnk,
    DENSE_RANK() OVER w_dept AS dense_rnk,
    SUM(salary) OVER w_dept AS running_sum,
    AVG(salary) OVER w_dept AS running_avg,
    -- WINDOW句を継承してフレームを変更
    SUM(salary) OVER (w_dept ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS dept_total
FROM employees
WINDOW w_dept AS (PARTITION BY department_id ORDER BY salary DESC)
ORDER BY department_id, salary DESC;

-- 複数ウィンドウの定義
SELECT
    name,
    department_id,
    salary,
    hired_date,
    ROW_NUMBER() OVER w_salary AS salary_rank,
    ROW_NUMBER() OVER w_tenure AS tenure_rank,
    SUM(salary) OVER w_dept_all AS dept_total
FROM employees
WINDOW
    w_salary AS (PARTITION BY department_id ORDER BY salary DESC),
    w_tenure AS (PARTITION BY department_id ORDER BY hired_date),
    w_dept_all AS (PARTITION BY department_id);
```

---

## 5. 実践的なパターン集

### コード例13: ギャップ検出とアイランド問題

```sql
-- 連続するログインの「島」を検出する（Islands and Gaps問題）
-- セッション間に1日以上の空白がある場合、新しい島とする

WITH login_groups AS (
    SELECT
        user_id,
        login_date,
        -- 連番からlogin_dateを引くことで、連続する日付は同じグループ値になる
        login_date - (ROW_NUMBER() OVER (
            PARTITION BY user_id ORDER BY login_date
        ))::INTEGER AS grp
    FROM user_logins
)
SELECT
    user_id,
    MIN(login_date) AS streak_start,
    MAX(login_date) AS streak_end,
    COUNT(*) AS consecutive_days
FROM login_groups
GROUP BY user_id, grp
HAVING COUNT(*) >= 7  -- 7日以上連続ログイン
ORDER BY user_id, streak_start;

-- ギャップの検出（欠落した日付を特定）
SELECT
    login_date AS last_login,
    LEAD(login_date) OVER (
        PARTITION BY user_id ORDER BY login_date
    ) AS next_login,
    LEAD(login_date) OVER (
        PARTITION BY user_id ORDER BY login_date
    ) - login_date - 1 AS gap_days
FROM user_logins
WHERE user_id = 42;
```

### コード例14: セッション化（Sessionization）

```sql
-- ページビューイベントを30分間隔でセッションに分割
WITH events_with_gap AS (
    SELECT
        user_id,
        event_time,
        page_url,
        -- 前回イベントとの時間差を計算
        event_time - LAG(event_time) OVER (
            PARTITION BY user_id ORDER BY event_time
        ) AS time_gap,
        -- 30分以上空いたら新セッション開始
        CASE WHEN event_time - LAG(event_time) OVER (
            PARTITION BY user_id ORDER BY event_time
        ) > INTERVAL '30 minutes'
        OR LAG(event_time) OVER (
            PARTITION BY user_id ORDER BY event_time
        ) IS NULL  -- 最初のイベント
        THEN 1 ELSE 0
        END AS new_session_flag
    FROM page_views
),
sessions AS (
    SELECT
        *,
        -- 累積SUMでセッションIDを生成
        SUM(new_session_flag) OVER (
            PARTITION BY user_id ORDER BY event_time
        ) AS session_id
    FROM events_with_gap
)
SELECT
    user_id,
    session_id,
    MIN(event_time) AS session_start,
    MAX(event_time) AS session_end,
    COUNT(*) AS page_views,
    MAX(event_time) - MIN(event_time) AS session_duration
FROM sessions
GROUP BY user_id, session_id
ORDER BY user_id, session_start;
```

### コード例15: 変化の検出（Change Detection）

```sql
-- 価格変更の履歴をフラット化
SELECT
    product_id,
    price,
    effective_from,
    LEAD(effective_from) OVER (
        PARTITION BY product_id ORDER BY effective_from
    ) - INTERVAL '1 day' AS effective_until,
    -- LEAD がNULLなら現在も有効
    COALESCE(
        LEAD(effective_from) OVER (
            PARTITION BY product_id ORDER BY effective_from
        ) - INTERVAL '1 day',
        CURRENT_DATE
    ) AS effective_until_safe,
    -- 前回価格との変動率
    ROUND(
        (price - LAG(price) OVER (
            PARTITION BY product_id ORDER BY effective_from
        ))::NUMERIC / NULLIF(LAG(price) OVER (
            PARTITION BY product_id ORDER BY effective_from
        ), 0) * 100, 2
    ) AS price_change_pct
FROM product_prices
ORDER BY product_id, effective_from;
```

---

## 6. クエリオプティマイザとウィンドウ関数の内部実装

### ウィンドウ関数の実行計画

```
┌──────── ウィンドウ関数の実行計画 ──────────────────────────────┐
│                                                                │
│  PostgreSQL EXPLAIN ANALYZE の読み方:                           │
│                                                                │
│  WindowAgg                                                     │
│  ├── Sort (PARTITION BY + ORDER BY に基づくソート)             │
│  │   └── Seq Scan on employees                                │
│  │       Filter: (status = 'active')                          │
│  │                                                            │
│  │  Sort Key: department_id, salary DESC                      │
│  │  Sort Method: quicksort  Memory: 25kB                      │
│  │                                                            │
│  実行の流れ:                                                   │
│  ① テーブルスキャン + WHEREフィルタ                            │
│  ② PARTITION BY + ORDER BY に基づいてソート                    │
│  ③ ソート済みデータ上でウィンドウ関数を計算                    │
│                                                                │
│  パーティションごとの処理:                                     │
│  ┌────────────────────────────────────┐                       │
│  │ パーティション1 (dept=10)          │                       │
│  │ ソート済み: [450000, 380000]       │                       │
│  │ → ROW_NUMBER: 1, 2                │                       │
│  │ → SUM (累積): 450000, 830000      │                       │
│  ├────────────────────────────────────┤                       │
│  │ パーティション2 (dept=20)          │                       │
│  │ ソート済み: [520000, 450000]       │                       │
│  │ → ROW_NUMBER: 1, 2                │                       │
│  │ → SUM (累積): 520000, 970000      │                       │
│  └────────────────────────────────────┘                       │
│                                                                │
│  同一PARTITION BY + ORDER BY のウィンドウ関数は                 │
│  1回のソートで複数関数を同時に計算できる                        │
└────────────────────────────────────────────────────────────────┘
```

### コード例16: 実行計画の確認と最適化

```sql
-- 実行計画の確認
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT
    name,
    department_id,
    salary,
    ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rn,
    SUM(salary) OVER (PARTITION BY department_id) AS dept_total
FROM employees
WHERE status = 'active';

-- 最適化: ウィンドウ関数用のインデックス
-- PARTITION BY + ORDER BY をカバーするインデックス
CREATE INDEX idx_emp_dept_salary
ON employees (department_id, salary DESC)
WHERE status = 'active';  -- 部分インデックスでWHEREもカバー

-- 確認: ソートが省略されているか
EXPLAIN (ANALYZE)
SELECT
    name, department_id, salary,
    ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) AS rn
FROM employees
WHERE status = 'active';
-- → Sort ノードが消えて Index Scan になればOK

-- 異なるウィンドウ定義が複数ある場合のソート回数
EXPLAIN (ANALYZE)
SELECT
    name,
    ROW_NUMBER() OVER (ORDER BY salary DESC) AS salary_rank,
    ROW_NUMBER() OVER (ORDER BY hired_date) AS tenure_rank
FROM employees;
-- → 2回のソートが発生する（ウィンドウ定義が異なるため）
```

### パフォーマンス最適化のガイドライン

```
┌──────── ウィンドウ関数のパフォーマンス最適化 ─────────────┐
│                                                           │
│  ■ ソートの最適化                                         │
│  ┌─────────────────────────────────────────────┐         │
│  │ 問題: 大量データのソートはメモリと時間を消費 │         │
│  │                                             │         │
│  │ 対策1: 複合インデックスの作成               │         │
│  │   CREATE INDEX ON t (partition_col, order_col)│        │
│  │   → ソートが不要になる                      │         │
│  │                                             │         │
│  │ 対策2: 同じウィンドウ定義の共有             │         │
│  │   WINDOW w AS (...) で1回のソート           │         │
│  │                                             │         │
│  │ 対策3: work_mem の調整                      │         │
│  │   SET work_mem = '256MB';                   │         │
│  │   → ディスクスピルを防ぐ                    │         │
│  └─────────────────────────────────────────────┘         │
│                                                           │
│  ■ ROWS vs RANGE のパフォーマンス                         │
│  ┌─────────────────────────────────────────────┐         │
│  │ ROWS: O(1) per row（フレーム内の行を直接参照）│        │
│  │ RANGE: ピアグループの探索が必要で若干遅い    │         │
│  │                                             │         │
│  │ 推奨: 明示的にROWSを指定する                │         │
│  │ （デフォルトのRANGEよりも高速で予測可能）    │         │
│  └─────────────────────────────────────────────┘         │
│                                                           │
│  ■ 大規模データでの注意                                   │
│  ┌─────────────────────────────────────────────┐         │
│  │ - ウィンドウ関数は全パーティションデータを    │         │
│  │   メモリに保持する必要がある                  │         │
│  │ - 巨大なパーティションは要注意               │         │
│  │ - WHERE句で先にフィルタして行数を減らす      │         │
│  │ - PARTITION BYがない場合、全行が1パーティション│        │
│  └─────────────────────────────────────────────┘         │
└───────────────────────────────────────────────────────────┘
```

---

## 7. RDBMS間の互換性

### ウィンドウ関数サポート比較表

| 機能 | PostgreSQL | MySQL 8.0+ | SQL Server | Oracle | SQLite 3.25+ |
|------|-----------|-------------|------------|--------|-------------|
| ROW_NUMBER | 対応 | 対応 | 対応 | 対応 | 対応 |
| RANK/DENSE_RANK | 対応 | 対応 | 対応 | 対応 | 対応 |
| NTILE | 対応 | 対応 | 対応 | 対応 | 対応 |
| LAG/LEAD | 対応 | 対応 | 対応 | 対応 | 対応 |
| FIRST_VALUE/LAST_VALUE | 対応 | 対応 | 対応 | 対応 | 対応 |
| NTH_VALUE | 対応 | 対応 | 非対応 | 対応 | 対応 |
| PERCENT_RANK | 対応 | 対応 | 対応 | 対応 | 対応 |
| CUME_DIST | 対応 | 対応 | 対応 | 対応 | 対応 |
| WINDOW句 | 対応 | 対応 | 非対応 | 非対応 | 対応 |
| GROUPS フレーム | 対応(11+) | 非対応 | 非対応 | 非対応 | 対応(3.28+) |
| EXCLUDE句 | 対応(11+) | 非対応 | 非対応 | 非対応 | 対応(3.28+) |
| FILTER句+ウィンドウ | 対応 | 非対応 | 非対応 | 非対応 | 対応 |
| RANGE + 数値範囲 | 対応 | 対応 | 対応 | 対応 | 対応 |
| RANGE + INTERVAL | 対応 | 非対応 | 非対応 | 対応 | 非対応 |

### RDBMS固有の構文

```sql
-- ■ SQL Server: ROW_NUMBER()を使ったページネーション
-- SQL ServerにはLIMIT/OFFSETがないためOFFSET...FETCHを使用
SELECT name, salary
FROM (
    SELECT name, salary,
           ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn
    FROM employees
) sub
WHERE rn BETWEEN 11 AND 20;  -- 2ページ目

-- SQL Server 2012+
SELECT name, salary
FROM employees
ORDER BY salary DESC
OFFSET 10 ROWS FETCH NEXT 10 ROWS ONLY;

-- ■ Oracle: 分析関数の拡張構文
-- KEEP (DENSE_RANK FIRST/LAST) — 集約関数との併用
SELECT
    department_id,
    MAX(salary) AS max_salary,
    MAX(name) KEEP (DENSE_RANK FIRST ORDER BY salary DESC) AS top_earner
FROM employees
GROUP BY department_id;

-- ■ MySQL 8.0+: ウィンドウ関数の制限
-- MySQL ではFILTER句やWINDOW句のチェーン（継承）が使えない
-- CASEで代替
SELECT
    name,
    SUM(CASE WHEN status = 'active' THEN salary ELSE 0 END) OVER (
        PARTITION BY department_id
    ) AS active_salary_total
FROM employees;

-- ■ PostgreSQL: FILTER句とウィンドウ関数の組み合わせ
SELECT
    name,
    department_id,
    salary,
    COUNT(*) FILTER (WHERE salary > 400000) OVER (
        PARTITION BY department_id
    ) AS high_earner_count
FROM employees;
```

---

## ウィンドウ関数一覧表

| 関数 | 分類 | 説明 | フレーム | 注意点 |
|------|------|------|---------|--------|
| ROW_NUMBER() | 順位 | 一意の連番 | 不要 | 同値の順序は非決定的 |
| RANK() | 順位 | 同順あり（飛び番） | 不要 | 同値後は番号が飛ぶ |
| DENSE_RANK() | 順位 | 同順あり（連番） | 不要 | 常に連続した番号 |
| NTILE(n) | 順位 | N等分に分割 | 不要 | 余りは先頭グループへ |
| PERCENT_RANK() | 順位 | 正規化順位 (0-1) | 不要 | (rank-1)/(rows-1) |
| CUME_DIST() | 順位 | 累積分布 (0-1) | 不要 | 値以下の行数/全行数 |
| LAG(col, n, default) | 前後参照 | N行前の値 | 不要 | n省略時は1、default省略時はNULL |
| LEAD(col, n, default) | 前後参照 | N行後の値 | 不要 | n省略時は1 |
| FIRST_VALUE(col) | 前後参照 | フレーム内の最初の値 | 使用可 | デフォルトフレームで動作 |
| LAST_VALUE(col) | 前後参照 | フレーム内の最後の値 | 要注意 | UNBOUNDED FOLLOWING必須 |
| NTH_VALUE(col, n) | 前後参照 | フレーム内のN番目の値 | 使用可 | SQL Serverでは未対応 |
| SUM/AVG/COUNT/MIN/MAX | 集約 | フレーム内の集約 | 使用可 | フレームで範囲制御 |

## ROWS vs RANGE vs GROUPS 比較表

| 項目 | ROWS | RANGE | GROUPS |
|------|------|-------|--------|
| 単位 | 物理的な行数 | 論理的な値の範囲 | ピアグループ数 |
| 同値の扱い | 個別に扱う | まとめて扱う | グループとして扱う |
| デフォルト | -- | ORDER BY指定時のデフォルト | -- |
| パフォーマンス | 最速 | やや遅い | 中間 |
| 推奨場面 | 移動平均、累積合計 | 値ベースの範囲集約 | グループ単位の分析 |
| 対応RDBMS | 全RDBMS | 全RDBMS | PostgreSQL 11+, SQLite 3.28+ |
| N PRECEDING/FOLLOWING | N行前/後 | 値がN以内 | Nグループ前/後 |

---

## エッジケース

### エッジケース1: NULLの扱い

```sql
-- ウィンドウ関数でのNULL
-- ORDER BYにNULLが含まれる場合
SELECT
    name,
    salary,
    ROW_NUMBER() OVER (ORDER BY salary) AS rn,
    RANK() OVER (ORDER BY salary) AS rnk
FROM employees;
-- → NULLは通常最後に来る（PostgreSQL: NULLS LAST がデフォルト）

-- NULLS FIRSTで制御
SELECT
    name,
    salary,
    ROW_NUMBER() OVER (ORDER BY salary NULLS FIRST) AS rn
FROM employees;

-- LAG/LEADとNULL
-- 前の行がNULLなのか、先頭行でLAGが無いからNULLなのか区別できない
SELECT
    sale_date,
    amount,
    LAG(amount) OVER (ORDER BY sale_date) AS prev_amount,
    -- 区別する方法: ROW_NUMBERで先頭行を判定
    CASE
        WHEN ROW_NUMBER() OVER (ORDER BY sale_date) = 1 THEN 'FIRST_ROW'
        WHEN LAG(amount) OVER (ORDER BY sale_date) IS NULL THEN 'NULL_VALUE'
        ELSE 'HAS_VALUE'
    END AS prev_status
FROM daily_sales;
```

### エッジケース2: ROW_NUMBERの非決定性

```sql
-- NG: 同値の場合、ROW_NUMBERの結果が実行ごとに変わりうる
SELECT
    name, salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn
FROM employees;
-- salary=450000が2人いる場合、rn=2,3の割り当ては非決定的

-- OK: タイブレーカーを追加して決定的にする
SELECT
    name, salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC, id ASC) AS rn
FROM employees;
-- idが一意なので結果が常に同一
```

### エッジケース3: パーティション内の行が1行のみ

```sql
-- 1行のみのパーティションでの各関数の動作
-- ROW_NUMBER: 1
-- RANK: 1
-- DENSE_RANK: 1
-- LAG: NULL（前の行がない）
-- LEAD: NULL（後の行がない）
-- NTILE(4): 1（全体が1グループ目に入る）
-- FIRST_VALUE: その行の値
-- LAST_VALUE: その行の値
-- SUM/AVG/COUNT: その行の値/1

-- 移動平均の先頭付近では窓が小さくなる
SELECT
    sale_date,
    amount,
    AVG(amount) OVER (ORDER BY sale_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS ma7,
    COUNT(*) OVER (ORDER BY sale_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS window_size
FROM daily_sales
ORDER BY sale_date
LIMIT 10;
-- 先頭7行ではwindow_sizeが1,2,3,...,7と増えていく
-- → 移動平均が安定しないことに注意
```

### エッジケース4: 空のパーティション

```sql
-- PARTITION BYの結果、一部パーティションに行がない場合
-- → そもそもウィンドウ関数の結果に行が出現しない（NULLが返るわけではない）
-- 全カテゴリの結果を出したい場合はLEFT JOINと組み合わせる

SELECT
    c.category_name,
    s.amount,
    s.sale_date,
    COALESCE(
        ROW_NUMBER() OVER (PARTITION BY c.id ORDER BY s.sale_date),
        0
    ) AS rn
FROM categories c
LEFT JOIN sales s ON c.id = s.category_id;
```

---

## アンチパターン

### アンチパターン1: LAST_VALUEのフレーム問題

```sql
-- NG: デフォルトフレームだとLAST_VALUEが期待通りに動かない
SELECT
    name, salary,
    LAST_VALUE(name) OVER (ORDER BY salary) AS last_name
FROM employees;
-- → デフォルトフレームは RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
-- → 各行の時点での「最後」＝現在行自身になってしまう

-- OK: フレームを明示的に指定
SELECT
    name, salary,
    LAST_VALUE(name) OVER (
        ORDER BY salary
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS last_name
FROM employees;

-- 実装メモ: FIRST_VALUEはデフォルトフレームで正しく動作する
-- なぜなら UNBOUNDED PRECEDING AND CURRENT ROW の先頭は常にパーティション先頭
```

### アンチパターン2: ウィンドウ関数の結果をWHEREで直接フィルタ

```sql
-- NG: WHERE句でウィンドウ関数は使用不可（SQL論理実行順序の制約）
SELECT name, salary, ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn
FROM employees
WHERE rn <= 10;  -- エラー！

-- OK: サブクエリで包む
SELECT * FROM (
    SELECT name, salary, ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn
    FROM employees
) ranked
WHERE rn <= 10;

-- OK: CTEを使う（推奨）
WITH ranked AS (
    SELECT name, salary, ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn
    FROM employees
)
SELECT * FROM ranked WHERE rn <= 10;

-- QUALIFY句（一部RDBMS対応: Snowflake, BigQuery, DuckDB）
-- SQLの拡張構文で、ウィンドウ関数の結果をフィルタできる
SELECT name, salary, ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn
FROM employees
QUALIFY rn <= 10;  -- PostgreSQL/MySQL/SQL Serverでは使用不可
```

### アンチパターン3: 異なるウィンドウ定義の乱用

```sql
-- NG: 多数の異なるウィンドウ定義（ソートが複数回発生）
SELECT
    name,
    ROW_NUMBER() OVER (ORDER BY salary DESC) AS salary_rank,
    ROW_NUMBER() OVER (ORDER BY hired_date ASC) AS tenure_rank,
    ROW_NUMBER() OVER (ORDER BY name ASC) AS alpha_rank,
    ROW_NUMBER() OVER (PARTITION BY department_id ORDER BY salary DESC) AS dept_rank
FROM employees;
-- → 4つの異なるソートが必要で非常にコストが高い

-- OK: 必要なものだけに限定し、可能なら同じウィンドウ定義を共有
SELECT
    name,
    ROW_NUMBER() OVER w AS salary_rank,
    RANK() OVER w AS salary_rnk,
    SUM(salary) OVER w AS running_sum
FROM employees
WINDOW w AS (ORDER BY salary DESC);
-- → 1回のソートで3つの関数を計算
```

### アンチパターン4: 自己結合で代替可能な場面でウィンドウ関数を使わない

```sql
-- NG: 自己結合で前月比を計算（非効率でエラーの温床）
SELECT
    a.month,
    a.revenue,
    b.revenue AS prev_revenue,
    a.revenue - b.revenue AS change
FROM monthly_sales a
LEFT JOIN monthly_sales b
    ON b.month = a.month - INTERVAL '1 month';
-- → 月の境界、営業日のみ等で結合条件が複雑化

-- OK: LAGで簡潔かつ正確
SELECT
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month) AS prev_revenue,
    revenue - LAG(revenue) OVER (ORDER BY month) AS change
FROM monthly_sales;
```

---

## セキュリティに関する注意

```
┌──────── セキュリティ上の考慮点 ────────────────────────┐
│                                                        │
│  ■ 行レベルセキュリティ (RLS) との相互作用             │
│  ┌──────────────────────────────────────────┐         │
│  │ RLSが有効なテーブルでウィンドウ関数を使う場合:│     │
│  │ - RLSフィルタはWHERE相当で先に適用される  │         │
│  │ - ユーザーAは自分の見える行の中でRANKが計算│         │
│  │ - 全体順位を知ることはできない（意図通り） │         │
│  │ - ただし、RANK()の飛び具合から他の行の存在 │         │
│  │   を推測される可能性がある → DENSE_RANKが安全│       │
│  └──────────────────────────────────────────┘         │
│                                                        │
│  ■ パフォーマンスDoS                                   │
│  ┌──────────────────────────────────────────┐         │
│  │ - PARTITION BYなしのウィンドウ関数は全行をソート│    │
│  │ - 大テーブルに対する無制限のウィンドウ関数は    │    │
│  │   メモリとCPUを大量消費する                    │     │
│  │ - ユーザー入力からORDER BY列を動的に生成する   │     │
│  │   場合はホワイトリスト検証が必須               │     │
│  └──────────────────────────────────────────┘         │
│                                                        │
│  ■ ビューを通したウィンドウ関数                        │
│  ┌──────────────────────────────────────────┐         │
│  │ ウィンドウ関数を含むビューは更新不可能         │     │
│  │ → INSERT/UPDATE/DELETEはエラーになる           │     │
│  │ → INSTEAD OFトリガーで回避可能（Oracle/PG）    │     │
│  └──────────────────────────────────────────┘         │
└────────────────────────────────────────────────────────┘
```

---

## 演習問題

### 演習1: 基礎 — 順位付けと前後参照

```sql
-- 問題: 以下のテーブルに対して要求されるクエリを書きなさい。
CREATE TABLE exam_scores (
    student_id   INTEGER,
    subject      VARCHAR(20),
    score        INTEGER,
    exam_date    DATE
);

-- Q1: 科目ごとの得点ランキング（同点は同順位、次の順位は飛ばない）
-- Q2: 各生徒の前回試験からの点数変化
-- Q3: 科目ごとのトップ3（同点の場合は全員含む）
```

**模範解答:**

```sql
-- Q1: DENSE_RANKで同点同順位・連番
SELECT
    student_id,
    subject,
    score,
    DENSE_RANK() OVER (PARTITION BY subject ORDER BY score DESC) AS rank
FROM exam_scores;

-- Q2: LAGで前回からの変化
SELECT
    student_id,
    subject,
    exam_date,
    score,
    LAG(score) OVER (
        PARTITION BY student_id, subject ORDER BY exam_date
    ) AS prev_score,
    score - LAG(score) OVER (
        PARTITION BY student_id, subject ORDER BY exam_date
    ) AS score_change
FROM exam_scores
ORDER BY student_id, subject, exam_date;

-- Q3: Top-3（同点含む）
WITH ranked AS (
    SELECT
        student_id,
        subject,
        score,
        DENSE_RANK() OVER (PARTITION BY subject ORDER BY score DESC) AS dr
    FROM exam_scores
)
SELECT * FROM ranked WHERE dr <= 3
ORDER BY subject, dr;
```

### 演習2: 応用 — 移動平均と累積分析

```sql
-- 問題: 以下の株価データに対してクエリを書きなさい。
CREATE TABLE stock_prices (
    ticker     VARCHAR(10),
    trade_date DATE,
    close_price DECIMAL(10, 2),
    volume      BIGINT
);

-- Q1: 各銘柄の5日移動平均と20日移動平均を計算
-- Q2: ゴールデンクロス（5日MAが20日MAを下から上に突き抜ける日）を検出
-- Q3: 各銘柄の年初来高値・安値からの乖離率を日次で計算
```

**模範解答:**

```sql
-- Q1: 移動平均
SELECT
    ticker,
    trade_date,
    close_price,
    ROUND(AVG(close_price) OVER (
        PARTITION BY ticker ORDER BY trade_date
        ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ), 2) AS ma5,
    ROUND(AVG(close_price) OVER (
        PARTITION BY ticker ORDER BY trade_date
        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
    ), 2) AS ma20
FROM stock_prices;

-- Q2: ゴールデンクロス検出
WITH ma AS (
    SELECT
        ticker,
        trade_date,
        close_price,
        AVG(close_price) OVER (
            PARTITION BY ticker ORDER BY trade_date
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        ) AS ma5,
        AVG(close_price) OVER (
            PARTITION BY ticker ORDER BY trade_date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ) AS ma20
    FROM stock_prices
),
with_prev AS (
    SELECT
        *,
        LAG(ma5) OVER (PARTITION BY ticker ORDER BY trade_date) AS prev_ma5,
        LAG(ma20) OVER (PARTITION BY ticker ORDER BY trade_date) AS prev_ma20
    FROM ma
)
SELECT ticker, trade_date, close_price, ma5, ma20
FROM with_prev
WHERE prev_ma5 < prev_ma20  -- 前日: 5日MA < 20日MA
  AND ma5 >= ma20;           -- 当日: 5日MA >= 20日MA（クロス）

-- Q3: 年初来高値・安値からの乖離率
SELECT
    ticker,
    trade_date,
    close_price,
    MAX(close_price) OVER (
        PARTITION BY ticker, EXTRACT(YEAR FROM trade_date)
        ORDER BY trade_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS ytd_high,
    MIN(close_price) OVER (
        PARTITION BY ticker, EXTRACT(YEAR FROM trade_date)
        ORDER BY trade_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS ytd_low,
    ROUND(
        (close_price - MAX(close_price) OVER (
            PARTITION BY ticker, EXTRACT(YEAR FROM trade_date)
            ORDER BY trade_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )) / MAX(close_price) OVER (
            PARTITION BY ticker, EXTRACT(YEAR FROM trade_date)
            ORDER BY trade_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) * 100, 2
    ) AS pct_from_ytd_high
FROM stock_prices;
```

### 演習3: 上級 — セッション化とギャップ検出

```sql
-- 問題: ECサイトのイベントログからユーザー行動を分析しなさい。
CREATE TABLE event_log (
    event_id    BIGSERIAL PRIMARY KEY,
    user_id     INTEGER,
    event_type  VARCHAR(20),  -- 'page_view', 'add_to_cart', 'purchase'
    event_time  TIMESTAMP,
    page_url    VARCHAR(500),
    amount      DECIMAL(10, 2)
);

-- Q1: 30分以上の空白をセッション境界として、セッションIDを振りなさい
-- Q2: 各セッション内で「page_view → add_to_cart → purchase」の
--     ファネル変換率を計算しなさい
-- Q3: 各ユーザーの購入間隔（日数）の平均と最大を計算しなさい
```

**模範解答:**

```sql
-- Q1: セッションID付与
WITH with_gap AS (
    SELECT
        *,
        CASE
            WHEN event_time - LAG(event_time) OVER (
                PARTITION BY user_id ORDER BY event_time
            ) > INTERVAL '30 minutes'
            OR LAG(event_time) OVER (
                PARTITION BY user_id ORDER BY event_time
            ) IS NULL
            THEN 1
            ELSE 0
        END AS is_new_session
    FROM event_log
)
SELECT
    *,
    SUM(is_new_session) OVER (
        PARTITION BY user_id ORDER BY event_time
    ) AS session_id
FROM with_gap;

-- Q2: ファネル分析
WITH sessions AS (
    -- Q1のCTEを再利用（省略）
    SELECT user_id, session_id, event_type FROM ...
),
funnel AS (
    SELECT
        session_id,
        COUNT(*) FILTER (WHERE event_type = 'page_view') AS views,
        COUNT(*) FILTER (WHERE event_type = 'add_to_cart') AS carts,
        COUNT(*) FILTER (WHERE event_type = 'purchase') AS purchases
    FROM sessions
    GROUP BY session_id
)
SELECT
    COUNT(*) AS total_sessions,
    COUNT(*) FILTER (WHERE views > 0) AS with_views,
    COUNT(*) FILTER (WHERE carts > 0) AS with_cart,
    COUNT(*) FILTER (WHERE purchases > 0) AS with_purchase,
    ROUND(COUNT(*) FILTER (WHERE carts > 0)::NUMERIC
        / NULLIF(COUNT(*) FILTER (WHERE views > 0), 0) * 100, 1) AS view_to_cart_pct,
    ROUND(COUNT(*) FILTER (WHERE purchases > 0)::NUMERIC
        / NULLIF(COUNT(*) FILTER (WHERE carts > 0), 0) * 100, 1) AS cart_to_purchase_pct
FROM funnel;

-- Q3: 購入間隔の分析
WITH purchases AS (
    SELECT
        user_id,
        event_time::DATE AS purchase_date,
        LAG(event_time::DATE) OVER (
            PARTITION BY user_id ORDER BY event_time
        ) AS prev_purchase_date
    FROM event_log
    WHERE event_type = 'purchase'
)
SELECT
    user_id,
    COUNT(*) AS purchase_count,
    ROUND(AVG(purchase_date - prev_purchase_date), 1) AS avg_interval_days,
    MAX(purchase_date - prev_purchase_date) AS max_interval_days,
    MIN(purchase_date - prev_purchase_date) AS min_interval_days
FROM purchases
WHERE prev_purchase_date IS NOT NULL
GROUP BY user_id
ORDER BY avg_interval_days;
```

---

## FAQ

### Q1: ウィンドウ関数はGROUP BYと併用できるか？

可能だが、ウィンドウ関数はGROUP BYの後に実行される。つまり、GROUP BYで集約された結果に対してウィンドウ関数が適用される。GROUP BY前の行にはアクセスできない。`SUM(COUNT(*)) OVER ()` のようにGROUP BY結果をさらにウィンドウ集約する場面で使用する。

### Q2: ウィンドウ関数はインデックスで高速化できるか？

PARTITION BY列とORDER BY列にインデックスがあると、ソート処理が省略される場合がある。特に `CREATE INDEX ON table (partition_col, order_col)` の複合インデックスが有効。ただし、ウィンドウ関数がクエリ内の唯一のアクセスパスではない場合、オプティマイザがインデックスを使わない判断をすることもある。EXPLAIN ANALYZEで確認すること。

### Q3: ROW_NUMBERでページネーションは推奨されるか？

大量データでは非推奨。ROW_NUMBERは全行をソートしてから番号を振るため、深いページ（OFFSET大）でパフォーマンスが劣化する。キーセットページネーション（`WHERE id > last_seen_id ORDER BY id LIMIT 20`）の方が効率的。ただし、小〜中規模のデータセットではROW_NUMBERも実用的。

### Q4: ウィンドウ関数は更新クエリ（UPDATE）で使用できるか？

PostgreSQLでは、FROM句のサブクエリ内でウィンドウ関数を使い、その結果でUPDATEすることが可能。直接SET句では使用不可。

```sql
UPDATE employees e
SET rank_in_dept = sub.rn
FROM (
    SELECT id, ROW_NUMBER() OVER (
        PARTITION BY department_id ORDER BY salary DESC
    ) AS rn
    FROM employees
) sub
WHERE e.id = sub.id;
```

### Q5: 同じクエリ内で複数のウィンドウ関数を使うとパフォーマンスは大丈夫か？

同じウィンドウ定義（PARTITION BY + ORDER BY）を共有する関数は1回のソートで計算されるため効率的。異なるウィンドウ定義の関数は追加のソートが必要になる。WINDOW句で定義を共有し、EXPLAINでソートの回数を確認すること。

### Q6: ウィンドウ関数の結果をORDER BYで使えるか？

使える。ORDER BY句はSELECTの後に評価されるため、ウィンドウ関数のエイリアスを参照可能。ただし、WHERE/HAVINGでは使えない（CTEやサブクエリで包む必要がある）。

---

## トラブルシューティング

### 症状1: LAST_VALUEが全行で同じ値にならない

**原因:** デフォルトフレーム（RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW）のため、「最後の値」が現在行までの最後 = 現在行自身になる。

**解決:** `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING` を明示指定する。

### 症状2: ROW_NUMBERの結果が実行ごとに異なる

**原因:** ORDER BY列に同値が存在し、タイブレーカーがないため非決定的。

**解決:** ORDER BYに一意列（PKやユニーク列）を追加する。`ORDER BY salary DESC, id ASC`

### 症状3: ウィンドウ関数を含むクエリが極端に遅い

**原因:** (1) 大テーブルに対するPARTITION BYなしのウィンドウ関数、(2) 異なるウィンドウ定義が多数、(3) work_memが不足しディスクスピル発生。

**解決:** (1) WHERE句で先にフィルタ、(2) WINDOW句でウィンドウ定義を統一、(3) `SET work_mem = '256MB'`、(4) 複合インデックスの追加。

### 症状4: COUNT(*) OVER()とCOUNT(col) OVER()の結果が異なる

**原因:** COUNT(col)はNULLを除外するため。パーティション内にNULLがある列で発生。

**解決:** 意図に応じてCOUNT(*)（NULL込み）またはCOUNT(col)（NULL除外）を選択する。

---

## まとめ

| 項目 | 要点 |
|------|------|
| ウィンドウ関数 | GROUP BYと異なり行を折りたたまない。SELECT後、ORDER BY前に評価 |
| ROW_NUMBER | 一意連番。Top-N、ページネーション、重複排除。タイブレーカー必須 |
| RANK/DENSE_RANK | 同順位の扱いが異なる。ランキング表示ではDENSE_RANKが多い |
| LAG/LEAD | 前後行参照。前期比、差分計算、ギャップ検出に必須 |
| FIRST_VALUE/LAST_VALUE | LAST_VALUEはフレーム明示必須。NTH_VALUEはN番目取得 |
| フレーム指定 | ROWS BETWEEN で集約範囲を制御。デフォルトRANGEに注意 |
| WINDOW句 | 同じウィンドウ定義の再利用。ソート回数削減で性能向上 |
| パフォーマンス | 複合インデックス、work_mem調整、WHERE句による事前フィルタが重要 |

---

## 次に読むべきガイド

- [01-cte-recursive.md](./01-cte-recursive.md) -- CTEとウィンドウ関数の組み合わせ
- [04-query-optimization.md](./04-query-optimization.md) -- ウィンドウ関数の実行計画
- [03-aggregation.md](../00-basics/03-aggregation.md) -- 集約関数との対比

---

## 参考文献

1. PostgreSQL Documentation -- "Window Functions" https://www.postgresql.org/docs/current/tutorial-window.html
2. PostgreSQL Documentation -- "Window Function Calls" https://www.postgresql.org/docs/current/sql-expressions.html#SYNTAX-WINDOW-FUNCTIONS
3. Winand, M. -- "Modern SQL: Window Functions" https://modern-sql.com/feature/window-functions
4. Molinaro, A. (2005). *SQL Cookbook*. O'Reilly Media. Chapter 12: Reporting and Warehousing.
5. Kline, K., Kline, D., & Hunt, B. (2008). *SQL in a Nutshell*. O'Reilly Media. Chapter 4: SQL Functions.
6. ISO/IEC 9075-2:2023 -- SQL Part 2: Foundation (Window Function Specification)
