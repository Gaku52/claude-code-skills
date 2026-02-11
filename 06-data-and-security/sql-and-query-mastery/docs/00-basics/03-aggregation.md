# 集約 — GROUP BY・HAVING・集約関数

> 集約操作は複数行のデータを要約して単一の値に変換する処理であり、レポーティングや分析クエリの基盤となる。

## この章で学ぶこと

1. 主要な集約関数（COUNT, SUM, AVG, MIN, MAX）の動作と注意点
2. GROUP BYとHAVINGの正しい使い方と実行順序の理解
3. GROUPING SETS、ROLLUP、CUBEによる多次元集約

---

## 1. 集約関数の基本

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
    region      VARCHAR(20)
);

-- 基本集約関数
SELECT
    COUNT(*)          AS total_rows,        -- 全行数
    COUNT(amount)     AS non_null_count,    -- NULL以外の行数
    COUNT(DISTINCT category) AS categories, -- ユニーク値数
    SUM(amount)       AS total_sales,       -- 合計
    AVG(amount)       AS avg_sale,          -- 平均
    MIN(amount)       AS min_sale,          -- 最小
    MAX(amount)       AS max_sale,          -- 最大
    MIN(sale_date)    AS first_sale,        -- 最古日
    MAX(sale_date)    AS last_sale          -- 最新日
FROM sales;
```

### コード例2: COUNT の3つの使い方

```sql
-- COUNT(*): NULLを含む全行を数える
SELECT COUNT(*) FROM employees;           -- → 100

-- COUNT(column): NULLを除いた行を数える
SELECT COUNT(phone) FROM employees;       -- → 85 (15人はNULL)

-- COUNT(DISTINCT column): ユニーク値を数える
SELECT COUNT(DISTINCT department_id) FROM employees;  -- → 8
```

---

## 2. GROUP BY

### コード例3: GROUP BYの基本と応用

```sql
-- 基本: カテゴリ別の売上集計
SELECT
    category,
    COUNT(*) AS sales_count,
    SUM(amount) AS total_amount,
    ROUND(AVG(amount), 2) AS avg_amount
FROM sales
GROUP BY category
ORDER BY total_amount DESC;

-- 複数列でグループ化
SELECT
    category,
    region,
    SUM(amount) AS total_amount
FROM sales
GROUP BY category, region
ORDER BY category, total_amount DESC;

-- 式でグループ化（月別集計）
SELECT
    DATE_TRUNC('month', sale_date) AS month,
    SUM(amount) AS monthly_total,
    COUNT(*) AS transaction_count
FROM sales
GROUP BY DATE_TRUNC('month', sale_date)
ORDER BY month;
```

### GROUP BY の実行フロー

```
┌─────────────────── GROUP BY 処理の流れ ──────────────────┐
│                                                          │
│  元テーブル (sales)                                       │
│  ┌──────────┬──────────┬──────────┐                     │
│  │ category │ region   │ amount   │                     │
│  ├──────────┼──────────┼──────────┤                     │
│  │ 食品     │ 東京     │ 1000     │                     │
│  │ 食品     │ 大阪     │ 1500     │                     │
│  │ 家電     │ 東京     │ 5000     │                     │
│  │ 食品     │ 東京     │ 2000     │                     │
│  │ 家電     │ 大阪     │ 3000     │                     │
│  └──────────┴──────────┴──────────┘                     │
│       │                                                  │
│       ▼ GROUP BY category                                │
│  ┌──────────┬─────────────────────┐                     │
│  │ category │ グループ内の行       │                     │
│  ├──────────┼─────────────────────┤                     │
│  │ 食品     │ {1000, 1500, 2000}  │ → SUM=4500, AVG=1500│
│  │ 家電     │ {5000, 3000}        │ → SUM=8000, AVG=4000│
│  └──────────┴─────────────────────┘                     │
│       │                                                  │
│       ▼ 集約関数の適用                                    │
│  ┌──────────┬───────┬───────┐                           │
│  │ category │ SUM   │ AVG   │                           │
│  ├──────────┼───────┼───────┤                           │
│  │ 食品     │ 4500  │ 1500  │                           │
│  │ 家電     │ 8000  │ 4000  │                           │
│  └──────────┴───────┴───────┘                           │
└──────────────────────────────────────────────────────────┘
```

---

## 3. HAVING

### コード例4: HAVINGによるグループのフィルタリング

```sql
-- WHEREは行をフィルタ、HAVINGはグループをフィルタ
SELECT
    category,
    COUNT(*) AS sales_count,
    SUM(amount) AS total_amount
FROM sales
WHERE sale_date >= '2024-01-01'     -- ① 行レベルのフィルタ
GROUP BY category
HAVING SUM(amount) >= 10000         -- ② グループレベルのフィルタ
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
```

### WHERE vs HAVING の実行タイミング

```
┌──────────── WHERE vs HAVING の違い ─────────────┐
│                                                   │
│  FROM → WHERE → GROUP BY → HAVING → SELECT       │
│           │                   │                   │
│           │                   │                   │
│    ┌──────┴──────┐    ┌──────┴──────┐            │
│    │  WHERE      │    │  HAVING     │            │
│    │  行をフィルタ│    │グループをフィルタ│         │
│    │  集約前      │    │  集約後      │            │
│    │  集約関数    │    │  集約関数    │            │
│    │  使用不可    │    │  使用可能    │            │
│    └─────────────┘    └─────────────┘            │
│                                                   │
│  例: WHERE amount > 100   ← 各行で判定           │
│      HAVING SUM(amount) > 10000 ← グループで判定 │
└───────────────────────────────────────────────────┘
```

---

## 4. 高度な集約

### コード例5: GROUPING SETS / ROLLUP / CUBE

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
SELECT
    COALESCE(category, '【総計】') AS category,
    COALESCE(region, '【小計】') AS region,
    SUM(amount) AS total
FROM sales
GROUP BY ROLLUP (category, region);

-- CUBE: 全組み合わせの集約
SELECT
    category,
    region,
    SUM(amount) AS total
FROM sales
GROUP BY CUBE (category, region);

-- FILTER句（PostgreSQL）: 条件付き集約
SELECT
    category,
    COUNT(*) AS total_count,
    COUNT(*) FILTER (WHERE region = '東京') AS tokyo_count,
    SUM(amount) FILTER (WHERE sale_date >= '2024-01-01') AS ytd_amount
FROM sales
GROUP BY category;
```

### コード例6: 統計関数

```sql
-- 分散と標準偏差
SELECT
    department_id,
    AVG(salary) AS avg_salary,
    STDDEV(salary) AS salary_stddev,       -- 標準偏差
    VARIANCE(salary) AS salary_variance,   -- 分散
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY salary) AS median_salary
FROM employees
GROUP BY department_id;

-- 文字列集約
SELECT
    department_id,
    STRING_AGG(name, ', ' ORDER BY name) AS member_list
FROM employees
GROUP BY department_id;

-- 配列集約（PostgreSQL）
SELECT
    department_id,
    ARRAY_AGG(name ORDER BY hired_date) AS members
FROM employees
GROUP BY department_id;
```

---

## 集約関数一覧表

| 関数 | 説明 | NULL扱い | 使用例 |
|------|------|---------|--------|
| COUNT(*) | 全行数 | 含む | `COUNT(*)` |
| COUNT(col) | 非NULL行数 | 除外 | `COUNT(email)` |
| COUNT(DISTINCT col) | ユニーク値数 | 除外 | `COUNT(DISTINCT category)` |
| SUM(col) | 合計 | 除外 | `SUM(amount)` |
| AVG(col) | 平均 | 除外 | `AVG(salary)` |
| MIN(col) | 最小値 | 除外 | `MIN(price)` |
| MAX(col) | 最大値 | 除外 | `MAX(created_at)` |
| STRING_AGG | 文字列結合 | 除外 | `STRING_AGG(name, ',')` |
| ARRAY_AGG | 配列化 | 含む | `ARRAY_AGG(tag)` |
| BOOL_AND/OR | 論理積/和 | 除外 | `BOOL_AND(is_active)` |

## ROLLUP vs CUBE vs GROUPING SETS 比較表

| 機能 | 生成される集約 | 行数目安 | 用途 |
|------|---------------|---------|------|
| GROUP BY A, B | (A, B) | グループ数 | 基本集約 |
| ROLLUP(A, B) | (A,B), (A), () | 階層的小計 | 帳票の小計行 |
| CUBE(A, B) | (A,B), (A), (B), () | 全組み合わせ | 多次元分析 |
| GROUPING SETS | 明示指定 | 指定数分 | 柔軟な集約 |

---

## アンチパターン

### アンチパターン1: GROUP BYに含めない列をSELECTする

```sql
-- NG: nameはGROUP BYに含まれていない
SELECT department_id, name, AVG(salary)
FROM employees
GROUP BY department_id;
-- → PostgreSQLではエラー、MySQLでは不定値が返る（ONLY_FULL_GROUP_BY無効時）

-- OK: GROUP BYに含めるか、集約関数で包む
SELECT department_id, MIN(name), AVG(salary)
FROM employees
GROUP BY department_id;

-- OK: ウィンドウ関数を使う
SELECT DISTINCT
    department_id,
    FIRST_VALUE(name) OVER (PARTITION BY department_id ORDER BY salary DESC),
    AVG(salary) OVER (PARTITION BY department_id)
FROM employees;
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
```

---

## FAQ

### Q1: WHEREとHAVINGはどう使い分けるか？

WHEREは個々の行に対するフィルタ（集約前）、HAVINGはグループに対するフィルタ（集約後）。集約関数を使わない条件は必ずWHEREに書く。WHEREで先にフィルタすることで処理対象行数が減り、パフォーマンスが向上する。

### Q2: COUNT(*)とCOUNT(1)に性能差はあるか？

現代のRDBMSでは差はない。オプティマイザが同じ実行計画を生成する。`COUNT(*)`が標準的で意図が明確なため推奨。

### Q3: GROUP BYで日付を扱う場合の注意点は？

TIMESTAMP型でGROUP BYすると秒単位でグループ化されるため、意図通りに集約されない。`DATE_TRUNC('day', timestamp_col)`や`CAST(... AS DATE)`で丸める必要がある。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 集約関数 | COUNT, SUM, AVG, MIN, MAX が基本5関数 |
| NULLの扱い | COUNT(*)以外はNULLを除外して計算 |
| GROUP BY | SELECT句の非集約列は全てGROUP BYに含める |
| HAVING | GROUP BY後のフィルタ。集約関数が使用可能 |
| ROLLUP/CUBE | 階層的/多次元的な小計を一度のクエリで取得 |
| FILTER句 | 条件別集約をCASE式より簡潔に書ける（PostgreSQL） |

---

## 次に読むべきガイド

- [04-subqueries.md](./04-subqueries.md) — 集約結果をサブクエリで活用
- [00-window-functions.md](../01-advanced/00-window-functions.md) — ウィンドウ関数で行ごとの集約
- [04-query-optimization.md](../01-advanced/04-query-optimization.md) — 集約クエリの最適化

---

## 参考文献

1. PostgreSQL Documentation — "Aggregate Functions" https://www.postgresql.org/docs/current/functions-aggregate.html
2. Celko, J. (2010). *Joe Celko's SQL for Smarties: Advanced SQL Programming*. Morgan Kaufmann.
3. Kline, K., Kline, D., & Hunt, B. (2008). *SQL in a Nutshell*. O'Reilly Media.
