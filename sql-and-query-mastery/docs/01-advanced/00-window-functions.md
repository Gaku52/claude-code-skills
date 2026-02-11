# ウィンドウ関数 — ROW_NUMBER・RANK・LAG/LEAD

> ウィンドウ関数はGROUP BYのように行を折りたたまず、各行に対して「窓」を定義し、その範囲内での集約・順位付け・前後行参照を行う機能である。

## この章で学ぶこと

1. ウィンドウ関数の構文（OVER句、PARTITION BY、ORDER BY、フレーム）を完全理解する
2. ROW_NUMBER、RANK、DENSE_RANK、NTILE の使い分け
3. LAG/LEAD、FIRST_VALUE/LAST_VALUE、累積集約の実践パターン

---

## 1. ウィンドウ関数の構文

```
┌─────────── ウィンドウ関数の構文構造 ──────────────┐
│                                                    │
│  関数名(...) OVER (                                │
│      PARTITION BY 列    -- グループ分割（省略可）   │
│      ORDER BY 列        -- 並び順（省略可）        │
│      ROWS/RANGE BETWEEN -- フレーム定義（省略可）  │
│          開始点 AND 終了点                          │
│  )                                                 │
│                                                    │
│  フレーム指定:                                      │
│  ┌─────────────────────────────────────────┐      │
│  │ UNBOUNDED PRECEDING  ── パーティション先頭│      │
│  │ N PRECEDING          ── N行前            │      │
│  │ CURRENT ROW          ── 現在行           │      │
│  │ N FOLLOWING          ── N行後            │      │
│  │ UNBOUNDED FOLLOWING  ── パーティション末尾│      │
│  └─────────────────────────────────────────┘      │
└────────────────────────────────────────────────────┘
```

### コード例1: ウィンドウ関数 vs GROUP BY

```sql
-- GROUP BY: 行が折りたたまれる（部署ごとに1行）
SELECT department_id, AVG(salary) AS avg_salary
FROM employees
GROUP BY department_id;

-- ウィンドウ関数: 全行が保持される（各行に部署平均が付与）
SELECT
    name,
    department_id,
    salary,
    AVG(salary) OVER (PARTITION BY department_id) AS dept_avg_salary,
    salary - AVG(salary) OVER (PARTITION BY department_id) AS diff_from_avg
FROM employees;

-- 結果イメージ:
-- name | dept_id | salary  | dept_avg | diff
-- 田中 |    10   | 450000  | 415000   | +35000
-- 佐藤 |    10   | 380000  | 415000   | -35000
-- 鈴木 |    20   | 520000  | 520000   |      0
```

---

## 2. 順位付け関数

### コード例2: ROW_NUMBER / RANK / DENSE_RANK

```sql
-- 3種類の順位付けの違い
SELECT
    name,
    department_id,
    salary,
    ROW_NUMBER() OVER (ORDER BY salary DESC) AS row_num,    -- 連番（重複なし）
    RANK()       OVER (ORDER BY salary DESC) AS rank,       -- 同順あり（飛び番）
    DENSE_RANK() OVER (ORDER BY salary DESC) AS dense_rank  -- 同順あり（連番）
FROM employees;

-- 結果例:
-- name   | salary  | row_num | rank | dense_rank
-- 鈴木   | 520000  |    1    |  1   |    1
-- 田中   | 450000  |    2    |  2   |    2
-- 高橋   | 450000  |    3    |  2   |    2      ← 同額
-- 佐藤   | 380000  |    4    |  4   |    3      ← rank=4飛び, dense=3連番
```

### 順位付けの違い図解

```
┌────────── ROW_NUMBER vs RANK vs DENSE_RANK ──────────┐
│                                                       │
│  データ: 100, 90, 90, 80, 70                          │
│                                                       │
│  値    ROW_NUMBER   RANK    DENSE_RANK                │
│  100       1          1         1                     │
│  90        2          2         2                     │
│  90        3          2         2     ← 同値の扱い    │
│  80        4          4         3     ← 飛びvs連番    │
│  70        5          5         4                     │
│                                                       │
│  ROW_NUMBER : 一意の連番。同値でも異なる番号          │
│  RANK       : 同値は同順。次の順位は飛ぶ              │
│  DENSE_RANK : 同値は同順。次の順位は連番              │
└───────────────────────────────────────────────────────┘
```

### コード例3: PARTITION BY + 順位付け（Top-N問題）

```sql
-- 部署ごとの給与Top3を取得
SELECT * FROM (
    SELECT
        name,
        department_id,
        salary,
        ROW_NUMBER() OVER (
            PARTITION BY department_id
            ORDER BY salary DESC
        ) AS rn
    FROM employees
) ranked
WHERE rn <= 3;

-- NTILE: N等分に分割
SELECT
    name,
    salary,
    NTILE(4) OVER (ORDER BY salary) AS quartile
FROM employees;
-- → 給与の低い順に4グループに分割（1=下位25%, 4=上位25%）
```

---

## 3. 前後行参照: LAG / LEAD

### コード例4: LAG / LEAD

```sql
-- 月次売上の前月比を計算
SELECT
    month,
    revenue,
    LAG(revenue, 1) OVER (ORDER BY month) AS prev_month,
    LEAD(revenue, 1) OVER (ORDER BY month) AS next_month,
    revenue - LAG(revenue, 1) OVER (ORDER BY month) AS mom_change,
    ROUND(
        (revenue - LAG(revenue, 1) OVER (ORDER BY month))::NUMERIC
        / LAG(revenue, 1) OVER (ORDER BY month) * 100, 1
    ) AS mom_pct
FROM monthly_sales;

-- LAGの第3引数: デフォルト値
SELECT
    month,
    revenue,
    LAG(revenue, 1, 0) OVER (ORDER BY month) AS prev_or_zero
FROM monthly_sales;

-- FIRST_VALUE / LAST_VALUE
SELECT
    name,
    department_id,
    salary,
    FIRST_VALUE(name) OVER (
        PARTITION BY department_id ORDER BY salary DESC
    ) AS highest_paid,
    LAST_VALUE(name) OVER (
        PARTITION BY department_id ORDER BY salary DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS lowest_paid
FROM employees;
```

---

## 4. フレーム指定と累積集約

### コード例5: 累積合計と移動平均

```sql
-- 累積合計（Running Total）
SELECT
    sale_date,
    amount,
    SUM(amount) OVER (
        ORDER BY sale_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total
FROM daily_sales;

-- 7日間移動平均
SELECT
    sale_date,
    amount,
    AVG(amount) OVER (
        ORDER BY sale_date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7d
FROM daily_sales;

-- パーティション内の割合
SELECT
    name,
    department_id,
    salary,
    ROUND(
        salary::NUMERIC / SUM(salary) OVER (PARTITION BY department_id) * 100, 1
    ) AS pct_of_dept,
    ROUND(
        salary::NUMERIC / SUM(salary) OVER () * 100, 1
    ) AS pct_of_total
FROM employees;
```

### フレーム指定の動作

```
┌──────── フレーム指定の視覚化 ────────┐
│                                       │
│  ROWS BETWEEN 2 PRECEDING             │
│               AND CURRENT ROW         │
│                                       │
│  行1: 100  ─┐                         │
│  行2: 200  ─┤ ← 行3のフレーム        │
│  行3: 150  ─┘ ← CURRENT ROW          │
│  行4: 300  ─┐                         │
│  行5: 250  ─┤ ← 行5のフレーム        │
│             ─┘                        │
│                                       │
│  ROWS BETWEEN UNBOUNDED PRECEDING     │
│               AND CURRENT ROW         │
│  → 先頭から現在行まで（累積）          │
│                                       │
│  ROWS BETWEEN 3 PRECEDING             │
│               AND 3 FOLLOWING         │
│  → 前後3行ずつ（7行の移動窓）         │
└───────────────────────────────────────┘
```

### コード例6: 名前付きウィンドウ（WINDOW句）

```sql
-- 同じウィンドウ定義を使い回す
SELECT
    name,
    department_id,
    salary,
    ROW_NUMBER() OVER w AS rn,
    RANK() OVER w AS rnk,
    SUM(salary) OVER w AS running_sum,
    AVG(salary) OVER w AS running_avg
FROM employees
WINDOW w AS (PARTITION BY department_id ORDER BY salary DESC)
ORDER BY department_id, salary DESC;
```

---

## ウィンドウ関数一覧表

| 関数 | 分類 | 説明 | フレーム |
|------|------|------|---------|
| ROW_NUMBER() | 順位 | 一意の連番 | 不要 |
| RANK() | 順位 | 同順あり（飛び番） | 不要 |
| DENSE_RANK() | 順位 | 同順あり（連番） | 不要 |
| NTILE(n) | 順位 | N等分に分割 | 不要 |
| LAG(col, n) | 前後参照 | N行前の値 | 不要 |
| LEAD(col, n) | 前後参照 | N行後の値 | 不要 |
| FIRST_VALUE(col) | 前後参照 | フレーム内の最初の値 | 使用可 |
| LAST_VALUE(col) | 前後参照 | フレーム内の最後の値 | 要注意 |
| NTH_VALUE(col, n) | 前後参照 | フレーム内のN番目の値 | 使用可 |
| SUM/AVG/COUNT/MIN/MAX | 集約 | フレーム内の集約 | 使用可 |

## ROWS vs RANGE 比較表

| 項目 | ROWS | RANGE |
|------|------|-------|
| 単位 | 物理的な行数 | 論理的な値の範囲 |
| 同値の扱い | 個別に扱う | まとめて扱う |
| デフォルト | — | ORDER BY指定時のデフォルト |
| パフォーマンス | 高速 | やや遅い |
| 推奨場面 | 移動平均、累積合計 | 値ベースの範囲集約 |

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
```

### アンチパターン2: ウィンドウ関数の結果をWHEREで直接フィルタ

```sql
-- NG: WHERE句でウィンドウ関数は使用不可
SELECT name, salary, ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn
FROM employees
WHERE rn <= 10;  -- エラー！

-- OK: サブクエリまたはCTEで包む
WITH ranked AS (
    SELECT name, salary, ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn
    FROM employees
)
SELECT * FROM ranked WHERE rn <= 10;
```

---

## FAQ

### Q1: ウィンドウ関数はGROUP BYと併用できるか？

可能だが、ウィンドウ関数はGROUP BY の後に実行される。つまり、GROUP BY で集約された結果に対してウィンドウ関数が適用される。GROUP BY前の行にはアクセスできない。

### Q2: ウィンドウ関数はインデックスで高速化できるか？

PARTITION BY列とORDER BY列にインデックスがあると、ソート処理が省略される場合がある。特に `CREATE INDEX ON table (partition_col, order_col)` の複合インデックスが有効。

### Q3: ROW_NUMBERでページネーションは推奨されるか？

大量データでは非推奨。ROW_NUMBERは全行をソートしてから番号を振るため、深いページ（OFFSET大）でパフォーマンスが劣化する。キーセットページネーション（`WHERE id > last_seen_id LIMIT 20`）の方が効率的。

---

## まとめ

| 項目 | 要点 |
|------|------|
| ウィンドウ関数 | GROUP BYと異なり行を折りたたまない |
| ROW_NUMBER | 一意連番。Top-N、ページネーション、重複排除 |
| RANK/DENSE_RANK | 同順位の扱いが異なる。ランキング表示 |
| LAG/LEAD | 前後行参照。前期比、差分計算 |
| フレーム指定 | ROWS BETWEEN で集約範囲を制御 |
| WINDOW句 | 同じウィンドウ定義の再利用で可読性向上 |

---

## 次に読むべきガイド

- [01-cte-recursive.md](./01-cte-recursive.md) — CTEとウィンドウ関数の組み合わせ
- [04-query-optimization.md](./04-query-optimization.md) — ウィンドウ関数の実行計画
- [03-aggregation.md](../00-basics/03-aggregation.md) — 集約関数との対比

---

## 参考文献

1. PostgreSQL Documentation — "Window Functions" https://www.postgresql.org/docs/current/tutorial-window.html
2. Winand, M. — "Modern SQL: Window Functions" https://modern-sql.com/feature/window-functions
3. Molinaro, A. (2005). *SQL Cookbook*. O'Reilly Media. Chapter 12: Reporting and Warehousing.
