# データモデリング — スター / スノーフレーク・次元モデル

> データモデリングはビジネス要件をデータ構造に変換する技法であり、OLTP向けの正規化モデルとOLAP向けの次元モデル（スター/スノーフレーク）を使い分けることで、トランザクション処理と分析の両方に最適な設計を実現する。

## この章で学ぶこと

1. OLTPモデルとOLAPモデルの本質的な違い
2. スタースキーマとスノーフレークスキーマの構造と使い分け
3. ファクトテーブルとディメンションテーブルの設計パターン

---

## 1. OLTP vs OLAP

```
┌────────────── OLTP vs OLAP ──────────────────┐
│                                               │
│  OLTP (Online Transaction Processing)         │
│  ┌─────────────────────────────────────────┐ │
│  │ 目的: 業務処理（注文、更新、削除）       │ │
│  │ 特徴: 少数行の読み書き、高頻度           │ │
│  │ 設計: 正規化（3NF）                      │ │
│  │ 例:   ECサイトの注文処理                 │ │
│  └─────────────────────────────────────────┘ │
│                     │                         │
│                ETL / ELT                      │
│                     │                         │
│  OLAP (Online Analytical Processing)          │
│  ┌─────────────────────────────────────────┐ │
│  │ 目的: 分析・レポーティング               │ │
│  │ 特徴: 大量行の読み取り、集約             │ │
│  │ 設計: 非正規化（スタースキーマ）          │ │
│  │ 例:   月次売上レポート、KPIダッシュボード │ │
│  └─────────────────────────────────────────┘ │
└───────────────────────────────────────────────┘
```

---

## 2. スタースキーマ

### コード例1: スタースキーマの実装

```sql
-- ファクトテーブル（中心）: 計測可能なイベント
CREATE TABLE fact_sales (
    sale_id         BIGSERIAL PRIMARY KEY,
    -- ディメンションキー（外部キー）
    date_key        INTEGER NOT NULL REFERENCES dim_date(date_key),
    product_key     INTEGER NOT NULL REFERENCES dim_product(product_key),
    customer_key    INTEGER NOT NULL REFERENCES dim_customer(customer_key),
    store_key       INTEGER NOT NULL REFERENCES dim_store(store_key),
    -- メジャー（計測値）
    quantity        INTEGER NOT NULL,
    unit_price      DECIMAL(10, 2) NOT NULL,
    discount_amount DECIMAL(10, 2) DEFAULT 0,
    total_amount    DECIMAL(12, 2) NOT NULL,
    cost_amount     DECIMAL(10, 2) NOT NULL,
    profit_amount   DECIMAL(10, 2) GENERATED ALWAYS AS
                    (total_amount - cost_amount) STORED
);

-- ディメンションテーブル（外周）: 分析の切り口
CREATE TABLE dim_date (
    date_key      INTEGER PRIMARY KEY,  -- YYYYMMDD形式
    full_date     DATE NOT NULL,
    year          SMALLINT NOT NULL,
    quarter       SMALLINT NOT NULL,
    month         SMALLINT NOT NULL,
    month_name    VARCHAR(10) NOT NULL,
    week          SMALLINT NOT NULL,
    day_of_week   SMALLINT NOT NULL,
    day_name      VARCHAR(10) NOT NULL,
    is_weekend    BOOLEAN NOT NULL,
    is_holiday    BOOLEAN DEFAULT FALSE,
    fiscal_year   SMALLINT,
    fiscal_quarter SMALLINT
);

CREATE TABLE dim_product (
    product_key     SERIAL PRIMARY KEY,
    product_id      VARCHAR(20) NOT NULL,  -- ビジネスキー
    product_name    VARCHAR(200) NOT NULL,
    category        VARCHAR(100),
    subcategory     VARCHAR(100),
    brand           VARCHAR(100),
    supplier        VARCHAR(200),
    unit_cost       DECIMAL(10, 2),
    -- SCD Type 2 用
    effective_from  DATE NOT NULL,
    effective_to    DATE DEFAULT '9999-12-31',
    is_current      BOOLEAN DEFAULT TRUE
);

CREATE TABLE dim_customer (
    customer_key    SERIAL PRIMARY KEY,
    customer_id     VARCHAR(20) NOT NULL,
    customer_name   VARCHAR(200) NOT NULL,
    segment         VARCHAR(50),
    city            VARCHAR(100),
    region          VARCHAR(50),
    country         VARCHAR(100),
    effective_from  DATE NOT NULL,
    effective_to    DATE DEFAULT '9999-12-31',
    is_current      BOOLEAN DEFAULT TRUE
);

CREATE TABLE dim_store (
    store_key   SERIAL PRIMARY KEY,
    store_id    VARCHAR(20) NOT NULL,
    store_name  VARCHAR(200) NOT NULL,
    store_type  VARCHAR(50),
    city        VARCHAR(100),
    region      VARCHAR(50),
    manager     VARCHAR(200)
);
```

### スタースキーマの構造

```
┌──────────────── スタースキーマ ─────────────────┐
│                                                  │
│          dim_date                                 │
│         ┌───────┐                                │
│         │ 日付  │                                │
│         │ 年/月 │                                │
│         │ 曜日  │                                │
│         └───┬───┘                                │
│             │                                    │
│  dim_product│        dim_customer                 │
│  ┌───────┐  │  ┌────────────┐  ┌───────┐        │
│  │ 商品  │──┼──│ fact_sales │──│ 顧客  │        │
│  │ カテゴリ│  │  │ (ファクト)  │  │ 地域  │        │
│  │ ブランド│  │  │  数量      │  │ セグメント│     │
│  └───────┘  │  │  金額      │  └───────┘        │
│             │  │  利益      │                    │
│         ┌───┴──│            │                    │
│         │      └────────────┘                    │
│  dim_store                                       │
│  ┌───────┐                                       │
│  │ 店舗  │  ★ 中心にファクト、周囲にディメンション │
│  │ 地域  │  → 星型（スター）に見える             │
│  └───────┘                                       │
└──────────────────────────────────────────────────┘
```

---

## 3. スノーフレークスキーマ

### コード例2: スノーフレークスキーマ

```sql
-- スノーフレーク: ディメンションをさらに正規化
CREATE TABLE dim_category (
    category_key  SERIAL PRIMARY KEY,
    category_name VARCHAR(100) NOT NULL
);

CREATE TABLE dim_subcategory (
    subcategory_key SERIAL PRIMARY KEY,
    subcategory_name VARCHAR(100) NOT NULL,
    category_key    INTEGER REFERENCES dim_category(category_key)
);

CREATE TABLE dim_product_snowflake (
    product_key     SERIAL PRIMARY KEY,
    product_name    VARCHAR(200) NOT NULL,
    subcategory_key INTEGER REFERENCES dim_subcategory(subcategory_key),
    brand           VARCHAR(100)
);

-- スノーフレークでの分析クエリ（JOINが増える）
SELECT
    c.category_name,
    sc.subcategory_name,
    SUM(f.total_amount) AS revenue
FROM fact_sales f
    JOIN dim_product_snowflake p ON f.product_key = p.product_key
    JOIN dim_subcategory sc ON p.subcategory_key = sc.subcategory_key
    JOIN dim_category c ON sc.category_key = c.category_key
GROUP BY c.category_name, sc.subcategory_name
ORDER BY revenue DESC;
```

---

## 4. SCD（Slowly Changing Dimensions）

### コード例3: SCD Type 1 / 2 / 3

```sql
-- SCD Type 1: 上書き（履歴なし）
UPDATE dim_customer
SET city = '大阪', region = '関西'
WHERE customer_id = 'C001' AND is_current = TRUE;

-- SCD Type 2: 新行追加（完全な履歴）
-- Step 1: 旧行を無効化
UPDATE dim_customer
SET effective_to = CURRENT_DATE - 1, is_current = FALSE
WHERE customer_id = 'C001' AND is_current = TRUE;

-- Step 2: 新行を追加
INSERT INTO dim_customer (customer_id, customer_name, segment, city, region,
                          country, effective_from, effective_to, is_current)
VALUES ('C001', '田中太郎', 'Premium', '大阪', '関西',
        '日本', CURRENT_DATE, '9999-12-31', TRUE);

-- SCD Type 2 のクエリ: 特定時点の状態を参照
SELECT * FROM dim_customer
WHERE customer_id = 'C001'
  AND '2024-06-15' BETWEEN effective_from AND effective_to;

-- SCD Type 3: 現在値と前回値を列で保持
ALTER TABLE dim_customer ADD COLUMN previous_city VARCHAR(100);
ALTER TABLE dim_customer ADD COLUMN city_changed_at DATE;

UPDATE dim_customer
SET previous_city = city,
    city = '大阪',
    city_changed_at = CURRENT_DATE
WHERE customer_id = 'C001';
```

### コード例4: 分析クエリの実践

```sql
-- 月別・カテゴリ別の売上分析
SELECT
    dd.year,
    dd.month_name,
    dp.category,
    COUNT(*) AS transactions,
    SUM(fs.quantity) AS total_quantity,
    SUM(fs.total_amount) AS revenue,
    SUM(fs.profit_amount) AS profit,
    ROUND(SUM(fs.profit_amount) / NULLIF(SUM(fs.total_amount), 0) * 100, 1)
        AS profit_margin_pct
FROM fact_sales fs
    JOIN dim_date dd ON fs.date_key = dd.date_key
    JOIN dim_product dp ON fs.product_key = dp.product_key
WHERE dd.year = 2024
  AND dp.is_current = TRUE
GROUP BY dd.year, dd.month, dd.month_name, dp.category
ORDER BY dd.month, revenue DESC;

-- YoY（前年比）比較
WITH yearly AS (
    SELECT
        dd.year,
        dp.category,
        SUM(fs.total_amount) AS revenue
    FROM fact_sales fs
        JOIN dim_date dd ON fs.date_key = dd.date_key
        JOIN dim_product dp ON fs.product_key = dp.product_key
    WHERE dd.year IN (2023, 2024)
    GROUP BY dd.year, dp.category
)
SELECT
    c.category,
    c.revenue AS current_year,
    p.revenue AS previous_year,
    ROUND((c.revenue - p.revenue) / p.revenue * 100, 1) AS yoy_growth
FROM yearly c
    JOIN yearly p ON c.category = p.category AND c.year = p.year + 1
ORDER BY yoy_growth DESC;
```

### コード例5: ETLパイプライン

```sql
-- ステージングテーブルへの生データロード
CREATE TABLE stg_sales_raw (
    transaction_id  VARCHAR(50),
    sale_date       DATE,
    product_code    VARCHAR(20),
    customer_code   VARCHAR(20),
    store_code      VARCHAR(20),
    quantity        INTEGER,
    unit_price      DECIMAL(10,2),
    discount_pct    DECIMAL(5,2),
    loaded_at       TIMESTAMPTZ DEFAULT NOW()
);

-- ステージングからファクトテーブルへの変換・ロード
INSERT INTO fact_sales (
    date_key, product_key, customer_key, store_key,
    quantity, unit_price, discount_amount, total_amount, cost_amount
)
SELECT
    TO_CHAR(s.sale_date, 'YYYYMMDD')::INTEGER AS date_key,
    dp.product_key,
    dc.customer_key,
    ds.store_key,
    s.quantity,
    s.unit_price,
    s.unit_price * s.quantity * s.discount_pct / 100 AS discount_amount,
    s.unit_price * s.quantity * (1 - s.discount_pct / 100) AS total_amount,
    dp.unit_cost * s.quantity AS cost_amount
FROM stg_sales_raw s
    JOIN dim_product dp ON dp.product_id = s.product_code AND dp.is_current
    JOIN dim_customer dc ON dc.customer_id = s.customer_code AND dc.is_current
    JOIN dim_store ds ON ds.store_id = s.store_code
WHERE s.loaded_at > (SELECT MAX(loaded_at) FROM fact_sales_load_log);
```

### コード例6: マテリアライズドビューの活用

```sql
-- 月次サマリーのマテリアライズドビュー
CREATE MATERIALIZED VIEW mv_monthly_summary AS
SELECT
    dd.year,
    dd.month,
    dp.category,
    dc.region,
    ds.store_name,
    COUNT(*) AS transaction_count,
    SUM(fs.quantity) AS total_quantity,
    SUM(fs.total_amount) AS revenue,
    SUM(fs.profit_amount) AS profit
FROM fact_sales fs
    JOIN dim_date dd ON fs.date_key = dd.date_key
    JOIN dim_product dp ON fs.product_key = dp.product_key
    JOIN dim_customer dc ON fs.customer_key = dc.customer_key
    JOIN dim_store ds ON fs.store_key = ds.store_key
WHERE dp.is_current AND dc.is_current
GROUP BY dd.year, dd.month, dp.category, dc.region, ds.store_name
WITH DATA;

CREATE UNIQUE INDEX idx_mv_monthly ON mv_monthly_summary
    (year, month, category, region, store_name);

-- 差分リフレッシュ（CONCURRENTLY = 読み取りブロックなし）
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_monthly_summary;
```

---

## スタースキーマ vs スノーフレークスキーマ 比較表

| 特徴 | スタースキーマ | スノーフレークスキーマ |
|------|-------------|-------------------|
| ディメンション構造 | 非正規化（1テーブル） | 正規化（複数テーブル） |
| JOINの数 | 少ない | 多い |
| クエリの複雑さ | シンプル | 複雑 |
| クエリ性能 | 高速 | やや遅い |
| ストレージ | 冗長（大きい） | 効率的（小さい） |
| ETLの複雑さ | シンプル | やや複雑 |
| 更新整合性 | 要注意 | 容易 |
| BI ツール互換性 | 高い | 中程度 |

## SCD (Slowly Changing Dimension) 比較表

| Type | 方式 | 履歴 | 実装複雑度 | 用途 |
|------|------|------|-----------|------|
| Type 0 | 変更しない | なし | 最も簡単 | 不変マスタ |
| Type 1 | 上書き | なし | 簡単 | 現在値のみ必要 |
| Type 2 | 新行追加 | 完全 | 複雑 | 時系列分析 |
| Type 3 | 前回値カラム | 直近1回 | 中程度 | 変更前後比較 |
| Type 4 | 履歴テーブル分離 | 完全 | やや複雑 | 大量履歴 |
| Type 6 | 1+2+3 の組合せ | 完全+現在値 | 最も複雑 | 高度な分析 |

---

## アンチパターン

### アンチパターン1: OLTPスキーマで分析クエリを実行

```sql
-- NG: 正規化されたOLTPスキーマで複雑な分析
SELECT
    EXTRACT(YEAR FROM o.order_date) AS year,
    c.name AS category,
    r.name AS region,
    SUM(oi.quantity * oi.unit_price) AS revenue
FROM orders o
    JOIN order_items oi ON o.id = oi.order_id
    JOIN products p ON oi.product_id = p.id
    JOIN categories c ON p.category_id = c.id
    JOIN customers cu ON o.customer_id = cu.id
    JOIN addresses a ON cu.address_id = a.id
    JOIN regions r ON a.region_id = r.id
WHERE o.status = 'delivered'
GROUP BY 1, 2, 3;
-- → 6テーブルJOIN、大量データで非常に遅い

-- OK: 分析用のスタースキーマを別途構築
SELECT year, category, region, SUM(revenue)
FROM mv_monthly_summary
GROUP BY year, category, region;
```

### アンチパターン2: ファクトテーブルに文字列を直接格納

```sql
-- NG: ファクトに文字列ディメンションを直接格納
CREATE TABLE fact_sales_bad (
    product_name   VARCHAR(200),  -- ← ディメンション属性がファクトに
    category_name  VARCHAR(100),
    customer_name  VARCHAR(200),
    amount         DECIMAL(10,2)
);
-- → 冗長、更新困難、ストレージ浪費

-- OK: サロゲートキーでディメンションを参照
CREATE TABLE fact_sales_good (
    product_key   INTEGER REFERENCES dim_product(product_key),
    customer_key  INTEGER REFERENCES dim_customer(customer_key),
    amount        DECIMAL(10,2)
);
```

---

## FAQ

### Q1: スタースキーマとスノーフレークのどちらを選ぶべきか？

BIツールとの連携やアドホックな分析にはスタースキーマが推奨される。JOINが少なく直感的でクエリ性能も高い。スノーフレークはストレージ効率を重視する場合や、ディメンションの更新頻度が高い場合に選択する。Kimball方式（スター）が業界標準。

### Q2: ファクトテーブルにはどの粒度を選ぶべきか？

最も細かい粒度（トランザクションレベル）で設計し、集約は後から行う。粒度を粗くすると後から詳細分析ができなくなる。ただし、ストレージとパフォーマンスの制約がある場合は日次/月次サマリーのファクトテーブルも併用する。

### Q3: 日付ディメンションテーブルは本当に必要か？

必須。DATE型のまま使うと年度、四半期、祝日、会計年度などの判定が毎クエリで必要になる。事前計算した日付ディメンションは分析を大幅に効率化する。通常20年分でも約7300行と小さい。

---

## まとめ

| 項目 | 要点 |
|------|------|
| OLTP vs OLAP | トランザクション→正規化、分析→次元モデル |
| スタースキーマ | ファクト（中心）+ ディメンション（外周） |
| スノーフレーク | ディメンションを正規化。JOINが増加 |
| ファクトテーブル | 計測値（メジャー）を格納。最も細かい粒度で |
| ディメンション | 分析の切り口。SCD で変更履歴を管理 |
| ETL | ステージング→変換→ロードの3段階 |

---

## 次に読むべきガイド

- [00-normalization.md](./00-normalization.md) — 正規化の理論（OLTP向け）
- [02-performance-tuning.md](../03-practical/02-performance-tuning.md) — 分析クエリのチューニング
- [00-postgresql-features.md](../03-practical/00-postgresql-features.md) — JSONB等の活用

---

## 参考文献

1. Kimball, R. & Ross, M. (2013). *The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling*. 3rd Edition. Wiley.
2. Inmon, W.H. (2005). *Building the Data Warehouse*. 4th Edition. Wiley.
3. Linstedt, D. & Olschimke, M. (2015). *Building a Scalable Data Warehouse with Data Vault 2.0*. Morgan Kaufmann.
