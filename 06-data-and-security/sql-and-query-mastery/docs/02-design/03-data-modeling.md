# データモデリング — スター / スノーフレーク・次元モデル

> データモデリングはビジネス要件をデータ構造に変換する技法であり、OLTP向けの正規化モデルとOLAP向けの次元モデル（スター/スノーフレーク）を使い分けることで、トランザクション処理と分析の両方に最適な設計を実現する。

## 前提知識

- SQL の基本（SELECT, JOIN, GROUP BY, 集約関数）
- 正規化の基本概念（[00-normalization.md](./00-normalization.md)）
- スキーマ設計の基礎（[01-schema-design.md](./01-schema-design.md)）

## この章で学ぶこと

1. OLTPモデルとOLAPモデルの本質的な違いと内部アーキテクチャ
2. スタースキーマとスノーフレークスキーマの構造と使い分け
3. ファクトテーブルとディメンションテーブルの設計パターン
4. SCD（Slowly Changing Dimensions）の全タイプと実装
5. Kimball方式 vs Inmon方式のデータウェアハウス設計
6. Data Vault 2.0 モデリング
7. ETL/ELTパイプラインの設計と実装
8. マテリアライズドビューの高度な活用

---

## 1. OLTP vs OLAP — 内部アーキテクチャの違い

### 1.1 概念的な違い

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

### 1.2 ストレージエンジンの違い

OLTPとOLAPの性能差は、ストレージエンジンのアーキテクチャに起因する。

```
┌───────────────────────────────────────────────────────────────┐
│               Row-Oriented vs Column-Oriented                 │
│                                                               │
│  Row Store (OLTP向け: PostgreSQL, MySQL)                      │
│  ┌─────────────────────────────────────────────┐             │
│  │  行1: [id=1, name="田中", age=30, city="東京"] │          │
│  │  行2: [id=2, name="鈴木", age=25, city="大阪"] │          │
│  │  行3: [id=3, name="佐藤", age=35, city="福岡"] │          │
│  │                                               │           │
│  │  → 1行の全カラムが連続して格納                 │           │
│  │  → INSERT/UPDATE/DELETE が高速                  │           │
│  │  → SELECT * WHERE id = 1 が高速               │           │
│  └─────────────────────────────────────────────┘             │
│                                                               │
│  Column Store (OLAP向け: ClickHouse, Redshift, BigQuery)      │
│  ┌─────────────────────────────────────────────┐             │
│  │  id列:   [1, 2, 3, ...]                      │            │
│  │  name列: ["田中", "鈴木", "佐藤", ...]        │            │
│  │  age列:  [30, 25, 35, ...]                   │            │
│  │  city列: ["東京", "大阪", "福岡", ...]        │            │
│  │                                               │           │
│  │  → 同一カラムが連続して格納                    │           │
│  │  → SUM(age), AVG(age) 等の集約が高速          │           │
│  │  → 圧縮効率が高い（同型データが連続）          │           │
│  └─────────────────────────────────────────────┘             │
└───────────────────────────────────────────────────────────────┘
```

### 1.3 詳細比較表

| 特性 | OLTP | OLAP |
|------|------|------|
| **主な操作** | INSERT/UPDATE/DELETE | SELECT（集約） |
| **対象行数** | 数行〜数十行 | 数万〜数億行 |
| **同時ユーザー** | 数千〜数万 | 数人〜数十人 |
| **レスポンス要件** | ミリ秒 | 秒〜分 |
| **正規化レベル** | 3NF/BCNF | 非正規化（スター） |
| **インデックス** | B-Tree（ポイントクエリ） | ビットマップ、Zone Map |
| **ストレージ** | Row Store | Column Store |
| **圧縮** | 低（行単位では非効率） | 高（同型データ連続） |
| **結合パターン** | 多テーブルJOIN | スター型JOIN |
| **トランザクション** | ACID 必須 | 結果整合性で十分 |
| **代表製品** | PostgreSQL, MySQL | BigQuery, Redshift, Snowflake |

---

## 2. スタースキーマ

### 2.1 設計原理 — Kimball 方式

Ralph Kimball が提唱した次元モデリングは、以下の4ステップで設計する。

```
┌───────── Kimball 次元モデリング 4ステップ ──────────┐
│                                                      │
│  Step 1: ビジネスプロセスの選択                       │
│  │  「何の分析をしたいか？」                         │
│  │  例: 売上分析、在庫分析、顧客行動分析             │
│  ▼                                                   │
│  Step 2: 粒度（Grain）の決定                         │
│  │  「ファクト1行は何を表すか？」                    │
│  │  例: 1商品・1トランザクション・1日                │
│  ▼                                                   │
│  Step 3: ディメンションの特定                        │
│  │  「どの切り口で分析するか？」                     │
│  │  例: 日付、商品、顧客、店舗                       │
│  ▼                                                   │
│  Step 4: ファクト（メジャー）の特定                   │
│     「何を計測するか？」                             │
│     例: 数量、金額、利益                             │
└──────────────────────────────────────────────────────┘
```

### 2.2 ファクトテーブルの3類型

ファクトテーブルは、記録する内容に応じて3つの類型に分類される。

```
┌──────────────────────────────────────────────────────────┐
│               ファクトテーブル 3類型                       │
│                                                          │
│  1. トランザクションファクト                              │
│     ┌────────────────────────────────────────┐          │
│     │ 1行 = 1イベント（購入、クリック等）     │          │
│     │ 粒度: 最も細かい                       │          │
│     │ 例: fact_sales（1購入 = 1行）          │          │
│     │ 特徴: 行数が最も多い、追記のみ          │          │
│     └────────────────────────────────────────┘          │
│                                                          │
│  2. 定期スナップショットファクト                          │
│     ┌────────────────────────────────────────┐          │
│     │ 1行 = 一定期間の状態                    │          │
│     │ 粒度: 日次/週次/月次                    │          │
│     │ 例: fact_daily_balance（日次残高）       │          │
│     │ 特徴: 期間ごとに固定行数                │          │
│     └────────────────────────────────────────┘          │
│                                                          │
│  3. 累積スナップショットファクト                          │
│     ┌────────────────────────────────────────┐          │
│     │ 1行 = ライフサイクル全体                │          │
│     │ 粒度: プロセス単位                      │          │
│     │ 例: fact_order_lifecycle（注文→出荷→配達）│         │
│     │ 特徴: 行が更新される（マイルストーン追加）│         │
│     └────────────────────────────────────────┘          │
└──────────────────────────────────────────────────────────┘
```

### コード例1: スタースキーマの実装（3類型すべて）

```sql
-- ==============================================
-- トランザクションファクト
-- ==============================================
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

-- パーティショニング（大量データ対応）
CREATE TABLE fact_sales_partitioned (
    sale_id         BIGSERIAL,
    date_key        INTEGER NOT NULL,
    product_key     INTEGER NOT NULL,
    customer_key    INTEGER NOT NULL,
    store_key       INTEGER NOT NULL,
    quantity        INTEGER NOT NULL,
    unit_price      DECIMAL(10, 2) NOT NULL,
    discount_amount DECIMAL(10, 2) DEFAULT 0,
    total_amount    DECIMAL(12, 2) NOT NULL,
    cost_amount     DECIMAL(10, 2) NOT NULL,
    profit_amount   DECIMAL(10, 2) GENERATED ALWAYS AS
                    (total_amount - cost_amount) STORED,
    PRIMARY KEY (sale_id, date_key)
) PARTITION BY RANGE (date_key);

-- 月別パーティション
CREATE TABLE fact_sales_202401 PARTITION OF fact_sales_partitioned
    FOR VALUES FROM (20240101) TO (20240201);
CREATE TABLE fact_sales_202402 PARTITION OF fact_sales_partitioned
    FOR VALUES FROM (20240201) TO (20240301);

-- ==============================================
-- 定期スナップショットファクト
-- ==============================================
CREATE TABLE fact_daily_inventory (
    date_key        INTEGER NOT NULL REFERENCES dim_date(date_key),
    product_key     INTEGER NOT NULL REFERENCES dim_product(product_key),
    store_key       INTEGER NOT NULL REFERENCES dim_store(store_key),
    -- スナップショットメジャー
    quantity_on_hand  INTEGER NOT NULL,
    quantity_on_order INTEGER NOT NULL DEFAULT 0,
    reorder_point     INTEGER,
    days_of_supply    DECIMAL(5, 1),
    PRIMARY KEY (date_key, product_key, store_key)
);

-- ==============================================
-- 累積スナップショットファクト
-- ==============================================
CREATE TABLE fact_order_lifecycle (
    order_key         BIGSERIAL PRIMARY KEY,
    order_id          VARCHAR(20) NOT NULL UNIQUE,
    customer_key      INTEGER NOT NULL REFERENCES dim_customer(customer_key),
    -- マイルストーン日付キー
    order_date_key    INTEGER REFERENCES dim_date(date_key),
    ship_date_key     INTEGER REFERENCES dim_date(date_key),
    delivery_date_key INTEGER REFERENCES dim_date(date_key),
    return_date_key   INTEGER REFERENCES dim_date(date_key),
    -- 期間メジャー
    days_to_ship      INTEGER,
    days_to_deliver   INTEGER,
    -- 金額メジャー
    order_amount      DECIMAL(12, 2) NOT NULL,
    shipping_cost     DECIMAL(10, 2),
    -- ステータス
    current_status    VARCHAR(20) NOT NULL DEFAULT 'ordered'
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
    fiscal_quarter SMALLINT,
    -- 分析用の追加属性
    year_month    VARCHAR(7) NOT NULL,    -- '2024-01'
    year_quarter  VARCHAR(7) NOT NULL,    -- '2024-Q1'
    day_of_year   SMALLINT NOT NULL,
    week_of_year  SMALLINT NOT NULL,
    is_month_end  BOOLEAN NOT NULL,
    is_quarter_end BOOLEAN NOT NULL
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

### 2.3 スタースキーマの構造

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

### 2.4 日付ディメンションの自動生成

```sql
-- 日付ディメンションを20年分自動生成
INSERT INTO dim_date
SELECT
    TO_CHAR(d, 'YYYYMMDD')::INTEGER AS date_key,
    d AS full_date,
    EXTRACT(YEAR FROM d)::SMALLINT AS year,
    EXTRACT(QUARTER FROM d)::SMALLINT AS quarter,
    EXTRACT(MONTH FROM d)::SMALLINT AS month,
    TO_CHAR(d, 'Month') AS month_name,
    EXTRACT(WEEK FROM d)::SMALLINT AS week,
    EXTRACT(DOW FROM d)::SMALLINT AS day_of_week,
    TO_CHAR(d, 'Day') AS day_name,
    EXTRACT(DOW FROM d) IN (0, 6) AS is_weekend,
    FALSE AS is_holiday,  -- 後で祝日マスタから更新
    -- 会計年度（4月始まり）
    CASE WHEN EXTRACT(MONTH FROM d) >= 4
         THEN EXTRACT(YEAR FROM d)::SMALLINT
         ELSE (EXTRACT(YEAR FROM d) - 1)::SMALLINT
    END AS fiscal_year,
    CASE WHEN EXTRACT(MONTH FROM d) >= 4
         THEN ((EXTRACT(MONTH FROM d) - 4) / 3 + 1)::SMALLINT
         ELSE ((EXTRACT(MONTH FROM d) + 8) / 3 + 1)::SMALLINT
    END AS fiscal_quarter,
    TO_CHAR(d, 'YYYY-MM') AS year_month,
    TO_CHAR(d, 'YYYY') || '-Q' || EXTRACT(QUARTER FROM d) AS year_quarter,
    EXTRACT(DOY FROM d)::SMALLINT AS day_of_year,
    EXTRACT(WEEK FROM d)::SMALLINT AS week_of_year,
    d = (DATE_TRUNC('month', d) + INTERVAL '1 month - 1 day')::DATE AS is_month_end,
    d = (DATE_TRUNC('quarter', d) + INTERVAL '3 months - 1 day')::DATE AS is_quarter_end
FROM generate_series('2015-01-01'::DATE, '2034-12-31'::DATE, '1 day') AS d;

-- 日本の祝日を反映（例）
UPDATE dim_date SET is_holiday = TRUE
WHERE full_date IN ('2024-01-01', '2024-01-08', '2024-02-11', '2024-02-12',
                    '2024-02-23', '2024-03-20', '2024-04-29', '2024-05-03',
                    '2024-05-04', '2024-05-05', '2024-05-06', '2024-07-15',
                    '2024-08-11', '2024-08-12', '2024-09-16', '2024-09-22',
                    '2024-09-23', '2024-10-14', '2024-11-03', '2024-11-04',
                    '2024-11-23');
```

---

## 3. スノーフレークスキーマ

### 3.1 構造と設計原理

スノーフレークスキーマは、ディメンションテーブルをさらに正規化した形式である。

```
┌──────── スノーフレークスキーマ ────────┐
│                                        │
│  dim_category                          │
│  ┌──────────┐                          │
│  │ カテゴリ  │                          │
│  └────┬─────┘                          │
│       │                                │
│  dim_subcategory                       │
│  ┌──────────┐    dim_date              │
│  │ サブカテゴリ│  ┌──────┐              │
│  └────┬─────┘  │ 日付  │              │
│       │        └──┬───┘              │
│  dim_product      │                    │
│  ┌──────────┐     │   dim_customer     │
│  │ 商品     │─────┼───│ 顧客     │    │
│  └──────────┘     │   └────┬─────┘    │
│              fact_sales    │           │
│              ┌────────┐   dim_region   │
│              │ファクト │   ┌──────┐    │
│              └────────┘   │ 地域  │    │
│                            └──────┘    │
│                                        │
│  ★ ディメンションが多段階に正規化      │
│  → 雪の結晶（スノーフレーク）に見える  │
└────────────────────────────────────────┘
```

### コード例2: スノーフレークスキーマ

```sql
-- スノーフレーク: ディメンションをさらに正規化
CREATE TABLE dim_category (
    category_key  SERIAL PRIMARY KEY,
    category_name VARCHAR(100) NOT NULL,
    department    VARCHAR(100)
);

CREATE TABLE dim_subcategory (
    subcategory_key  SERIAL PRIMARY KEY,
    subcategory_name VARCHAR(100) NOT NULL,
    category_key     INTEGER REFERENCES dim_category(category_key)
);

CREATE TABLE dim_brand (
    brand_key  SERIAL PRIMARY KEY,
    brand_name VARCHAR(100) NOT NULL,
    origin_country VARCHAR(100)
);

CREATE TABLE dim_product_snowflake (
    product_key     SERIAL PRIMARY KEY,
    product_name    VARCHAR(200) NOT NULL,
    subcategory_key INTEGER REFERENCES dim_subcategory(subcategory_key),
    brand_key       INTEGER REFERENCES dim_brand(brand_key),
    unit_cost       DECIMAL(10, 2)
);

-- 地域の正規化
CREATE TABLE dim_country (
    country_key  SERIAL PRIMARY KEY,
    country_name VARCHAR(100) NOT NULL,
    continent    VARCHAR(50)
);

CREATE TABLE dim_region (
    region_key   SERIAL PRIMARY KEY,
    region_name  VARCHAR(50) NOT NULL,
    country_key  INTEGER REFERENCES dim_country(country_key)
);

CREATE TABLE dim_city (
    city_key    SERIAL PRIMARY KEY,
    city_name   VARCHAR(100) NOT NULL,
    region_key  INTEGER REFERENCES dim_region(region_key),
    population  INTEGER
);

-- スノーフレークでの分析クエリ（JOINが増える）
SELECT
    c.category_name,
    sc.subcategory_name,
    b.brand_name,
    SUM(f.total_amount) AS revenue,
    SUM(f.quantity)     AS total_quantity
FROM fact_sales f
    JOIN dim_product_snowflake p ON f.product_key = p.product_key
    JOIN dim_subcategory sc ON p.subcategory_key = sc.subcategory_key
    JOIN dim_category c ON sc.category_key = c.category_key
    JOIN dim_brand b ON p.brand_key = b.brand_key
GROUP BY c.category_name, sc.subcategory_name, b.brand_name
ORDER BY revenue DESC;
```

### 3.2 スターとスノーフレークの選択フロー

```
┌───────── スキーマ選択フロー ─────────┐
│                                      │
│  Q: ディメンションの更新頻度は？     │
│  │                                   │
│  ├─ 高い（日次以上）                 │
│  │  → スノーフレーク候補             │
│  │  （正規化で更新箇所を局所化）     │
│  │                                   │
│  └─ 低い（月次以下）                 │
│     │                               │
│     Q: BIツールのユーザーは？        │
│     │                               │
│     ├─ 非技術者が多い               │
│     │  → スター推奨                 │
│     │  （JOINが少なくシンプル）      │
│     │                               │
│     └─ エンジニアが多い             │
│        │                            │
│        Q: ストレージ制約は？         │
│        │                            │
│        ├─ 厳しい → スノーフレーク   │
│        └─ 余裕   → スター          │
└──────────────────────────────────────┘
```

---

## 4. SCD（Slowly Changing Dimensions）— 全タイプ詳解

### 4.1 SCD全タイプの概要

```
┌─────────────── SCD タイプ一覧 ────────────────┐
│                                                │
│  Type 0: 固定（変更しない）                     │
│  Type 1: 上書き（履歴なし）                     │
│  Type 2: 新行追加（完全な履歴）                 │
│  Type 3: 前回値カラム（直近1回の変更）          │
│  Type 4: 履歴テーブル分離                       │
│  Type 6: ハイブリッド（1+2+3の組合せ）          │
│                                                │
│  ※ Type 5, Type 7 もあるが実務ではまれ         │
└────────────────────────────────────────────────┘
```

### コード例3: SCD Type 1 / 2 / 3

```sql
-- ==============================================
-- SCD Type 0: 固定 — 初回ロード後に変更しない
-- ==============================================
-- 生年月日、初回登録日など不変の属性に使用
-- 特別な処理は不要。UPDATE を行わないだけ。

-- ==============================================
-- SCD Type 1: 上書き（履歴なし）
-- ==============================================
UPDATE dim_customer
SET city = '大阪', region = '関西'
WHERE customer_id = 'C001' AND is_current = TRUE;

-- メリット: シンプル、ストレージ効率が良い
-- デメリット: 変更前の状態を一切追跡できない

-- ==============================================
-- SCD Type 2: 新行追加（完全な履歴）
-- ==============================================
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

-- SCD Type 2 の自動化（トリガー/プロシージャ）
CREATE OR REPLACE FUNCTION scd2_update_customer()
RETURNS TRIGGER AS $$
BEGIN
    -- 旧行を無効化
    UPDATE dim_customer
    SET effective_to = CURRENT_DATE - 1, is_current = FALSE
    WHERE customer_id = NEW.customer_id AND is_current = TRUE;

    -- 新行のメタデータを設定
    NEW.effective_from := CURRENT_DATE;
    NEW.effective_to := '9999-12-31';
    NEW.is_current := TRUE;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ==============================================
-- SCD Type 3: 現在値と前回値を列で保持
-- ==============================================
ALTER TABLE dim_customer ADD COLUMN previous_city VARCHAR(100);
ALTER TABLE dim_customer ADD COLUMN city_changed_at DATE;

UPDATE dim_customer
SET previous_city = city,
    city = '大阪',
    city_changed_at = CURRENT_DATE
WHERE customer_id = 'C001';

-- 変更前後の比較クエリ
SELECT customer_id, customer_name,
       previous_city AS "旧住所",
       city AS "新住所",
       city_changed_at AS "変更日"
FROM dim_customer
WHERE previous_city IS NOT NULL;

-- ==============================================
-- SCD Type 4: 履歴テーブル分離
-- ==============================================
-- 現在テーブル（常に最新）
CREATE TABLE dim_customer_current (
    customer_key    SERIAL PRIMARY KEY,
    customer_id     VARCHAR(20) NOT NULL UNIQUE,
    customer_name   VARCHAR(200) NOT NULL,
    segment         VARCHAR(50),
    city            VARCHAR(100),
    region          VARCHAR(50)
);

-- 履歴テーブル（変更ログ）
CREATE TABLE dim_customer_history (
    history_id      BIGSERIAL PRIMARY KEY,
    customer_id     VARCHAR(20) NOT NULL,
    customer_name   VARCHAR(200),
    segment         VARCHAR(50),
    city            VARCHAR(100),
    region          VARCHAR(50),
    effective_from  TIMESTAMPTZ NOT NULL,
    effective_to    TIMESTAMPTZ,
    change_type     VARCHAR(10) NOT NULL  -- INSERT/UPDATE/DELETE
);

-- 変更時のトリガー
CREATE OR REPLACE FUNCTION scd4_track_changes()
RETURNS TRIGGER AS $$
BEGIN
    -- 旧行の終了時刻を設定
    UPDATE dim_customer_history
    SET effective_to = NOW()
    WHERE customer_id = OLD.customer_id AND effective_to IS NULL;

    -- 新行を履歴に追加
    INSERT INTO dim_customer_history
        (customer_id, customer_name, segment, city, region,
         effective_from, change_type)
    VALUES
        (NEW.customer_id, NEW.customer_name, NEW.segment, NEW.city, NEW.region,
         NOW(), TG_OP);

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_customer_scd4
    AFTER INSERT OR UPDATE ON dim_customer_current
    FOR EACH ROW EXECUTE FUNCTION scd4_track_changes();

-- ==============================================
-- SCD Type 6: ハイブリッド（1+2+3）
-- ==============================================
CREATE TABLE dim_customer_type6 (
    customer_key    SERIAL PRIMARY KEY,
    customer_id     VARCHAR(20) NOT NULL,
    customer_name   VARCHAR(200) NOT NULL,
    -- Type 2 属性
    historical_city VARCHAR(100) NOT NULL,  -- 行作成時の値
    effective_from  DATE NOT NULL,
    effective_to    DATE DEFAULT '9999-12-31',
    is_current      BOOLEAN DEFAULT TRUE,
    -- Type 1 属性（全行に最新値をバックフィル）
    current_city    VARCHAR(100) NOT NULL,   -- 常に最新値
    -- Type 3 属性
    previous_city   VARCHAR(100)
);

-- Type 6 更新手順
-- 1. 旧行を無効化
UPDATE dim_customer_type6
SET effective_to = CURRENT_DATE - 1, is_current = FALSE,
    current_city = '大阪',  -- Type 1: 全行の current を更新
    previous_city = historical_city  -- Type 3: 前回値保持
WHERE customer_id = 'C001' AND is_current = TRUE;

-- 2. 新行を追加 (Type 2)
INSERT INTO dim_customer_type6
    (customer_id, customer_name, historical_city, current_city,
     previous_city, effective_from)
VALUES ('C001', '田中太郎', '大阪', '大阪', '東京', CURRENT_DATE);

-- 3. 過去の全行の current_city も更新 (Type 1)
UPDATE dim_customer_type6
SET current_city = '大阪'
WHERE customer_id = 'C001';
```

### SCD (Slowly Changing Dimension) 比較表

| Type | 方式 | 履歴 | 実装複雑度 | ストレージ | 用途 |
|------|------|------|-----------|-----------|------|
| Type 0 | 変更しない | なし | 最も簡単 | 最小 | 不変マスタ（生年月日等） |
| Type 1 | 上書き | なし | 簡単 | 最小 | 現在値のみ必要（typo修正等） |
| Type 2 | 新行追加 | 完全 | 複雑 | 大 | 時系列分析 |
| Type 3 | 前回値カラム | 直近1回 | 中程度 | 小 | 変更前後比較 |
| Type 4 | 履歴テーブル分離 | 完全 | やや複雑 | 中 | 大量履歴・監査要件 |
| Type 6 | 1+2+3 の組合せ | 完全+現在値 | 最も複雑 | 最大 | 高度な分析 |

---

## 5. Kimball 方式 vs Inmon 方式

### 5.1 アーキテクチャの比較

```
┌──────────────── Kimball 方式 ─────────────────┐
│                                                │
│  OLTP ──┐                                      │
│  OLTP ──┼─ ETL → ┌──────────────────────┐     │
│  OLTP ──┘        │  Data Warehouse      │     │
│                   │  (次元モデル)         │     │
│                   │  ┌─────────────────┐ │     │
│                   │  │ スタースキーマ   │ │     │
│                   │  │ ファクト+ディメン │ │     │
│                   │  └─────────────────┘ │     │
│                   └──────────┬───────────┘     │
│                              │                  │
│                          BIツール               │
│  ★ ボトムアップ: 業務部門のニーズから構築      │
│  ★ 短期で成果: 数ヶ月で1つのデータマートを構築  │
│  ★ 非技術者にも分かりやすい                    │
└────────────────────────────────────────────────┘

┌──────────────── Inmon 方式 ────────────────────┐
│                                                │
│  OLTP ──┐                                      │
│  OLTP ──┼─ ETL → ┌──────────────────────┐     │
│  OLTP ──┘        │  EDW (Enterprise DW) │     │
│                   │  (3NF 正規化)        │     │
│                   └──────────┬───────────┘     │
│                              │                  │
│              ┌───────────────┼───────────────┐  │
│              ▼               ▼               ▼  │
│        ┌──────────┐  ┌──────────┐  ┌──────────┐│
│        │Data Mart │  │Data Mart │  │Data Mart ││
│        │(売上)    │  │(在庫)    │  │(顧客)    ││
│        │スタースキーマ│  │スタースキーマ│  │スタースキーマ││
│        └──────────┘  └──────────┘  └──────────┘│
│                                                │
│  ★ トップダウン: 全社の統合データモデルを先に設計│
│  ★ 長期投資: 構築に時間がかかるが一貫性が高い   │
│  ★ EDWが「単一の真実の源」                     │
└────────────────────────────────────────────────┘
```

### 5.2 比較表

| 特性 | Kimball 方式 | Inmon 方式 |
|------|-------------|-----------|
| **アプローチ** | ボトムアップ | トップダウン |
| **中心構造** | 次元モデル（スター） | 3NF 正規化 EDW |
| **構築期間** | 短い（数ヶ月/マート） | 長い（年単位） |
| **初期コスト** | 低い | 高い |
| **スケーラビリティ** | データマート追加で拡張 | EDW を拡張 |
| **データ統合** | Conformed Dimensions で統合 | EDW で一元管理 |
| **冗長性** | 高い（非正規化） | 低い（正規化） |
| **ユーザー** | 業務部門中心 | IT部門中心 |
| **適用規模** | 中小〜中規模 | 大企業 |
| **業界標準** | 現在の主流 | 金融・通信等の大企業 |

---

## 6. Data Vault 2.0

### 6.1 概要

Data Vault は、Hub（ビジネスキー）、Link（リレーション）、Satellite（属性・履歴）の3要素で構成される。

```
┌──────────── Data Vault 2.0 構造 ──────────────┐
│                                                │
│  Hub (ビジネスキー)                             │
│  ┌────────────────────────┐                    │
│  │ hub_customer           │                    │
│  │ - hub_customer_hk (PK) │ ← ハッシュキー     │
│  │ - customer_id (BK)     │ ← ビジネスキー     │
│  │ - load_date            │                    │
│  │ - record_source        │                    │
│  └───────────┬────────────┘                    │
│              │                                  │
│  Satellite (属性・履歴)                         │
│  ┌────────────────────────┐                    │
│  │ sat_customer           │                    │
│  │ - hub_customer_hk (FK) │                    │
│  │ - load_date (PK)       │                    │
│  │ - name, city, region   │ ← 属性群           │
│  │ - hash_diff            │ ← 変更検知用       │
│  │ - record_source        │                    │
│  └────────────────────────┘                    │
│                                                │
│  Link (リレーション)                            │
│  ┌────────────────────────┐                    │
│  │ lnk_customer_order     │                    │
│  │ - link_hk (PK)         │                    │
│  │ - hub_customer_hk (FK) │                    │
│  │ - hub_order_hk (FK)    │                    │
│  │ - load_date            │                    │
│  │ - record_source        │                    │
│  └────────────────────────┘                    │
└────────────────────────────────────────────────┘
```

### コード例4: Data Vault 2.0 実装

```sql
-- Hub: ビジネスキーの一意管理
CREATE TABLE hub_customer (
    hub_customer_hk  CHAR(32) PRIMARY KEY,  -- MD5ハッシュ
    customer_id      VARCHAR(20) NOT NULL UNIQUE,
    load_date        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    record_source    VARCHAR(100) NOT NULL
);

CREATE TABLE hub_product (
    hub_product_hk   CHAR(32) PRIMARY KEY,
    product_id       VARCHAR(20) NOT NULL UNIQUE,
    load_date        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    record_source    VARCHAR(100) NOT NULL
);

-- Satellite: 属性と履歴
CREATE TABLE sat_customer (
    hub_customer_hk  CHAR(32) NOT NULL REFERENCES hub_customer(hub_customer_hk),
    load_date        TIMESTAMPTZ NOT NULL,
    load_end_date    TIMESTAMPTZ DEFAULT '9999-12-31',
    customer_name    VARCHAR(200),
    segment          VARCHAR(50),
    city             VARCHAR(100),
    region           VARCHAR(50),
    hash_diff        CHAR(32) NOT NULL,  -- 属性のハッシュ（変更検知用）
    record_source    VARCHAR(100) NOT NULL,
    PRIMARY KEY (hub_customer_hk, load_date)
);

-- Link: エンティティ間のリレーション
CREATE TABLE lnk_sale (
    lnk_sale_hk      CHAR(32) PRIMARY KEY,
    hub_customer_hk   CHAR(32) REFERENCES hub_customer(hub_customer_hk),
    hub_product_hk    CHAR(32) REFERENCES hub_product(hub_product_hk),
    load_date         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    record_source     VARCHAR(100) NOT NULL
);

-- Link Satellite: リンクの属性
CREATE TABLE sat_sale (
    lnk_sale_hk   CHAR(32) NOT NULL REFERENCES lnk_sale(lnk_sale_hk),
    load_date      TIMESTAMPTZ NOT NULL,
    quantity       INTEGER,
    unit_price     DECIMAL(10, 2),
    total_amount   DECIMAL(12, 2),
    hash_diff      CHAR(32) NOT NULL,
    record_source  VARCHAR(100) NOT NULL,
    PRIMARY KEY (lnk_sale_hk, load_date)
);

-- ハッシュキー生成の例
INSERT INTO hub_customer (hub_customer_hk, customer_id, record_source)
VALUES (
    MD5('C001'),  -- ビジネスキーのハッシュ
    'C001',
    'erp_system'
)
ON CONFLICT (hub_customer_hk) DO NOTHING;  -- 冪等性
```

### 6.2 モデリング手法比較

| 特性 | Star Schema | Data Vault 2.0 | 3NF (Inmon EDW) |
|------|------------|-----------------|-----------------|
| **目的** | 分析・レポーティング | 統合・監査・履歴 | 統合データ管理 |
| **柔軟性** | 低（スキーマ変更大） | 高（Hub/Sat追加で拡張） | 中 |
| **履歴管理** | SCD（要設計） | 自動（Satellite） | 要設計 |
| **ロード速度** | 中（変換が必要） | 高速（並列ロード） | 中 |
| **クエリ性能** | 最高（非正規化） | 低（多数JOIN） | 中 |
| **監査追跡** | 限定的 | 完全（record_source） | 限定的 |
| **冪等性** | 要設計 | 自然に実現 | 要設計 |
| **学習コスト** | 低 | 高 | 中 |
| **用途** | BIレイヤー | ストレージレイヤー | EDW |

---

## 7. コード例4: 分析クエリの実践

```sql
-- =============================================
-- 月別・カテゴリ別の売上分析
-- =============================================
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

-- =============================================
-- YoY（前年比）比較
-- =============================================
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

-- =============================================
-- RFM分析（Recency, Frequency, Monetary）
-- =============================================
WITH customer_rfm AS (
    SELECT
        dc.customer_id,
        dc.customer_name,
        dc.segment,
        -- Recency: 最終購入からの日数
        CURRENT_DATE - MAX(dd.full_date) AS recency_days,
        -- Frequency: 購入回数
        COUNT(DISTINCT dd.full_date) AS frequency,
        -- Monetary: 総購入金額
        SUM(fs.total_amount) AS monetary
    FROM fact_sales fs
        JOIN dim_customer dc ON fs.customer_key = dc.customer_key
        JOIN dim_date dd ON fs.date_key = dd.date_key
    WHERE dc.is_current = TRUE
    GROUP BY dc.customer_id, dc.customer_name, dc.segment
),
rfm_scored AS (
    SELECT *,
        NTILE(5) OVER (ORDER BY recency_days ASC) AS r_score,   -- 最近ほど高い
        NTILE(5) OVER (ORDER BY frequency DESC)    AS f_score,   -- 頻度高いほど高い
        NTILE(5) OVER (ORDER BY monetary DESC)     AS m_score    -- 金額大きいほど高い
    FROM customer_rfm
)
SELECT
    customer_id,
    customer_name,
    segment,
    recency_days,
    frequency,
    monetary,
    r_score || f_score || m_score AS rfm_segment,
    CASE
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'VIP'
        WHEN r_score >= 4 AND f_score >= 3 THEN 'ロイヤル'
        WHEN r_score >= 3 AND f_score <= 2 THEN '新規有望'
        WHEN r_score <= 2 AND f_score >= 3 THEN '休眠（要再活性化）'
        WHEN r_score <= 2 AND f_score <= 2 THEN '離反リスク'
        ELSE '一般'
    END AS customer_type
FROM rfm_scored
ORDER BY monetary DESC;

-- =============================================
-- コホート分析（月別リテンション率）
-- =============================================
WITH first_purchase AS (
    SELECT
        dc.customer_key,
        DATE_TRUNC('month', MIN(dd.full_date)) AS cohort_month
    FROM fact_sales fs
        JOIN dim_customer dc ON fs.customer_key = dc.customer_key
        JOIN dim_date dd ON fs.date_key = dd.date_key
    WHERE dc.is_current = TRUE
    GROUP BY dc.customer_key
),
monthly_activity AS (
    SELECT
        fp.cohort_month,
        DATE_TRUNC('month', dd.full_date) AS activity_month,
        COUNT(DISTINCT dc.customer_key) AS active_customers
    FROM fact_sales fs
        JOIN dim_customer dc ON fs.customer_key = dc.customer_key
        JOIN dim_date dd ON fs.date_key = dd.date_key
        JOIN first_purchase fp ON dc.customer_key = fp.customer_key
    WHERE dc.is_current = TRUE
    GROUP BY fp.cohort_month, DATE_TRUNC('month', dd.full_date)
),
cohort_sizes AS (
    SELECT cohort_month, COUNT(*) AS cohort_size
    FROM first_purchase
    GROUP BY cohort_month
)
SELECT
    TO_CHAR(ma.cohort_month, 'YYYY-MM') AS cohort,
    cs.cohort_size,
    EXTRACT(MONTH FROM AGE(ma.activity_month, ma.cohort_month))::INT AS month_number,
    ma.active_customers,
    ROUND(100.0 * ma.active_customers / cs.cohort_size, 1) AS retention_pct
FROM monthly_activity ma
    JOIN cohort_sizes cs ON ma.cohort_month = cs.cohort_month
ORDER BY cohort, month_number;
```

---

## 8. コード例5: ETLパイプライン

```sql
-- =============================================
-- ステージングテーブルへの生データロード
-- =============================================
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

-- データ品質チェック用テーブル
CREATE TABLE etl_quality_log (
    log_id          BIGSERIAL PRIMARY KEY,
    batch_id        VARCHAR(50) NOT NULL,
    check_name      VARCHAR(100) NOT NULL,
    check_result    VARCHAR(10) NOT NULL,  -- PASS / FAIL / WARN
    affected_rows   INTEGER,
    details         TEXT,
    checked_at      TIMESTAMPTZ DEFAULT NOW()
);

-- =============================================
-- データ品質チェック
-- =============================================
CREATE OR REPLACE FUNCTION etl_data_quality_check(p_batch_id VARCHAR)
RETURNS TABLE(check_name VARCHAR, result VARCHAR, affected INT) AS $$
BEGIN
    -- Check 1: NULL チェック
    INSERT INTO etl_quality_log (batch_id, check_name, check_result, affected_rows, details)
    SELECT p_batch_id, 'null_check',
           CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END,
           COUNT(*),
           'transaction_id, sale_date, product_code に NULL あり'
    FROM stg_sales_raw
    WHERE loaded_at > (SELECT COALESCE(MAX(checked_at), '1970-01-01') FROM etl_quality_log)
      AND (transaction_id IS NULL OR sale_date IS NULL OR product_code IS NULL);

    -- Check 2: 数値範囲チェック
    INSERT INTO etl_quality_log (batch_id, check_name, check_result, affected_rows, details)
    SELECT p_batch_id, 'range_check',
           CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'WARN' END,
           COUNT(*),
           'quantity <= 0 または unit_price <= 0 の行あり'
    FROM stg_sales_raw
    WHERE quantity <= 0 OR unit_price <= 0;

    -- Check 3: 参照整合性チェック
    INSERT INTO etl_quality_log (batch_id, check_name, check_result, affected_rows, details)
    SELECT p_batch_id, 'referential_check',
           CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END,
           COUNT(*),
           '対応するディメンションが見つからないレコード'
    FROM stg_sales_raw s
    WHERE NOT EXISTS (
        SELECT 1 FROM dim_product dp
        WHERE dp.product_id = s.product_code AND dp.is_current
    );

    -- Check 4: 重複チェック
    INSERT INTO etl_quality_log (batch_id, check_name, check_result, affected_rows, details)
    SELECT p_batch_id, 'duplicate_check',
           CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'WARN' END,
           COUNT(*),
           '重複 transaction_id あり'
    FROM (
        SELECT transaction_id, COUNT(*) AS cnt
        FROM stg_sales_raw
        GROUP BY transaction_id
        HAVING COUNT(*) > 1
    ) dup;

    RETURN QUERY
    SELECT eql.check_name::VARCHAR, eql.check_result::VARCHAR, eql.affected_rows
    FROM etl_quality_log eql
    WHERE eql.batch_id = p_batch_id;
END;
$$ LANGUAGE plpgsql;

-- =============================================
-- ステージングからファクトテーブルへの変換・ロード
-- =============================================
CREATE OR REPLACE FUNCTION etl_load_fact_sales(p_batch_id VARCHAR)
RETURNS INTEGER AS $$
DECLARE
    v_rows_loaded INTEGER;
BEGIN
    -- 品質チェック実行
    PERFORM etl_data_quality_check(p_batch_id);

    -- FAILがあれば中止
    IF EXISTS (
        SELECT 1 FROM etl_quality_log
        WHERE batch_id = p_batch_id AND check_result = 'FAIL'
    ) THEN
        RAISE EXCEPTION 'データ品質チェック失敗: batch_id=%', p_batch_id;
    END IF;

    -- ファクトテーブルへロード
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
    WHERE s.loaded_at > (SELECT COALESCE(MAX(loaded_at), '1970-01-01')
                         FROM fact_sales_load_log);

    GET DIAGNOSTICS v_rows_loaded = ROW_COUNT;

    -- ロードログ記録
    INSERT INTO fact_sales_load_log (batch_id, rows_loaded, loaded_at)
    VALUES (p_batch_id, v_rows_loaded, NOW());

    RETURN v_rows_loaded;
END;
$$ LANGUAGE plpgsql;
```

### 8.1 ETL vs ELT の比較

```
┌──────────── ETL vs ELT ──────────────────┐
│                                           │
│  ETL (Extract-Transform-Load)             │
│  ┌────────┐  ┌──────────┐  ┌────────┐   │
│  │Extract │→ │Transform │→ │ Load   │   │
│  │(抽出)  │  │(変換)    │  │(ロード)│   │
│  └────────┘  └──────────┘  └────────┘   │
│  外部ツールで変換後にDWHにロード          │
│  例: Informatica, Talend, dbt           │
│                                           │
│  ELT (Extract-Load-Transform)             │
│  ┌────────┐  ┌────────┐  ┌──────────┐   │
│  │Extract │→ │ Load   │→ │Transform │   │
│  │(抽出)  │  │(ロード)│  │(変換)    │   │
│  └────────┘  └────────┘  └──────────┘   │
│  生データをDWHにロード後、DWH内で変換      │
│  例: BigQuery + dbt, Snowflake + dbt    │
│                                           │
│  近年はELTが主流（DWHの処理能力が向上）    │
└───────────────────────────────────────────┘
```

| 特性 | ETL | ELT |
|------|-----|-----|
| **変換場所** | 外部サーバー | DWH内 |
| **スケーラビリティ** | 変換サーバーに依存 | DWHのスケールを活用 |
| **生データ保持** | 通常は変換後のみ | 生データも保持可能 |
| **柔軟性** | 変換ロジック変更→再処理が大変 | SQLで変換→再処理が容易 |
| **コスト** | ETLサーバーのコスト | DWHの計算コスト |
| **代表ツール** | Informatica, Talend | dbt, Dataform |
| **適用場面** | レガシー環境 | クラウドDWH |

---

## 9. コード例6: マテリアライズドビューの活用

```sql
-- =============================================
-- 月次サマリーのマテリアライズドビュー
-- =============================================
CREATE MATERIALIZED VIEW mv_monthly_summary AS
SELECT
    dd.year,
    dd.month,
    dd.year_month,
    dp.category,
    dc.region,
    ds.store_name,
    COUNT(*) AS transaction_count,
    SUM(fs.quantity) AS total_quantity,
    SUM(fs.total_amount) AS revenue,
    SUM(fs.profit_amount) AS profit,
    AVG(fs.total_amount) AS avg_transaction_value,
    COUNT(DISTINCT dc.customer_key) AS unique_customers
FROM fact_sales fs
    JOIN dim_date dd ON fs.date_key = dd.date_key
    JOIN dim_product dp ON fs.product_key = dp.product_key
    JOIN dim_customer dc ON fs.customer_key = dc.customer_key
    JOIN dim_store ds ON fs.store_key = ds.store_key
WHERE dp.is_current AND dc.is_current
GROUP BY dd.year, dd.month, dd.year_month, dp.category, dc.region, ds.store_name
WITH DATA;

CREATE UNIQUE INDEX idx_mv_monthly ON mv_monthly_summary
    (year, month, category, region, store_name);

-- 差分リフレッシュ（CONCURRENTLY = 読み取りブロックなし）
REFRESH MATERIALIZED VIEW CONCURRENTLY mv_monthly_summary;

-- =============================================
-- マテリアライズドビューの自動リフレッシュ（pg_cron）
-- =============================================
-- pg_cron 拡張を使った定期リフレッシュ
-- CREATE EXTENSION pg_cron;  -- 要インストール
-- SELECT cron.schedule(
--     'refresh_monthly_summary',
--     '0 2 * * *',  -- 毎日2時に実行
--     'REFRESH MATERIALIZED VIEW CONCURRENTLY mv_monthly_summary'
-- );

-- =============================================
-- 階層的マテリアライズドビュー戦略
-- =============================================
-- Level 1: 日次集計（最も細かい集約）
CREATE MATERIALIZED VIEW mv_daily_sales AS
SELECT
    date_key,
    product_key,
    store_key,
    SUM(quantity) AS total_quantity,
    SUM(total_amount) AS revenue,
    SUM(profit_amount) AS profit,
    COUNT(*) AS transactions
FROM fact_sales
GROUP BY date_key, product_key, store_key
WITH DATA;

-- Level 2: 月次集計（日次ビューから構築 — より高速にリフレッシュ可能）
CREATE MATERIALIZED VIEW mv_monthly_product AS
SELECT
    dd.year,
    dd.month,
    dp.product_name,
    dp.category,
    SUM(ds.total_quantity) AS total_quantity,
    SUM(ds.revenue) AS revenue,
    SUM(ds.profit) AS profit,
    SUM(ds.transactions) AS transactions
FROM mv_daily_sales ds
    JOIN dim_date dd ON ds.date_key = dd.date_key
    JOIN dim_product dp ON ds.product_key = dp.product_key
WHERE dp.is_current
GROUP BY dd.year, dd.month, dp.product_name, dp.category
WITH DATA;
```

### 9.1 RDBMS別マテリアライズドビュー比較

| 機能 | PostgreSQL | Oracle | SQL Server | MySQL |
|------|-----------|--------|-----------|-------|
| **マテリアライズドビュー** | あり | あり | Indexed View | なし（手動実装） |
| **CONCURRENTLY リフレッシュ** | あり | 一部（ON COMMIT） | N/A | N/A |
| **自動リフレッシュ** | pg_cron で実現 | ON COMMIT / ON DEMAND | N/A | N/A |
| **差分リフレッシュ** | UNIQUE INDEX 必須 | マテリアライズドビューログ | 自動 | N/A |
| **クエリリライト** | なし | あり（自動適用） | あり（自動適用） | なし |

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
| ディメンション数 | 少なくて済む | テーブル数が増加 |
| 学習コスト | 低い | 中程度 |

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
-- → OLTPのロック競合でトランザクション処理にも悪影響

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
-- → 商品名変更時に全行UPDATEが必要

-- OK: サロゲートキーでディメンションを参照
CREATE TABLE fact_sales_good (
    product_key   INTEGER REFERENCES dim_product(product_key),
    customer_key  INTEGER REFERENCES dim_customer(customer_key),
    amount        DECIMAL(10,2)
);
-- → INTEGER は4バイト、VARCHAR(200) は最大200バイト
-- → ファクトテーブルが1億行の場合、約18GB以上の差
```

### アンチパターン3: 粒度の混在

```sql
-- NG: 1つのファクトテーブルに異なる粒度のデータを混在
CREATE TABLE fact_mixed_bad (
    date_key     INTEGER,
    product_key  INTEGER,
    store_key    INTEGER,
    -- トランザクションレベルの属性
    transaction_id  VARCHAR(50),
    line_item_qty   INTEGER,
    -- 日次集計レベルの属性
    daily_total     DECIMAL(12,2),
    daily_count     INTEGER
);
-- → 粒度が不明確でJOINや集約が正しく行えない

-- OK: 粒度ごとにテーブルを分離
CREATE TABLE fact_sales_transaction (  -- トランザクション粒度
    sale_id BIGSERIAL PRIMARY KEY,
    date_key INTEGER, product_key INTEGER, store_key INTEGER,
    quantity INTEGER, amount DECIMAL(12,2)
);
CREATE TABLE fact_sales_daily (  -- 日次スナップショット粒度
    date_key INTEGER, product_key INTEGER, store_key INTEGER,
    total_quantity INTEGER, total_amount DECIMAL(12,2),
    transaction_count INTEGER,
    PRIMARY KEY (date_key, product_key, store_key)
);
```

---

## エッジケース

### エッジケース1: 遅延データ（Late Arriving Facts/Dimensions）

```sql
-- 問題: 12/15の売上データが12/20に届いた場合
-- ファクトは正しい date_key を使えば問題ない

-- 問題: ディメンションが遅延（顧客登録前に購入データが到着）
-- 対策: "Unknown" ディメンション行を用意
INSERT INTO dim_customer (customer_key, customer_id, customer_name, segment,
                          city, region, country, effective_from, is_current)
VALUES (0, 'UNKNOWN', '不明', '不明', '不明', '不明', '不明',
        '1900-01-01', TRUE);

-- 後で正しいディメンションが到着したらファクトを更新
UPDATE fact_sales
SET customer_key = (
    SELECT customer_key FROM dim_customer
    WHERE customer_id = 'C999' AND is_current
)
WHERE customer_key = 0;  -- Unknown を正しいキーに差し替え
```

### エッジケース2: 多対多ディメンション（Bridge Table）

```sql
-- 問題: 1つの商品に複数のプロモーションが同時適用
-- → ファクトとディメンションが多対多

-- Bridge テーブルで解決
CREATE TABLE bridge_promotion (
    promotion_group_key INTEGER NOT NULL,
    promotion_key       INTEGER NOT NULL REFERENCES dim_promotion(promotion_key),
    weight_factor       DECIMAL(5,4) NOT NULL,  -- 寄与率（合計=1.0）
    PRIMARY KEY (promotion_group_key, promotion_key)
);

-- ファクトは promotion_group_key を参照
ALTER TABLE fact_sales ADD COLUMN promotion_group_key INTEGER;

-- クエリ時
SELECT dp.promotion_name,
       SUM(fs.total_amount * bp.weight_factor) AS attributed_revenue
FROM fact_sales fs
    JOIN bridge_promotion bp ON fs.promotion_group_key = bp.promotion_group_key
    JOIN dim_promotion dp ON bp.promotion_key = dp.promotion_key
GROUP BY dp.promotion_name;
```

### エッジケース3: Junk Dimension（低基数属性の統合）

```sql
-- 問題: is_online, is_gift_wrapped, payment_type 等の
-- 低基数のフラグ/コード属性がファクトに散在

-- Junk Dimension で統合
CREATE TABLE dim_transaction_profile (
    profile_key     SERIAL PRIMARY KEY,
    is_online       BOOLEAN NOT NULL,
    is_gift_wrapped BOOLEAN NOT NULL,
    payment_type    VARCHAR(20) NOT NULL,
    delivery_type   VARCHAR(20) NOT NULL,
    -- 2 * 2 * 4 * 3 = 48 行で全組み合わせをカバー
    UNIQUE (is_online, is_gift_wrapped, payment_type, delivery_type)
);

-- 全組み合わせを事前投入
INSERT INTO dim_transaction_profile
    (is_online, is_gift_wrapped, payment_type, delivery_type)
SELECT
    online, gift, pay, delivery
FROM
    (VALUES (TRUE), (FALSE)) AS o(online),
    (VALUES (TRUE), (FALSE)) AS g(gift),
    (VALUES ('credit'), ('debit'), ('cash'), ('transfer')) AS p(pay),
    (VALUES ('standard'), ('express'), ('pickup')) AS d(delivery);

-- ファクトテーブルは profile_key 1つで参照
ALTER TABLE fact_sales ADD COLUMN profile_key INTEGER
    REFERENCES dim_transaction_profile(profile_key);
```

---

## 演習問題

### 演習1: 基礎 — スタースキーマ設計

以下の要件でスタースキーマを設計せよ。

**要件**: オンライン書店の売上分析
- 分析の切り口: 書籍（ジャンル、著者、出版社）、顧客（年齢層、地域）、日付、配送方法
- 計測値: 販売数量、売上金額、割引額、配送コスト

**解答例**:

```sql
-- ファクトテーブル
CREATE TABLE fact_book_sales (
    sale_id         BIGSERIAL PRIMARY KEY,
    date_key        INTEGER NOT NULL REFERENCES dim_date(date_key),
    book_key        INTEGER NOT NULL REFERENCES dim_book(book_key),
    customer_key    INTEGER NOT NULL REFERENCES dim_reader(reader_key),
    delivery_key    INTEGER NOT NULL REFERENCES dim_delivery(delivery_key),
    quantity        INTEGER NOT NULL,
    sale_amount     DECIMAL(10, 2) NOT NULL,
    discount_amount DECIMAL(10, 2) DEFAULT 0,
    delivery_cost   DECIMAL(8, 2) NOT NULL
);

-- ディメンション: 書籍
CREATE TABLE dim_book (
    book_key     SERIAL PRIMARY KEY,
    isbn         VARCHAR(20) NOT NULL,
    title        VARCHAR(300) NOT NULL,
    genre        VARCHAR(100),
    sub_genre    VARCHAR(100),
    author       VARCHAR(200),
    publisher    VARCHAR(200),
    publish_date DATE,
    list_price   DECIMAL(8, 2),
    effective_from DATE NOT NULL,
    effective_to   DATE DEFAULT '9999-12-31',
    is_current     BOOLEAN DEFAULT TRUE
);

-- ディメンション: 読者
CREATE TABLE dim_reader (
    reader_key   SERIAL PRIMARY KEY,
    reader_id    VARCHAR(20) NOT NULL,
    reader_name  VARCHAR(200),
    age_group    VARCHAR(20),  -- '10代', '20代', ...
    prefecture   VARCHAR(20),
    region       VARCHAR(20),  -- '関東', '関西', ...
    member_since DATE,
    effective_from DATE NOT NULL,
    effective_to   DATE DEFAULT '9999-12-31',
    is_current     BOOLEAN DEFAULT TRUE
);

-- ディメンション: 配送方法
CREATE TABLE dim_delivery (
    delivery_key    SERIAL PRIMARY KEY,
    delivery_method VARCHAR(50) NOT NULL,  -- '通常配送', '翌日配送', 'コンビニ受取'
    carrier         VARCHAR(100),
    is_free         BOOLEAN DEFAULT FALSE
);
```

### 演習2: 応用 — SCD Type 2 の実装

dim_book テーブルで、書籍の価格改定が発生した場合の SCD Type 2 更新を SQL で実装せよ。

**解答例**:

```sql
-- トランザクション内で実行（原子性を保証）
BEGIN;

-- 旧行を無効化
UPDATE dim_book
SET effective_to = CURRENT_DATE - 1,
    is_current = FALSE
WHERE isbn = '978-4-XXX-XXXXX-X'
  AND is_current = TRUE;

-- 新行を追加（価格のみ変更）
INSERT INTO dim_book (isbn, title, genre, sub_genre, author, publisher,
                      publish_date, list_price, effective_from, is_current)
SELECT isbn, title, genre, sub_genre, author, publisher,
       publish_date,
       1980.00,  -- 新価格
       CURRENT_DATE,
       TRUE
FROM dim_book
WHERE isbn = '978-4-XXX-XXXXX-X'
  AND effective_to = CURRENT_DATE - 1
  AND is_current = FALSE
ORDER BY effective_to DESC
LIMIT 1;

COMMIT;

-- 検証: 時点別の価格を確認
SELECT isbn, title, list_price, effective_from, effective_to, is_current
FROM dim_book
WHERE isbn = '978-4-XXX-XXXXX-X'
ORDER BY effective_from;
```

### 演習3: 発展 — ETL パイプラインの品質チェック設計

以下の要件を満たすデータ品質チェックプロシージャを設計せよ。

**要件**:
1. ステージングテーブルの NULL チェック（必須カラム）
2. 数値の範囲チェック（数量 > 0、金額 > 0）
3. 参照整合性チェック（全ディメンションに対応するキーが存在するか）
4. チェック結果をログテーブルに記録
5. FAIL があれば後続のロードを中止

**解答例**: 上記コード例5の `etl_data_quality_check` 関数および `etl_load_fact_sales` 関数を参照。追加で以下のような時間整合性チェックも有効。

```sql
-- 時間整合性チェック: 未来の日付がないか
INSERT INTO etl_quality_log (batch_id, check_name, check_result, affected_rows)
SELECT 'batch_001', 'future_date_check',
       CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END,
       COUNT(*)
FROM stg_sales_raw
WHERE sale_date > CURRENT_DATE;

-- 重複トランザクションチェック: ファクトテーブルとの重複
INSERT INTO etl_quality_log (batch_id, check_name, check_result, affected_rows)
SELECT 'batch_001', 'fact_duplicate_check',
       CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'WARN' END,
       COUNT(*)
FROM stg_sales_raw s
WHERE EXISTS (
    SELECT 1 FROM fact_sales f
    -- ビジネスキーの組み合わせで重複判定
    WHERE f.date_key = TO_CHAR(s.sale_date, 'YYYYMMDD')::INTEGER
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

### Q4: Data Vault と Star Schema の使い分けは？

Data Vault はデータウェアハウスの「ストレージレイヤー」として使い、エンドユーザー向けの「プレゼンテーションレイヤー」にはスタースキーマを使うのが一般的。Data Vault は監査証跡や履歴の完全な保持に優れるが、クエリが複雑になるため直接BIツールに接続することは推奨されない。

### Q5: dbt を使う場合、次元モデルはどう構築する？

dbt ではモデルを `staging`（生データ整形）、`intermediate`（中間変換）、`marts`（最終的な次元モデル）の3層に分けるのが一般的。ファクト/ディメンションは `marts` レイヤーに配置し、SCD Type 2 は dbt の `snapshot` 機能で自動化できる。

---

## トラブルシューティング

| 症状 | 原因 | 対策 |
|------|------|------|
| ファクトテーブルのJOINが遅い | ディメンションキーにインデックスがない | FK列にインデックスを作成 |
| マテリアライズドビューのリフレッシュが遅い | UNIQUE INDEX がない | CONCURRENTLY 用のUNIQUE INDEX を作成 |
| SCD Type 2 で行が重複 | is_current の更新漏れ | トランザクション内で UPDATE → INSERT |
| ETL後にファクトの行数が想定より少ない | INNER JOIN で不一致行が除外 | LEFT JOIN + NULL チェックで原因特定 |
| 日付ディメンションに歯抜けがある | generate_series の範囲不足 | 十分な期間で再生成 |
| 集計クエリの結果が正しくない | ファクトの粒度とJOINの不整合 | 粒度の確認、GROUP BY の見直し |
| カラムストア（Redshift等）での UPDATE が遅い | カラムストアは UPDATE 非効率 | DELETE + INSERT（再挿入）パターンに変更 |

---

## パフォーマンス考慮事項

1. **ファクトテーブルのパーティショニング**: date_key でのRANGEパーティションが最も効果的。月別または四半期別に分割することで、不要なパーティションの読み飛ばし（Partition Pruning）が発生し、大幅な高速化が可能。

2. **ビットマップインデックス**: OLAP環境では、低基数カラム（性別、地域等）にはビットマップインデックスが有効。ただし PostgreSQL にはネイティブのビットマップインデックスはなく、実行時にBitmap Index Scanとして最適化される。Oracle では CREATE BITMAP INDEX が利用可能。

3. **圧縮**: ファクトテーブルは同一パターンのデータが多いため、テーブルレベルの圧縮が効果的。PostgreSQL では TOAST 圧縮が自動適用、Redshift では ENCODE 指定、BigQuery は自動圧縮。

4. **統計情報の更新**: 大量ロード後は必ず ANALYZE を実行。クエリプランナーが正確な統計に基づいてJOIN順序やスキャン方式を選択できるようにする。

---

## セキュリティ考慮事項

1. **行レベルセキュリティ（RLS）**: 部門ごとに閲覧可能なデータを制限する場合、RLS を活用。

```sql
-- 部門別のデータアクセス制御
ALTER TABLE fact_sales ENABLE ROW LEVEL SECURITY;

CREATE POLICY sales_region_policy ON fact_sales
    USING (store_key IN (
        SELECT store_key FROM dim_store
        WHERE region = current_setting('app.user_region')
    ));
```

2. **個人情報の分離**: 顧客ディメンションの個人識別情報（PII）は別テーブルに分離し、分析用ディメンションには匿名化/集約された属性のみを含める。

3. **マスキング**: 開発環境やテスト環境では、本番データのコピーに対してデータマスキングを適用する。

---

## まとめ

| 項目 | 要点 |
|------|------|
| OLTP vs OLAP | トランザクション→正規化、分析→次元モデル |
| スタースキーマ | ファクト（中心）+ ディメンション（外周）、業界標準 |
| スノーフレーク | ディメンションを正規化。JOINが増加、ストレージ効率は向上 |
| ファクトテーブル | 3類型: トランザクション / スナップショット / 累積 |
| ディメンション | 分析の切り口。SCD で変更履歴を管理（Type 0-6） |
| Kimball vs Inmon | ボトムアップ vs トップダウン。現在は Kimball が主流 |
| Data Vault 2.0 | Hub/Link/Satellite。監査・履歴に強い。ストレージレイヤー向け |
| ETL/ELT | クラウド時代はELT（dbt等）が主流 |
| マテリアライズドビュー | 階層的構築で効率的なリフレッシュを実現 |

---

## 次に読むべきガイド

- [00-normalization.md](./00-normalization.md) — 正規化の理論（OLTP向け）
- [02-performance-tuning.md](../03-practical/02-performance-tuning.md) — 分析クエリのチューニング
- [00-postgresql-features.md](../03-practical/00-postgresql-features.md) — JSONB等の活用
- [01-schema-design.md](./01-schema-design.md) — テーブル設計の実践パターン

---

## 参考文献

1. Kimball, R. & Ross, M. (2013). *The Data Warehouse Toolkit: The Definitive Guide to Dimensional Modeling*. 3rd Edition. Wiley.
2. Inmon, W.H. (2005). *Building the Data Warehouse*. 4th Edition. Wiley.
3. Linstedt, D. & Olschimke, M. (2015). *Building a Scalable Data Warehouse with Data Vault 2.0*. Morgan Kaufmann.
4. Agosta, L. (2023). *The Data Warehouse Mentor: Practical Data Warehouse and Business Intelligence Insights*. Technics Publications.
5. dbt Labs. "How we structure our dbt projects" — https://docs.getdbt.com/guides/best-practices/how-we-structure/1-guide-overview
