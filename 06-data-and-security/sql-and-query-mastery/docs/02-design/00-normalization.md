# 正規化 — 1NF〜BCNF・非正規化

> 正規化はデータの冗長性を排除し更新異常を防ぐためのリレーショナルデータベース設計手法であり、非正規化はパフォーマンスとのトレードオフとして意図的に冗長性を導入する手法である。

## この章で学ぶこと

1. 第1正規形（1NF）から第3正規形（3NF）、BCNFまでの段階的な正規化プロセス
2. 正規化によって解決される更新異常の種類
3. 非正規化の判断基準と実践的なパターン

---

## 1. 正規化の目的と更新異常

```
┌─────────── 正規化で解決する3つの更新異常 ─────────┐
│                                                     │
│  挿入異常（Insertion Anomaly）                      │
│  ─────────────────────────────                     │
│  まだ社員がいない部署を登録できない                  │
│  （社員テーブルに部署情報が含まれている場合）        │
│                                                     │
│  更新異常（Update Anomaly）                         │
│  ─────────────────────────────                     │
│  部署名を変更するとき、その部署の全社員の行を        │
│  更新する必要がある（1行でも漏れると不整合）        │
│                                                     │
│  削除異常（Deletion Anomaly）                       │
│  ─────────────────────────────                     │
│  最後の社員を削除すると部署情報も失われる            │
│                                                     │
│  → 正規化によってテーブルを適切に分割すれば          │
│    これらの異常を防止できる                          │
└─────────────────────────────────────────────────────┘
```

---

## 2. 正規化の段階

### コード例1: 非正規形から第1正規形（1NF）

```sql
-- 非正規形: 繰り返し項目がある
-- ┌────┬──────┬─────────────────────┐
-- │ id │ name │ phones              │
-- ├────┼──────┼─────────────────────┤
-- │ 1  │ 田中 │ 090-1111, 03-2222   │ ← 1セルに複数値
-- └────┴──────┴─────────────────────┘

-- 第1正規形（1NF）: 各セルに原子値（Atomic Value）のみ
CREATE TABLE contacts (
    id    INTEGER,
    name  VARCHAR(100),
    phone VARCHAR(20),
    PRIMARY KEY (id, phone)  -- 複合主キー
);

INSERT INTO contacts VALUES (1, '田中', '090-1111-2222');
INSERT INTO contacts VALUES (1, '田中', '03-2222-3333');

-- 1NF の要件:
-- 1. 各列の値が原子的（分割不可能）
-- 2. 繰り返しグループがない
-- 3. 行の順序に意味がない
```

### コード例2: 第2正規形（2NF）— 部分関数従属の排除

```sql
-- 1NFだが2NFでない例:
-- 注文明細テーブル
-- PK = (order_id, product_id)
-- ┌──────────┬────────────┬──────────────┬──────────┬───────┐
-- │ order_id │ product_id │ product_name │ quantity │ price │
-- └──────────┴────────────┴──────────────┴──────────┴───────┘
--   product_name は product_id のみに従属（部分関数従属）

-- 第2正規形（2NF）: 部分関数従属を排除
CREATE TABLE products (
    product_id   INTEGER PRIMARY KEY,
    product_name VARCHAR(100)           -- product_id のみに従属
);

CREATE TABLE order_items (
    order_id   INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(product_id),
    quantity   INTEGER,
    price      DECIMAL(10,2),          -- 注文時の価格
    PRIMARY KEY (order_id, product_id)  -- 全キーに従属
);

-- 2NF の要件:
-- 1. 1NFを満たす
-- 2. 非キー属性が主キーの一部にのみ従属しない
--    （主キーが単一列なら自動的に2NF）
```

### コード例3: 第3正規形（3NF）— 推移的関数従属の排除

```sql
-- 2NFだが3NFでない例:
-- ┌────┬──────┬─────────────┬────────────────┐
-- │ id │ name │ dept_id     │ dept_name      │
-- └────┴──────┴─────────────┴────────────────┘
-- dept_name は dept_id に従属し、dept_id は id に従属
-- → dept_name は id に推移的に従属

-- 第3正規形（3NF）: 推移的関数従属を排除
CREATE TABLE departments (
    dept_id   INTEGER PRIMARY KEY,
    dept_name VARCHAR(100)
);

CREATE TABLE employees (
    id      INTEGER PRIMARY KEY,
    name    VARCHAR(100),
    dept_id INTEGER REFERENCES departments(dept_id)
);

-- 3NF の要件:
-- 1. 2NFを満たす
-- 2. 非キー属性が他の非キー属性に従属しない
--    （非キー→非キーの関数従属がない）
```

### 正規化の段階図解

```
┌─────────────── 正規化の段階 ───────────────────┐
│                                                 │
│  非正規形                                       │
│    │  繰り返し項目の排除                        │
│    ▼                                            │
│  第1正規形 (1NF)                                │
│    │  部分関数従属の排除                        │
│    ▼                                            │
│  第2正規形 (2NF)                                │
│    │  推移的関数従属の排除                      │
│    ▼                                            │
│  第3正規形 (3NF) ← ここまでが一般的な目標      │
│    │  非自明な関数従属の候補キー依存            │
│    ▼                                            │
│  ボイス・コッド正規形 (BCNF)                    │
│    │  多値従属の排除                            │
│    ▼                                            │
│  第4正規形 (4NF)                                │
│    │  結合従属の排除                            │
│    ▼                                            │
│  第5正規形 (5NF)                                │
│                                                 │
│  ※ 実務では3NFまたはBCNFが実用的な上限         │
└─────────────────────────────────────────────────┘
```

### コード例4: BCNF（ボイス・コッド正規形）

```sql
-- 3NFだがBCNFでない例:
-- 学生の講義登録（1講義に複数の教員が担当可能、
--                 各教員は1つの講義のみ担当）
-- PK = (student_id, course_id)
-- 関数従属: teacher_id → course_id
--          （教員がどの講義を担当するかは一意に決まる）

-- 3NFではteacher_idは非キーだがcourse_id（キーの一部）を決定
-- → BCNFに違反

-- BCNF化:
CREATE TABLE teacher_courses (
    teacher_id INTEGER PRIMARY KEY,
    course_id  INTEGER REFERENCES courses(id)
);

CREATE TABLE enrollments (
    student_id INTEGER REFERENCES students(id),
    teacher_id INTEGER REFERENCES teacher_courses(teacher_id),
    PRIMARY KEY (student_id, teacher_id)
);
```

---

## 3. 非正規化

### コード例5: 意図的な非正規化パターン

```sql
-- パターン1: 計算済みカラム（集約結果のキャッシュ）
ALTER TABLE orders ADD COLUMN item_count INTEGER DEFAULT 0;
ALTER TABLE orders ADD COLUMN total_amount DECIMAL(12,2) DEFAULT 0;

-- トリガーで自動更新
CREATE OR REPLACE FUNCTION update_order_totals()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE orders SET
        item_count = (SELECT COUNT(*) FROM order_items WHERE order_id = NEW.order_id),
        total_amount = (SELECT SUM(price * quantity) FROM order_items WHERE order_id = NEW.order_id)
    WHERE id = NEW.order_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- パターン2: マテリアライズドビュー
CREATE MATERIALIZED VIEW monthly_sales_summary AS
SELECT
    DATE_TRUNC('month', order_date) AS month,
    category,
    COUNT(*) AS order_count,
    SUM(total_amount) AS revenue
FROM orders o
    JOIN products p ON o.product_id = p.id
GROUP BY 1, 2;

-- 定期的にリフレッシュ
REFRESH MATERIALIZED VIEW CONCURRENTLY monthly_sales_summary;
```

### コード例6: 正規化 vs 非正規化の実例比較

```sql
-- 正規化されたスキーマ（3NF）
-- 6テーブルをJOINして注文詳細を取得
SELECT
    o.id, o.order_date,
    c.name AS customer, c.email,
    p.name AS product, p.sku,
    cat.name AS category,
    oi.quantity, oi.unit_price,
    a.city, a.postal_code
FROM orders o
    JOIN customers c ON o.customer_id = c.id
    JOIN order_items oi ON o.id = oi.order_id
    JOIN products p ON oi.product_id = p.id
    JOIN categories cat ON p.category_id = cat.id
    JOIN addresses a ON o.shipping_address_id = a.id;

-- 非正規化されたスキーマ（読み取り最適化）
-- 1テーブルで完結
SELECT
    order_id, order_date,
    customer_name, customer_email,
    product_name, product_sku,
    category_name,
    quantity, unit_price,
    shipping_city, shipping_postal_code
FROM order_details_denormalized
WHERE order_id = 42;
```

---

## 正規化レベル比較表

| 正規形 | 排除する問題 | 適用条件 | 実用性 |
|--------|------------|---------|--------|
| 1NF | 繰り返し項目、非原子値 | 各セルが原子値 | 必須 |
| 2NF | 部分関数従属 | 非キーがキー全体に従属 | 必須 |
| 3NF | 推移的関数従属 | 非キー間の従属排除 | 推奨 |
| BCNF | 全ての非自明な関数従属 | 決定項が候補キー | 推奨 |
| 4NF | 多値従属 | 独立した多値関係の分離 | 稀 |
| 5NF | 結合従属 | 無損失結合分解 | 極稀 |

## 正規化 vs 非正規化 比較表

| 観点 | 正規化 | 非正規化 |
|------|--------|---------|
| データ冗長性 | なし | あり |
| 更新異常 | なし | リスクあり |
| 書き込み性能 | 高い | 低い（複数箇所更新） |
| 読み取り性能 | JOIN必要（やや低い） | 高い（1テーブル） |
| ストレージ | 効率的 | 冗長（大きい） |
| スキーマ変更 | 容易 | 困難 |
| データ整合性 | 高い | 自力で維持が必要 |
| 適する用途 | OLTP | OLAP / レポーティング |

---

## アンチパターン

### アンチパターン1: EAV（Entity-Attribute-Value）パターン

```sql
-- NG: 汎用的だが正規化の恩恵を全く受けられない
CREATE TABLE entity_attributes (
    entity_id  INTEGER,
    attr_name  VARCHAR(100),
    attr_value TEXT,
    PRIMARY KEY (entity_id, attr_name)
);

-- 問題点:
-- 1. 型安全性がない（全てTEXT）
-- 2. 制約が使えない（NOT NULL、CHECK等）
-- 3. JOINが複雑化（属性ごとにSelf JOIN）
-- 4. クエリが非効率

-- OK: JSONBでスキーマレスな部分を分離
CREATE TABLE products (
    id    SERIAL PRIMARY KEY,
    name  VARCHAR(100) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    attrs JSONB  -- 可変属性はJSONBに格納
);
```

### アンチパターン2: 過度な正規化

```sql
-- NG: 都道府県や性別まで別テーブルに分離
CREATE TABLE genders (id INT PRIMARY KEY, name VARCHAR(10));
CREATE TABLE prefectures (id INT PRIMARY KEY, name VARCHAR(10));
-- → JOINが増え、クエリが複雑化し、パフォーマンスが低下

-- OK: 変更されない小さなマスタはENUMやCHECK制約で十分
CREATE TABLE users (
    id       SERIAL PRIMARY KEY,
    gender   VARCHAR(10) CHECK (gender IN ('male', 'female', 'other')),
    prefecture VARCHAR(10) NOT NULL
);
```

---

## FAQ

### Q1: 3NFまで正規化すれば十分か？

多くの実務アプリケーションでは3NFで十分。BCNFまで進める場合は、候補キーが複数存在し、非キー属性がキーの一部を決定するような特殊な状況に限られる。過度な正規化はJOINの増加とパフォーマンスの低下を招く。

### Q2: いつ非正規化すべきか？

(1) 読み取り頻度が書き込み頻度を大幅に上回る場合、(2) JOINのコストが許容できないレベルの場合、(3) レポーティング/分析用途。ただし、マテリアライズドビューやキャッシュ層で対応できないか先に検討すべき。

### Q3: 配列型やJSONB型は1NF違反か？

厳密なリレーショナル理論では1NF違反だが、PostgreSQLの配列型やJSONB型はインデックス対応しており、実務では有用な場面が多い。タグやメタデータなど、個別のテーブルに分離するコストが高い場合に適切に使用する。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 正規化の目的 | データ冗長性の排除と更新異常の防止 |
| 1NF | 各セルに原子値、繰り返し項目なし |
| 2NF | 非キーがキー全体に従属 |
| 3NF | 非キー間の従属がない。実務の目標 |
| BCNF | 全ての決定項が候補キー |
| 非正規化 | 読み取り性能のため意図的に冗長性導入 |
| 判断基準 | OLTP → 正規化、OLAP → 非正規化を検討 |

---

## 次に読むべきガイド

- [01-schema-design.md](./01-schema-design.md) — 制約とパーティションを含むスキーマ設計
- [03-data-modeling.md](./03-data-modeling.md) — スター/スノーフレークスキーマ
- [02-migration.md](./02-migration.md) — 正規化変更のマイグレーション

---

## 参考文献

1. Codd, E.F. (1972). "Further Normalization of the Data Base Relational Model". *IBM Research Report*.
2. Date, C.J. (2019). *Database Design and Relational Theory*. O'Reilly Media.
3. Karwin, B. (2010). *SQL Antipatterns*. Chapter 15: Entity-Attribute-Value. Pragmatic Bookshelf.
