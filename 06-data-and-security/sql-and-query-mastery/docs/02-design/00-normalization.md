# 正規化 — 1NF〜BCNF・非正規化

> 正規化はデータの冗長性を排除し更新異常を防ぐためのリレーショナルデータベース設計手法であり、非正規化はパフォーマンスとのトレードオフとして意図的に冗長性を導入する手法である。

## この章で学ぶこと

1. 第1正規形（1NF）から第3正規形（3NF）、BCNFまでの段階的な正規化プロセス
2. 正規化によって解決される更新異常の種類と内部的メカニズム
3. 第4正規形（4NF）、第5正規形（5NF）の理論的背景と実例
4. 非正規化の判断基準と実践的なパターン
5. RDBMS間の正規化関連機能の差異

## 前提知識

- SQLの基本構文（CREATE TABLE、INSERT、SELECT）
- リレーショナルモデルの基礎概念（テーブル、行、列）
- [01-schema-design.md](./01-schema-design.md) の概要理解があると望ましい

---

## 1. 正規化の目的と理論的背景

### 1.1 関数従属性（Functional Dependency）

正規化理論の基盤は関数従属性にある。属性Xの値が決まると属性Yの値が一意に決まるとき、「YはXに関数従属する」と言い、`X → Y` と記述する。

```
┌──────────── 関数従属性の種類 ────────────────────┐
│                                                    │
│  完全関数従属（Full Functional Dependency）         │
│  ────────────────────────────────────              │
│  {A, B} → C であり、A → C でも B → C でもない     │
│  例: {student_id, course_id} → grade               │
│  （学生IDだけでも科目IDだけでも成績は決まらない）   │
│                                                    │
│  部分関数従属（Partial Functional Dependency）       │
│  ────────────────────────────────────              │
│  {A, B} → C であり、A → C または B → C が成立     │
│  例: {order_id, product_id} → product_name          │
│  （product_idだけでproduct_nameが決まる）           │
│                                                    │
│  推移的関数従属（Transitive Functional Dependency） │
│  ────────────────────────────────────              │
│  A → B かつ B → C なら A → C が成立               │
│  例: emp_id → dept_id → dept_name                  │
│  （社員IDで部署IDが決まり、部署IDで部署名が決まる） │
│                                                    │
│  多値従属（Multi-Valued Dependency）                │
│  ────────────────────────────────────              │
│  A →→ B: Aの値に対してBの値の集合が一意に決まる    │
│  例: employee →→ skill, employee →→ language       │
│  （スキルと言語は互いに独立だが社員に紐づく）       │
└────────────────────────────────────────────────────┘
```

### 1.2 更新異常の詳細分析

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

#### 更新異常の具体例とコスト分析

```sql
-- 非正規化テーブル: 更新異常が発生する構造
CREATE TABLE emp_dept_denormalized (
    emp_id    INTEGER PRIMARY KEY,
    emp_name  VARCHAR(100),
    dept_id   INTEGER,
    dept_name VARCHAR(100),
    dept_loc  VARCHAR(100)
);

INSERT INTO emp_dept_denormalized VALUES
(1, '田中', 10, '開発部', '東京'),
(2, '鈴木', 10, '開発部', '東京'),
(3, '佐藤', 20, '営業部', '大阪'),
(4, '高橋', 10, '開発部', '東京');

-- 挿入異常の実演: 社員なしで新部署を登録するには？
-- emp_idがPRIMARY KEYなのでNULLにできない → 登録不可能
-- INSERT INTO emp_dept_denormalized VALUES (NULL, NULL, 30, '経理部', '名古屋');
-- → ERROR: null value in column "emp_id"

-- 更新異常の実演: 開発部を東京から横浜に移転
UPDATE emp_dept_denormalized SET dept_loc = '横浜' WHERE dept_id = 10;
-- → 3行を更新する必要がある（漏れると不整合）
-- 10万人の部署なら10万行の更新が必要

-- 削除異常の実演: 佐藤を削除すると営業部の情報も消失
DELETE FROM emp_dept_denormalized WHERE emp_id = 3;
-- → dept_id=20の情報が完全に失われる
```

### 1.3 候補キーとスーパーキー

正規化を正確に理解するにはキーの概念が不可欠である。

```
┌──────────── キーの階層構造 ──────────────────────┐
│                                                    │
│  スーパーキー (Superkey)                           │
│  ┌──────────────────────────────────────────┐    │
│  │ 行を一意に識別できる属性の集合            │    │
│  │ 例: {emp_id}, {emp_id, name},             │    │
│  │     {emp_id, name, dept_id}, ...          │    │
│  │                                            │    │
│  │  候補キー (Candidate Key)                  │    │
│  │  ┌──────────────────────────────────┐    │    │
│  │  │ 極小のスーパーキー（余分な属性なし）│    │    │
│  │  │ 例: {emp_id}, {email}            │    │    │
│  │  │                                    │    │    │
│  │  │  主キー (Primary Key)             │    │    │
│  │  │  ┌──────────────────────────┐    │    │    │
│  │  │  │ 候補キーから1つ選択       │    │    │    │
│  │  │  │ 例: emp_id               │    │    │    │
│  │  │  └──────────────────────────┘    │    │    │
│  │  └──────────────────────────────────┘    │    │
│  └──────────────────────────────────────────┘    │
│                                                    │
│  代替キー (Alternate Key)                         │
│  = 主キーに選ばれなかった候補キー                  │
│  例: email (UNIQUEで制約)                         │
└────────────────────────────────────────────────────┘
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
-- 4. 各行が一意に識別可能（主キーが存在）
```

#### 1NFの内部実装への影響

```
┌──────── 1NF違反がストレージに与える影響 ────────┐
│                                                    │
│  非正規形（カンマ区切り格納）:                     │
│  ┌─────────────────────────────────┐              │
│  │ Page内: "090-1111,03-2222" を   │              │
│  │ 可変長TEXTとして1つのタプルに格納│              │
│  │ → 検索時にパース処理が必要       │              │
│  │ → インデックスが効かない         │              │
│  │ → 個別の電話番号でのWHERE不可能  │              │
│  └─────────────────────────────────┘              │
│                                                    │
│  1NF（各値が独立行）:                              │
│  ┌─────────────────────────────────┐              │
│  │ Page内: 各タプルが独立           │              │
│  │ → B-Treeインデックスで高速検索   │              │
│  │ → WHERE phone = '090-1111-2222'  │              │
│  │   がインデックススキャンで実行可能│              │
│  └─────────────────────────────────┘              │
└────────────────────────────────────────────────────┘
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
--
-- 関数従属の分析:
--   {order_id, product_id} → quantity（完全関数従属 ✓）
--   {order_id, product_id} → price（完全関数従属 ✓）
--   product_id → product_name（部分関数従属 ✗）

-- 第2正規形（2NF）: 部分関数従属を排除
CREATE TABLE products (
    product_id   INTEGER PRIMARY KEY,
    product_name VARCHAR(100)           -- product_id のみに従属
);

CREATE TABLE order_items (
    order_id   INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(product_id),
    quantity   INTEGER,
    price      DECIMAL(10,2),          -- 注文時の価格（スナップショット）
    PRIMARY KEY (order_id, product_id)  -- 全キーに従属
);

-- 2NF の要件:
-- 1. 1NFを満たす
-- 2. 非キー属性が主キーの一部にのみ従属しない
--    （主キーが単一列なら自動的に2NF）

-- 重要な注意: priceは注文時のスナップショットとして
-- order_itemsに残すのが正しい設計（商品の現在価格とは別）
```

### コード例3: 第3正規形（3NF）— 推移的関数従属の排除

```sql
-- 2NFだが3NFでない例:
-- ┌────┬──────┬─────────────┬────────────────┐
-- │ id │ name │ dept_id     │ dept_name      │
-- └────┴──────┴─────────────┴────────────────┘
-- dept_name は dept_id に従属し、dept_id は id に従属
-- → dept_name は id に推移的に従属
--
-- 関数従属の分析:
--   id → name（直接従属 ✓）
--   id → dept_id（直接従属 ✓）
--   dept_id → dept_name（推移的従属 ✗）
--   ∴ id → dept_id → dept_name

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

-- 3NFの形式的定義（Coddの定義）:
-- テーブルRが3NFであるとは、全ての非自明な関数従属 X → A について
-- 以下のいずれかが成り立つことである:
-- (a) Xがスーパーキーである
-- (b) Aがいずれかの候補キーの一部（素属性）である
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
│    │  ドメインキー制約の排除                    │
│    ▼                                            │
│  第6正規形 (6NF) ← テンポラルデータ向け        │
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

-- BCNF の定義:
-- 全ての非自明な関数従属 X → Y について、Xがスーパーキーである
--
-- 3NFとBCNFの違い:
-- 3NFは「Yが素属性ならOK」という例外を許す
-- BCNFは例外なく「決定項はスーパーキー」を要求

-- BCNFの注意点:
-- BCNFに分解すると、元の関数従属が保存されない場合がある
-- （従属性保存分解ができないケースがある）
```

### コード例5: 3NFとBCNFの違いを示す実践例

```sql
-- より具体的なBCNF違反の例: 配送スケジュール
--
-- 前提条件:
-- 1. 各配送地域には複数の配送業者が対応可能
-- 2. 各配送業者は1つの配送地域のみを担当
-- 3. 1つの注文に対して1つの配送業者が割り当てられる
--
-- テーブル: deliveries
-- PK = (order_id, area_id)
-- 関数従属:
--   {order_id, area_id} → carrier_id（完全関数従属）
--   carrier_id → area_id（業者が担当地域を決定）
--
-- carrier_idは非キーだがarea_id（キーの一部）を決定 → BCNF違反

-- 3NF（BCNF違反を含む）
CREATE TABLE deliveries_3nf (
    order_id   INTEGER,
    area_id    INTEGER,
    carrier_id INTEGER,
    PRIMARY KEY (order_id, area_id)
    -- carrier_id → area_id の従属が問題
);

-- BCNF（分解後）
CREATE TABLE carrier_areas (
    carrier_id INTEGER PRIMARY KEY,
    area_id    INTEGER NOT NULL
);

CREATE TABLE order_carriers (
    order_id   INTEGER,
    carrier_id INTEGER REFERENCES carrier_areas(carrier_id),
    PRIMARY KEY (order_id, carrier_id)
);

-- 分解の結果:
-- carrier_areas: carrier_id → area_id（carrier_idがキー ✓ BCNF）
-- order_carriers: 候補キー = {order_id, carrier_id}（BCNF ✓）
```

---

## 3. 高次正規形

### 3.1 第4正規形（4NF）— 多値従属の排除

```sql
-- 4NF違反の例:
-- 社員が複数のスキルと複数の言語を持つ
-- スキルと言語は互いに独立

-- 4NF違反テーブル
CREATE TABLE emp_skills_languages_bad (
    emp_id   INTEGER,
    skill    VARCHAR(50),
    language VARCHAR(50),
    PRIMARY KEY (emp_id, skill, language)
);

-- データ例:
-- emp_id=1 が skill={Java, Python} と language={日本語, 英語} を持つ場合
INSERT INTO emp_skills_languages_bad VALUES
(1, 'Java',   '日本語'),
(1, 'Java',   '英語'),
(1, 'Python', '日本語'),
(1, 'Python', '英語');
-- → 2 × 2 = 4行（直積）が必要 → 冗長

-- 多値従属: emp_id →→ skill, emp_id →→ language
-- skillとlanguageは互いに独立なのに、直積を保持する必要がある

-- 4NF（多値従属を排除）
CREATE TABLE emp_skills (
    emp_id INTEGER,
    skill  VARCHAR(50),
    PRIMARY KEY (emp_id, skill)
);

CREATE TABLE emp_languages (
    emp_id   INTEGER,
    language VARCHAR(50),
    PRIMARY KEY (emp_id, language)
);

INSERT INTO emp_skills VALUES (1, 'Java'), (1, 'Python');
INSERT INTO emp_languages VALUES (1, '日本語'), (1, '英語');
-- → 2 + 2 = 4行で済む（直積なら4行必要だった）

-- 4NF の定義:
-- 全ての非自明な多値従属 X →→ Y について、Xがスーパーキーである
```

### 3.2 第5正規形（5NF）— 結合従属の排除

```sql
-- 5NF違反の例:
-- 代理店が供給者から商品を仕入れる関係
-- ただし、3者間の関係が2者間の関係から復元できない場合

-- 以下の3つの2項関係が成り立つ:
-- 代理店Aは供給者Xから仕入れる
-- 供給者Xは商品Pを供給する
-- 代理店Aは商品Pを扱う
-- → これらが成り立っても、「代理店Aが供給者Xから商品Pを仕入れる」
--   とは限らない（結合従属）

CREATE TABLE supply_3way (
    agent_id    INTEGER,
    supplier_id INTEGER,
    product_id  INTEGER,
    PRIMARY KEY (agent_id, supplier_id, product_id)
);

-- 5NFでは3者間関係は分解できない場合がある
-- ただし、ビジネスルールにより分解可能な場合もある:
-- 「代理店が供給者と取引し、その供給者が扱う商品を
--   代理店も扱っているなら、必ずその経路で仕入れる」
-- というルールがあれば、3つの2項テーブルに分解可能

-- 5NF の定義:
-- 全ての非自明な結合従属が候補キーにより暗示される
```

```
┌──────── 4NF/5NFの実務での判断フロー ──────────┐
│                                                  │
│  Q: テーブルに3つ以上の属性の組み合わせがある？   │
│  │                                              │
│  ├── No → 3NF/BCNFで十分                       │
│  │                                              │
│  └── Yes                                        │
│      │                                          │
│      Q: 属性間に独立した多値関係がある？          │
│      │                                          │
│      ├── Yes → 4NF違反 → テーブルを分割        │
│      │                                          │
│      └── No                                     │
│          │                                      │
│          Q: 3者以上の関係が2者関係から           │
│             復元できる？                         │
│          │                                      │
│          ├── Yes → 5NF分解可能                 │
│          │                                      │
│          └── No → 3者間テーブルを維持           │
└──────────────────────────────────────────────────┘
```

---

## 4. 非正規化

### 4.1 非正規化の判断基準

```
┌────────── 非正規化を検討すべき条件 ──────────────┐
│                                                    │
│  1. 読み取り/書き込み比率                          │
│     読み取り >>> 書き込み (100:1以上)               │
│     → 非正規化の効果が大きい                       │
│                                                    │
│  2. クエリパターンが固定的                         │
│     特定のJOINパターンが全体の80%以上               │
│     → そのJOINを非正規化で排除                     │
│                                                    │
│  3. レイテンシ要件                                  │
│     JOINによるレイテンシが許容できない              │
│     → キャッシュ/MVで対応できないか先に検討        │
│                                                    │
│  4. データサイズ                                    │
│     JOINするテーブルが数千万行以上                  │
│     → パーティショニング検討が先                   │
│                                                    │
│  判断フロー:                                       │
│  正規化 → インデックス追加 → MV/キャッシュ         │
│  → パーティション → リードレプリカ                 │
│  → 最後の手段として非正規化                        │
└────────────────────────────────────────────────────┘
```

### コード例6: 意図的な非正規化パターン

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

CREATE TRIGGER trg_update_order_totals
    AFTER INSERT OR UPDATE OR DELETE ON order_items
    FOR EACH ROW
    EXECUTE FUNCTION update_order_totals();

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

-- パターン3: 非正規化カラムの追加
-- 頻繁にJOINされるカラムを冗長に保持
ALTER TABLE orders ADD COLUMN customer_name VARCHAR(200);
ALTER TABLE orders ADD COLUMN customer_email VARCHAR(255);

-- トリガーでcustomersテーブルとの同期を維持
CREATE OR REPLACE FUNCTION sync_customer_denorm()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_TABLE_NAME = 'customers' THEN
        UPDATE orders SET
            customer_name = NEW.name,
            customer_email = NEW.email
        WHERE customer_id = NEW.id;
    ELSIF TG_TABLE_NAME = 'orders' THEN
        SELECT name, email INTO NEW.customer_name, NEW.customer_email
        FROM customers WHERE id = NEW.customer_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- パターン4: JSONBによる半構造化データの格納
-- 頻繁にJOINされる関連データをJSONBとして埋め込む
ALTER TABLE orders ADD COLUMN items_snapshot JSONB;

-- 注文確定時にスナップショットを作成
UPDATE orders SET items_snapshot = (
    SELECT jsonb_agg(jsonb_build_object(
        'product_name', p.name,
        'quantity', oi.quantity,
        'price', oi.price
    ))
    FROM order_items oi
    JOIN products p ON oi.product_id = p.id
    WHERE oi.order_id = orders.id
)
WHERE id = 42;
```

### コード例7: 正規化 vs 非正規化の実例比較

```sql
-- 正規化されたスキーマ（3NF）
-- 6テーブルをJOINして注文詳細を取得
EXPLAIN ANALYZE
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
    JOIN addresses a ON o.shipping_address_id = a.id
WHERE o.id = 42;
-- → 実行計画: 5つのNested Loop Join、推定50ms

-- 非正規化されたスキーマ（読み取り最適化）
-- 1テーブルで完結
EXPLAIN ANALYZE
SELECT
    order_id, order_date,
    customer_name, customer_email,
    product_name, product_sku,
    category_name,
    quantity, unit_price,
    shipping_city, shipping_postal_code
FROM order_details_denormalized
WHERE order_id = 42;
-- → 実行計画: Index Scan のみ、推定2ms

-- 中間的なアプローチ: マテリアライズドビュー
CREATE MATERIALIZED VIEW mv_order_details AS
SELECT
    o.id AS order_id, o.order_date,
    c.name AS customer_name, c.email AS customer_email,
    p.name AS product_name, p.sku AS product_sku,
    cat.name AS category_name,
    oi.quantity, oi.unit_price,
    a.city AS shipping_city, a.postal_code AS shipping_postal_code
FROM orders o
    JOIN customers c ON o.customer_id = c.id
    JOIN order_items oi ON o.id = oi.order_id
    JOIN products p ON oi.product_id = p.id
    JOIN categories cat ON p.category_id = cat.id
    JOIN addresses a ON o.shipping_address_id = a.id;

CREATE UNIQUE INDEX idx_mv_order_details_order_id ON mv_order_details(order_id);
-- → 読み取りは高速、データ更新はREFRESHで制御
```

### 4.2 RDBMS間の非正規化機能比較

```
┌──────── RDBMS間の非正規化サポート比較 ─────────┐
│                                                   │
│  機能                  │ PG │ MySQL │ Oracle │ SS│
│  ─────────────────────┼────┼───────┼────────┼───│
│  マテリアライズドビュー│ ✓  │ ✗*    │ ✓      │ ✓ │
│  生成列(GENERATED)    │ ✓  │ ✓     │ ✓      │ ✓ │
│  JSONB型             │ ✓  │ JSON  │ JSON   │ JSON│
│  配列型              │ ✓  │ ✗     │ ✗      │ ✗ │
│  トリガー            │ ✓  │ ✓     │ ✓      │ ✓ │
│  計算列(STORED)      │ ✓  │ ✓     │ ✓      │ ✓ │
│                                                   │
│  PG = PostgreSQL, SS = SQL Server                 │
│  * MySQLはMVの代わりにサマリーテーブル+イベントで対応│
└───────────────────────────────────────────────────┘
```

---

## 5. 正規化のオプティマイザへの影響

### 5.1 JOINの実行コストモデル

```
┌──────── JOINアルゴリズムと正規化の関係 ────────┐
│                                                   │
│  正規化するとJOINが増える → 実行コストへの影響    │
│                                                   │
│  Nested Loop Join:                                │
│  外側テーブルN行 × 内側テーブルの検索コスト        │
│  → インデックスがあれば O(N * log M)              │
│  → 小テーブル同士のJOINに最適                     │
│                                                   │
│  Hash Join:                                       │
│  ビルドフェーズ: O(N)  プローブフェーズ: O(M)      │
│  → 大テーブル同士のJOINに最適                     │
│  → work_memが十分なら高速                         │
│                                                   │
│  Merge Join:                                      │
│  ソート済みデータ同士: O(N + M)                    │
│  → インデックスでソート済みならソート不要           │
│                                                   │
│  結論:                                            │
│  適切なインデックスがあれば、3NF分解による          │
│  JOIN増加のコストは多くの場合許容範囲内            │
└───────────────────────────────────────────────────┘
```

### コード例8: 正規化テーブルのJOIN最適化

```sql
-- 3NFテーブルでも適切なインデックスで高速化
CREATE INDEX idx_employees_dept ON employees(department_id);
CREATE INDEX idx_departments_pk ON departments(dept_id);

-- オプティマイザはNested Loop + Index Scanを選択
EXPLAIN ANALYZE
SELECT e.name, d.dept_name
FROM employees e
    JOIN departments d ON e.department_id = d.dept_id
WHERE e.salary > 500000;

-- 結果例:
-- Nested Loop  (actual time=0.030..0.150 rows=100 loops=1)
--   -> Index Scan using idx_emp_salary on employees e
--      Filter: (salary > 500000)
--      Rows Removed by Filter: 400
--   -> Index Scan using departments_pkey on departments d
--      Index Cond: (dept_id = e.department_id)
-- Execution Time: 0.200 ms
-- → 正規化テーブルでも十分高速（インデックスが適切なら）
```

---

## 正規化レベル比較表

| 正規形 | 排除する問題 | 適用条件 | 実用性 | 分解の可逆性 |
|--------|------------|---------|--------|-------------|
| 1NF | 繰り返し項目、非原子値 | 各セルが原子値 | 必須 | - |
| 2NF | 部分関数従属 | 非キーがキー全体に従属 | 必須 | 可逆（無損失） |
| 3NF | 推移的関数従属 | 非キー間の従属排除 | 推奨 | 可逆（従属性保存） |
| BCNF | 全ての非自明な関数従属 | 決定項が候補キー | 推奨 | 可逆（従属性非保存の場合あり） |
| 4NF | 多値従属 | 独立した多値関係の分離 | 稀 | 可逆（無損失） |
| 5NF | 結合従属 | 無損失結合分解 | 極稀 | 可逆 |
| 6NF | 非自明な結合従属が全くない | 完全な分解 | テンポラルDB | 可逆 |

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
| インデックス設計 | シンプル | 複合的 |
| バックアップサイズ | 小さい | 大きい |
| トランザクション | 単純 | 複雑（複数テーブル更新） |

## RDBMS別の正規化関連機能比較表

| 機能 | PostgreSQL | MySQL (InnoDB) | Oracle | SQL Server |
|------|-----------|----------------|--------|------------|
| CHECK制約 | 完全サポート | 8.0.16+ | 完全サポート | 完全サポート |
| 外部キー | 完全サポート | 完全サポート | 完全サポート | 完全サポート |
| 排他制約 | ✓（EXCLUDE） | ✗ | ✗ | ✗ |
| GENERATED列 | ✓（STORED） | ✓（STORED/VIRTUAL） | ✓（VIRTUAL） | ✓（PERSISTED） |
| 配列型 | ✓ | ✗ | ✓（VARRAY） | ✗ |
| JSON型 | JSONB（バイナリ） | JSON（テキスト） | JSON（21c+） | JSON（2016+） |
| 部分インデックス | ✓ | ✗ | ✗（関数ベース） | ✓（フィルター付き） |
| MV | ✓ | ✗ | ✓（自動リフレッシュ） | ✓（インデックス付きビュー） |

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
-- 5. 外部キー制約が使えない
-- 6. 集約関数が使えない（文字列なのでSUM不可）

-- EAVでピボットクエリを書く場合の困難さ:
SELECT
    e.entity_id,
    MAX(CASE WHEN ea.attr_name = 'name' THEN ea.attr_value END) AS name,
    MAX(CASE WHEN ea.attr_name = 'email' THEN ea.attr_value END) AS email,
    MAX(CASE WHEN ea.attr_name = 'age' THEN ea.attr_value END)::INTEGER AS age
FROM entities e
    LEFT JOIN entity_attributes ea ON e.id = ea.entity_id
GROUP BY e.entity_id;
-- → 属性が増えるたびにクエリの修正が必要

-- OK: JSONBでスキーマレスな部分を分離
CREATE TABLE products (
    id    SERIAL PRIMARY KEY,
    name  VARCHAR(100) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    attrs JSONB  -- 可変属性はJSONBに格納
);

-- JSONBならGINインデックスで効率的に検索可能
CREATE INDEX idx_products_attrs ON products USING GIN (attrs);

-- 特定属性で検索
SELECT * FROM products WHERE attrs @> '{"color": "red"}';

-- 属性の存在確認
SELECT * FROM products WHERE attrs ? 'weight';
```

### アンチパターン2: 過度な正規化

```sql
-- NG: 都道府県や性別まで別テーブルに分離
CREATE TABLE genders (id INT PRIMARY KEY, name VARCHAR(10));
CREATE TABLE prefectures (id INT PRIMARY KEY, name VARCHAR(10));
-- → JOINが増え、クエリが複雑化し、パフォーマンスが低下
-- → 47都道府県のマスタテーブルのためにJOINが1つ増える

-- OK: 変更されない小さなマスタはENUMやCHECK制約で十分
CREATE TABLE users (
    id       SERIAL PRIMARY KEY,
    gender   VARCHAR(10) CHECK (gender IN ('male', 'female', 'other')),
    prefecture VARCHAR(10) NOT NULL
);

-- 判断基準:
-- マスタデータが以下の全てを満たすなら正規化不要:
-- ✓ 値の数が少ない（50以下）
-- ✓ 変更頻度が極めて低い
-- ✓ 追加の属性を持たない
-- ✓ 他のテーブルから参照されない
```

### アンチパターン3: 正規化なしのログテーブル

```sql
-- NG: ログテーブルに冗長データを無秩序に格納
CREATE TABLE activity_logs (
    id         BIGSERIAL PRIMARY KEY,
    user_id    INTEGER,
    user_name  VARCHAR(100),   -- usersテーブルと冗長
    user_email VARCHAR(255),   -- usersテーブルと冗長
    action     VARCHAR(50),
    target_id  INTEGER,
    target_type VARCHAR(50),
    target_name VARCHAR(200),  -- 対象テーブルと冗長
    ip_address INET,
    created_at TIMESTAMPTZ
);
-- → ユーザー名変更時に過去ログとの不整合が発生
-- → ストレージが急速に肥大化

-- OK: ログテーブルはイベント時点のスナップショットとして設計
CREATE TABLE activity_logs (
    id         BIGSERIAL PRIMARY KEY,
    user_id    INTEGER NOT NULL,
    action     VARCHAR(50) NOT NULL,
    target_id  INTEGER,
    target_type VARCHAR(50),
    -- スナップショット（意図的な非正規化）
    snapshot   JSONB NOT NULL DEFAULT '{}',
    ip_address INET,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
) PARTITION BY RANGE (created_at);

-- snapshotには「その時点の状態」を記録
-- → 正規化テーブルとは別の設計意図
-- → 監査・コンプライアンス要件を満たす
```

---

## エッジケース

### エッジケース1: 1NFとPostgreSQLの配列型

```sql
-- PostgreSQLの配列型は厳密には1NF違反だが、実務では有用
CREATE TABLE articles (
    id   SERIAL PRIMARY KEY,
    title VARCHAR(200),
    tags  TEXT[] NOT NULL DEFAULT '{}'
);

-- GINインデックスで効率的に検索可能
CREATE INDEX idx_articles_tags ON articles USING GIN (tags);

-- 配列内の値で検索
SELECT * FROM articles WHERE tags @> ARRAY['SQL', 'データベース'];

-- いつ配列型を使うべきか:
-- ✓ 値の集合を1つのエンティティに関連付ける場合
-- ✓ 値に追加属性がない場合（タグ名だけなど）
-- ✓ 値の個数が少ない場合（数十個以下）
-- ✗ 値に属性がある場合は正規化（中間テーブル）を使う
-- ✗ 値が他のエンティティへの参照の場合は正規化
```

### エッジケース2: 時系列データと正規化

```sql
-- 時系列データは正規化の適用が難しい
-- センサーデータの例: 1秒に1行、100万行/日

-- 純粋な正規化アプローチ
CREATE TABLE sensors (
    sensor_id   SERIAL PRIMARY KEY,
    sensor_name VARCHAR(100),
    location    VARCHAR(200)
);

CREATE TABLE sensor_readings (
    sensor_id INTEGER REFERENCES sensors(sensor_id),
    ts        TIMESTAMPTZ NOT NULL,
    value     DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (sensor_id, ts)
);

-- 時系列DBアプローチ（TimescaleDB: PostgreSQL拡張）
-- ハイパーテーブルで自動パーティショニング
-- SELECT create_hypertable('sensor_readings', 'ts');

-- 時系列データの正規化のポイント:
-- 1. メタデータ（センサー情報）は正規化
-- 2. 計測値は時系列最適化された構造で格納
-- 3. 集約結果は非正規化（CAGG: 連続集約）で保持
```

### エッジケース3: 多対多関係の属性付き中間テーブル

```sql
-- 中間テーブルに追加属性がある場合の正規化
CREATE TABLE students (
    id   SERIAL PRIMARY KEY,
    name VARCHAR(100)
);

CREATE TABLE courses (
    id   SERIAL PRIMARY KEY,
    name VARCHAR(100)
);

-- 中間テーブルに属性がある
CREATE TABLE enrollments (
    student_id INTEGER REFERENCES students(id),
    course_id  INTEGER REFERENCES courses(id),
    grade      CHAR(2),         -- 成績
    enrolled_at DATE,           -- 登録日
    instructor VARCHAR(100),    -- 担当教員
    PRIMARY KEY (student_id, course_id)
);

-- 問題: instructorが courses に対して関数従属する場合
-- course_id → instructor なら3NF違反
-- → instructorをcoursesテーブルに移動すべき

-- ただし、同じ科目でも学期によって教員が変わる場合は
-- {student_id, course_id} → instructor は完全関数従属
-- → 3NFを満たす（中間テーブルに残して良い）
```

---

## 演習

### 演習1（基礎）: 1NF〜3NFの実践

以下の非正規形テーブルを3NFまで正規化せよ。

```sql
-- 非正規形テーブル
CREATE TABLE orders_raw (
    order_id     INTEGER,
    order_date   DATE,
    customer_name VARCHAR(100),
    customer_email VARCHAR(255),
    customer_phone VARCHAR(20),
    items        TEXT,  -- "商品A:3個:1000円, 商品B:1個:2000円"
    total_amount DECIMAL(10,2)
);
```

**ヒント**: 関数従属を洗い出し、段階的に分解する。

<details>
<summary>解答例</summary>

```sql
-- 1NF: 繰り返し項目の排除
CREATE TABLE customers_1nf (
    customer_id   SERIAL PRIMARY KEY,
    customer_name VARCHAR(100) NOT NULL,
    customer_email VARCHAR(255) UNIQUE NOT NULL,
    customer_phone VARCHAR(20)
);

CREATE TABLE orders_1nf (
    order_id     SERIAL PRIMARY KEY,
    order_date   DATE NOT NULL,
    customer_id  INTEGER NOT NULL,
    total_amount DECIMAL(10,2)
);

CREATE TABLE order_items_1nf (
    order_id      INTEGER,
    product_name  VARCHAR(200),
    quantity      INTEGER,
    unit_price    DECIMAL(10,2),
    PRIMARY KEY (order_id, product_name)
);

-- 2NF: order_items_1nfのproduct_nameを商品テーブルに分離
-- （product_nameがproduct_idのみに従属する場合）
CREATE TABLE products_2nf (
    product_id   SERIAL PRIMARY KEY,
    product_name VARCHAR(200) NOT NULL
);

CREATE TABLE order_items_2nf (
    order_id   INTEGER,
    product_id INTEGER REFERENCES products_2nf(product_id),
    quantity   INTEGER NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    PRIMARY KEY (order_id, product_id)
);

-- 3NF: customersのphone/emailは既にcustomer_idに直接従属
-- total_amountはorder_itemsから計算可能なので排除可能
-- （ただし、パフォーマンスのため残す場合は意図的な非正規化）
CREATE TABLE orders_3nf (
    order_id    SERIAL PRIMARY KEY,
    order_date  DATE NOT NULL,
    customer_id INTEGER NOT NULL REFERENCES customers_1nf(customer_id)
    -- total_amountは計算で求めるか、非正規化として残す
);
```
</details>

### 演習2（応用）: BCNF分解

以下のテーブルの関数従属を分析し、BCNF に分解せよ。

```sql
-- 教室予約テーブル
-- 前提:
-- 1. 各教員は1つの科目のみを教える
-- 2. 各科目は複数の教員が教えることができる
-- 3. 各教室は1時間枠につき1つの授業のみ
CREATE TABLE classroom_bookings (
    room_id    INTEGER,
    time_slot  INTEGER,
    teacher_id INTEGER,
    subject    VARCHAR(100),
    PRIMARY KEY (room_id, time_slot)
);
```

**ヒント**: `teacher_id → subject` の関数従属がBCNF違反を引き起こす。

<details>
<summary>解答例</summary>

```sql
-- 関数従属の分析:
-- {room_id, time_slot} → teacher_id（主キーから一意に決まる）
-- {room_id, time_slot} → subject（主キーから一意に決まる）
-- teacher_id → subject（教員は1科目のみ担当）
--
-- teacher_id は主キーの部分集合ではないが、subjectを決定する
-- teacher_id はスーパーキーではない → BCNF違反

-- BCNF分解:
CREATE TABLE teacher_subjects (
    teacher_id INTEGER PRIMARY KEY,
    subject    VARCHAR(100) NOT NULL
);

CREATE TABLE room_bookings (
    room_id    INTEGER,
    time_slot  INTEGER,
    teacher_id INTEGER REFERENCES teacher_subjects(teacher_id),
    PRIMARY KEY (room_id, time_slot)
);

-- 検証: 全ての関数従属の決定項がスーパーキー
-- teacher_subjects: teacher_id → subject（teacher_idがPK = スーパーキー ✓）
-- room_bookings: {room_id, time_slot} → teacher_id（PKがスーパーキー ✓）
```
</details>

### 演習3（発展）: 非正規化の設計判断

以下の要件に基づき、正規化スキーマと非正規化スキーマの両方を設計し、トレードオフを分析せよ。

**要件**:
- ECサイトの商品レビューシステム
- 商品ページに表示する平均評価とレビュー数（1秒以内のレスポンス必須）
- レビューの投稿/編集/削除は1日1万件程度
- 商品ページの閲覧は1日100万PV
- レビューには「役に立った」ボタン（投票数も表示）

<details>
<summary>解答例</summary>

```sql
-- 正規化スキーマ（3NF）
CREATE TABLE products (
    id   SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL
);

CREATE TABLE reviews (
    id         SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL REFERENCES products(id),
    user_id    INTEGER NOT NULL REFERENCES users(id),
    rating     SMALLINT NOT NULL CHECK (rating BETWEEN 1 AND 5),
    title      VARCHAR(200),
    body       TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (product_id, user_id)  -- 1ユーザー1レビュー
);

CREATE TABLE review_votes (
    review_id INTEGER REFERENCES reviews(id),
    user_id   INTEGER REFERENCES users(id),
    helpful   BOOLEAN NOT NULL,
    PRIMARY KEY (review_id, user_id)
);

-- 商品ページ表示クエリ（正規化版: 重い）
SELECT
    p.name,
    AVG(r.rating) AS avg_rating,
    COUNT(r.id) AS review_count,
    (SELECT COUNT(*) FROM review_votes rv WHERE rv.review_id = r.id AND rv.helpful)
        AS helpful_count
FROM products p
    LEFT JOIN reviews r ON r.product_id = p.id
WHERE p.id = 42
GROUP BY p.id, p.name;
-- → レビュー数が多いと集約が重い

-- 非正規化スキーマ: 集約結果をキャッシュ
ALTER TABLE products ADD COLUMN avg_rating DECIMAL(3,2) DEFAULT 0;
ALTER TABLE products ADD COLUMN review_count INTEGER DEFAULT 0;
ALTER TABLE reviews ADD COLUMN helpful_count INTEGER DEFAULT 0;

-- トリガーで自動更新
CREATE OR REPLACE FUNCTION update_product_review_stats()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE products SET
        avg_rating = (SELECT AVG(rating) FROM reviews WHERE product_id = COALESCE(NEW.product_id, OLD.product_id)),
        review_count = (SELECT COUNT(*) FROM reviews WHERE product_id = COALESCE(NEW.product_id, OLD.product_id))
    WHERE id = COALESCE(NEW.product_id, OLD.product_id);
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_review_stats
    AFTER INSERT OR UPDATE OR DELETE ON reviews
    FOR EACH ROW EXECUTE FUNCTION update_product_review_stats();

-- 商品ページ表示クエリ（非正規化版: 高速）
SELECT name, avg_rating, review_count
FROM products WHERE id = 42;
-- → Index Scan のみ、1ms以下

-- トレードオフ分析:
-- 読み取り: 100万PV/日 → 非正規化で大幅に高速化
-- 書き込み: 1万件/日 → トリガーのオーバーヘッドは許容範囲
-- 判定: 読み取り/書き込み比 = 100:1 → 非正規化が適切
```
</details>

---

## FAQ

### Q1: 3NFまで正規化すれば十分か？

多くの実務アプリケーションでは3NFで十分。BCNFまで進める場合は、候補キーが複数存在し、非キー属性がキーの一部を決定するような特殊な状況に限られる。過度な正規化はJOINの増加とパフォーマンスの低下を招く。

### Q2: いつ非正規化すべきか？

(1) 読み取り頻度が書き込み頻度を大幅に上回る場合、(2) JOINのコストが許容できないレベルの場合、(3) レポーティング/分析用途。ただし、マテリアライズドビューやキャッシュ層で対応できないか先に検討すべき。非正規化は最後の手段である。

### Q3: 配列型やJSONB型は1NF違反か？

厳密なリレーショナル理論では1NF違反だが、PostgreSQLの配列型やJSONB型はインデックス対応しており、実務では有用な場面が多い。タグやメタデータなど、個別のテーブルに分離するコストが高い場合に適切に使用する。

### Q4: NoSQLでは正規化は不要か？

NoSQL（MongoDB、DynamoDB等）では非正規化が基本的な設計方針となる。JOINがないため、読み取りパターンに合わせてデータを冗長に埋め込む。ただし、これは「正規化が不要」なのではなく「非正規化を前提とした設計」であり、更新整合性はアプリケーション層で担保する必要がある。

### Q5: 正規化の度合いはマイグレーションで変更できるか？

可能だが、データ移行が必要となる。正規化の強化（テーブル分割）は比較的安全だが、非正規化（テーブル統合）はデータの整合性確認が重要。大規模な正規化レベルの変更には [02-migration.md](./02-migration.md) のオンラインマイグレーション手法を参照。

### Q6: 正規化とパフォーマンスの関係をどう測定すべきか？

`EXPLAIN ANALYZE` で実行計画を確認し、JOINの実行コスト（特にHash Join vs Nested Loop の選択）を分析する。実際の本番データ量で負荷テストを行い、正規化版と非正規化版のスループットとレイテンシを比較する。統計的に有意な差がない場合は正規化を維持すべき。

---

## トラブルシューティング

### 正規化に関する一般的な問題と対処法

| 問題 | 原因 | 対処法 |
|------|------|--------|
| JOINが多すぎて遅い | 過度な正規化 or インデックス不足 | まずインデックスを追加、改善しなければMV検討 |
| 更新が遅い | 非正規化によるトリガー連鎖 | トリガーの実行計画を確認、非同期更新を検討 |
| ストレージ肥大化 | 非正規化によるデータ冗長 | パーティショニングで古いデータをアーカイブ |
| データ不整合 | 非正規化テーブルの同期漏れ | トリガーの完全性を確認、制約を追加 |
| 統計情報の不一致 | ANALYZE未実行 | `ANALYZE table_name` で統計更新 |
| デッドロック | トリガーによる循環更新 | トリガーの更新順序を統一、ロック戦略の見直し |

---

## セキュリティに関する考察

### 正規化とデータアクセス制御

```sql
-- 正規化されたスキーマではRow Level Security（RLS）が適用しやすい
-- 部署テーブルと社員テーブルが分離されていれば、
-- 部署ごとのアクセス制御が容易

ALTER TABLE employees ENABLE ROW LEVEL SECURITY;

CREATE POLICY emp_dept_policy ON employees
    USING (department_id IN (
        SELECT dept_id FROM user_dept_access
        WHERE user_id = current_setting('app.current_user_id')::INTEGER
    ));

-- 非正規化テーブルではRLSの条件が複雑化する可能性がある
-- → セキュリティ要件も正規化レベルの判断材料とすべき
```

---

## まとめ

| 項目 | 要点 |
|------|------|
| 正規化の目的 | データ冗長性の排除と更新異常の防止 |
| 関数従属性 | 正規化理論の基盤。完全/部分/推移的従属を区別 |
| 1NF | 各セルに原子値、繰り返し項目なし |
| 2NF | 非キーがキー全体に従属（部分関数従属の排除） |
| 3NF | 非キー間の従属がない。実務の目標 |
| BCNF | 全ての決定項が候補キー。従属性保存が犠牲になる場合あり |
| 4NF/5NF | 多値従属/結合従属の排除。実務では稀 |
| 非正規化 | 読み取り性能のため意図的に冗長性導入。最後の手段 |
| 判断基準 | OLTP → 正規化、OLAP → 非正規化を検討 |
| 実装 | MV・トリガー・JSONB等で非正規化を制御 |

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
4. Kent, W. (1983). "A Simple Guide to Five Normal Forms in Relational Database Theory". *Communications of the ACM*, 26(2), 120-125.
5. Bernstein, P.A. (1976). "Synthesizing Third Normal Form Relations from Functional Dependencies". *ACM TODS*, 1(4), 277-298.
6. PostgreSQL Documentation — "Data Definition" — https://www.postgresql.org/docs/current/ddl.html
7. Fagin, R. (1977). "Multivalued Dependencies and a New Normal Form for Relational Databases". *ACM TODS*, 2(3), 262-278.
