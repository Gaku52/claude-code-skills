# PostgreSQL固有機能 — JSONB・配列型・範囲型・全文検索・拡張機能

> PostgreSQLは「世界で最も先進的なオープンソースRDBMS」を標榜し、JSONB、配列型、範囲型、全文検索、拡張機能（Extension）など他のRDBMSにはない豊富な機能を提供する。これらを適切に使いこなすことで、NoSQLの柔軟性とRDBMSのトランザクション整合性を両立した堅牢なデータ基盤を構築できる。本章ではこれらのPostgreSQL固有機能について、内部実装レベルの仕組みからWHYの理解まで含めて徹底的に解説する。

---

## この章で学ぶこと

1. **JSONB型の内部構造と操作** — バイナリ格納形式の仕組み、演算子体系、GINインデックスの内部構造、jsonpathクエリを理解し、リレーショナルデータとの使い分けができるようになる
2. **配列型・範囲型・複合型の実践活用** — PostgreSQL固有のコレクション型を使って、中間テーブルを使わないタグ管理や日付範囲の重複検出ができるようになる
3. **全文検索とトライグラム検索の実装** — tsvector/tsquery の仕組みと日本語対応、pg_trgmによるあいまい検索を組み込み全文検索として実装できるようになる
4. **Extension（拡張機能）による機能拡張** — uuid-ossp、pgcrypto、PostGIS、pg_stat_statementsなど主要拡張の選定と導入判断ができるようになる
5. **GENERATED COLUMNS・LISTEN/NOTIFY・テーブル継承** — 計算列、リアルタイム通知、継承による分類を適切な場面で活用できるようになる

---

## 前提知識

本章を理解するには以下の知識が必要です。

- [SQLの基礎](../00-basics/00-sql-overview.md) — SELECT/INSERT/UPDATE/DELETEの基本操作
- [JOINの理解](../00-basics/02-joins.md) — テーブル結合の概念
- [インデックスの基礎](../01-advanced/03-indexing.md) — B-Treeインデックスの仕組み
- [正規化の基礎](../02-design/00-normalization.md) — リレーショナルモデルの基本原則

---

## 1. JSONB型 — バイナリJSON の全体像

### 1.1 なぜJSONBが必要なのか

リレーショナルデータベースは厳格なスキーマ（テーブル定義）が強みだが、現実のアプリケーションでは「スキーマが事前に決まらないデータ」を格納する必要がある。たとえば以下のようなケースだ。

- ECサイトの商品属性（衣類はサイズ・色、PCはCPU・RAM・SSD）
- ユーザーの設定値（人によって異なる設定項目）
- 外部APIからの応答データ（スキーマが頻繁に変わる）
- イベントのメタデータ（種類によって含まれる情報が異なる）

従来はEAV（Entity-Attribute-Value）パターンやシリアライズされたテキストで対応していたが、これらは検索性能が悪く、型安全性もない。PostgreSQLのJSONB型は「リレーショナルの堅牢性 + ドキュメントの柔軟性」を両立する解決策だ。

### 1.2 JSONB の内部構造

```
┌──────────────── JSONB の内部構造 ─────────────────────┐
│                                                        │
│  入力: {"name": "田中", "age": 30, "tags": ["A","B"]}  │
│                                                        │
│  ┌──── JSON型（テキスト格納）────┐                     │
│  │ そのまま文字列として保存      │                     │
│  │ 読み取り時に毎回パース        │                     │
│  │ 重複キー保持、順序保持        │                     │
│  └──────────────────────────────┘                     │
│                                                        │
│  ┌──── JSONB型（バイナリ格納）───┐                     │
│  │ パース済みバイナリで保存      │                     │
│  │ ヘッダ + エントリ配列         │                     │
│  │  ┌─ JEntry ─────────────┐    │                     │
│  │  │ type: object          │    │                     │
│  │  │ num_pairs: 3          │    │                     │
│  │  │ pairs[0]:             │    │                     │
│  │  │   key_offset → "age"  │    │                     │
│  │  │   val_offset → 30     │    │                     │
│  │  │ pairs[1]:             │    │                     │
│  │  │   key_offset → "name" │    │                     │
│  │  │   val_offset → "田中" │    │                     │
│  │  │ pairs[2]:             │    │                     │
│  │  │   key_offset → "tags" │    │                     │
│  │  │   val_offset → [...]  │    │                     │
│  │  └───────────────────────┘    │                     │
│  │ ※ キーはソート済み            │                     │
│  │ ※ 重複キーは最後の値のみ      │                     │
│  │ ※ バイナリサーチで高速アクセス │                     │
│  └──────────────────────────────┘                     │
│                                                        │
│  WHY バイナリ？                                        │
│  → 読み取り時にパース不要（O(1)でキーアクセス）        │
│  → GINインデックスの構築が可能                         │
│  → 演算子による高速な包含検索・存在確認                 │
└────────────────────────────────────────────────────────┘
```

### コード例1: JSONB の基本操作 — CRUD

```sql
-- テーブル作成: メタデータをJSONBで持つ商品テーブル
CREATE TABLE products (
    id       SERIAL PRIMARY KEY,
    name     VARCHAR(200) NOT NULL,
    category VARCHAR(100) NOT NULL,
    price    DECIMAL(10, 2) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'
);

-- データ挿入: 商品ごとに異なる属性をJSONBに格納
INSERT INTO products (name, category, price, metadata) VALUES
('ThinkPad X1 Carbon', 'laptop', 198000, '{
    "brand": "Lenovo",
    "specs": {"cpu": "i7-1365U", "ram_gb": 16, "ssd_gb": 512},
    "tags": ["business", "lightweight", "14inch"],
    "weight_kg": 1.12,
    "ports": {"usb_c": 2, "usb_a": 2, "hdmi": 1}
}'),
('iPhone 15 Pro', 'smartphone', 159800, '{
    "brand": "Apple",
    "specs": {"chip": "A17 Pro", "ram_gb": 8, "storage_gb": 256},
    "tags": ["premium", "camera", "titanium"],
    "weight_g": 187,
    "colors": ["natural", "blue", "white", "black"]
}'),
('Pixel 8', 'smartphone', 82500, '{
    "brand": "Google",
    "specs": {"chip": "Tensor G3", "ram_gb": 8, "storage_gb": 128},
    "tags": ["ai", "camera", "affordable"],
    "weight_g": 187
}');

-- ===== フィールドアクセス演算子 =====

-- -> : JSONBオブジェクトを返す（ネスト参照に使う）
-- ->> : TEXT型で返す（最終的な値の取得に使う）
SELECT
    name,
    metadata->>'brand' AS brand,                    -- TEXT型で取得
    metadata->'specs'->>'cpu' AS cpu,                -- ネストしたキーの参照
    (metadata->'specs'->>'ram_gb')::INTEGER AS ram,  -- 型キャスト
    metadata->'tags'->0 AS first_tag,               -- 配列の0番目（JSONB型）
    metadata->'tags'->>0 AS first_tag_text,         -- 配列の0番目（TEXT型）
    metadata#>>'{specs,ram_gb}' AS ram_by_path       -- パス形式で取得
FROM products;

-- ===== JSONB の検索演算子 =====

-- @> : 包含（右辺のJSONが左辺に含まれるか）
SELECT * FROM products
WHERE metadata @> '{"brand": "Apple"}';

-- ? : キーの存在確認
SELECT * FROM products
WHERE metadata ? 'colors';

-- ?& : 全てのキーが存在するか
SELECT * FROM products
WHERE metadata ?& ARRAY['brand', 'specs', 'weight_g'];

-- ?| : いずれかのキーが存在するか
SELECT * FROM products
WHERE metadata ?| ARRAY['weight_kg', 'weight_g'];

-- ===== JSONB の更新 =====

-- || : マージ（既存キーは上書き、新規キーは追加）
UPDATE products
SET metadata = metadata || '{"warranty_years": 2, "in_stock": true}'
WHERE id = 1;

-- jsonb_set : 特定パスの値を更新
UPDATE products
SET metadata = jsonb_set(metadata, '{specs,ram_gb}', '32')
WHERE id = 1;

-- jsonb_set の第4引数 create_if_missing（デフォルトtrue）
UPDATE products
SET metadata = jsonb_set(metadata, '{specs,gpu}', '"RTX 4060"', true)
WHERE id = 1;

-- - : キーの削除
UPDATE products
SET metadata = metadata - 'warranty_years'
WHERE id = 1;

-- #- : ネストしたキーの削除（パス指定）
UPDATE products
SET metadata = metadata #- '{specs,gpu}'
WHERE id = 1;

-- ===== JSONB の集約 =====

-- ブランド別の商品数と平均価格
SELECT
    metadata->>'brand' AS brand,
    COUNT(*) AS product_count,
    AVG(price) AS avg_price
FROM products
GROUP BY metadata->>'brand'
ORDER BY avg_price DESC;
```

### コード例2: JSONB のインデックスと高度な検索

```sql
-- ===== GINインデックスの2つのオプション =====

-- 1. デフォルトGIN: @>, ?, ?|, ?& の全演算子に対応
CREATE INDEX idx_products_metadata
ON products USING GIN (metadata);

-- 2. jsonb_path_ops GIN: @> 演算子に特化（高速・省メモリ）
-- パスごとにハッシュ値を格納するため、包含検索が高速
CREATE INDEX idx_products_metadata_path
ON products USING GIN (metadata jsonb_path_ops);

-- WHY 2種類あるのか？
-- デフォルトGIN: キーの存在確認（?演算子）もインデックスで処理可能
-- jsonb_path_ops: @> 包含検索に特化、インデックスサイズが20-30%小さい
-- → アプリで使う演算子に応じて選択する

-- 3. 式インデックス: 特定フィールドのB-Tree（等値・範囲検索に最適）
CREATE INDEX idx_products_brand
ON products ((metadata->>'brand'));

-- 複合インデックス: ブランド + 価格帯の検索を高速化
CREATE INDEX idx_products_brand_price
ON products ((metadata->>'brand'), price);

-- ===== jsonpath クエリ（SQL:2016標準、PostgreSQL 12+）=====

-- @@ : jsonpathの条件式で検索
SELECT * FROM products
WHERE metadata @@ '$.specs.ram_gb > 8';

-- jsonb_path_query: パスに一致する値を全て返す
SELECT name, jsonb_path_query(metadata, '$.tags[*]') AS tag
FROM products;

-- jsonb_path_query_array: 結果を配列として返す
SELECT name, jsonb_path_query_array(metadata, '$.tags[*]') AS all_tags
FROM products;

-- jsonb_path_exists: パスが存在するか
SELECT name FROM products
WHERE jsonb_path_exists(metadata, '$.specs.cpu');

-- フィルタ付きjsonpath
SELECT name,
       jsonb_path_query(metadata, '$.tags[*] ? (@ like_regex "^cam")')
       AS camera_tag
FROM products;

-- ===== JSONB関数 =====

-- jsonb_each: キーと値のペアに展開
SELECT p.name, kv.key, kv.value
FROM products p,
     jsonb_each(p.metadata->'specs') AS kv(key, value)
WHERE p.id = 1;

-- jsonb_object_keys: キー一覧を取得
SELECT DISTINCT jsonb_object_keys(metadata) AS top_level_key
FROM products
ORDER BY top_level_key;

-- jsonb_array_elements: 配列の各要素に展開
SELECT p.name, tag.value AS tag
FROM products p,
     jsonb_array_elements_text(p.metadata->'tags') AS tag(value);

-- jsonb_strip_nulls: null値を含むキーを除去
SELECT jsonb_strip_nulls('{"a": 1, "b": null, "c": 3}'::JSONB);
-- → {"a": 1, "c": 3}

-- jsonb_typeof: 値の型を判定
SELECT jsonb_typeof(metadata->'specs') AS specs_type,
       jsonb_typeof(metadata->'tags') AS tags_type,
       jsonb_typeof(metadata->'weight_kg') AS weight_type
FROM products WHERE id = 1;
-- → object, array, number

-- jsonb_build_object / jsonb_build_array: JSONBの動的構築
SELECT jsonb_build_object(
    'product_name', name,
    'brand', metadata->>'brand',
    'price', price
) AS summary
FROM products;
```

### コード例3: JSONB のバリデーションとCHECK制約

```sql
-- PostgreSQL 17+ の JSON Schema バリデーション（将来対応）
-- 現時点ではCHECK制約とIS JSON述語で代用する

-- IS JSON述語（PostgreSQL 16+）
ALTER TABLE products
ADD CONSTRAINT check_metadata_is_object
CHECK (metadata IS JSON OBJECT);

-- CHECK制約でJSONBの構造を検証
ALTER TABLE products
ADD CONSTRAINT check_metadata_has_brand
CHECK (metadata ? 'brand');

-- CHECK制約でJSONBの値を検証
ALTER TABLE products
ADD CONSTRAINT check_metadata_brand_not_empty
CHECK (length(metadata->>'brand') > 0);

-- トリガーによる高度なバリデーション
CREATE OR REPLACE FUNCTION validate_product_metadata()
RETURNS TRIGGER AS $$
BEGIN
    -- brandフィールドは必須
    IF NOT (NEW.metadata ? 'brand') THEN
        RAISE EXCEPTION 'metadata must contain "brand" key';
    END IF;

    -- specsフィールドがある場合はオブジェクトであること
    IF NEW.metadata ? 'specs'
       AND jsonb_typeof(NEW.metadata->'specs') != 'object' THEN
        RAISE EXCEPTION 'metadata.specs must be an object';
    END IF;

    -- tagsフィールドがある場合は配列であること
    IF NEW.metadata ? 'tags'
       AND jsonb_typeof(NEW.metadata->'tags') != 'array' THEN
        RAISE EXCEPTION 'metadata.tags must be an array';
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_validate_product_metadata
    BEFORE INSERT OR UPDATE ON products
    FOR EACH ROW EXECUTE FUNCTION validate_product_metadata();
```

---

## 2. 配列型 — PostgreSQL のコレクション型

### 2.1 配列型を使うべき場面

配列型は「少数の値のリストを1行に格納する」場合に有用だ。ただし、配列の要素に外部キー制約を設定することはできないため、使い分けの判断基準を理解しておく必要がある。

```
┌───── 配列型 vs 中間テーブル 判断フロー ──────────┐
│                                                    │
│  Q1: 要素に外部キー制約が必要？                     │
│  ├── Yes → 中間テーブル                            │
│  └── No → Q2へ                                     │
│                                                    │
│  Q2: 要素の変更頻度は？                             │
│  ├── 高い → 中間テーブル                           │
│  └── 低い → Q3へ                                   │
│                                                    │
│  Q3: 要素数の上限は？                               │
│  ├── 数百以上 → 中間テーブル                       │
│  └── 数十以下 → 配列型が適切                       │
│                                                    │
│  配列型が適切な例:                                  │
│  - タグ（文字列リスト）                             │
│  - 電話番号（複数保持）                             │
│  - 設定値（選択肢リスト）                           │
│  - スコア履歴（数値リスト）                         │
│                                                    │
│  中間テーブルが適切な例:                             │
│  - ユーザーとロールの関係（M:N）                    │
│  - 注文と商品の関係（付帯情報あり）                  │
│  - フォロー関係（相互参照あり）                      │
└────────────────────────────────────────────────────┘
```

### コード例4: 配列型の完全操作ガイド

```sql
-- ===== テーブル作成 =====
CREATE TABLE blog_posts (
    id         SERIAL PRIMARY KEY,
    title      VARCHAR(200) NOT NULL,
    body       TEXT,
    tags       TEXT[] NOT NULL DEFAULT '{}',
    scores     INTEGER[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ===== データ挿入（2つの構文）=====
INSERT INTO blog_posts (title, body, tags, scores) VALUES
('PostgreSQL入門', 'PostgreSQLの基礎を解説します',
 ARRAY['database', 'postgresql', 'sql'],              -- ARRAY構文
 ARRAY[85, 92, 78]),
('React入門', 'Reactの基礎を解説します',
 '{"frontend", "react", "javascript"}',                -- リテラル構文
 '{90, 88, 95}'),
('TypeScript実践', 'TypeScriptの型システムについて',
 ARRAY['frontend', 'typescript', 'javascript'],
 ARRAY[95, 91, 89]);

-- ===== 配列演算子 =====

-- @> : 左辺が右辺を含むか（包含）
SELECT * FROM blog_posts WHERE tags @> ARRAY['postgresql'];

-- <@ : 左辺が右辺に含まれるか
SELECT * FROM blog_posts WHERE tags <@ ARRAY['database', 'postgresql', 'sql', 'mysql'];

-- && : 共通要素があるか（重複チェック）
SELECT * FROM blog_posts WHERE tags && ARRAY['react', 'sql'];

-- = ANY() : いずれかの要素に一致
SELECT * FROM blog_posts WHERE 'javascript' = ANY(tags);

-- インデックスアクセス（1始まり！PostgreSQLの配列は1-indexed）
SELECT title, tags[1] AS first_tag, tags[2] AS second_tag
FROM blog_posts;

-- スライス
SELECT title, tags[1:2] AS first_two_tags FROM blog_posts;

-- ===== 配列関数 =====

-- array_length: 配列の長さ（第2引数は次元、通常1）
SELECT title, array_length(tags, 1) AS tag_count FROM blog_posts;

-- array_to_string: 配列を文字列に結合
SELECT title, array_to_string(tags, ', ') AS tag_list FROM blog_posts;

-- array_position: 要素の位置を返す（なければNULL）
SELECT title, array_position(tags, 'javascript') AS js_pos FROM blog_posts;

-- array_remove: 特定の要素を除去した新しい配列を返す
SELECT title, array_remove(tags, 'javascript') AS tags_no_js FROM blog_posts;

-- array_cat: 配列の結合
SELECT array_cat(ARRAY[1,2,3], ARRAY[4,5,6]);  -- → {1,2,3,4,5,6}

-- array_append / array_prepend: 要素の追加
UPDATE blog_posts
SET tags = array_append(tags, 'tutorial')
WHERE id = 1;

UPDATE blog_posts
SET tags = array_prepend('featured', tags)
WHERE id = 1;

-- ===== UNNEST: 配列の展開（最重要関数）=====

-- 全記事のタグを行に展開
SELECT p.title, t.tag
FROM blog_posts p, UNNEST(p.tags) AS t(tag);

-- タグの出現回数を集計
SELECT tag, COUNT(*) AS usage_count
FROM blog_posts, UNNEST(tags) AS tag
GROUP BY tag
ORDER BY usage_count DESC;

-- 全タグの重複排除リスト（ソート済み）
SELECT ARRAY_AGG(DISTINCT tag ORDER BY tag) AS all_tags
FROM blog_posts, UNNEST(tags) AS tag;

-- スコアの統計
SELECT
    title,
    array_length(scores, 1) AS num_scores,
    (SELECT AVG(s) FROM UNNEST(scores) AS s) AS avg_score,
    (SELECT MAX(s) FROM UNNEST(scores) AS s) AS max_score,
    (SELECT MIN(s) FROM UNNEST(scores) AS s) AS min_score
FROM blog_posts;

-- ===== 配列用GINインデックス =====
CREATE INDEX idx_posts_tags ON blog_posts USING GIN (tags);

-- インデックス使用確認
EXPLAIN ANALYZE
SELECT * FROM blog_posts WHERE tags @> ARRAY['postgresql'];
-- → Bitmap Index Scan on idx_posts_tags（GINインデックスが使われる）
```

---

## 3. 範囲型・全文検索・ネットワーク型

### 3.1 範囲型（Range Types）

範囲型は「ある値の区間」を表すPostgreSQL固有のデータ型で、日付範囲、数値範囲、タイムスタンプ範囲などを直接扱える。EXCLUDE制約と組み合わせることで、予約やスケジュールの重複をデータベースレベルで防止できる。

### コード例5: 範囲型の実践活用

```sql
-- ===== 範囲型の種類 =====
-- INT4RANGE  : 整数範囲
-- INT8RANGE  : bigint範囲
-- NUMRANGE   : 数値範囲
-- DATERANGE  : 日付範囲
-- TSRANGE    : タイムスタンプ範囲（タイムゾーンなし）
-- TSTZRANGE  : タイムスタンプ範囲（タイムゾーンあり）

-- ===== 予約システム: 部屋の重複チェック =====
CREATE EXTENSION IF NOT EXISTS btree_gist;  -- EXCLUDE制約に必要

CREATE TABLE room_reservations (
    id       SERIAL PRIMARY KEY,
    room_id  INTEGER NOT NULL,
    period   DATERANGE NOT NULL,
    guest    VARCHAR(100) NOT NULL,
    -- EXCLUDE制約: 同じ部屋で期間が重複する予約を禁止
    EXCLUDE USING GIST (room_id WITH =, period WITH &&)
);

-- 正常な予約
INSERT INTO room_reservations (room_id, period, guest) VALUES
(101, '[2024-03-01, 2024-03-05)', '田中太郎'),   -- 3/1〜3/4
(101, '[2024-03-05, 2024-03-08)', '鈴木一郎'),   -- 3/5〜3/7（隣接はOK）
(102, '[2024-03-01, 2024-03-05)', '佐藤花子');   -- 別の部屋はOK

-- エラーになる予約（重複）
-- INSERT INTO room_reservations (room_id, period, guest) VALUES
-- (101, '[2024-03-04, 2024-03-06)', '山田次郎');
-- → ERROR: conflicting key value violates exclusion constraint

-- ===== 範囲演算子 =====

-- @> : 範囲が値を含むか
SELECT * FROM room_reservations
WHERE period @> '2024-03-03'::DATE;

-- && : 範囲同士が重複するか
SELECT * FROM room_reservations
WHERE period && '[2024-03-03, 2024-03-06)'::DATERANGE;

-- << : 左辺が右辺より完全に前か
-- >> : 左辺が右辺より完全に後か
-- -|- : 隣接しているか

-- 範囲の演算
SELECT
    '[2024-03-01, 2024-03-05)'::DATERANGE
    * '[2024-03-03, 2024-03-08)'::DATERANGE AS intersection,  -- 共通部分
    '[2024-03-01, 2024-03-05)'::DATERANGE
    + '[2024-03-03, 2024-03-08)'::DATERANGE AS union_range;    -- 和集合

-- lower/upper: 範囲の下限・上限を取得
SELECT
    guest,
    lower(period) AS check_in,
    upper(period) AS check_out,
    upper(period) - lower(period) AS stay_days
FROM room_reservations;

-- ===== 勤務シフト管理 =====
CREATE TABLE work_shifts (
    id          SERIAL PRIMARY KEY,
    employee_id INTEGER NOT NULL,
    shift_time  TSTZRANGE NOT NULL,
    EXCLUDE USING GIST (employee_id WITH =, shift_time WITH &&)
);

INSERT INTO work_shifts (employee_id, shift_time) VALUES
(1, '[2024-03-01 09:00, 2024-03-01 17:00)'::TSTZRANGE),
(1, '[2024-03-01 18:00, 2024-03-01 22:00)'::TSTZRANGE);  -- 隣接しないのでOK

-- ===== 価格帯テーブル =====
CREATE TABLE price_tiers (
    id          SERIAL PRIMARY KEY,
    tier_name   VARCHAR(50) NOT NULL,
    quantity    INT4RANGE NOT NULL,
    unit_price  DECIMAL(10, 2) NOT NULL,
    EXCLUDE USING GIST (quantity WITH &&)  -- 数量範囲の重複禁止
);

INSERT INTO price_tiers (tier_name, quantity, unit_price) VALUES
('個人', '[1, 10)', 1000),
('小口', '[10, 100)', 800),
('大口', '[100, 1000)', 600),
('卸売', '[1000,)', 400);  -- 上限なし

-- 数量に応じた単価を取得
SELECT tier_name, unit_price
FROM price_tiers
WHERE quantity @> 50;  -- → '小口', 800
```

### 3.2 全文検索

### コード例6: 全文検索とトライグラム検索

```sql
-- ===== tsvector / tsquery ベースの全文検索 =====

CREATE TABLE articles (
    id            SERIAL PRIMARY KEY,
    title         VARCHAR(300) NOT NULL,
    body          TEXT NOT NULL,
    search_vector TSVECTOR,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- search_vectorをトリガーで自動更新
CREATE OR REPLACE FUNCTION update_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector :=
        setweight(to_tsvector('english', COALESCE(NEW.title, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.body, '')), 'B');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_search_vector
    BEFORE INSERT OR UPDATE OF title, body ON articles
    FOR EACH ROW EXECUTE FUNCTION update_search_vector();

-- GINインデックス
CREATE INDEX idx_articles_search ON articles USING GIN (search_vector);

-- データ挿入（トリガーでsearch_vectorが自動設定される）
INSERT INTO articles (title, body) VALUES
('PostgreSQL Full-Text Search Guide',
 'PostgreSQL provides built-in full-text search using tsvector and tsquery. It supports stemming, ranking, and highlighting.'),
('Index Optimization Techniques',
 'Understanding B-Tree, GIN, and GiST indexes is essential for PostgreSQL performance tuning.'),
('Database Security Best Practices',
 'Implementing row-level security, encryption, and audit logging protects your PostgreSQL database.');

-- ===== 検索クエリ =====

-- 基本検索
SELECT title, ts_rank(search_vector, query) AS rank
FROM articles,
     to_tsquery('english', 'postgresql & index') AS query
WHERE search_vector @@ query
ORDER BY rank DESC;

-- フレーズ検索（PostgreSQL 13+）
SELECT title
FROM articles
WHERE search_vector @@ phraseto_tsquery('english', 'full text search');

-- OR検索
SELECT title FROM articles
WHERE search_vector @@ to_tsquery('english', 'security | encryption');

-- NOT検索（securityを含むがencryptionを含まない）
SELECT title FROM articles
WHERE search_vector @@ to_tsquery('english', 'security & !encryption');

-- ハイライト表示
SELECT
    title,
    ts_headline('english', body,
        to_tsquery('english', 'postgresql'),
        'StartSel=<b>, StopSel=</b>, MaxWords=35, MinWords=15'
    ) AS highlighted
FROM articles
WHERE search_vector @@ to_tsquery('english', 'postgresql');

-- ===== pg_trgm: トライグラム（あいまい検索）=====

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- トライグラムの確認
SELECT show_trgm('PostgreSQL');
-- → {"  p"," po","gre","osq","pos","ostg","pgr","que","res","sql","stg","tgr"}

-- トライグラムGINインデックス
CREATE INDEX idx_articles_title_trgm
ON articles USING GIN (title gin_trgm_ops);

-- 類似度検索（タイポ許容）
SELECT title, similarity(title, 'Postgre') AS sim
FROM articles
WHERE title % 'Postgre'  -- デフォルト閾値 0.3 以上
ORDER BY sim DESC;

-- 類似度の閾値変更
SET pg_trgm.similarity_threshold = 0.1;

-- LIKE/ILIKEでもトライグラムインデックスが使える
EXPLAIN ANALYZE
SELECT * FROM articles WHERE title ILIKE '%security%';
-- → Bitmap Index Scan on idx_articles_title_trgm

-- ===== 日本語全文検索の課題と対策 =====
-- to_tsvectorのデフォルトパーサーは空白区切りのため、
-- 日本語のように空白なしの言語は対応が必要

-- 方式1: pg_bigm拡張（バイグラムベース）
-- CREATE EXTENSION IF NOT EXISTS pg_bigm;
-- CREATE INDEX idx_title_bigm ON articles USING GIN (title gin_bigm_ops);
-- SELECT * FROM articles WHERE title LIKE '%データベース%';

-- 方式2: pgroonga拡張（MeCab形態素解析ベース）
-- CREATE EXTENSION IF NOT EXISTS pgroonga;
-- CREATE INDEX idx_articles_pgroonga ON articles USING pgroonga (title, body);
-- SELECT * FROM articles WHERE title &@~ 'データベース';

-- 方式3: simpleパーサー + LIKE（小規模データ向け）
SELECT * FROM articles
WHERE to_tsvector('simple', title) @@ to_tsquery('simple', 'postgresql');
```

### 3.3 ネットワーク型

```sql
-- ===== INET / CIDR / MACADDR 型 =====

CREATE TABLE access_logs (
    id          BIGSERIAL PRIMARY KEY,
    client_ip   INET NOT NULL,
    subnet      CIDR,
    mac_address MACADDR,
    accessed_at TIMESTAMPTZ DEFAULT NOW(),
    path        VARCHAR(500)
);

INSERT INTO access_logs (client_ip, subnet, path) VALUES
('192.168.1.100', '192.168.1.0/24', '/api/users'),
('10.0.0.50', '10.0.0.0/8', '/api/orders'),
('192.168.1.200', '192.168.1.0/24', '/api/products');

-- サブネットに含まれるか
SELECT * FROM access_logs
WHERE client_ip << '192.168.1.0/24';

-- サブネットに含まれるか（等しいも含む）
SELECT * FROM access_logs
WHERE client_ip <<= '192.168.0.0/16';

-- IPアドレスのホスト部分を取得
SELECT client_ip, host(client_ip), masklen(subnet)
FROM access_logs;

-- GiSTインデックスでIP範囲検索を高速化
CREATE INDEX idx_access_logs_ip ON access_logs USING GIST (client_ip inet_ops);
```

---

## 4. 拡張機能（Extension）

### 4.1 Extension のアーキテクチャ

```
┌──────── PostgreSQL Extension アーキテクチャ ────────┐
│                                                      │
│  contrib拡張（PostgreSQL本体に同梱）                  │
│  ├── pg_stat_statements   クエリ統計（必須級）        │
│  ├── uuid-ossp / pgcrypto UUID生成・暗号化           │
│  ├── pg_trgm              トライグラム検索           │
│  ├── hstore               簡易KVS                    │
│  ├── btree_gist           GiSTでB-Tree演算子利用     │
│  ├── btree_gin            GINでB-Tree演算子利用      │
│  ├── tablefunc            crosstab（ピボット）       │
│  ├── postgres_fdw         外部PostgreSQLへの接続     │
│  └── file_fdw             外部ファイルの参照         │
│                                                      │
│  サードパーティ拡張（別途インストール）                │
│  ├── PostGIS              地理空間データ             │
│  ├── TimescaleDB          時系列データ               │
│  ├── pg_partman           パーティション自動管理     │
│  ├── pgvector             ベクトル類似度検索         │
│  ├── pg_bigm              日本語全文検索             │
│  ├── pgroonga             高速日本語全文検索         │
│  ├── pgAudit              監査ログ                   │
│  └── Citus                分散DB化                   │
│                                                      │
│  WHY Extensionが重要か？                             │
│  → PostgreSQL本体をスリムに保ちつつ                  │
│  → 必要な機能だけを追加できるモジュラー設計          │
│  → contrib拡張はPostgreSQL本体と同じ品質保証         │
└──────────────────────────────────────────────────────┘
```

### コード例7: 主要拡張機能の活用

```sql
-- ===== 利用可能な拡張の確認 =====
SELECT name, default_version, comment
FROM pg_available_extensions
WHERE comment IS NOT NULL
ORDER BY name
LIMIT 20;

-- インストール済み拡張の確認
SELECT extname, extversion FROM pg_extension ORDER BY extname;

-- ===== uuid-ossp: UUID生成 =====
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

SELECT uuid_generate_v4();                    -- ランダムUUID
SELECT uuid_generate_v1();                    -- タイムスタンプベースUUID
SELECT gen_random_uuid();                     -- PostgreSQL 13+標準関数

-- UUIDを主キーとして使う
CREATE TABLE users_v2 (
    id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name  VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE
);

-- ===== pgcrypto: 暗号化 =====
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- パスワードハッシュ（bcrypt）
SELECT crypt('my_password', gen_salt('bf', 12));
-- → $2a$12$xxxx...（60文字のbcryptハッシュ）

-- パスワード検証
SELECT (crypt('my_password', stored_hash) = stored_hash) AS is_valid;

-- 対称鍵暗号（AES-256）
SELECT pgp_sym_encrypt('機密データ', 'encryption_key');
SELECT pgp_sym_decrypt(
    pgp_sym_encrypt('機密データ', 'encryption_key'),
    'encryption_key'
);

-- ランダムバイト列
SELECT encode(gen_random_bytes(32), 'hex') AS random_token;

-- ===== pg_trgm: あいまい検索 =====
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE customers (
    id   SERIAL PRIMARY KEY,
    name VARCHAR(200) NOT NULL
);

INSERT INTO customers (name) VALUES
('田中太郎'), ('田中太朗'), ('田仲太郎'), ('鈴木一郎');

CREATE INDEX idx_customers_name_trgm ON customers USING GIN (name gin_trgm_ops);

-- 類似度検索
SELECT name, similarity(name, '田中太郎') AS sim
FROM customers
ORDER BY sim DESC;
-- → 田中太郎: 1.0, 田中太朗: 0.75, 田仲太郎: 0.5, 鈴木一郎: 0.0

-- ===== hstore: 簡易キーバリューストア =====
CREATE EXTENSION IF NOT EXISTS hstore;

CREATE TABLE app_settings (
    user_id  INTEGER PRIMARY KEY,
    prefs    HSTORE
);

INSERT INTO app_settings VALUES
(1, 'theme => dark, lang => ja, font_size => 14'),
(2, 'theme => light, lang => en');

SELECT user_id, prefs->'theme' AS theme, prefs->'lang' AS lang
FROM app_settings;

-- ※ 現在はJSONBの方が推奨。hstoreはネストできない

-- ===== PostGIS: 地理空間（別途インストールが必要）=====
-- CREATE EXTENSION IF NOT EXISTS postgis;
--
-- CREATE TABLE shops (
--     id       SERIAL PRIMARY KEY,
--     name     VARCHAR(200),
--     location GEOMETRY(Point, 4326)
-- );
--
-- INSERT INTO shops (name, location) VALUES
-- ('東京本店', ST_SetSRID(ST_MakePoint(139.7671, 35.6812), 4326)),
-- ('大阪支店', ST_SetSRID(ST_MakePoint(135.5023, 34.6937), 4326));
--
-- -- 東京駅から5km以内の店舗
-- SELECT name, ST_Distance(
--     location::geography,
--     ST_SetSRID(ST_MakePoint(139.7671, 35.6812), 4326)::geography
-- ) AS distance_m
-- FROM shops
-- WHERE ST_DWithin(
--     location::geography,
--     ST_SetSRID(ST_MakePoint(139.7671, 35.6812), 4326)::geography,
--     5000
-- );

-- ===== pg_stat_statements: クエリ統計（本番環境では必須）=====
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- トップ10の遅いクエリ
SELECT
    query,
    calls,
    round(total_exec_time::NUMERIC, 2) AS total_ms,
    round(mean_exec_time::NUMERIC, 2) AS avg_ms,
    rows,
    round((100.0 * shared_blks_hit / NULLIF(shared_blks_hit + shared_blks_read, 0))::NUMERIC, 2)
        AS cache_hit_pct
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- ===== pgvector: ベクトル類似度検索（AI/ML向け）=====
-- CREATE EXTENSION IF NOT EXISTS vector;
--
-- CREATE TABLE document_embeddings (
--     id        SERIAL PRIMARY KEY,
--     content   TEXT NOT NULL,
--     embedding vector(1536)  -- OpenAI text-embedding-3-small の次元数
-- );
--
-- CREATE INDEX ON document_embeddings
--     USING ivfflat (embedding vector_cosine_ops)
--     WITH (lists = 100);
--
-- -- コサイン類似度で最近傍検索
-- SELECT content, 1 - (embedding <=> '[0.1, 0.2, ...]'::vector) AS similarity
-- FROM document_embeddings
-- ORDER BY embedding <=> '[0.1, 0.2, ...]'::vector
-- LIMIT 5;
```

---

## 5. 高度なPostgreSQL機能

### コード例8: GENERATED COLUMNS・LISTEN/NOTIFY・テーブル継承

```sql
-- ===== GENERATED COLUMNS（計算列、PostgreSQL 12+）=====

CREATE TABLE order_items (
    id          SERIAL PRIMARY KEY,
    product_name VARCHAR(200) NOT NULL,
    quantity    INTEGER NOT NULL CHECK (quantity > 0),
    unit_price  DECIMAL(10, 2) NOT NULL CHECK (unit_price > 0),
    tax_rate    DECIMAL(5, 4) NOT NULL DEFAULT 0.10,
    -- STORED: ディスクに保存される計算列
    subtotal    DECIMAL(12, 2) GENERATED ALWAYS AS
                (quantity * unit_price) STORED,
    tax_amount  DECIMAL(12, 2) GENERATED ALWAYS AS
                (quantity * unit_price * tax_rate) STORED,
    total       DECIMAL(12, 2) GENERATED ALWAYS AS
                (quantity * unit_price * (1 + tax_rate)) STORED
);

INSERT INTO order_items (product_name, quantity, unit_price) VALUES
('ノートPC', 2, 150000),
('マウス', 5, 3000);

SELECT product_name, quantity, unit_price, subtotal, tax_amount, total
FROM order_items;
-- ノートPC | 2 | 150000 | 300000 | 30000 | 330000
-- マウス   | 5 | 3000   | 15000  | 1500  | 16500

-- WHY GENERATED COLUMNS？
-- → アプリケーション側で計算不要（バグ防止）
-- → 計算値にインデックスを張れる
-- → STORED列は物理的に保存されるのでSELECT時の計算コストゼロ
-- → VIRTUAL列（PostgreSQL未対応、MySQL対応）はSELECT時に計算

-- ===== LISTEN / NOTIFY（リアルタイム通知）=====

-- セッション1（リスナー）:
LISTEN order_created;
LISTEN order_status_changed;

-- セッション2（通知側）:
NOTIFY order_created, '{"order_id": 42, "amount": 5000}';

-- トリガーで自動通知する実践例
CREATE TABLE orders (
    id          SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    status      VARCHAR(20) NOT NULL DEFAULT 'pending',
    total       DECIMAL(12, 2) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE OR REPLACE FUNCTION notify_order_event()
RETURNS TRIGGER AS $$
DECLARE
    payload JSONB;
BEGIN
    IF TG_OP = 'INSERT' THEN
        payload := jsonb_build_object(
            'event', 'order_created',
            'order_id', NEW.id,
            'customer_id', NEW.customer_id,
            'total', NEW.total
        );
        PERFORM pg_notify('order_events', payload::TEXT);
    ELSIF TG_OP = 'UPDATE' AND OLD.status != NEW.status THEN
        payload := jsonb_build_object(
            'event', 'order_status_changed',
            'order_id', NEW.id,
            'old_status', OLD.status,
            'new_status', NEW.status
        );
        PERFORM pg_notify('order_events', payload::TEXT);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_notify_order
    AFTER INSERT OR UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION notify_order_event();

-- WHY LISTEN/NOTIFY？
-- → ポーリング不要のリアルタイム通知
-- → 外部メッセージキューが不要な軽量なイベント通知
-- → トランザクションに連動（COMMITされた変更のみ通知される）
-- 制限:
-- → ペイロードは8KB以内
-- → 永続化されない（リスナーが未接続なら通知は失われる）
-- → 大規模イベント処理にはKafka/RabbitMQが適切

-- ===== テーブル継承（Table Inheritance）=====

-- 基底テーブル
CREATE TABLE events (
    id         BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    user_id    INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 子テーブル（基底テーブルのカラムを継承）
CREATE TABLE click_events (
    element_id  VARCHAR(100),
    page_url    VARCHAR(500)
) INHERITS (events);

CREATE TABLE purchase_events (
    product_id  INTEGER NOT NULL,
    amount      DECIMAL(10, 2) NOT NULL,
    currency    VARCHAR(3) DEFAULT 'JPY'
) INHERITS (events);

-- 子テーブルへの挿入
INSERT INTO click_events (event_type, user_id, element_id, page_url)
VALUES ('click', 1, 'btn-submit', '/checkout');

INSERT INTO purchase_events (event_type, user_id, product_id, amount)
VALUES ('purchase', 1, 42, 5000);

-- 親テーブルを検索すると全子テーブルのデータも返る
SELECT * FROM events;

-- 親テーブルのみ検索（子テーブルを除外）
SELECT * FROM ONLY events;

-- WHY テーブル継承？
-- → 共通カラムの定義を一箇所に集約
-- → 親テーブルへのクエリで全子テーブルを検索可能
-- 注意:
-- → 現在はパーティショニングの方が推奨（PostgreSQL 10+）
-- → 継承ではUNIQUE制約やFKが子テーブルに適用されない
-- → 新規プロジェクトではDECLARATIVE PARTITIONINGを使うこと
```

---

## PostgreSQLのデータ型体系（全体図）

```
┌──────────────── PostgreSQL の特殊データ型 ─────────────────────┐
│                                                                 │
│  構造化データ                                                    │
│  ├── JSONB       → バイナリJSON。GINインデックス対応。推奨       │
│  ├── JSON        → テキストJSON。パース済みでない。非推奨        │
│  ├── HSTORE      → 単純KVS。ネスト不可。JSONBで代替推奨         │
│  └── XML         → XMLドキュメント。XMLサポートが必要な場合のみ  │
│                                                                 │
│  コレクション                                                    │
│  ├── ARRAY       → 配列型（全型で利用可能）。GINインデックス対応 │
│  └── COMPOSITE   → 複合型（行型）。UDTの定義に使用               │
│                                                                 │
│  範囲型                                                          │
│  ├── INT4RANGE   → 整数範囲        EXCLUDE制約で重複防止可能     │
│  ├── INT8RANGE   → bigint範囲                                    │
│  ├── NUMRANGE    → 数値範囲                                      │
│  ├── DATERANGE   → 日付範囲        予約システムの重複チェック     │
│  ├── TSRANGE     → タイムスタンプ範囲（TZなし）                  │
│  └── TSTZRANGE   → タイムスタンプ範囲（TZあり）シフト管理に最適  │
│                                                                 │
│  ネットワーク型                                                  │
│  ├── INET        → IPアドレス（v4/v6）  サブネット検索対応       │
│  ├── CIDR        → ネットワークアドレス                          │
│  └── MACADDR     → MACアドレス                                   │
│                                                                 │
│  空間・検索型                                                    │
│  ├── POINT       → 2D座標                                       │
│  ├── GEOMETRY    → PostGIS地理空間（拡張）                       │
│  ├── TSVECTOR    → 全文検索用ベクトル（GINインデックス）         │
│  └── TSQUERY     → 全文検索用クエリ                              │
│                                                                 │
│  識別子・特殊型                                                  │
│  ├── UUID        → 128ビット一意識別子。gen_random_uuid()        │
│  ├── BIT/VARBIT  → ビット列                                     │
│  ├── BYTEA       → バイナリデータ                                │
│  └── ENUM        → 列挙型。ALTER TYPE ... ADD VALUEで追加可能    │
│                                                                 │
│  ベクトル型（拡張）                                              │
│  └── VECTOR      → pgvector。AI/ML向けベクトル類似度検索        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 比較表

### JSON vs JSONB 比較表

| 特徴 | JSON | JSONB |
|------|------|-------|
| 格納形式 | テキスト（入力そのまま） | バイナリ（パース済み） |
| 格納サイズ | やや小さい | やや大きい（メタデータ分） |
| 挿入速度 | 高速（変換不要） | やや遅い（パースが必要） |
| 読み取り速度 | 遅い（毎回パース） | 高速（パース済み） |
| キーアクセス | O(n) 線形走査 | O(log n) バイナリサーチ |
| インデックス | 不可 | GIN対応（@>, ?, @@） |
| 重複キー | 保持 | 最後のキーのみ保持 |
| キー順序 | 保持（入力順） | ソートされる |
| 空白 | 保持 | 除去 |
| 演算子 | 少ない（->, ->>のみ） | 豊富（@>, ?, ?&, ?|, @@） |
| jsonpath | 非対応 | 対応（PostgreSQL 12+） |
| 推奨度 | 入力テキストの完全保存が必要な場合のみ | ほぼ全てのユースケースで推奨 |

### 主要Extension 比較表

| Extension | カテゴリ | 用途 | サイズ影響 | 本番利用 | 推奨度 |
|-----------|---------|------|----------|---------|--------|
| pg_stat_statements | 監視 | クエリ統計・スロークエリ分析 | 極小 | 必須 | 必須 |
| uuid-ossp | 識別子 | UUID生成（v1,v4） | なし | 安全 | 高（13+はgen_random_uuid()で代替可） |
| pgcrypto | 暗号化 | パスワードハッシュ・暗号化 | なし | 安全 | 高 |
| pg_trgm | 検索 | トライグラムあいまい検索 | インデックス依存 | 安全 | 高 |
| btree_gist | インデックス | GiSTでB-Tree演算子利用 | 小 | 安全 | EXCLUDE制約使用時に必要 |
| PostGIS | 地理空間 | 座標・距離・面積計算 | 大 | 安全 | 地理空間データがあるなら必須 |
| pgvector | AI/ML | ベクトル類似度検索 | 中 | 安全 | RAG/セマンティック検索なら必須 |
| pg_partman | 管理 | パーティション自動管理 | 小 | 安全 | 大規模テーブルで推奨 |
| TimescaleDB | 時系列 | 時系列データの高速処理 | 中 | 安全 | 時系列データが主なら推奨 |
| pg_bigm | 検索 | 日本語全文検索（バイグラム） | インデックス依存 | 安全 | 日本語検索が必要なら推奨 |
| hstore | KVS | 単純キーバリュー格納 | 小 | 安全 | 低（JSONBで代替可） |
| pgAudit | 監査 | 詳細な監査ログ記録 | 中（ログ量依存） | 安全 | コンプライアンス要件があれば必須 |

### インデックス型 vs 対応するデータ型・演算子

| インデックス型 | 対応データ型 | 対応演算子 | ユースケース |
|--------------|------------|-----------|------------|
| B-Tree | 全スカラー型 | =, <, >, <=, >= | 等値・範囲検索（デフォルト） |
| GIN | JSONB, ARRAY, TSVECTOR | @>, ?, &&, @@ | 全文検索、JSON検索、配列包含 |
| GIN (jsonb_path_ops) | JSONB | @> のみ | JSONB包含検索に特化 |
| GIN (gin_trgm_ops) | TEXT | %, LIKE, ILIKE | あいまい検索 |
| GiST | 範囲型, INET, 幾何型 | @>, &&, <<, >> | 範囲検索、空間検索 |
| SP-GiST | INET, TEXT | <<, >> | 基数木構造の検索 |
| BRIN | タイムスタンプ, 整数 | <, >, =, <= | 大規模テーブルの範囲検索（省サイズ） |
| IVFFlat (pgvector) | vector | <=> (コサイン距離) | ベクトル近傍検索 |
| HNSW (pgvector) | vector | <=> | 高精度ベクトル検索 |

---

## アンチパターン

### アンチパターン1: リレーショナルデータを全てJSONBに格納する

```sql
-- NG: 本来テーブルで管理すべきデータを全てJSONBに
CREATE TABLE bad_orders (
    id   SERIAL PRIMARY KEY,
    data JSONB  -- 顧客情報、商品情報、配送先、全てJSONに
);

INSERT INTO bad_orders (data) VALUES ('{
    "customer": {"name": "田中", "email": "tanaka@example.com"},
    "items": [{"product": "PC", "price": 150000}],
    "shipping": {"address": "東京都...", "method": "express"}
}');

-- 問題点:
-- 1. 外部キー制約が効かない → データ整合性を保証できない
-- 2. JOINできない → 集計・分析が困難
-- 3. 型安全性がない → 数値が文字列として入るバグ
-- 4. スキーマの自己文書化機能が失われる
-- 5. NOT NULL制約・CHECK制約が使えない

-- OK: リレーショナルデータはテーブル、可変部分のみJSONB
CREATE TABLE good_orders (
    id          SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    status      VARCHAR(20) NOT NULL DEFAULT 'pending',
    total       DECIMAL(12,2) NOT NULL,
    metadata    JSONB DEFAULT '{}'  -- メモ、タグ、外部APIレスポンス等の可変情報のみ
);

-- 判断基準:
-- ・WHERE句で頻繁に検索する → テーブルカラム
-- ・外部キー制約が必要 → テーブルカラム
-- ・スキーマが固定 → テーブルカラム
-- ・スキーマが可変/任意 → JSONB
-- ・表示のみで検索しない → JSONB
```

### アンチパターン2: 配列型でM:N関係を表現する

```sql
-- NG: 配列でM:N関係を表現
CREATE TABLE bad_articles (
    id       SERIAL PRIMARY KEY,
    title    VARCHAR(200),
    tag_ids  INTEGER[]  -- FK制約が効かない！
);

-- 問題点:
-- 1. tag_ids内の値が実在するtagsレコードか検証できない
-- 2. tags.idが削除されても配列内の参照は残る（孤立参照）
-- 3. タグの付帯情報（追加日時等）を持てない
-- 4. タグの変更時にUPDATEが配列操作になり複雑
-- 5. パフォーマンス: JOINの最適化が効かない

-- OK: 中間テーブルでM:N関係を表現
CREATE TABLE tags (
    id   SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE
);

CREATE TABLE good_articles (
    id    SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL
);

CREATE TABLE article_tags (
    article_id INTEGER REFERENCES good_articles(id) ON DELETE CASCADE,
    tag_id     INTEGER REFERENCES tags(id) ON DELETE CASCADE,
    added_at   TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (article_id, tag_id)
);

-- 外部キー制約、カスケード削除、JOIN最適化が全て可能

-- ただし、タグが単なる文字列ラベルで
-- 外部キーが不要な場合は配列型がシンプルで適切
CREATE TABLE simple_articles (
    id    SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    tags  TEXT[] NOT NULL DEFAULT '{}'  -- 文字列タグのリスト
);
-- tags用にGINインデックスを張れば検索も高速
CREATE INDEX idx_simple_articles_tags
ON simple_articles USING GIN (tags);
```

### アンチパターン3: JSONB に対する全件スキャン

```sql
-- NG: GINインデックスなしでJSONB検索
SELECT * FROM products
WHERE metadata @> '{"brand": "Apple"}';
-- → Seq Scan: テーブル全体をスキャンしてJSONBの中身を比較

-- OK: GINインデックスを作成してから検索
CREATE INDEX idx_products_metadata ON products USING GIN (metadata);
-- または特定のブランドで頻繁に検索するなら式インデックス
CREATE INDEX idx_products_brand ON products ((metadata->>'brand'));
-- → Index Scan に変わり、大幅に高速化

-- EXPLAIN ANALYZEでインデックスの利用を確認する習慣をつける
EXPLAIN ANALYZE
SELECT * FROM products WHERE metadata @> '{"brand": "Apple"}';
```

---

## 実践演習

### 演習1（基礎）: ECサイトの商品JSONB設計

**課題**: 以下の要件を満たすECサイトの商品テーブルを設計し、検索クエリを書いてください。

- 商品名、カテゴリ、価格は通常のカラム
- 商品属性（色、サイズ、重量など）はJSONBで格納
- タグは配列型で格納
- 以下のクエリを作成:
  1. 特定のブランド（"Apple"）の商品を検索
  2. 価格が10万円以上でRAMが8GB以上の商品を検索
  3. "camera"タグを持つ商品の一覧
  4. ブランド別の商品数と平均価格の集計

<details>
<summary>模範解答</summary>

```sql
-- テーブル定義
CREATE TABLE ec_products (
    id         SERIAL PRIMARY KEY,
    name       VARCHAR(200) NOT NULL,
    category   VARCHAR(100) NOT NULL,
    price      DECIMAL(12, 2) NOT NULL,
    tags       TEXT[] NOT NULL DEFAULT '{}',
    attributes JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- インデックス
CREATE INDEX idx_ec_products_category ON ec_products (category);
CREATE INDEX idx_ec_products_price ON ec_products (price);
CREATE INDEX idx_ec_products_tags ON ec_products USING GIN (tags);
CREATE INDEX idx_ec_products_attrs ON ec_products USING GIN (attributes);
CREATE INDEX idx_ec_products_brand ON ec_products ((attributes->>'brand'));

-- テストデータ
INSERT INTO ec_products (name, category, price, tags, attributes) VALUES
('MacBook Pro 14', 'laptop', 298000,
 ARRAY['apple', 'professional', 'creative'],
 '{"brand": "Apple", "specs": {"cpu": "M3 Pro", "ram_gb": 18, "ssd_gb": 512}, "color": "space_black"}'),
('iPhone 15 Pro', 'smartphone', 159800,
 ARRAY['apple', 'camera', 'premium'],
 '{"brand": "Apple", "specs": {"chip": "A17 Pro", "ram_gb": 8, "storage_gb": 256}, "color": "natural"}'),
('Galaxy S24 Ultra', 'smartphone', 189800,
 ARRAY['samsung', 'camera', 'ai', 'premium'],
 '{"brand": "Samsung", "specs": {"chip": "Snapdragon 8 Gen 3", "ram_gb": 12, "storage_gb": 256}, "color": "titanium_gray"}'),
('ThinkPad X1 Carbon', 'laptop', 198000,
 ARRAY['lenovo', 'business', 'lightweight'],
 '{"brand": "Lenovo", "specs": {"cpu": "i7-1365U", "ram_gb": 16, "ssd_gb": 512}, "weight_kg": 1.12}');

-- 1. 特定のブランドの商品を検索
SELECT name, price, attributes->>'brand' AS brand
FROM ec_products
WHERE attributes @> '{"brand": "Apple"}';

-- 2. 価格10万円以上かつRAM 8GB以上
SELECT name, price,
       (attributes->'specs'->>'ram_gb')::INTEGER AS ram_gb
FROM ec_products
WHERE price >= 100000
  AND (attributes->'specs'->>'ram_gb')::INTEGER >= 8;

-- 3. "camera"タグを持つ商品
SELECT name, price, tags
FROM ec_products
WHERE tags @> ARRAY['camera'];

-- 4. ブランド別集計
SELECT
    attributes->>'brand' AS brand,
    COUNT(*) AS product_count,
    ROUND(AVG(price), 0) AS avg_price,
    MIN(price) AS min_price,
    MAX(price) AS max_price
FROM ec_products
GROUP BY attributes->>'brand'
ORDER BY avg_price DESC;
```

</details>

### 演習2（応用）: 会議室予約システムの範囲型活用

**課題**: 以下の要件を満たす会議室予約システムを設計してください。

- 同じ会議室の同じ時間帯に重複予約ができないこと（EXCLUDE制約）
- 特定の日時に空いている会議室を検索できること
- 予約の重複チェックが高速であること
- 予約の作成・検索・キャンセル機能を実装

<details>
<summary>模範解答</summary>

```sql
-- 拡張の有効化
CREATE EXTENSION IF NOT EXISTS btree_gist;

-- 会議室マスタ
CREATE TABLE meeting_rooms (
    id       SERIAL PRIMARY KEY,
    name     VARCHAR(100) NOT NULL UNIQUE,
    capacity INTEGER NOT NULL CHECK (capacity > 0),
    floor    INTEGER NOT NULL,
    features JSONB DEFAULT '{}' -- {"projector": true, "whiteboard": true}
);

-- 予約テーブル
CREATE TABLE meeting_reservations (
    id          SERIAL PRIMARY KEY,
    room_id     INTEGER NOT NULL REFERENCES meeting_rooms(id),
    time_range  TSTZRANGE NOT NULL,
    organizer   VARCHAR(100) NOT NULL,
    title       VARCHAR(200) NOT NULL,
    attendees   INTEGER NOT NULL DEFAULT 1,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    -- 重複予約の防止
    EXCLUDE USING GIST (room_id WITH =, time_range WITH &&),
    -- 過去の予約は不可
    CHECK (lower(time_range) >= NOW() - interval '1 hour')
);

-- テストデータ
INSERT INTO meeting_rooms (name, capacity, floor, features) VALUES
('会議室A', 10, 3, '{"projector": true, "whiteboard": true}'),
('会議室B', 6, 3, '{"whiteboard": true}'),
('大会議室', 30, 5, '{"projector": true, "whiteboard": true, "video_conf": true}');

-- 予約の作成
INSERT INTO meeting_reservations (room_id, time_range, organizer, title, attendees) VALUES
(1, '[2024-12-01 10:00, 2024-12-01 11:00)'::TSTZRANGE, '田中', 'チームMTG', 5),
(1, '[2024-12-01 13:00, 2024-12-01 14:30)'::TSTZRANGE, '鈴木', '設計レビュー', 8),
(2, '[2024-12-01 10:00, 2024-12-01 12:00)'::TSTZRANGE, '佐藤', '1on1', 2);

-- 特定の日時に空いている会議室を検索
-- 12/1 11:00-12:00 で空いている部屋
SELECT r.name, r.capacity, r.floor
FROM meeting_rooms r
WHERE r.id NOT IN (
    SELECT room_id FROM meeting_reservations
    WHERE time_range && '[2024-12-01 11:00, 2024-12-01 12:00)'::TSTZRANGE
)
ORDER BY r.capacity;

-- 特定の部屋の予約一覧
SELECT
    mr.title,
    mr.organizer,
    lower(mr.time_range) AS start_time,
    upper(mr.time_range) AS end_time,
    mr.attendees
FROM meeting_reservations mr
JOIN meeting_rooms rm ON mr.room_id = rm.id
WHERE rm.name = '会議室A'
  AND mr.time_range && '[2024-12-01, 2024-12-02)'::TSTZRANGE
ORDER BY lower(mr.time_range);

-- 予約のキャンセル
DELETE FROM meeting_reservations WHERE id = 1;
```

</details>

### 演習3（発展）: リアルタイム通知付き在庫管理システム

**課題**: 以下の要件を満たす在庫管理システムを設計してください。

- 商品の在庫数が閾値（10個）以下になったらLISTEN/NOTIFYで自動通知
- 在庫変動の履歴をJSONBで記録
- 在庫状況のリアルタイムビューを作成
- GENERATED COLUMNSで在庫ステータスを自動計算

<details>
<summary>模範解答</summary>

```sql
-- 商品テーブル
CREATE TABLE inventory_products (
    id            SERIAL PRIMARY KEY,
    sku           VARCHAR(50) NOT NULL UNIQUE,
    name          VARCHAR(200) NOT NULL,
    current_stock INTEGER NOT NULL DEFAULT 0 CHECK (current_stock >= 0),
    reorder_point INTEGER NOT NULL DEFAULT 10,
    max_stock     INTEGER NOT NULL DEFAULT 100,
    -- GENERATED COLUMN: 在庫ステータスを自動計算
    stock_status  VARCHAR(20) GENERATED ALWAYS AS (
        CASE
            WHEN current_stock = 0 THEN 'out_of_stock'
            WHEN current_stock <= reorder_point THEN 'low_stock'
            WHEN current_stock >= max_stock THEN 'overstocked'
            ELSE 'in_stock'
        END
    ) STORED,
    metadata      JSONB DEFAULT '{}',
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);

-- 在庫変動履歴テーブル
CREATE TABLE inventory_movements (
    id           BIGSERIAL PRIMARY KEY,
    product_id   INTEGER NOT NULL REFERENCES inventory_products(id),
    movement_type VARCHAR(20) NOT NULL CHECK (
        movement_type IN ('receipt', 'sale', 'adjustment', 'return')
    ),
    quantity     INTEGER NOT NULL,  -- 正=入庫、負=出庫
    previous_stock INTEGER NOT NULL,
    new_stock    INTEGER NOT NULL,
    details      JSONB DEFAULT '{}',
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- 在庫変動時の通知トリガー
CREATE OR REPLACE FUNCTION notify_low_stock()
RETURNS TRIGGER AS $$
DECLARE
    product RECORD;
    payload JSONB;
BEGIN
    SELECT * INTO product FROM inventory_products WHERE id = NEW.product_id;

    -- 在庫が閾値以下になったら通知
    IF NEW.new_stock <= product.reorder_point AND NEW.previous_stock > product.reorder_point THEN
        payload := jsonb_build_object(
            'event', 'low_stock_alert',
            'product_id', product.id,
            'sku', product.sku,
            'name', product.name,
            'current_stock', NEW.new_stock,
            'reorder_point', product.reorder_point,
            'timestamp', NOW()
        );
        PERFORM pg_notify('inventory_alerts', payload::TEXT);
    END IF;

    -- 在庫切れの場合は緊急通知
    IF NEW.new_stock = 0 AND NEW.previous_stock > 0 THEN
        payload := jsonb_build_object(
            'event', 'out_of_stock_alert',
            'product_id', product.id,
            'sku', product.sku,
            'name', product.name,
            'timestamp', NOW()
        );
        PERFORM pg_notify('inventory_alerts', payload::TEXT);
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_notify_low_stock
    AFTER INSERT ON inventory_movements
    FOR EACH ROW EXECUTE FUNCTION notify_low_stock();

-- 在庫変動を安全に記録する関数
CREATE OR REPLACE FUNCTION update_stock(
    p_product_id INTEGER,
    p_movement_type VARCHAR,
    p_quantity INTEGER,
    p_details JSONB DEFAULT '{}'
) RETURNS INTEGER AS $$
DECLARE
    v_current_stock INTEGER;
    v_new_stock INTEGER;
BEGIN
    -- 行ロックで排他制御
    SELECT current_stock INTO v_current_stock
    FROM inventory_products
    WHERE id = p_product_id
    FOR UPDATE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Product not found: %', p_product_id;
    END IF;

    v_new_stock := v_current_stock + p_quantity;

    IF v_new_stock < 0 THEN
        RAISE EXCEPTION 'Insufficient stock. Current: %, Requested: %',
            v_current_stock, p_quantity;
    END IF;

    -- 在庫数を更新
    UPDATE inventory_products
    SET current_stock = v_new_stock, updated_at = NOW()
    WHERE id = p_product_id;

    -- 変動履歴を記録
    INSERT INTO inventory_movements
        (product_id, movement_type, quantity, previous_stock, new_stock, details)
    VALUES
        (p_product_id, p_movement_type, p_quantity, v_current_stock, v_new_stock, p_details);

    RETURN v_new_stock;
END;
$$ LANGUAGE plpgsql;

-- テストデータ
INSERT INTO inventory_products (sku, name, current_stock, reorder_point) VALUES
('SKU-001', 'ノートPC A', 50, 10),
('SKU-002', 'マウス B', 8, 10),
('SKU-003', 'キーボード C', 0, 5);

-- 在庫操作
SELECT update_stock(1, 'sale', -5, '{"order_id": 1001}');    -- 45個に
SELECT update_stock(1, 'sale', -37, '{"order_id": 1002}');   -- 8個に → 通知発火
SELECT update_stock(1, 'receipt', 20, '{"po_number": "PO-001"}'); -- 28個に

-- 在庫状況ビュー
CREATE VIEW v_inventory_dashboard AS
SELECT
    p.sku,
    p.name,
    p.current_stock,
    p.reorder_point,
    p.stock_status,
    (SELECT COUNT(*) FROM inventory_movements m
     WHERE m.product_id = p.id
       AND m.created_at >= NOW() - INTERVAL '24 hours') AS movements_24h,
    (SELECT SUM(quantity) FROM inventory_movements m
     WHERE m.product_id = p.id
       AND m.movement_type = 'sale'
       AND m.created_at >= NOW() - INTERVAL '7 days') AS sold_7d
FROM inventory_products p
ORDER BY
    CASE p.stock_status
        WHEN 'out_of_stock' THEN 1
        WHEN 'low_stock' THEN 2
        WHEN 'in_stock' THEN 3
        WHEN 'overstocked' THEN 4
    END;

SELECT * FROM v_inventory_dashboard;
```

</details>

---

## FAQ

### Q1: JSONBとMongoDBのどちらを使うべきか？

リレーショナルデータが中心でスキーマレスな部分が一部ならPostgreSQLのJSONBが適切。全データがドキュメント指向で、スキーマが頻繁に変わり、水平スケーリング（シャーディング）が必要ならMongoDBを検討する。

PostgreSQL JSONB の利点:
- トランザクション（ACID）の完全サポート
- リレーショナルデータとのJOIN
- SQL標準のクエリ言語
- 一つのDBで完結（運用コスト低）

MongoDB の利点:
- ネイティブなドキュメントストア（より自然なAPIと集計パイプライン）
- 組み込みの水平シャーディング
- Change Streamsによるリアルタイム処理
- Atlas Search（組み込みElasticsearch相当）

データの80%以上がリレーショナルならPostgreSQL + JSONB、80%以上がドキュメントならMongoDBが適切。詳細は[NoSQL比較](./04-nosql-comparison.md)を参照。

### Q2: pg_trgmとto_tsvectorの違いは？

| 比較項目 | pg_trgm（トライグラム） | to_tsvector（全文検索） |
|---------|----------------------|----------------------|
| 仕組み | 3文字の部分文字列で類似度計算 | 形態素解析で語句のベクトル化 |
| 用途 | タイポ許容・あいまい検索 | 語句の意味的な検索 |
| 日本語対応 | そのまま使える（部分一致） | 別途パーサーが必要（pg_bigm, pgroonga） |
| 演算子 | %, LIKE, ILIKE | @@ |
| ランキング | similarity()で類似度スコア | ts_rank()で関連度スコア |
| インデックス | GIN (gin_trgm_ops) | GIN (tsvector) |
| 推奨場面 | ユーザー名・住所検索 | ドキュメント・記事検索 |

### Q3: PostgreSQLの拡張機能は本番で使っても問題ないか？

contrib拡張（pg_stat_statements、uuid-ossp、pgcryptoなど）はPostgreSQL本体と同じリリースサイクル・品質基準で開発されており、本番利用に全く問題ない。特にpg_stat_statementsはクエリ統計の取得に必須級で、本番環境には必ず導入すべきだ。

サードパーティ拡張の判断基準:
- **PostGIS**: 成熟度20年以上、地理空間のデファクト。安心して利用可能
- **TimescaleDB**: 時系列データ処理の標準。エンタープライズ実績豊富
- **pgvector**: AI/ML向けベクトル検索。2023年以降急速に普及、本番利用事例多数
- **Citus**: 分散PostgreSQL。Microsoft買収後にメンテナンスが安定

注意: AWS RDSやCloud SQLでは利用可能な拡張が制限される。マネージドサービスの対応拡張リストを事前に確認すること。

### Q4: GINインデックスのデフォルトとjsonb_path_opsはどちらを使うべきか？

| 比較項目 | デフォルトGIN | jsonb_path_ops |
|---------|-------------|---------------|
| 対応演算子 | @>, ?, ?&, ?| | @> のみ |
| インデックスサイズ | 大きい | 20-30%小さい |
| 構築速度 | 遅い | 速い |
| 検索速度 | 高速 | @> に限れば最速 |
| 推奨場面 | キー存在確認も必要 | 包含検索のみ使う場合 |

アプリケーションで `?`（キー存在確認）演算子を使うならデフォルトGIN、`@>`（包含検索）のみならjsonb_path_opsが効率的。どちらを使うか迷ったらデフォルトGINを選択すれば問題ない。

### Q5: 配列型のサイズ上限は？パフォーマンスへの影響は？

PostgreSQLの配列型にはサイズ上限の明示的な制限はないが、1つのフィールドは最大1GBまでという一般的な制限がある。ただし、実運用での推奨は以下の通り:

- 要素数が数十以下: 配列型で問題なし
- 要素数が数百: GINインデックスで検索は可能だが、更新時のコスト増加に注意
- 要素数が数千以上: 中間テーブルに移行すべき（GINインデックスの更新コストが高くなる）

配列の更新（array_append等）は配列全体のコピーが発生するため、要素数が多いほどコストが高い。読み取り中心なら配列、書き込み頻繁なら中間テーブルが適切。

---

## まとめ

| 項目 | 要点 |
|------|------|
| JSONB | バイナリJSON。GINインデックスで高速検索。リレーショナル + 可変部分の設計に最適 |
| 配列型 | コレクション格納。GINインデックス対応。少数要素のリストに適する |
| 範囲型 | 期間・区間を直接表現。EXCLUDE制約で重複防止。予約・シフト管理に必須 |
| 全文検索 | TSVECTOR + GINで組み込み全文検索。pg_trgmであいまい検索も可能 |
| ネットワーク型 | INET/CIDRでIPアドレスを直接扱える。サブネット検索にGiST対応 |
| Extension | pg_stat_statementsは本番必須。pgvectorでAI/ML対応も可能 |
| GENERATED COLUMNS | 計算列を自動管理。インデックスも張れる |
| LISTEN/NOTIFY | 軽量なリアルタイム通知。トランザクション連動 |
| テーブル継承 | 新規はDECLARATIVE PARTITIONINGを推奨 |
| 使い分け | リレーショナル基盤 + 可変部分にJSONB。アクセスパターンで型を選択 |

---

## 次に読むべきガイド

- [01-security.md](./01-security.md) — PostgreSQLのセキュリティ設定（RLS、暗号化、監査）
- [02-performance-tuning.md](./02-performance-tuning.md) — パフォーマンスチューニング（EXPLAIN ANALYZE、インデックス設計）
- [03-orm-comparison.md](./03-orm-comparison.md) — ORMからのPostgreSQL固有機能活用
- [04-nosql-comparison.md](./04-nosql-comparison.md) — NoSQLとの比較とポリグロット永続化
- [インデックス設計](../01-advanced/03-indexing.md) — B-Tree、GIN、GiSTインデックスの詳細
- [クエリ最適化](../01-advanced/04-query-optimization.md) — EXPLAIN ANALYZEの読み方と最適化

---

## 参考文献

1. PostgreSQL Documentation — "JSON Types" https://www.postgresql.org/docs/current/datatype-json.html
2. PostgreSQL Documentation — "Additional Supplied Modules" https://www.postgresql.org/docs/current/contrib.html
3. PostgreSQL Documentation — "Range Types" https://www.postgresql.org/docs/current/rangetypes.html
4. PostgreSQL Documentation — "Full Text Search" https://www.postgresql.org/docs/current/textsearch.html
5. PostgreSQL Documentation — "GIN Indexes" https://www.postgresql.org/docs/current/gin.html
6. Schonig, H.-J. (2023). *Mastering PostgreSQL 16*. Packt Publishing.
7. Riggs, S. & Ciolli, G. (2022). *PostgreSQL 14 Administration Cookbook*. Packt Publishing.
