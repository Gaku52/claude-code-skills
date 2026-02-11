# PostgreSQL機能 — JSONB・Array・拡張

> PostgreSQLは「世界で最も先進的なオープンソースリレーショナルデータベース」を標榜し、JSONB、配列型、全文検索、拡張機能など、他のRDBMSにはない豊富な機能を提供する。

## この章で学ぶこと

1. JSONB型の操作、インデックス、パフォーマンス特性
2. 配列型、範囲型、複合型などの高度なデータ型
3. 拡張機能（Extension）によるPostgreSQLの能力拡張

---

## 1. JSONB型

### コード例1: JSONB の基本操作

```sql
CREATE TABLE products (
    id       SERIAL PRIMARY KEY,
    name     VARCHAR(200) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'
);

INSERT INTO products (name, metadata) VALUES
('ノートPC', '{
    "brand": "ThinkPad",
    "specs": {"cpu": "i7", "ram": 16, "ssd": 512},
    "tags": ["business", "lightweight"],
    "price": {"jpy": 198000, "usd": 1350}
}');

-- フィールドアクセス演算子
SELECT
    name,
    metadata->>'brand' AS brand,                  -- TEXT型で取得
    metadata->'specs'->>'cpu' AS cpu,              -- ネスト参照
    (metadata->'specs'->>'ram')::INTEGER AS ram_gb, -- 型キャスト
    metadata->'tags'->0 AS first_tag,             -- 配列の0番目
    metadata#>>'{price,jpy}' AS price_jpy          -- パスで取得
FROM products;

-- JSONB の検索演算子
SELECT * FROM products
WHERE metadata @> '{"brand": "ThinkPad"}';         -- 包含

SELECT * FROM products
WHERE metadata ? 'brand';                           -- キー存在確認

SELECT * FROM products
WHERE metadata ?& ARRAY['brand', 'specs'];          -- 全キー存在

SELECT * FROM products
WHERE metadata ?| ARRAY['discontinued', 'brand'];   -- いずれかのキー存在

-- JSONB の更新
UPDATE products
SET metadata = metadata || '{"color": "black"}'     -- マージ
WHERE id = 1;

UPDATE products
SET metadata = jsonb_set(metadata, '{specs,ram}', '32')  -- 特定パス更新
WHERE id = 1;

UPDATE products
SET metadata = metadata - 'color'                   -- キー削除
WHERE id = 1;
```

### コード例2: JSONB のインデックスと高度な検索

```sql
-- GINインデックス（汎用）
CREATE INDEX idx_products_metadata ON products USING GIN (metadata);

-- GINインデックス（jsonb_path_ops: @>演算子に特化、高速・省メモリ）
CREATE INDEX idx_products_metadata_path
ON products USING GIN (metadata jsonb_path_ops);

-- 式インデックス（特定フィールドのB-Tree）
CREATE INDEX idx_products_brand
ON products ((metadata->>'brand'));

-- jsonpath クエリ（SQL:2016標準）
SELECT * FROM products
WHERE metadata @@ '$.specs.ram > 8';

SELECT jsonb_path_query(metadata, '$.tags[*]') AS tag
FROM products;

-- JSONB集約
SELECT
    metadata->>'brand' AS brand,
    COUNT(*) AS count,
    AVG((metadata#>>'{price,jpy}')::NUMERIC) AS avg_price
FROM products
GROUP BY metadata->>'brand';
```

---

## 2. 配列型

### コード例3: 配列型の操作

```sql
CREATE TABLE posts (
    id    SERIAL PRIMARY KEY,
    title VARCHAR(200),
    tags  TEXT[] NOT NULL DEFAULT '{}'
);

INSERT INTO posts (title, tags) VALUES
('PostgreSQL入門', ARRAY['database', 'postgresql', 'sql']),
('React入門', '{"frontend", "react", "javascript"}');

-- 配列演算子
SELECT * FROM posts WHERE tags @> ARRAY['postgresql'];    -- 含む
SELECT * FROM posts WHERE tags && ARRAY['react', 'sql'];  -- 重複あり
SELECT * FROM posts WHERE 'sql' = ANY(tags);              -- 要素一致
SELECT * FROM posts WHERE tags[1] = 'database';           -- インデックス参照

-- 配列関数
SELECT
    title,
    array_length(tags, 1) AS tag_count,
    array_to_string(tags, ', ') AS tag_list,
    array_position(tags, 'sql') AS sql_position
FROM posts;

-- 配列の展開（UNNEST）
SELECT p.title, t.tag
FROM posts p, UNNEST(p.tags) AS t(tag);

-- GINインデックス
CREATE INDEX idx_posts_tags ON posts USING GIN (tags);

-- 配列の集約
SELECT ARRAY_AGG(DISTINCT tag ORDER BY tag) AS all_tags
FROM posts, UNNEST(tags) AS tag;
```

---

## 3. 範囲型・全文検索・その他の型

### コード例4: 範囲型と全文検索

```sql
-- 範囲型: 予約の重複チェック
CREATE TABLE room_reservations (
    id       SERIAL PRIMARY KEY,
    room_id  INTEGER NOT NULL,
    period   DATERANGE NOT NULL,
    guest    VARCHAR(100),
    EXCLUDE USING GIST (room_id WITH =, period WITH &&)
);

INSERT INTO room_reservations (room_id, period, guest) VALUES
(101, '[2024-03-01, 2024-03-05)', '田中'),
(101, '[2024-03-05, 2024-03-08)', '鈴木');  -- OK: 隣接
-- (101, '[2024-03-04, 2024-03-06)', '佐藤');  -- NG: 重複でエラー

-- 範囲演算子
SELECT * FROM room_reservations
WHERE period @> '2024-03-03'::DATE;  -- 日付が範囲に含まれるか

-- 全文検索
ALTER TABLE posts ADD COLUMN search_vector TSVECTOR;

UPDATE posts SET search_vector =
    to_tsvector('english', title) || to_tsvector('english', COALESCE(body, ''));

CREATE INDEX idx_posts_search ON posts USING GIN (search_vector);

SELECT title, ts_rank(search_vector, query) AS rank
FROM posts, to_tsquery('english', 'postgresql & index') AS query
WHERE search_vector @@ query
ORDER BY rank DESC;

-- INET/CIDR型: IPアドレス
CREATE TABLE access_logs (
    id         BIGSERIAL PRIMARY KEY,
    client_ip  INET NOT NULL,
    accessed_at TIMESTAMPTZ DEFAULT NOW()
);

SELECT * FROM access_logs
WHERE client_ip << '192.168.1.0/24';  -- サブネットに含まれるか
```

### コード例5: 拡張機能（Extension）

```sql
-- 利用可能な拡張を確認
SELECT name, default_version, comment
FROM pg_available_extensions
ORDER BY name;

-- 主要な拡張機能

-- uuid-ossp: UUID生成
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
SELECT uuid_generate_v4();

-- pgcrypto: 暗号化
CREATE EXTENSION IF NOT EXISTS pgcrypto;
SELECT crypt('password123', gen_salt('bf'));

-- pg_trgm: 類似文字列検索（あいまい検索）
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX idx_users_name_trgm ON users USING GIN (name gin_trgm_ops);
SELECT * FROM users WHERE name % 'タナカ';  -- 類似度で検索
SELECT similarity('田中太郎', '田中太朗');   -- → 0.75

-- hstore: キーバリューストア
CREATE EXTENSION IF NOT EXISTS hstore;
CREATE TABLE settings (
    user_id INTEGER PRIMARY KEY,
    prefs   HSTORE
);
INSERT INTO settings VALUES (1, 'theme => dark, lang => ja');

-- PostGIS: 地理空間（別途インストール必要）
-- CREATE EXTENSION IF NOT EXISTS postgis;
-- SELECT ST_Distance(
--     ST_GeomFromText('POINT(139.7671 35.6812)', 4326),  -- 東京駅
--     ST_GeomFromText('POINT(135.5023 34.6937)', 4326)   -- 大阪駅
-- );

-- pg_stat_statements: クエリ統計
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
SELECT query, calls, mean_exec_time, rows
FROM pg_stat_statements ORDER BY total_exec_time DESC LIMIT 10;
```

### コード例6: 高度なPostgreSQL機能

```sql
-- GENERATED COLUMNS（計算列）
CREATE TABLE orders (
    id          SERIAL PRIMARY KEY,
    quantity    INTEGER NOT NULL,
    unit_price  DECIMAL(10, 2) NOT NULL,
    tax_rate    DECIMAL(3, 2) NOT NULL DEFAULT 0.10,
    subtotal    DECIMAL(12, 2) GENERATED ALWAYS AS (quantity * unit_price) STORED,
    tax_amount  DECIMAL(12, 2) GENERATED ALWAYS AS
                (quantity * unit_price * tax_rate) STORED,
    total       DECIMAL(12, 2) GENERATED ALWAYS AS
                (quantity * unit_price * (1 + tax_rate)) STORED
);

-- LISTEN / NOTIFY（リアルタイム通知）
-- セッション1:
LISTEN order_created;

-- セッション2:
NOTIFY order_created, '{"order_id": 42, "amount": 5000}';

-- トリガーで自動通知
CREATE OR REPLACE FUNCTION notify_order() RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify('order_created', json_build_object(
        'id', NEW.id, 'amount', NEW.total
    )::TEXT);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- テーブル継承（Table Inheritance）
CREATE TABLE events (
    id         BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE click_events (
    element_id VARCHAR(100),
    page_url   VARCHAR(500)
) INHERITS (events);

CREATE TABLE purchase_events (
    product_id INTEGER,
    amount     DECIMAL(10, 2)
) INHERITS (events);
```

### PostgreSQLのデータ型体系

```
┌──────── PostgreSQL の特殊データ型 ────────────────┐
│                                                    │
│  構造化データ                                       │
│  ├── JSONB      → ドキュメント格納、インデックス可  │
│  ├── JSON       → テキスト格納（非推奨）           │
│  ├── HSTORE     → 単純KVS                         │
│  └── XML        → XMLドキュメント                  │
│                                                    │
│  コレクション                                       │
│  ├── ARRAY      → 配列型（全型で使用可）           │
│  └── COMPOSITE  → 複合型（行型）                   │
│                                                    │
│  範囲・空間                                         │
│  ├── INT4RANGE  → 整数範囲                         │
│  ├── DATERANGE  → 日付範囲                         │
│  ├── TSRANGE    → タイムスタンプ範囲               │
│  ├── POINT      → 2D座標                           │
│  └── PostGIS    → 地理空間（拡張）                 │
│                                                    │
│  ネットワーク                                       │
│  ├── INET       → IPアドレス                       │
│  ├── CIDR       → ネットワークアドレス             │
│  └── MACADDR    → MACアドレス                      │
│                                                    │
│  その他                                             │
│  ├── UUID       → 128ビット一意識別子              │
│  ├── TSVECTOR   → 全文検索用ベクトル               │
│  └── BIT/VARBIT → ビット列                         │
└────────────────────────────────────────────────────┘
```

---

## JSON vs JSONB 比較表

| 特徴 | JSON | JSONB |
|------|------|-------|
| 格納形式 | テキスト（入力そのまま） | バイナリ（パース済み） |
| 挿入速度 | 高速（変換不要） | やや遅い（パースが必要） |
| 読み取り速度 | 遅い（毎回パース） | 高速（パース済み） |
| インデックス | 不可 | GIN対応 |
| 重複キー | 保持 | 最後のキーのみ |
| 順序保持 | 保持 | 保持しない |
| 演算子 | 少ない | 豊富（@>, ?, @@） |
| 推奨度 | 非推奨 | 推奨 |

## 主要Extension 比較表

| Extension | 用途 | サイズ影響 | 推奨度 |
|-----------|------|----------|--------|
| pg_stat_statements | クエリ統計 | 極小 | 必須 |
| uuid-ossp / pgcrypto | UUID/暗号 | なし | 高 |
| pg_trgm | あいまい検索 | インデックス依存 | 高 |
| PostGIS | 地理空間 | 大 | 用途次第 |
| hstore | KVS | 小 | JSONBで代替可 |
| pg_partman | パーティション自動管理 | 小 | 高 |
| timescaledb | 時系列データ | 中 | 時系列なら必須 |

---

## アンチパターン

### アンチパターン1: リレーショナルデータをJSONBに格納

```sql
-- NG: 本来テーブルで管理すべきデータをJSONBに
CREATE TABLE orders (
    id   SERIAL PRIMARY KEY,
    data JSONB  -- 顧客情報、商品情報、全てJSON
);
-- → 制約なし、JOIN不可、型安全性なし

-- OK: リレーショナルデータはテーブル、可変部分のみJSONB
CREATE TABLE orders (
    id          SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    total       DECIMAL(12,2) NOT NULL,
    metadata    JSONB DEFAULT '{}'  -- メモ、タグ等の可変情報のみ
);
```

### アンチパターン2: 配列型でM:N関係を表現

```sql
-- NG: 配列でM:N関係
CREATE TABLE articles (
    id       SERIAL PRIMARY KEY,
    title    VARCHAR(200),
    tag_ids  INTEGER[]  -- FK制約が効かない！
);

-- OK: 中間テーブルでM:N関係
CREATE TABLE article_tags (
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    tag_id     INTEGER REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (article_id, tag_id)
);
-- → 外部キー制約、カスケード削除、JOIN最適化が全て可能
```

---

## FAQ

### Q1: JSONBとMongoDBのどちらを使うべきか？

リレーショナルデータが中心でスキーマレスな部分が一部ならPostgreSQLのJSONBが適切。全データがドキュメント指向で、スキーマが頻繁に変わり、分散が必要ならMongoDBを検討する。PostgreSQLのJSONBはトランザクション、JOIN、SQLとの統合という利点がある。

### Q2: pg_trgmとto_tsvectorの違いは？

pg_trgmはトライグラム（3文字の部分文字列）ベースの類似度検索で、タイポ許容やあいまい検索に適する。to_tsvectorは形態素解析ベースの全文検索で、語句の意味的な検索に適する。日本語にはpg_bigmやMeCab辞書との組み合わせが必要。

### Q3: PostgreSQLの拡張機能は本番で使っても問題ないか？

pg_stat_statements、uuid-ossp、pgcryptoなどのcontrib拡張はPostgreSQL本体と同じ品質で提供されており、本番利用に問題ない。サードパーティ拡張（PostGIS、TimescaleDBなど）もコミュニティが大きく成熟しているものは問題ない。

---

## まとめ

| 項目 | 要点 |
|------|------|
| JSONB | バイナリJSON。GINインデックスで高速検索 |
| 配列型 | コレクション格納。GINインデックス対応 |
| 範囲型 | 期間の重複チェックにEXCLUDE制約 |
| 全文検索 | TSVECTOR + GINで組み込み全文検索 |
| Extension | pg_stat_statements は必須。用途に応じて追加 |
| 使い分け | リレーショナル基盤 + 可変部分にJSONB |

---

## 次に読むべきガイド

- [01-security.md](./01-security.md) — PostgreSQLのセキュリティ設定
- [02-performance-tuning.md](./02-performance-tuning.md) — パフォーマンスチューニング
- [03-orm-comparison.md](./03-orm-comparison.md) — ORMからのPostgreSQL活用

---

## 参考文献

1. PostgreSQL Documentation — "JSON Types" https://www.postgresql.org/docs/current/datatype-json.html
2. PostgreSQL Documentation — "Additional Supplied Modules" https://www.postgresql.org/docs/current/contrib.html
3. Schonig, H.-J. (2023). *Mastering PostgreSQL 16*. Packt Publishing.
