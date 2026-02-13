# インデックス — B-Tree・GiST・GIN・部分インデックス・カバリングインデックス

> B-Tree、GiST、GIN、BRIN、部分インデックス、カバリングインデックスなど、データベースインデックスの内部構造と設計戦略を理解し、クエリ性能を最適化する。本章では、インデックスが「なぜ速いのか」を内部実装レベルで理解し、適切なインデックスを選択・設計するための判断基準を体系的に習得する。

---

## この章で学ぶこと

1. **インデックスの内部構造** — B-Treeの仕組み、ページ分割、検索アルゴリズム、計算量の理論的根拠
2. **特殊インデックス** — GiST（空間検索）、GIN（全文検索・JSONB）、BRIN（大規模時系列データ）、Hash、SP-GiST
3. **インデックス設計戦略** — 部分インデックス、カバリングインデックス、複合インデックスの列順序最適化
4. **インデックスの保守と監視** — 膨張検出、REINDEX、未使用インデックスの特定と削除

---

## 前提知識

| トピック | 内容 | 参照先 |
|---------|------|--------|
| SQL基礎 | SELECT/WHERE/JOIN の基本構文 | [00-basics/](../00-basics/) |
| EXPLAIN | 実行計画の基本的な読み方 | [04-query-optimization.md](./04-query-optimization.md) |
| テーブル設計 | PRIMARY KEY、FOREIGN KEY、制約 | [01-schema-design.md](../02-design/01-schema-design.md) |
| トランザクション | ロックの基本概念 | [02-transactions.md](./02-transactions.md) |

---

## 1. B-Tree インデックスの内部構造

### なぜインデックスが速いのか

テーブルの全行を順番に読む（Sequential Scan）場合、100万行のテーブルから1行を探すには平均50万行の読み取りが必要（O(N)）。B-Treeインデックスを使えば、わずか3-4回のページ読み取りで目的の行に到達できる（O(log N)）。

```
B-Tree の構造 (次数=3 の例)
==============================

           [30 | 60]              <-- ルートノード (1ページ = 8KB)
          /    |    \
   [10|20]  [40|50]  [70|80]     <-- 内部ノード
   / | \    / | \    / | \
  L  L  L  L  L  L  L  L  L     <-- リーフノード (実データへのポインタ)

検索: WHERE id = 45
  1. ルート: 30 < 45 < 60 --> 中央の子へ     (1回目のI/O)
  2. 内部: 40 < 45 < 50  --> 中央の子へ       (2回目のI/O)
  3. リーフ: id=45 のTID(ページ,オフセット)取得 (3回目のI/O)
  4. テーブル: TIDでヒープタプルを直接取得      (4回目のI/O)

計算量: O(log N)
  N = 1,000,000行 → log_500(1,000,000) ≈ 2.2 → 約3ページ読み取り
  N = 100,000,000行 → log_500(100,000,000) ≈ 2.9 → 約3-4ページ読み取り

  ※ 次数(1ページに入るキー数)が大きいほど木の高さが低くなる
  ※ 8KBページにINTEGER(4bytes)なら約500キー格納可能
  ※ ルートと上位内部ノードはバッファキャッシュに載るため実I/Oは少ない
```

### B-Treeの特性

```
┌─────────── B-Treeの重要な特性 ───────────────────┐
│                                                    │
│  1. バランス木                                     │
│     全てのリーフノードが同じ深さにある              │
│     → 最悪ケースでもO(log N)を保証                 │
│                                                    │
│  2. ソート済み                                     │
│     リーフノードが双方向リンクリストで接続           │
│     → 範囲検索（BETWEEN, <, >）が効率的            │
│     → ORDER BYもインデックスで解決可能              │
│                                                    │
│  3. ページ分割                                     │
│     ノードが満杯になるとページ分割が発生            │
│     → INSERT時のオーバーヘッド                     │
│     → 単調増加キー(SERIAL)では右端のみ分割          │
│       → 分割コストが低い                           │
│                                                    │
│  4. 対応する演算子                                 │
│     =, <, >, <=, >=, BETWEEN                       │
│     IN (値リスト)                                   │
│     LIKE 'abc%'（前方一致のみ）                    │
│     IS NULL / IS NOT NULL                           │
│     ORDER BY, GROUP BY                              │
│     MIN(), MAX()（インデックスの端を直接読み取り）   │
└────────────────────────────────────────────────────┘
```

### コード例1: 基本的なインデックス作成

```sql
-- テスト用テーブル
CREATE TABLE users (
    id         SERIAL PRIMARY KEY,
    email      VARCHAR(255) NOT NULL,
    username   VARCHAR(50) NOT NULL,
    status     VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE orders (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id),
    status      VARCHAR(20) NOT NULL DEFAULT 'pending',
    total       DECIMAL(10, 2) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- B-Tree インデックス（デフォルト）
CREATE INDEX idx_users_email ON users (email);

-- ユニークインデックス（重複不可制約 + インデックスの2役）
CREATE UNIQUE INDEX idx_users_email_unique ON users (email);

-- 複合インデックス（列の順序が重要 — 後述）
CREATE INDEX idx_orders_user_date ON orders (user_id, created_at DESC);

-- 降順インデックス
CREATE INDEX idx_orders_recent ON orders (created_at DESC);

-- 式インデックス（関数の結果にインデックス）
CREATE INDEX idx_users_email_lower ON users (LOWER(email));
-- → WHERE LOWER(email) = 'test@example.com' で使用される

-- インデックスの確認
SELECT
    indexname,
    indexdef,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS size
FROM pg_indexes
WHERE tablename = 'orders';
```

### コード例2: EXPLAIN ANALYZEによる効果測定

```sql
-- インデックスなし（Sequential Scan）
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT * FROM orders WHERE user_id = 12345;
-- Seq Scan on orders  (cost=0.00..25000.00 rows=50 width=120)
--   actual time=45.2..180.5 rows=50 loops=1
--   Filter: (user_id = 12345)
--   Rows Removed by Filter: 999950
--   Buffers: shared hit=15000 read=10000
-- Planning Time: 0.1 ms
-- Execution Time: 180.6 ms

-- インデックス作成
CREATE INDEX idx_orders_user_id ON orders (user_id);

-- インデックスあり（Index Scan）
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT * FROM orders WHERE user_id = 12345;
-- Index Scan using idx_orders_user_id on orders
--   (cost=0.43..8.50 rows=50 width=120)
--   actual time=0.03..0.15 rows=50 loops=1
--   Index Cond: (user_id = 12345)
--   Buffers: shared hit=5
-- Planning Time: 0.2 ms
-- Execution Time: 0.2 ms

-- 結果比較:
-- Sequential Scan: 180.6 ms, 25000 buffers
-- Index Scan:      0.2 ms,   5 buffers
-- → 約900倍の高速化、I/Oが約5000分の1に削減
```

---

## 2. 複合インデックスの列順序

### なぜ列順序が重要なのか

複合インデックスは電話帳のようなものである。電話帳が「姓→名」の順にソートされている場合、「田中」で始まる人は簡単に見つかるが、名前が「太郎」の人を姓に関係なく検索するには全ページを見る必要がある。

```
複合インデックスの動作原理（左端プレフィックスルール）
========================================================

CREATE INDEX idx ON orders (user_id, status, created_at);

インデックスの内部ソート順（辞書順）:
  user_id=1, status='active',   created_at='2024-01-01'
  user_id=1, status='active',   created_at='2024-01-15'
  user_id=1, status='shipped',  created_at='2024-01-10'
  user_id=2, status='active',   created_at='2024-01-05'
  user_id=2, status='pending',  created_at='2024-01-20'

利用可能なクエリパターン:
  [OK] WHERE user_id = 1                         (左端1列)
  [OK] WHERE user_id = 1 AND status = 'active'   (左端2列)
  [OK] WHERE user_id = 1 AND status = 'active'
       AND created_at > '2024-01-01'              (全3列)
  [OK] WHERE user_id = 1 ORDER BY status          (プレフィックス + ソート)

  [NG] WHERE status = 'active'                    (左端スキップ)
  [NG] WHERE created_at > '2024-01-01'            (左端スキップ)
  [NG] WHERE user_id = 1 AND created_at > '...'   (status をスキップ)
       → user_id=1 までは利用、created_atは Filter として適用

列順序の設計原則:
  1. 等値条件 (=) の列を先に
  2. 範囲条件 (<, >, BETWEEN) の列を後に
  3. 選択率が高い（値の種類が多い）列を先に
  4. ORDER BY の列を最後に

例: WHERE user_id = ? AND status = ? AND created_at > ?
    → (user_id, status, created_at) が最適
```

### コード例3: 複合インデックスの効果測定

```sql
-- 悪い列順序
CREATE INDEX idx_orders_bad ON orders (created_at, user_id, status);

-- 良い列順序（等値条件を先に、範囲条件を後に）
CREATE INDEX idx_orders_good ON orders (user_id, status, created_at);

-- テストクエリ
EXPLAIN ANALYZE
SELECT * FROM orders
WHERE user_id = 42
  AND status = 'shipped'
  AND created_at > '2024-01-01';

-- idx_orders_bad の場合:
-- Index Scan using idx_orders_bad on orders
--   Index Cond: (created_at > '2024-01-01')
--   Filter: (user_id = 42 AND status = 'shipped')
--   Rows Removed by Filter: 4500  ← 多くの行をフィルタで除外
--   → created_at でしか絞れない

-- idx_orders_good の場合:
-- Index Scan using idx_orders_good on orders
--   Index Cond: (user_id = 42 AND status = 'shipped'
--                AND created_at > '2024-01-01')
--   Rows Removed by Filter: 0     ← 全条件がインデックスで処理
--   → 3列全てが Index Cond として使用される
```

---

## 3. 特殊インデックス

### インデックス種類比較表

| インデックス型 | 対応演算子 | ユースケース | サイズ | 構築速度 |
|---|---|---|---|---|
| **B-Tree** | =, <, >, BETWEEN, LIKE 'abc%' | 汎用、範囲検索、ソート | 中 | 速い |
| **Hash** | = のみ | 等値検索のみ（B-Treeで代替可能） | 小 | 速い |
| **GiST** | 包含(&&)、重複、近傍(<->) | 地理空間、範囲型、排他制約 | 大 | 遅い |
| **GIN** | 含む(@>)、配列要素、全文検索(@@) | 全文検索、JSONB、配列 | 大 | 遅い |
| **BRIN** | =, <, >, BETWEEN | 時系列、物理ソート済み大テーブル | 極小 | 極速 |
| **SP-GiST** | パーティション検索 | 電話番号、IPアドレス、四分木 | 中 | 中 |

### コード例4: GINインデックス（JSONB・全文検索・配列）

```sql
-- ===== JSONB検索用GINインデックス =====
CREATE TABLE products (
    id          SERIAL PRIMARY KEY,
    name        VARCHAR(200) NOT NULL,
    price       DECIMAL(10, 2) NOT NULL,
    attributes  JSONB DEFAULT '{}'
);

-- GINインデックスの作成
-- jsonb_ops: @>, ?, ?&, ?| をサポート（デフォルト）
CREATE INDEX idx_products_attrs ON products USING GIN (attributes);

-- jsonb_path_ops: @> のみだがサイズが小さく高速
CREATE INDEX idx_products_attrs_path ON products
    USING GIN (attributes jsonb_path_ops);

-- JSONB の検索（GINインデックスが使用される）
SELECT * FROM products
WHERE attributes @> '{"color": "red", "size": "L"}';

-- キーの存在確認
SELECT * FROM products
WHERE attributes ? 'wireless';  -- 'wireless'キーが存在するか

-- 複数キーの存在確認
SELECT * FROM products
WHERE attributes ?& ARRAY['color', 'size'];  -- 両方のキーが存在


-- ===== 全文検索用GINインデックス =====
CREATE TABLE articles (
    id            SERIAL PRIMARY KEY,
    title         VARCHAR(500) NOT NULL,
    body          TEXT NOT NULL,
    search_vector tsvector  -- 全文検索用の事前計算カラム
);

-- tsvectorカラムの更新トリガー
CREATE OR REPLACE FUNCTION update_search_vector() RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector :=
        setweight(to_tsvector('english', COALESCE(NEW.title, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.body, '')), 'B');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_articles_search
BEFORE INSERT OR UPDATE ON articles
FOR EACH ROW EXECUTE FUNCTION update_search_vector();

-- GINインデックスの作成
CREATE INDEX idx_articles_search ON articles USING GIN (search_vector);

-- 全文検索クエリ
SELECT title, ts_rank(search_vector, query) AS rank
FROM articles, to_tsquery('english', 'PostgreSQL & index') AS query
WHERE search_vector @@ query
ORDER BY rank DESC
LIMIT 10;


-- ===== 配列検索用GINインデックス =====
CREATE TABLE posts (
    id    SERIAL PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    tags  TEXT[] DEFAULT '{}'
);

CREATE INDEX idx_posts_tags ON posts USING GIN (tags);

-- 配列の包含検索
SELECT * FROM posts WHERE tags @> ARRAY['rust', 'wasm'];
-- → tagsに'rust'と'wasm'の両方を含む投稿

-- 配列の重複検索
SELECT * FROM posts WHERE tags && ARRAY['python', 'javascript'];
-- → tagsに'python'または'javascript'のいずれかを含む投稿
```

### コード例5: GiSTインデックス（空間検索・範囲型）

```sql
-- ===== PostGIS: 空間インデックス =====
-- PostGIS拡張のインストール
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE stores (
    id        SERIAL PRIMARY KEY,
    name      VARCHAR(200) NOT NULL,
    location  GEOGRAPHY(POINT, 4326) NOT NULL  -- WGS84座標系
);

-- GiSTインデックスの作成
CREATE INDEX idx_stores_location ON stores USING GiST (location);

-- 半径5km以内の店舗検索
SELECT
    name,
    ST_Distance(
        location,
        ST_SetSRID(ST_MakePoint(139.7671, 35.6812), 4326)::geography
    ) AS distance_m
FROM stores
WHERE ST_DWithin(
    location,
    ST_SetSRID(ST_MakePoint(139.7671, 35.6812), 4326)::geography,
    5000  -- 5000m = 5km
)
ORDER BY distance_m;

-- k近傍検索（最も近い10店舗）
SELECT name, location <-> ST_SetSRID(ST_MakePoint(139.7671, 35.6812), 4326)::geography AS dist
FROM stores
ORDER BY location <-> ST_SetSRID(ST_MakePoint(139.7671, 35.6812), 4326)::geography
LIMIT 10;


-- ===== 範囲型の重複検索（予約管理） =====
CREATE TABLE reservations (
    id        SERIAL PRIMARY KEY,
    room_id   INTEGER NOT NULL,
    guest     VARCHAR(100) NOT NULL,
    check_in  DATE NOT NULL,
    check_out DATE NOT NULL
);

-- 範囲型のGiSTインデックス
CREATE INDEX idx_reservations_period ON reservations
    USING GiST (room_id, daterange(check_in, check_out));

-- 期間が重複する予約の検索
SELECT * FROM reservations
WHERE room_id = 101
  AND daterange(check_in, check_out) && daterange('2024-03-01', '2024-03-10');
-- → 3月1日〜10日と重複する予約をインデックスで効率的に検索


-- ===== 排他制約（同じ部屋の予約重複を禁止） =====
CREATE EXTENSION IF NOT EXISTS btree_gist;  -- 排他制約にB-Tree演算子を使うため必要

ALTER TABLE reservations ADD CONSTRAINT excl_room_period
    EXCLUDE USING GiST (
        room_id WITH =,
        daterange(check_in, check_out) WITH &&
    );
-- → 同じroom_idで期間が重複するINSERT/UPDATEを自動的に拒否
```

### コード例6: BRINインデックス（大規模時系列データ）

```sql
-- BRINインデックスは物理的に順序付けされたデータに最適
-- 例: ログテーブル（created_atが挿入順に増加）
CREATE TABLE access_logs (
    id         BIGSERIAL PRIMARY KEY,
    user_id    INTEGER,
    action     VARCHAR(50),
    ip_address INET,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- BRINインデックスの作成
-- pages_per_range: 何ページごとに要約情報を保持するか（デフォルト128）
CREATE INDEX idx_logs_created_brin ON access_logs
    USING BRIN (created_at)
    WITH (pages_per_range = 32);

-- サイズ比較（1億行のテーブルの場合の目安）:
-- B-Tree: 約2GB
-- BRIN:   約200KB（B-Treeの約1/10000）

-- BRINが効果的なクエリ
SELECT COUNT(*) FROM access_logs
WHERE created_at BETWEEN '2024-01-01' AND '2024-01-31';
-- → BRINが該当ページブロックだけをスキャン

-- BRINが効果的でないクエリ
SELECT * FROM access_logs WHERE user_id = 42;
-- → user_idは物理的な順序と相関がないため、BRINは効果なし

-- BRINの内部動作
-- 128ページごとに min_val, max_val を保持
-- WHERE created_at > '2024-06-01' の場合:
--   Block 0-127:   min=2024-01, max=2024-02 → スキップ
--   Block 128-255: min=2024-03, max=2024-04 → スキップ
--   Block 256-383: min=2024-05, max=2024-06 → スキャン（該当の可能性あり）
--   Block 384-511: min=2024-07, max=2024-08 → スキャン
```

---

## 4. 部分インデックスとカバリングインデックス

### コード例7: 部分インデックス（Partial Index）

```sql
-- 部分インデックス: WHERE句付きのインデックス
-- テーブルの一部の行だけにインデックスを作成する

-- アクティブユーザーのみインデックス（全体の10%）
CREATE INDEX idx_users_active_email ON users (email)
WHERE status = 'active';
-- → フルインデックスの約1/10のサイズ
-- → INSERT/UPDATE時のオーバーヘッドも約1/10

-- 未処理注文のみ（全体の5%）
CREATE INDEX idx_orders_pending ON orders (created_at)
WHERE status = 'pending';
-- 使用されるクエリ:
SELECT * FROM orders WHERE status = 'pending' ORDER BY created_at;

-- NULLでないカラムのみ（90%がNULLの場合）
CREATE INDEX idx_users_phone ON users (phone)
WHERE phone IS NOT NULL;

-- 条件付き一意制約（部分ユニークインデックス）
-- アクティブなユーザーの中でのみメールアドレスの一意性を保証
CREATE UNIQUE INDEX idx_users_active_email_unique ON users (email)
WHERE deleted_at IS NULL;
-- → 論理削除されたユーザーは同じメールアドレスを持てる

-- 部分インデックスの効果を確認
SELECT
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS size
FROM pg_indexes
WHERE tablename = 'users';
-- idx_users_email:        120 MB  ← フルインデックス
-- idx_users_active_email:  12 MB  ← 部分インデックス（1/10）
```

### コード例8: カバリングインデックス（INCLUDE）

```sql
-- カバリングインデックス: テーブルアクセスを完全に回避
-- INCLUDE句でインデックスに追加カラムを格納（検索キーとしては使わない）

CREATE INDEX idx_orders_covering ON orders (user_id, status)
INCLUDE (total, created_at);

-- このクエリは Index Only Scan で応答（テーブル読み取り不要）
EXPLAIN ANALYZE
SELECT user_id, status, total, created_at
FROM orders
WHERE user_id = 12345 AND status = 'shipped';

-- 出力:
-- Index Only Scan using idx_orders_covering on orders
--   (cost=0.43..5.50 rows=10 width=40)
--   actual time=0.02..0.05 rows=10 loops=1
--   Index Cond: (user_id = 12345 AND status = 'shipped')
--   Heap Fetches: 0  <-- テーブルアクセスなし！

-- INCLUDEとキー列の違い:
-- キー列: 検索条件として使用可能、ソートされる、B-Treeのノードに格納
-- INCLUDE列: 検索条件には使用不可、リーフノードにのみ格納
-- → INCLUDEの方がインデックスサイズが小さい（内部ノードに影響しない）

-- INCLUDEなしで同等の効果を得る方法（非推奨）
CREATE INDEX idx_orders_covering_old ON orders (user_id, status, total, created_at);
-- → 全列がキーになるためインデックスが大きくなり、更新コストも増加
```

### コード例9: 式インデックス（Expression Index）

```sql
-- 関数の結果にインデックスを作成
-- 大文字小文字を区別しないメール検索
CREATE INDEX idx_users_email_lower ON users (LOWER(email));

-- クエリでも同じ式を使う必要がある
SELECT * FROM users WHERE LOWER(email) = 'test@example.com';
-- → Index Scan using idx_users_email_lower

-- NG: 式が一致しないとインデックスが使われない
SELECT * FROM users WHERE email = 'test@example.com';
-- → Seq Scan（LOWER()がないため）

-- JSONBの特定キーにインデックス
CREATE INDEX idx_products_color ON products ((attributes->>'color'));
SELECT * FROM products WHERE attributes->>'color' = 'red';

-- 日付の年月部分にインデックス
CREATE INDEX idx_orders_yearmonth ON orders (DATE_TRUNC('month', created_at));
SELECT * FROM orders WHERE DATE_TRUNC('month', created_at) = '2024-06-01';

-- テキストの先頭N文字にインデックス（前方一致検索の高速化）
CREATE INDEX idx_users_name_prefix ON users (LEFT(username, 3));
```

---

## 5. インデックスの保守と監視

### インデックス肥大化の仕組み

```
インデックス肥大化の仕組み
============================

DELETE/UPDATE が繰り返されると:

初期状態:  [1][2][3][4][5][6][7][8]  (ページ使用率 100%)
           ↓ DELETE 3,5,7
中間状態:  [1][2][ ][4][ ][6][ ][8]  (ページ使用率 62.5%)
           ↓ 新しいINSERTの値が空きに入らない場合
肥大化:    多数の空きページが残存 --> インデックスサイズ増大

            ページフィルファクター (fillfactor):
            デフォルト90%。UPDATE頻度が高いテーブルでは
            70-80%に下げると、HOT Update（同一ページ内更新）
            の確率が上がり、インデックス更新を回避できる

対策:
  1. REINDEX: インデックスを最初から再構築
  2. pg_repack: オンラインでテーブル/インデックスを再編成
  3. fillfactorの調整: 更新頻度に応じて設定
```

### コード例10: インデックスの監視と保守

```sql
-- ===== インデックスサイズの確認 =====
SELECT
    schemaname || '.' || tablename AS table_name,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
    idx_scan AS total_scans,           -- このインデックスが使われた回数
    idx_tup_read AS tuples_read,       -- インデックスから読まれたタプル数
    idx_tup_fetch AS tuples_fetched    -- テーブルから取得されたタプル数
FROM pg_stat_user_indexes
JOIN pg_indexes USING (indexname)
ORDER BY pg_relation_size(indexrelid) DESC
LIMIT 20;


-- ===== 未使用インデックスの検出 =====
SELECT
    indexrelname AS index_name,
    relname AS table_name,
    idx_scan AS scans,
    pg_size_pretty(pg_relation_size(indexrelid)) AS size,
    pg_size_pretty(pg_total_relation_size(relid)) AS table_size
FROM pg_stat_user_indexes
JOIN pg_statio_user_indexes USING (indexrelid)
WHERE idx_scan = 0
  AND indexrelid NOT IN (
      SELECT indexrelid FROM pg_constraint
      WHERE contype IN ('p', 'u')  -- PRIMARY KEY, UNIQUE制約は除外
  )
ORDER BY pg_relation_size(indexrelid) DESC;

-- 注意: 統計情報はpg_stat_reset()で初期化される
-- レプリカではインデックスが使われないため、プライマリで確認すること


-- ===== 重複インデックスの検出 =====
-- 同じ列の組み合わせで作られたインデックスを特定
SELECT
    a.indexrelid::regclass AS index1,
    b.indexrelid::regclass AS index2,
    a.indrelid::regclass AS table_name
FROM pg_index a
JOIN pg_index b ON a.indrelid = b.indrelid
    AND a.indexrelid < b.indexrelid
    AND a.indkey::text = b.indkey::text;


-- ===== REINDEX（オンライン再構築、PostgreSQL 12+）=====
-- 通常のREINDEX: テーブルへの書き込みをロック
REINDEX INDEX idx_orders_user_date;

-- CONCURRENTLY: ロックなしで再構築（推奨）
REINDEX INDEX CONCURRENTLY idx_orders_user_date;

-- テーブルの全インデックスを再構築
REINDEX TABLE CONCURRENTLY orders;


-- ===== インデックス膨張率の推定 =====
-- pgstattupleエクステンションを使用
CREATE EXTENSION IF NOT EXISTS pgstattuple;

SELECT
    indexrelid::regclass AS index_name,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
    leaf_fragmentation
FROM pgstatindex('idx_orders_user_date');
-- leaf_fragmentation > 30% なら REINDEX を検討
```

---

## 6. インデックスの落とし穴

### コード例11: インデックスが使われないケース

```sql
-- ===== ケース1: 関数を適用している =====
-- NG: インデックスは email カラムに存在するが関数で包まれている
SELECT * FROM users WHERE UPPER(email) = 'TEST@EXAMPLE.COM';
-- → Seq Scan（式インデックスが必要）

-- OK: 式インデックスを作成するか、アプリ側で正規化
CREATE INDEX idx_users_email_upper ON users (UPPER(email));


-- ===== ケース2: 暗黙の型変換 =====
-- NG: user_idがINTEGERなのにTEXTで検索
SELECT * FROM orders WHERE user_id = '12345';
-- PostgreSQLでは暗黙変換で動作するが、他のDBでは問題になることがある

-- NG: VARCHARカラムにINTEGERで検索
SELECT * FROM products WHERE sku = 12345;
-- → 型が不一致でインデックスが使われない可能性


-- ===== ケース3: 選択率が低い（大部分の行が該当） =====
-- NG: status='active' が全体の90%を占める場合
SELECT * FROM users WHERE status = 'active';
-- → オプティマイザがSeq Scanを選択（インデックス経由より全走査が速い）

-- OK: 部分インデックスにして、少数派の条件で使用
CREATE INDEX idx_users_inactive ON users (email) WHERE status = 'inactive';


-- ===== ケース4: LIKE '%中間%'（中間一致・後方一致） =====
-- NG: 前方以外のLIKEはB-Treeで使えない
SELECT * FROM users WHERE username LIKE '%田中%';
-- → Seq Scan

-- OK: pg_trgm拡張でトライグラムインデックス
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX idx_users_name_trgm ON users USING GIN (username gin_trgm_ops);
SELECT * FROM users WHERE username LIKE '%田中%';
-- → GIN Scanで高速検索

-- OK: 全文検索（tsvector + GIN）
-- → 前述のコード例4参照


-- ===== ケース5: OR条件 =====
-- NG: ORでインデックスが効きにくい
SELECT * FROM orders WHERE user_id = 42 OR status = 'pending';
-- → Bitmap OR（各インデックスのビットマップを論理和）
-- → またはSeq Scan

-- OK: UNION ALLに書き換え
SELECT * FROM orders WHERE user_id = 42
UNION ALL
SELECT * FROM orders WHERE status = 'pending' AND user_id != 42;


-- ===== ケース6: NULL の扱い =====
-- B-TreeインデックスはIS NULLとIS NOT NULLに対応
SELECT * FROM users WHERE deleted_at IS NULL;
-- → Index Scanが使用可能（PostgreSQL 8.3+）

-- ただし、NULLが大多数の場合は部分インデックスが効果的
CREATE INDEX idx_users_deleted ON users (deleted_at)
WHERE deleted_at IS NOT NULL;
```

---

## インデックス選択フロー比較表

| 条件 | 推奨インデックス | 理由 |
|---|---|---|
| 等値/範囲検索（汎用） | B-Tree | 最も汎用的、デフォルト |
| JSONBの検索 (@>) | GIN (jsonb_ops / jsonb_path_ops) | @>, ?, ?& 演算子に対応 |
| 全文検索 (@@) | GIN (tsvector) | トークンの逆引きインデックス |
| 地理空間検索 | GiST (PostGIS) | 空間的な包含・近傍クエリ |
| 時系列データ（物理ソート済み） | BRIN | 極小サイズで効率的 |
| 等値検索のみ（高頻度） | Hash | B-Treeより若干高速（差は小さい） |
| 大テーブルの少数レコード検索 | 部分インデックス | サイズと保守コストを削減 |
| テーブルアクセス不要の参照 | カバリングインデックス (INCLUDE) | Index Only Scan を実現 |
| 中間一致 (LIKE '%xxx%') | GIN (pg_trgm) | トライグラムによる部分文字列検索 |
| 期間の重複検出 | GiST (範囲型) | && 演算子に対応、排他制約も可能 |

## B-Tree vs 特殊インデックス 性能比較表

| 項目 | B-Tree | GIN | GiST | BRIN |
|------|--------|-----|------|------|
| 構築速度 | 速い | 遅い (2-5x) | 遅い (2-3x) | 極速 |
| 更新コスト | 中 | 高い (Pending List) | 中 | 極低 |
| ディスクサイズ (1億行) | ~2GB | ~3-5GB | ~2-4GB | ~200KB |
| 等値検索 | O(log N) | O(1)* | O(log N) | O(N/R)** |
| 範囲検索 | O(log N + M) | 非対応 | O(log N + M) | O(N/R + M) |
| VACUUM影響 | 中 | 大 | 中 | 小 |

*GINはハッシュ的な検索。**R=pages_per_range

---

## アンチパターン

### アンチパターン1: すべてのカラムにインデックスを作成

```sql
-- NG: 思考停止で全カラムにインデックス
CREATE INDEX idx_users_id ON users (id);           -- PKで既にある
CREATE INDEX idx_users_email ON users (email);      -- 必要
CREATE INDEX idx_users_username ON users (username); -- 検索しない場合は不要
CREATE INDEX idx_users_status ON users (status);     -- 値の種類が少ない
CREATE INDEX idx_users_created ON users (created_at); -- 検索しない場合は不要
CREATE INDEX idx_users_updated ON users (updated_at); -- 検索しない場合は不要

-- 問題点:
-- 1. 各INSERTで6個のインデックスを更新 → 書き込み性能30-50%低下
-- 2. ストレージ消費がテーブルの2-3倍に
-- 3. VACUUMの負荷増大
-- 4. 未使用インデックスが多くの場合50%以上

-- OK: 実際のクエリパターンに基づいてインデックスを作成
-- pg_stat_statementsで頻出クエリを分析
SELECT query, calls, mean_exec_time
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;
-- → 上位クエリのWHERE句/JOIN条件に必要なインデックスだけを作成
```

### アンチパターン2: 複合インデックスの列順序ミス

```sql
-- NG: 範囲条件の列が先、等値条件の列が後
CREATE INDEX idx_orders_bad ON orders (created_at, user_id);

-- このクエリでは user_id の条件がインデックスで処理されない
SELECT * FROM orders
WHERE user_id = 42 AND created_at > '2024-01-01';
-- → Index Cond: (created_at > '2024-01-01')
-- → Filter: (user_id = 42)  ← インデックス外のフィルタ

-- OK: 等値条件を先、範囲条件を後
CREATE INDEX idx_orders_good ON orders (user_id, created_at);
-- → Index Cond: (user_id = 42 AND created_at > '2024-01-01')  ← 全条件がインデックス内
```

### アンチパターン3: CONCURRENTLYを使わないインデックス作成

```sql
-- NG: プロダクションで通常のCREATE INDEX
CREATE INDEX idx_orders_email ON orders (email);
-- → テーブルへの書き込みがロックされる（大テーブルでは数分〜数十分）

-- OK: CONCURRENTLYを使用
CREATE INDEX CONCURRENTLY idx_orders_email ON orders (email);
-- → ロックなしで構築（約2倍の時間がかかるが書き込み可能）
-- 注意: トランザクション内では使用不可
-- 注意: 構築に失敗した場合、INVALID状態のインデックスが残る
--       → DROP INDEX CONCURRENTLY で削除
```

---

## 実践演習

### 演習1（基礎）: 適切なインデックスの選択

以下のテーブルとクエリに対して、最適なインデックスを設計してください。

```sql
CREATE TABLE events (
    id          BIGSERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL,
    event_type  VARCHAR(50) NOT NULL,
    payload     JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- クエリ1: 特定ユーザーの最近のイベント
SELECT * FROM events
WHERE user_id = 42
ORDER BY created_at DESC
LIMIT 20;

-- クエリ2: 特定タイプのイベントをJSONBフィルタ付きで検索
SELECT * FROM events
WHERE event_type = 'purchase'
  AND payload @> '{"amount_gte": 10000}';

-- クエリ3: 日付範囲での集計
SELECT event_type, COUNT(*)
FROM events
WHERE created_at BETWEEN '2024-01-01' AND '2024-01-31'
GROUP BY event_type;
```

<details>
<summary>模範解答</summary>

```sql
-- クエリ1用: 複合インデックス（等値→ソート順）
CREATE INDEX idx_events_user_recent ON events (user_id, created_at DESC);
-- → Index Scan (Backward) で LIMIT 20 が効率的に処理される
-- → ORDER BY + LIMIT がインデックスで完結

-- クエリ2用: 複合B-Tree + GINインデックス
-- 方法A: event_type用B-Tree + payload用GIN（別々のインデックス）
CREATE INDEX idx_events_type ON events (event_type);
CREATE INDEX idx_events_payload ON events USING GIN (payload);
-- → Bitmap AND で両方のインデックスを組み合わせ

-- 方法B: 部分インデックス（event_typeの値が限定的な場合）
CREATE INDEX idx_events_purchase_payload ON events USING GIN (payload)
WHERE event_type = 'purchase';
-- → 'purchase'イベントのpayloadだけにGINインデックス（サイズ最小）

-- クエリ3用: BRINインデックス（時系列データに最適）
CREATE INDEX idx_events_created_brin ON events USING BRIN (created_at)
WITH (pages_per_range = 32);
-- → 日付範囲の絞り込みに極小サイズで対応
-- → GROUP BY event_type は Seq Scan + Hash Aggregate で処理
-- 注意: 結果行数が多い場合はBRINの方がB-Treeより効率的
--       （ランダムI/Oを避けられるため）
```

**解説**: クエリパターンごとに最適なインデックス種類を選択することが重要。クエリ1は複合B-Tree、クエリ2はGIN（部分インデックスがベスト）、クエリ3はBRINが最適。全てのクエリに対して単一のインデックスで対応しようとすると、どのクエリにも中途半端な結果になる。

</details>

### 演習2（応用）: インデックスの効果測定と改善

以下の遅いクエリを分析し、適切なインデックスを設計してEXPLAIN ANALYZEで効果を検証してください。

```sql
-- 遅いクエリ（100万行のテーブル）
EXPLAIN ANALYZE
SELECT o.id, o.total, u.email, u.username
FROM orders o
JOIN users u ON o.user_id = u.id
WHERE o.status = 'shipped'
  AND o.created_at > '2024-06-01'
  AND u.status = 'active'
ORDER BY o.created_at DESC
LIMIT 50;

-- 現在の実行計画:
-- Limit  (cost=50000.00..50000.12 rows=50 width=100)
--   ->  Sort  (cost=50000.00..50100.00 rows=10000 width=100)
--         Sort Key: o.created_at DESC
--         ->  Hash Join  (cost=1000.00..49000.00 rows=10000 width=100)
--               ->  Seq Scan on orders o  (Filter: status='shipped' AND created_at > ...)
--               ->  Hash
--                     ->  Seq Scan on users u  (Filter: status='active')
```

<details>
<summary>模範解答</summary>

```sql
-- Step 1: ordersテーブルの最適化
-- 部分インデックス + カバリング + ソート順
CREATE INDEX idx_orders_shipped_recent ON orders (created_at DESC)
INCLUDE (user_id, total)
WHERE status = 'shipped';
-- → status='shipped' の行のみインデックス化
-- → created_at DESC でソート済み → ORDER BY + LIMIT が高速
-- → INCLUDE (user_id, total) で必要なカラムをインデックス内に保持

-- Step 2: usersテーブルの最適化
-- アクティブユーザーの部分インデックス
CREATE INDEX idx_users_active ON users (id)
INCLUDE (email, username)
WHERE status = 'active';
-- → JOIN条件(id)でIndex Only Scan
-- → email, username もインデックス内から取得

-- 改善後の実行計画:
-- Limit  (cost=0.86..100.00 rows=50 width=100)
--   ->  Nested Loop  (cost=0.86..2000.00 rows=50 width=100)
--         ->  Index Only Scan using idx_orders_shipped_recent on orders o
--               (actual time=0.02..0.30 rows=50)
--               Heap Fetches: 0
--         ->  Index Only Scan using idx_users_active on users u
--               Index Cond: (id = o.user_id)
--               Heap Fetches: 0

-- 結果: Seq Scan × 2 → Index Only Scan × 2
--       50000ms → 0.5ms（約100,000倍の高速化）
```

**解説**: 最も効果的な改善ポイントは以下の3つ:
1. **部分インデックス**: shipped注文は全体の20%程度なので、インデックスサイズが1/5に
2. **カバリングインデックス (INCLUDE)**: テーブルアクセスを完全回避（Heap Fetches: 0）
3. **ソート順の一致**: created_at DESC でインデックスを作成することで、ORDER BY + LIMIT がインデックスのスキャン順で処理される

</details>

### 演習3（発展）: インデックス保守の自動化

以下の要件を満たす、インデックス保守の監視クエリセットを作成してください。

**要件**:
1. 未使用インデックスの一覧（PRIMARY KEY/UNIQUE除外）
2. 重複インデックスの検出
3. インデックス膨張率の推定
4. テーブルサイズに対するインデックスサイズの比率
5. 改善推奨のレポート出力

<details>
<summary>模範解答</summary>

```sql
-- ===== インデックス保守レポート =====

-- 1. 未使用インデックスの検出
WITH unused_indexes AS (
    SELECT
        s.indexrelname AS index_name,
        s.relname AS table_name,
        s.idx_scan AS scans,
        pg_size_pretty(pg_relation_size(s.indexrelid)) AS index_size,
        pg_relation_size(s.indexrelid) AS index_size_bytes
    FROM pg_stat_user_indexes s
    WHERE s.idx_scan = 0
      AND s.indexrelid NOT IN (
          SELECT indexrelid FROM pg_constraint
          WHERE contype IN ('p', 'u')
      )
)
SELECT '未使用インデックス' AS category, index_name, table_name,
       index_size, scans
FROM unused_indexes
ORDER BY index_size_bytes DESC;

-- 2. 重複インデックスの検出
WITH index_cols AS (
    SELECT
        i.indexrelid,
        i.indrelid,
        i.indexrelid::regclass AS index_name,
        i.indrelid::regclass AS table_name,
        array_agg(a.attname ORDER BY k.ord) AS columns
    FROM pg_index i
    CROSS JOIN LATERAL unnest(i.indkey) WITH ORDINALITY AS k(attnum, ord)
    JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = k.attnum
    GROUP BY i.indexrelid, i.indrelid
)
SELECT
    '重複インデックス' AS category,
    a.index_name AS index1,
    b.index_name AS index2,
    a.table_name,
    a.columns
FROM index_cols a
JOIN index_cols b ON a.indrelid = b.indrelid
    AND a.indexrelid < b.indexrelid
    AND a.columns = b.columns;

-- 3. テーブルサイズに対するインデックスサイズの比率
SELECT
    'サイズ比率' AS category,
    relname AS table_name,
    pg_size_pretty(pg_relation_size(relid)) AS table_size,
    pg_size_pretty(pg_indexes_size(relid)) AS total_index_size,
    CASE
        WHEN pg_relation_size(relid) > 0 THEN
            ROUND(100.0 * pg_indexes_size(relid) / pg_relation_size(relid), 1)
        ELSE 0
    END AS index_ratio_pct,
    (SELECT COUNT(*) FROM pg_index WHERE indrelid = relid) AS num_indexes
FROM pg_stat_user_tables
WHERE pg_relation_size(relid) > 1024 * 1024  -- 1MB以上のテーブル
ORDER BY pg_indexes_size(relid) DESC
LIMIT 20;

-- 4. 改善推奨サマリー
SELECT
    CASE
        WHEN idx_scan = 0 AND indexrelid NOT IN
            (SELECT indexrelid FROM pg_constraint WHERE contype IN ('p','u'))
        THEN 'DELETE INDEX（未使用）'
        WHEN pg_relation_size(indexrelid) > pg_relation_size(relid) * 0.5
        THEN 'REINDEX（サイズ過大）'
        ELSE 'OK'
    END AS recommendation,
    indexrelname AS index_name,
    relname AS table_name,
    idx_scan AS total_scans,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE pg_relation_size(indexrelid) > 1024 * 1024
ORDER BY
    CASE WHEN idx_scan = 0 THEN 0 ELSE 1 END,
    pg_relation_size(indexrelid) DESC;
```

**解説**: インデックスの保守は定期的に実施する必要がある。特に重要なのは:
1. **未使用インデックスの削除**: 書き込み性能の改善とストレージ節約
2. **重複インデックスの統合**: (a, b) と (a, b, c) がある場合、前者は不要
3. **膨張率の監視**: 30%以上なら REINDEX CONCURRENTLY を検討
4. **インデックス/テーブル比率**: 200%を超えたらインデックス構成を見直す

</details>

---

## FAQ

### Q1: インデックスを追加すると書き込みはどれくらい遅くなりますか？

B-Treeインデックス1つにつき、INSERTは約10-20%遅くなります。ただし影響はインデックスサイズとテーブルサイズに依存します。GINインデックスはPending Listを使って遅延更新するため、INSERTへの影響はB-Treeより小さいですが、定期的なクリーンアップ（`gin_pending_list_limit`）が必要です。読み取りの高速化が書き込みの劣化を上回るかどうかで判断してください。

### Q2: BRINインデックスはいつ使うべきですか？

テーブルの物理的な行の順序とインデックス対象カラムの値が相関している場合に最適です。典型例は時系列データで、`created_at` カラムの値が挿入順に増加する場合です。B-Treeの1/10000以下のサイズで同等の範囲検索性能を実現できます。ただし、等値検索（`WHERE id = 42`）には不向きです。`correlation`（相関係数）が0.9以上であればBRINが有効です。

```sql
-- 相関係数の確認
SELECT attname, correlation
FROM pg_stats
WHERE tablename = 'access_logs' AND attname = 'created_at';
-- correlation > 0.9 なら BRIN が有効
```

### Q3: CREATE INDEX CONCURRENTLYは常に使うべきですか？

プロダクション環境では推奨します。通常の`CREATE INDEX`はテーブルへの書き込みをロックしますが、`CONCURRENTLY`はロックなしでインデックスを構築します。ただし以下の注意点があります:
- 構築時間が約2倍になる
- トランザクション内では使用できない
- 構築が途中で失敗すると`INVALID`状態のインデックスが残る（手動削除が必要）
- 2回のテーブルスキャンが必要なため、大量の同時書き込みがあると構築時間が長くなる

### Q4: HOT Update（Heap-Only Tuple Update）とは何ですか？

HOT Updateは、更新された行がインデックスに影響しない場合に、インデックスの更新を完全にスキップする最適化です。条件は:
1. 更新されたカラムがどのインデックスにも含まれていない
2. 新しいタプルが元のタプルと同じページに収まる

`fillfactor`を下げる（デフォルト100→70-80）と、ページに空きスペースができ、HOT Updateの確率が上がります。UPDATE頻度が高いテーブルでは効果的です。

```sql
-- HOT Update率の確認
SELECT relname, n_tup_upd, n_tup_hot_upd,
       ROUND(100.0 * n_tup_hot_upd / NULLIF(n_tup_upd, 0), 1) AS hot_pct
FROM pg_stat_user_tables
WHERE n_tup_upd > 0
ORDER BY n_tup_upd DESC;
-- hot_pct が低い場合、fillfactor の調整を検討
```

### Q5: マルチカラムインデックスと複数の単一カラムインデックス、どちらが良いですか？

**マルチカラムインデックス**が有利な場合:
- 常に同じカラムの組み合わせで検索する（`WHERE a = ? AND b = ?`）
- ソート順と検索条件が一致する（`WHERE a = ? ORDER BY b`）
- Index Only Scanを実現したい

**複数の単一カラムインデックス**が有利な場合:
- カラムの組み合わせが不定（`WHERE a = ?` の時もあれば `WHERE b = ?` の時もある）
- PostgreSQLのBitmap AND/ORで組み合わせ可能

---

## まとめ

| 項目 | 要点 |
|---|---|
| B-Tree | 汎用インデックス。等値・範囲検索・ソート。O(log N) |
| GIN | 全文検索、JSONB、配列の検索。逆引きインデックス |
| GiST | 空間検索、範囲型の重複検索、排他制約 |
| BRIN | 物理ソート済みデータの範囲検索。極小サイズ(B-Treeの1/10000) |
| 部分インデックス | WHERE句付き。テーブルの一部のみインデックス化してサイズ・保守コスト削減 |
| カバリング | INCLUDE で Index Only Scan を実現。テーブルアクセス不要 |
| 複合インデックス | 列順序が重要。等値条件を先、範囲条件を後、ソートキーを最後 |
| 式インデックス | 関数の結果にインデックス。クエリでも同じ式を使うこと |
| 保守 | 未使用インデックスの定期的な検出と削除。REINDEX CONCURRENTLY |

---

## 次に読むべきガイド

- [04-query-optimization.md](./04-query-optimization.md) — EXPLAINの読み方とクエリリライト
- [02-transactions.md](./02-transactions.md) — FOR UPDATEとインデックスロックの関係
- [02-migration.md](../02-design/02-migration.md) — インデックス追加のゼロダウンタイム手法（CONCURRENTLY）
- [02-performance-tuning.md](../03-practical/02-performance-tuning.md) — 総合的なパフォーマンスチューニング
- [04-nosql-comparison.md](../03-practical/04-nosql-comparison.md) — インデックス不要なデータモデル

---

## 参考文献

1. **PostgreSQL公式ドキュメント**: [Indexes](https://www.postgresql.org/docs/current/indexes.html) — インデックス種類と詳細仕様
2. **Markus Winand**: [Use The Index, Luke](https://use-the-index-luke.com/) — SQLインデックスの包括的ガイド（必読）
3. **Cybertec**: [PostgreSQL Index Types](https://www.cybertec-postgresql.com/en/postgresql-indexes-overview/) — PostgreSQL固有のインデックス解説
4. **PostgreSQL Wiki**: [Index Maintenance](https://wiki.postgresql.org/wiki/Index_Maintenance) — インデックス保守のベストプラクティス
5. **Citus Data Blog**: [PostgreSQL Index Tips](https://www.citusdata.com/blog/) — 実践的なインデックス最適化事例
