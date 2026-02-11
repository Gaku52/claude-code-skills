# インデックス

> B-Tree、GiST、GIN、部分インデックスなど、データベースインデックスの内部構造と設計戦略を理解し、クエリ性能を最適化する

## この章で学ぶこと

1. **インデックスの内部構造** — B-Tree の仕組み、ページ分割、検索アルゴリズム
2. **特殊インデックス** — GiST（空間検索）、GIN（全文検索・JSON）、BRIN（大規模テーブル）
3. **インデックス設計戦略** — 部分インデックス、カバリングインデックス、複合インデックスの列順序

---

## 1. B-Tree インデックスの内部構造

```
B-Tree の構造 (次数=3 の例)
==============================

           [30 | 60]              <-- ルートノード
          /    |    \
   [10|20]  [40|50]  [70|80]     <-- 内部ノード
   / | \    / | \    / | \
  L  L  L  L  L  L  L  L  L     <-- リーフノード (実データへのポインタ)

検索: WHERE id = 45
  1. ルート: 30 < 45 < 60 --> 中央の子へ
  2. 内部: 40 < 45 < 50  --> 中央の子へ
  3. リーフ: id=45 のレコードを取得

計算量: O(log N)
  1億行のテーブルでも約3-4回のページ読み取りで到達
```

### コード例 1: 基本的なインデックス作成

```sql
-- B-Tree インデックス（デフォルト）
CREATE INDEX idx_users_email ON users (email);

-- ユニークインデックス（重複不可制約）
CREATE UNIQUE INDEX idx_users_email_unique ON users (email);

-- 複合インデックス（列の順序が重要）
CREATE INDEX idx_orders_user_date ON orders (user_id, created_at DESC);

-- 降順インデックス
CREATE INDEX idx_orders_recent ON orders (created_at DESC);

-- インデックスの確認
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'orders';
```

### コード例 2: EXPLAIN ANALYZE による効果測定

```sql
-- インデックスなし
EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM orders WHERE user_id = 12345;
-- Seq Scan on orders  (cost=0.00..25000.00 rows=50 width=120)
--   actual time=45.2..180.5 rows=50 loops=1
--   Buffers: shared hit=15000 read=10000

-- インデックス作成後
CREATE INDEX idx_orders_user_id ON orders (user_id);

EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM orders WHERE user_id = 12345;
-- Index Scan using idx_orders_user_id on orders
--   (cost=0.43..8.50 rows=50 width=120)
--   actual time=0.03..0.15 rows=50 loops=1
--   Buffers: shared hit=5
```

---

## 2. 複合インデックスの列順序

```
複合インデックスの動作原理
============================

CREATE INDEX idx ON orders (user_id, status, created_at);

インデックスの内部ソート順:
  user_id=1, status='active',   created_at='2026-01-01'
  user_id=1, status='active',   created_at='2026-01-15'
  user_id=1, status='shipped',  created_at='2026-01-10'
  user_id=2, status='active',   created_at='2026-01-05'
  user_id=2, status='pending',  created_at='2026-01-20'

利用可能なクエリ:
  [OK] WHERE user_id = 1                         (左端プレフィックス)
  [OK] WHERE user_id = 1 AND status = 'active'   (プレフィックス一致)
  [OK] WHERE user_id = 1 AND status = 'active'
       AND created_at > '2026-01-01'              (全列利用)
  [NG] WHERE status = 'active'                    (左端スキップ)
  [NG] WHERE created_at > '2026-01-01'            (左端スキップ)
```

### インデックス種類比較表

| インデックス型 | 対応演算 | ユースケース | サイズ |
|---|---|---|---|
| **B-Tree** | =, <, >, BETWEEN, LIKE 'abc%' | 汎用、範囲検索 | 中 |
| **Hash** | = のみ | 等値検索のみ | 小 |
| **GiST** | 包含、重複、近傍 | 地理空間、範囲型 | 大 |
| **GIN** | 含む、配列要素 | 全文検索、JSONB、配列 | 大 |
| **BRIN** | 範囲検索 | 時系列、物理ソート済みデータ | 極小 |
| **SP-GiST** | パーティション検索 | 電話番号、IP アドレス | 中 |

---

## 3. 特殊インデックス

### コード例 3: GIN インデックス（JSONB・全文検索）

```sql
-- JSONB 検索用 GIN インデックス
CREATE INDEX idx_products_attrs ON products USING GIN (attributes);

-- JSONB の検索
SELECT * FROM products
WHERE attributes @> '{"color": "red", "size": "L"}';

-- 全文検索用 GIN インデックス
ALTER TABLE articles ADD COLUMN search_vector tsvector;

UPDATE articles SET search_vector =
  setweight(to_tsvector('japanese', title), 'A') ||
  setweight(to_tsvector('japanese', body), 'B');

CREATE INDEX idx_articles_search ON articles USING GIN (search_vector);

-- 全文検索クエリ
SELECT title, ts_rank(search_vector, query) AS rank
FROM articles, to_tsquery('japanese', 'PostgreSQL & インデックス') AS query
WHERE search_vector @@ query
ORDER BY rank DESC
LIMIT 10;

-- 配列検索用 GIN
CREATE INDEX idx_posts_tags ON posts USING GIN (tags);
SELECT * FROM posts WHERE tags @> ARRAY['rust', 'wasm'];
```

### コード例 4: GiST インデックス（空間検索）

```sql
-- PostGIS: 空間インデックス
CREATE INDEX idx_stores_location ON stores USING GiST (location);

-- 半径5km以内の店舗検索
SELECT name, ST_Distance(
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

-- 範囲型の重複検索
CREATE INDEX idx_reservations_period ON reservations
  USING GiST (daterange(check_in, check_out));

SELECT * FROM reservations
WHERE daterange(check_in, check_out) && daterange('2026-03-01', '2026-03-10');
```

---

## 4. 部分インデックスとカバリングインデックス

### コード例 5: 部分インデックス

```sql
-- アクティブユーザーのみインデックス（全体の10%）
CREATE INDEX idx_users_active_email ON users (email)
WHERE status = 'active';

-- 未処理注文のみ（全体の5%）
CREATE INDEX idx_orders_pending ON orders (created_at)
WHERE status = 'pending';

-- NULL でないカラムのみ
CREATE INDEX idx_users_phone ON users (phone)
WHERE phone IS NOT NULL;

-- 効果: フルインデックスの 1/10 ~ 1/20 のサイズ
--        書き込みオーバーヘッドも比例して減少
```

### コード例 6: カバリングインデックス（INCLUDE）

```sql
-- カバリングインデックス: テーブルアクセスを完全回避
CREATE INDEX idx_orders_covering ON orders (user_id, status)
INCLUDE (total_amount, created_at);

-- Index Only Scan で応答（テーブル読み取り不要）
SELECT user_id, status, total_amount, created_at
FROM orders
WHERE user_id = 12345 AND status = 'shipped';

-- EXPLAIN で確認
-- Index Only Scan using idx_orders_covering on orders
--   Heap Fetches: 0  <-- テーブルアクセスなし
```

---

## 5. インデックスの保守

```
インデックス肥大化の仕組み
============================

DELETE/UPDATE が繰り返されると:

初期状態:  [1][2][3][4][5][6][7][8]  (ページ使用率 100%)
           ↓ DELETE 3,5,7
中間状態:  [1][2][ ][4][ ][6][ ][8]  (ページ使用率 62.5%)
           ↓ 新しい INSERT は空き領域に入らない場合も
肥大化:    多数の空きページが残存 --> インデックスサイズ増大

対策: REINDEX または pg_repack で再構築
```

### コード例 7: インデックスの監視と保守

```sql
-- インデックスサイズの確認
SELECT
  schemaname || '.' || tablename AS table,
  indexname,
  pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
  idx_scan AS scans,
  idx_tup_read AS tuples_read,
  idx_tup_fetch AS tuples_fetched
FROM pg_stat_user_indexes
JOIN pg_indexes USING (indexname)
ORDER BY pg_relation_size(indexrelid) DESC
LIMIT 20;

-- 未使用インデックスの検出
SELECT indexrelname, idx_scan, pg_size_pretty(pg_relation_size(indexrelid))
FROM pg_stat_user_indexes
WHERE idx_scan = 0
  AND indexrelid NOT IN (
    SELECT indexrelid FROM pg_constraint
    WHERE contype IN ('p', 'u')  -- PK, UNIQUE は除外
  )
ORDER BY pg_relation_size(indexrelid) DESC;

-- REINDEX（オンライン、PostgreSQL 12+）
REINDEX INDEX CONCURRENTLY idx_orders_user_date;
```

---

## インデックス選択フロー比較表

| 条件 | 推奨インデックス | 理由 |
|---|---|---|
| 等値/範囲検索（汎用） | B-Tree | 最も汎用的、デフォルト |
| JSONB の検索 | GIN | `@>`, `?`, `?&` 演算子に対応 |
| 全文検索 | GIN (tsvector) | トークンの逆引きインデックス |
| 地理空間検索 | GiST (PostGIS) | 空間的な包含・近傍クエリ |
| 時系列データ（物理ソート済み） | BRIN | 極小サイズで効率的 |
| 等値検索のみ（高頻度） | Hash | B-Tree より若干高速 |
| 大テーブルの少数レコード検索 | 部分インデックス | サイズと保守コストを削減 |

---

## アンチパターン

### 1. すべてのカラムにインデックスを作成

**問題**: インデックスは INSERT/UPDATE/DELETE のたびに更新される。不要なインデックスは書き込み性能を劣化させ、ストレージを消費する。

**対策**: 実際のクエリパターンを `pg_stat_statements` で分析し、必要なインデックスだけを作成する。未使用インデックスは定期的に検出して削除する。

### 2. 複合インデックスの列順序ミス

**問題**: `CREATE INDEX ON orders (created_at, user_id)` の順序だと、`WHERE user_id = ?` のクエリでインデックスが使われない（左端プレフィックスルール）。

**対策**: 等値条件のカラムを先に、範囲条件のカラムを後に配置する。`WHERE user_id = ? AND created_at > ?` なら `(user_id, created_at)` の順序が正しい。

---

## FAQ

### Q1: インデックスを追加すると書き込みはどれくらい遅くなりますか？

**A**: B-Tree インデックス1つにつき、INSERT は約 10-20% 遅くなります。ただし影響はインデックスサイズとテーブルサイズに依存します。読み取りの高速化が書き込みの劣化を上回るかどうかで判断してください。

### Q2: BRIN インデックスはいつ使うべきですか？

**A**: テーブルの物理的な行の順序とインデックス対象カラムの値が相関している場合に最適です。典型例は時系列データで、`created_at` カラムの値が挿入順に増加する場合です。B-Tree の 1/100 以下のサイズで同等の範囲検索性能を実現できます。

### Q3: CREATE INDEX CONCURRENTLY は常に使うべきですか？

**A**: プロダクション環境では推奨します。通常の `CREATE INDEX` はテーブルへの書き込みをロックしますが、`CONCURRENTLY` はロックなしでインデックスを構築します。ただし構築時間は約2倍になり、トランザクション内では使用できません。

---

## まとめ

| 項目 | 要点 |
|---|---|
| B-Tree | 汎用インデックス。等値・範囲検索。O(log N) |
| GIN | 全文検索、JSONB、配列の検索に最適 |
| GiST | 空間検索、範囲型の重複検索に最適 |
| BRIN | 物理ソート済みデータの範囲検索。極小サイズ |
| 部分インデックス | 条件付きインデックスでサイズと保守コスト削減 |
| カバリング | INCLUDE で Index Only Scan を実現 |
| 複合インデックス | 列順序が重要。等値条件を先、範囲条件を後 |
| 保守 | 未使用インデックスの定期的な検出と削除 |

## 次に読むべきガイド

- [マイグレーション](../02-design/02-migration.md) — インデックス追加のゼロダウンタイム手法
- [NoSQL 比較](../03-practical/04-nosql-comparison.md) — インデックス不要なデータモデル

## 参考文献

1. **PostgreSQL 公式ドキュメント**: [Indexes](https://www.postgresql.org/docs/current/indexes.html) — インデックス種類と詳細仕様
2. **Markus Winand**: [Use The Index, Luke](https://use-the-index-luke.com/) — SQL インデックスの包括的ガイド
3. **Cybertec**: [PostgreSQL Index Types](https://www.cybertec-postgresql.com/en/postgresql-indexes-overview/) — PostgreSQL 固有のインデックス解説
