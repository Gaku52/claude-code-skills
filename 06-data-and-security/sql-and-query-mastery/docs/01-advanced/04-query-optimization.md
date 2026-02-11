# クエリ最適化 — EXPLAIN・実行計画・統計

> クエリ最適化はデータベースのパフォーマンス問題の根本原因を特定し、論理的に解決するプロセスであり、EXPLAINコマンドによる実行計画の解読がその第一歩となる。

## この章で学ぶこと

1. EXPLAIN / EXPLAIN ANALYZEの出力を正確に読み解く
2. 主要なスキャン方式と結合方式の特性を理解する
3. 統計情報の仕組みとクエリリライトによる最適化手法

---

## 1. EXPLAIN の基本

### コード例1: EXPLAIN と EXPLAIN ANALYZE

```sql
-- EXPLAIN: 実行計画を推定（実行しない）
EXPLAIN
SELECT e.name, d.name AS department
FROM employees e
    INNER JOIN departments d ON e.department_id = d.id
WHERE e.salary > 500000;

-- 出力例:
-- Hash Join  (cost=1.09..2.24 rows=3 width=64)
--   Hash Cond: (e.department_id = d.id)
--   ->  Seq Scan on employees e  (cost=0.00..1.12 rows=3 width=40)
--         Filter: (salary > 500000)
--   ->  Hash  (cost=1.05..1.05 rows=5 width=36)
--         ->  Seq Scan on departments d  (cost=0.00..1.05 rows=5 width=36)

-- EXPLAIN ANALYZE: 実際に実行して測定
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT e.name, d.name AS department
FROM employees e
    INNER JOIN departments d ON e.department_id = d.id
WHERE e.salary > 500000;

-- 出力例:
-- Hash Join  (cost=1.09..2.24 rows=3 width=64)
--            (actual time=0.035..0.042 rows=5 loops=1)
--   Hash Cond: (e.department_id = d.id)
--   Buffers: shared hit=2
--   ->  Seq Scan on employees e  (cost=0.00..1.12 rows=3 width=40)
--                                (actual time=0.012..0.018 rows=5 loops=1)
--         Filter: (salary > 500000)
--         Rows Removed by Filter: 95
--   ->  Hash  (cost=1.05..1.05 rows=5 width=36)
--             (actual time=0.008..0.008 rows=5 loops=1)
--         Buckets: 1024  Batches: 1
--         ->  Seq Scan on departments d  (cost=0.00..1.05 rows=5 width=36)
```

### EXPLAIN出力の読み方

```
┌─────────── EXPLAIN 出力の構成要素 ──────────────┐
│                                                   │
│  Hash Join (cost=1.09..2.24 rows=3 width=64)      │
│  ~~~~~~~~  ~~~~~  ~~~~~~~~ ~~~~~ ~~~~~~~~         │
│  ノード名   起動   総コスト 推定行 行幅(bytes)     │
│             コスト                                 │
│                                                   │
│  (actual time=0.035..0.042 rows=5 loops=1)        │
│  ~~~~~~~~~~~~~~~  ~~~~~~~~ ~~~~~ ~~~~~~~          │
│  実際の起動時間   実際の総時間 実行行 ループ回数    │
│                                                   │
│  ※ コスト単位: seq_page_cost = 1.0 が基準         │
│  ※ actual time: ミリ秒                            │
│  ※ rows推定 vs actual rows の乖離 → 統計の問題    │
│                                                   │
│  ツリーの読み方:                                   │
│  ・インデントが深い = 先に実行される                │
│  ・下から上に読む                                  │
│  ・各ノードは子ノードの結果を受け取る              │
└───────────────────────────────────────────────────┘
```

---

## 2. スキャン方式

### コード例2: 各スキャン方式の比較

```sql
-- Sequential Scan: テーブル全体を読む
EXPLAIN SELECT * FROM employees WHERE status = 'active';
-- Seq Scan on employees  (cost=0.00..1.12 rows=80 width=200)
--   Filter: (status = 'active')

-- Index Scan: インデックスを使って行を特定
EXPLAIN SELECT * FROM employees WHERE id = 42;
-- Index Scan using employees_pkey on employees  (cost=0.15..8.17 rows=1 width=200)
--   Index Cond: (id = 42)

-- Index Only Scan: インデックスだけで回答（テーブル不要）
EXPLAIN SELECT id, name FROM employees WHERE id BETWEEN 1 AND 100;
-- Index Only Scan using idx_emp_covering on employees  (cost=...)

-- Bitmap Index Scan: 複数条件をビットマップで合成
EXPLAIN SELECT * FROM employees
WHERE department_id = 10 AND salary > 500000;
-- Bitmap Heap Scan on employees
--   Recheck Cond: (department_id = 10)
--   Filter: (salary > 500000)
--   ->  Bitmap Index Scan on idx_emp_dept
--         Index Cond: (department_id = 10)
```

---

## 3. 結合方式

### コード例3: 結合方式の理解

```sql
-- Nested Loop Join: 小テーブル×インデックス付きテーブル
-- 外側の各行に対して内側テーブルをスキャン
-- 適する: 小結果セット、内側にインデックスあり
EXPLAIN SELECT * FROM orders o
    INNER JOIN customers c ON o.customer_id = c.id
WHERE o.id = 42;
-- Nested Loop  (cost=0.56..16.59 rows=1 width=400)

-- Hash Join: 一方のテーブルでハッシュ表構築、もう一方で突き合わせ
-- 適する: 等値結合、中〜大テーブル
EXPLAIN SELECT * FROM orders o
    INNER JOIN customers c ON o.customer_id = c.id;
-- Hash Join  (cost=1.09..35.24 rows=1000 width=400)
--   ->  Seq Scan on orders o
--   ->  Hash
--         ->  Seq Scan on customers c

-- Merge Join: 両テーブルをソートして突き合わせ
-- 適する: 大テーブル同士、ソート済みデータ
EXPLAIN SELECT * FROM large_table_a a
    INNER JOIN large_table_b b ON a.key = b.key
    ORDER BY a.key;
-- Merge Join  (cost=...)
--   ->  Sort  ->  Seq Scan on large_table_a
--   ->  Sort  ->  Seq Scan on large_table_b
```

### 結合方式の動作図

```
┌────────── 3つの結合方式 ──────────────────────┐
│                                                │
│  Nested Loop (ネステッドループ)                 │
│  外側の各行に対して内側をスキャン               │
│  計算量: O(N × M)  ※インデックスあればO(N log M)│
│  ┌─────┐                                      │
│  │ A(1)│ → B全体を走査して一致行を検索         │
│  │ A(2)│ → B全体を走査して一致行を検索         │
│  │ A(3)│ → B全体を走査して一致行を検索         │
│  └─────┘                                      │
│                                                │
│  Hash Join (ハッシュ結合)                       │
│  小テーブルでハッシュ表構築→大テーブルで照合    │
│  計算量: O(N + M)                              │
│  ┌─────┐        ┌───────────┐                 │
│  │  B  │ ──────►│Hash Table │                 │
│  └─────┘ build  └─────┬─────┘                 │
│  ┌─────┐        probe│                        │
│  │  A  │ ────────────┘                        │
│  └─────┘                                      │
│                                                │
│  Merge Join (マージ結合)                       │
│  両方をソート済みにして並行走査                  │
│  計算量: O(N log N + M log M)                  │
│  ┌─────┐  ┌─────┐                             │
│  │A(sorted)│B(sorted)│  → 並行して比較         │
│  └─────┘  └─────┘                             │
└────────────────────────────────────────────────┘
```

---

## 4. 統計情報とクエリリライト

### コード例4: 統計情報の確認と更新

```sql
-- テーブルの統計情報を確認
SELECT
    attname AS column,
    n_distinct,           -- ユニーク値数（負数は割合）
    null_frac,            -- NULL率
    avg_width,            -- 平均バイト幅
    most_common_vals,     -- 最頻値
    most_common_freqs     -- 最頻値の出現率
FROM pg_stats
WHERE tablename = 'employees' AND attname = 'department_id';

-- 統計情報の更新（手動ANALYZE）
ANALYZE employees;

-- 統計情報の精度を上げる（デフォルト100、最大10000）
ALTER TABLE employees ALTER COLUMN department_id SET STATISTICS 1000;
ANALYZE employees;

-- テーブルの行数推定 vs 実際
SELECT
    relname,
    reltuples::BIGINT AS estimated_rows,  -- 推定行数
    pg_stat_get_live_tuples(oid) AS actual_rows  -- 実際の行数
FROM pg_class
WHERE relname = 'employees';
```

### コード例5: クエリリライトによる最適化

```sql
-- NG: OR条件でインデックスが効きにくい
SELECT * FROM orders
WHERE customer_id = 42 OR product_id = 100;

-- OK: UNION ALLに書き換え（各条件でインデックスが使える）
SELECT * FROM orders WHERE customer_id = 42
UNION ALL
SELECT * FROM orders WHERE product_id = 100
  AND customer_id != 42;  -- 重複排除

-- NG: IN (サブクエリ) が非効率な場合
SELECT * FROM products
WHERE id IN (SELECT product_id FROM order_items WHERE quantity > 10);

-- OK: EXISTS に書き換え
SELECT * FROM products p
WHERE EXISTS (
    SELECT 1 FROM order_items oi
    WHERE oi.product_id = p.id AND oi.quantity > 10
);

-- OK: JOIN に書き換え
SELECT DISTINCT p.*
FROM products p
    INNER JOIN order_items oi ON p.id = oi.product_id
WHERE oi.quantity > 10;
```

### コード例6: パフォーマンス分析クエリ

```sql
-- 最も遅いクエリの特定（pg_stat_statements拡張）
SELECT
    calls,
    ROUND(total_exec_time::NUMERIC, 2) AS total_ms,
    ROUND(mean_exec_time::NUMERIC, 2) AS avg_ms,
    ROUND(stddev_exec_time::NUMERIC, 2) AS stddev_ms,
    rows,
    LEFT(query, 100) AS query_preview
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;

-- 実行計画のJSON形式（プログラムからの解析用）
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT * FROM employees WHERE department_id = 10;

-- auto_explainで自動的にスロークエリの実行計画をログ
-- postgresql.conf:
-- shared_preload_libraries = 'auto_explain'
-- auto_explain.log_min_duration = '100ms'
```

---

## スキャン方式比較表

| スキャン方式 | 適する場面 | 計算量 | I/O特性 |
|------------|-----------|--------|---------|
| Sequential Scan | 全行/大部分の取得 | O(N) | 順次読み |
| Index Scan | 少数行の取得 | O(log N) | ランダム読み |
| Index Only Scan | カバリングインデックスあり | O(log N) | インデックスのみ |
| Bitmap Index Scan | 中程度の行数 | O(N log N) | バッチ読み |
| Parallel Seq Scan | 大テーブルの全走査 | O(N/P) | 並列順次 |

## 結合方式比較表

| 結合方式 | 適する場面 | 計算量 | メモリ使用量 |
|---------|-----------|--------|------------|
| Nested Loop | 小テーブル + インデックス | O(N*M) | 小 |
| Hash Join | 等値結合、中テーブル | O(N+M) | 中（ハッシュ表） |
| Merge Join | 大テーブル、ソート済み | O(NlogN+MlogM) | 小 |

---

## アンチパターン

### アンチパターン1: EXPLAIN なしでインデックスを追加

```sql
-- NG: 「遅いからインデックスを追加」という安易な対応
CREATE INDEX idx_guess ON orders (status);
-- 効果がない理由: statusの選択率が低い（80%が'completed'）
-- → Seq Scan の方が速い

-- OK: EXPLAIN ANALYZE で原因を特定してから対策
EXPLAIN ANALYZE
SELECT * FROM orders WHERE status = 'pending';
-- → Seq Scan、Filter: Rows Removed: 95000
-- → 選択率 5% → インデックスは有効

-- さらに部分インデックスで最適化
CREATE INDEX idx_orders_pending ON orders (created_at)
WHERE status = 'pending';
```

### アンチパターン2: 推定行数と実際の行数の乖離を無視

```sql
-- 推定 rows=10 vs 実際 rows=50000 のような乖離
-- → オプティマイザが間違った実行計画を選択する原因

-- 対策1: ANALYZE を実行して統計を更新
ANALYZE orders;

-- 対策2: 統計精度を上げる
ALTER TABLE orders ALTER COLUMN status SET STATISTICS 1000;
ANALYZE orders;

-- 対策3: 拡張統計（複数列の相関）
CREATE STATISTICS stat_orders_status_date (dependencies)
ON status, order_date FROM orders;
ANALYZE orders;
```

---

## FAQ

### Q1: EXPLAIN と EXPLAIN ANALYZE の違いは？

EXPLAINは実行計画を推定するだけでクエリを実行しない（安全）。EXPLAIN ANALYZEは実際にクエリを実行して実測値を表示する。UPDATE/DELETEにEXPLAIN ANALYZEを使う場合は、トランザクション内で実行してROLLBACKすること。

### Q2: コスト値の単位は何か？

PostgreSQLのコストは抽象的な単位で、`seq_page_cost = 1.0`（シーケンシャルページ読み取り1回）を基準とする。`random_page_cost`はデフォルト4.0（SSDでは1.1〜1.5に下げることが推奨）。コスト値同士の比較は有意だが、絶対値は実時間と直接対応しない。

### Q3: パラレルクエリはいつ有効か？

大テーブルのSeq Scan、大量行の集約、大テーブル同士のHash Joinなどで有効。`max_parallel_workers_per_gather`（デフォルト2）で並列度を制御。小テーブルや索引アクセスではオーバーヘッドの方が大きい。

---

## まとめ

| 項目 | 要点 |
|------|------|
| EXPLAIN ANALYZE | 実行計画と実測値の両方を確認。最適化の第一歩 |
| 推定 vs 実際 | rows の乖離 → ANALYZEで統計更新 |
| スキャン方式 | Seq Scan / Index Scan / Bitmap の使い分け |
| 結合方式 | Nested Loop / Hash / Merge の特性理解 |
| 統計情報 | pg_stats、SET STATISTICS、拡張統計 |
| 監視 | pg_stat_statements でスロークエリ特定 |

---

## 次に読むべきガイド

- [03-indexing.md](./03-indexing.md) — インデックスの詳細設計
- [02-performance-tuning.md](../03-practical/02-performance-tuning.md) — 総合チューニング
- [00-normalization.md](../02-design/00-normalization.md) — スキーマ設計と性能

---

## 参考文献

1. PostgreSQL Documentation — "Using EXPLAIN" https://www.postgresql.org/docs/current/using-explain.html
2. Winand, M. (2012). *SQL Performance Explained*. https://use-the-index-luke.com/
3. Citus Data — "PostgreSQL Query Optimization" https://www.citusdata.com/blog/
