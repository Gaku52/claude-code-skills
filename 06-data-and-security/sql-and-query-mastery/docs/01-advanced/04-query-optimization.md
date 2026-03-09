# クエリ最適化 — EXPLAIN・実行計画・統計情報・クエリリライト

> クエリ最適化はデータベースのパフォーマンス問題の根本原因を特定し、論理的に解決するプロセスであり、EXPLAINコマンドによる実行計画の解読がその第一歩となる。本章では、EXPLAIN出力の各要素を正確に読み解き、スキャン方式・結合方式の選択基準を理解し、統計情報とクエリリライトによる最適化手法を体系的に習得する。

---

## この章で学ぶこと

1. **EXPLAIN / EXPLAIN ANALYZEの出力を正確に読み解く** — コスト計算、推定行数と実際の行数の乖離、バッファ情報の解釈
2. **主要なスキャン方式と結合方式の特性を理解する** — Sequential Scan、Index Scan、Bitmap Scan、Nested Loop、Hash Join、Merge Join
3. **統計情報の仕組みとクエリリライトによる最適化手法** — pg_stats、拡張統計、IN→EXISTS書き換え、CTE最適化
4. **実践的なパフォーマンス分析ワークフロー** — pg_stat_statements、auto_explain、ボトルネック特定手法

---

## 前提知識

| トピック | 内容 | 参照先 |
|---------|------|--------|
| SQL基礎 | SELECT/JOIN/サブクエリの構文 | [00-basics/](../00-basics/) |
| インデックス | B-Tree、GIN、GiST の基本 | [03-indexing.md](./03-indexing.md) |
| テーブル設計 | 正規化、制約の基本 | [00-normalization.md](../02-design/00-normalization.md) |

---

## 1. EXPLAINの基本

### なぜEXPLAINが重要か

「遅いからインデックスを追加する」という安易な対応は、問題の根本原因を見落とす。EXPLAINは「なぜこのクエリが遅いのか」を科学的に診断するツールであり、データベースパフォーマンスチューニングの唯一の出発点である。

### コード例1: EXPLAINとEXPLAIN ANALYZE

```sql
-- テスト用テーブルの準備
CREATE TABLE employees (
    id            SERIAL PRIMARY KEY,
    name          VARCHAR(100) NOT NULL,
    department_id INTEGER NOT NULL,
    salary        INTEGER NOT NULL,
    hired_date    DATE NOT NULL DEFAULT CURRENT_DATE
);

CREATE TABLE departments (
    id   SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

-- EXPLAIN: 実行計画を推定（実行しない — 安全）
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


-- EXPLAIN ANALYZE: 実際に実行して測定（DMLに注意）
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
--         Buffers: shared hit=1
--   ->  Hash  (cost=1.05..1.05 rows=5 width=36)
--             (actual time=0.008..0.008 rows=5 loops=1)
--         Buckets: 1024  Batches: 1
--         ->  Seq Scan on departments d  (cost=0.00..1.05 rows=5 width=36)
--                                        (actual time=0.003..0.004 rows=5 loops=1)
--         Buffers: shared hit=1
-- Planning Time: 0.15 ms
-- Execution Time: 0.08 ms
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
│  コスト:                                          │
│  ・起動コスト: 最初の行を返すまでのコスト           │
│  ・総コスト: 全行を返すまでのコスト                │
│  ・単位: seq_page_cost = 1.0 が基準               │
│  ・random_page_cost = 4.0 (HDD) / 1.1 (SSD)      │
│                                                   │
│  実際の時間:                                      │
│  ・ミリ秒単位                                     │
│  ・loops > 1 の場合、表示値は1ループ分の平均       │
│    → 実際の合計時間 = time × loops                │
│                                                   │
│  推定行 vs 実際の行（rows推定 vs actual rows）:   │
│  ・乖離が大きい → 統計情報の問題                  │
│  ・ANALYZE を実行して統計を更新                    │
│                                                   │
│  Buffers:                                         │
│  ・shared hit: バッファキャッシュからの読み取り     │
│  ・shared read: ディスクからの読み取り             │
│  ・shared dirtied: 書き込まれたバッファ            │
│  ・shared written: ディスクに書き出されたバッファ  │
│                                                   │
│  ツリーの読み方:                                   │
│  ・インデントが深い = 先に実行される                │
│  ・下から上に読む                                  │
│  ・各ノードは子ノードの結果を受け取る              │
└───────────────────────────────────────────────────┘
```

### EXPLAIN のオプション比較表

| オプション | 説明 | 安全性 | 用途 |
|-----------|------|:---:|------|
| `EXPLAIN` | 推定のみ（実行しない） | 安全 | 計画の確認 |
| `EXPLAIN ANALYZE` | 実際に実行して実測値を表示 | DML注意 | パフォーマンス分析 |
| `BUFFERS` | バッファI/O情報を表示 | - | I/Oボトルネック特定 |
| `FORMAT JSON` | JSON形式で出力 | - | プログラム解析用 |
| `FORMAT YAML` | YAML形式で出力 | - | 可読性重視 |
| `SETTINGS` | デフォルトと異なる設定を表示 | - | 設定影響の確認 |
| `WAL` | WAL使用量を表示 (PG13+) | - | 書き込み負荷の確認 |

---

## 2. スキャン方式

### コード例2: 各スキャン方式の比較

```sql
-- ===== Sequential Scan (Seq Scan) =====
-- テーブルの全ページを先頭から順に読む
-- 適する: テーブル全体の大部分を取得する場合
EXPLAIN ANALYZE SELECT * FROM employees WHERE status = 'active';
-- Seq Scan on employees  (cost=0.00..1.12 rows=80 width=200)
--   Filter: (status = 'active')
--   Rows Removed by Filter: 20  ← フィルタで除外された行数


-- ===== Index Scan =====
-- インデックスで行の位置(TID)を特定し、テーブルから行を取得
-- 適する: 少数の行を取得する場合（選択率 < ~5%）
EXPLAIN ANALYZE SELECT * FROM employees WHERE id = 42;
-- Index Scan using employees_pkey on employees
--   (cost=0.15..8.17 rows=1 width=200)
--   Index Cond: (id = 42)
--   Buffers: shared hit=3  ← インデックス2ページ + テーブル1ページ


-- ===== Index Only Scan =====
-- インデックスだけで回答（テーブルアクセス不要）
-- 適する: カバリングインデックスが存在し、必要カラムが全てインデックス内
EXPLAIN ANALYZE SELECT id, name FROM employees WHERE id BETWEEN 1 AND 100;
-- Index Only Scan using idx_emp_covering on employees
--   Index Cond: (id >= 1 AND id <= 100)
--   Heap Fetches: 0  ← テーブルアクセスなし（VACUUMが最新の場合）
--   Heap Fetches: 15 ← VACUUMが遅れていると可視性チェックが必要


-- ===== Bitmap Index Scan + Bitmap Heap Scan =====
-- 複数のインデックスをビットマップで合成してから一括取得
-- 適する: 中程度の行数、複数条件のAND/OR
EXPLAIN ANALYZE
SELECT * FROM employees
WHERE department_id = 10 AND salary > 500000;
-- Bitmap Heap Scan on employees
--   Recheck Cond: (department_id = 10)
--   Filter: (salary > 500000)
--   Rows Removed by Filter: 5
--   ->  Bitmap Index Scan on idx_emp_dept
--         Index Cond: (department_id = 10)
-- → まずビットマップでページ位置を収集、
--   次にページ順にソートしてテーブルを読む（ランダムI/Oを削減）


-- ===== Parallel Seq Scan =====
-- 複数のワーカーで並列にテーブルを読む
-- 適する: 大テーブルの全走査
EXPLAIN ANALYZE SELECT COUNT(*) FROM employees WHERE salary > 300000;
-- Finalize Aggregate
--   ->  Gather  (Workers Planned: 2)
--         ->  Partial Aggregate
--               ->  Parallel Seq Scan on employees
--                     Filter: (salary > 300000)
-- → 2ワーカー + リーダーで並列スキャン
```

### スキャン方式の選択フロー

```
┌──────── オプティマイザのスキャン方式選択 ─────────┐
│                                                    │
│  SELECT * FROM T WHERE condition                   │
│                   │                                │
│          選択率はどれくらい？                       │
│          (条件に該当する行の割合)                   │
│                   │                                │
│    ┌──────────────┼──────────────┐                 │
│    │              │              │                 │
│  ~100%         5-30%          <5%                  │
│    │              │              │                 │
│ Seq Scan    Bitmap Scan    Index Scan              │
│  (全走査)   (バッチ読み)   (ランダム読み)          │
│                                                    │
│  追加の判断基準:                                   │
│  ・テーブルサイズが小さい → Seq Scan が速い        │
│  ・SELECT に必要な列が全てインデックス内            │
│    → Index Only Scan                              │
│  ・テーブルサイズが大きい + CPUコア数が多い         │
│    → Parallel Seq Scan                            │
│  ・OR条件で複数インデックスを使いたい              │
│    → BitmapOr                                     │
└────────────────────────────────────────────────────┘
```

---

## 3. 結合方式

### コード例3: 結合方式の理解

```sql
-- ===== Nested Loop Join =====
-- 外側テーブルの各行に対して内側テーブルをスキャン
-- 適する: 外側が小、内側にインデックスあり
-- 計算量: O(N × M)（インデックスあれば O(N × log M)）
EXPLAIN ANALYZE
SELECT * FROM orders o
    INNER JOIN customers c ON o.customer_id = c.id
WHERE o.id = 42;
-- Nested Loop  (cost=0.56..16.59 rows=1 width=400)
--   ->  Index Scan using orders_pkey on orders o  (rows=1)
--   ->  Index Scan using customers_pkey on customers c  (rows=1)
-- → 外側1行 × 内側1行 = 1回のIndex Scan


-- ===== Hash Join =====
-- 小テーブルでハッシュ表を構築、大テーブルで照合
-- 適する: 等値結合、中〜大テーブル
-- 計算量: O(N + M)、メモリ使用: O(min(N, M))
EXPLAIN ANALYZE
SELECT * FROM orders o
    INNER JOIN customers c ON o.customer_id = c.id;
-- Hash Join  (cost=1.09..35.24 rows=1000 width=400)
--   Hash Cond: (o.customer_id = c.id)
--   ->  Seq Scan on orders o        ← 大テーブル（probe側）
--   ->  Hash                         ← ハッシュ表の構築
--         Buckets: 1024  Batches: 1  Memory Usage: 40kB
--         ->  Seq Scan on customers c  ← 小テーブル（build側）


-- ===== Merge Join =====
-- 両テーブルをソートして並行走査
-- 適する: 大テーブル同士の結合、ソート済みデータ
-- 計算量: O(N log N + M log M)（既ソートなら O(N + M)）
EXPLAIN ANALYZE
SELECT * FROM large_orders o
    INNER JOIN large_customers c ON o.customer_id = c.id
ORDER BY o.customer_id;
-- Merge Join  (cost=...)
--   Merge Cond: (o.customer_id = c.id)
--   ->  Sort  ->  Seq Scan on large_orders
--   ->  Sort  ->  Seq Scan on large_customers
```

### 結合方式の動作図

```
┌────────── 3つの結合方式 ──────────────────────┐
│                                                │
│  Nested Loop (ネステッドループ)                 │
│  ┌─────┐                                      │
│  │ A(1)│ → B のインデックスで一致行を検索      │
│  │ A(2)│ → B のインデックスで一致行を検索      │
│  │ A(3)│ → B のインデックスで一致行を検索      │
│  └─────┘                                      │
│  最適: 外側が少行 + 内側にインデックス          │
│  最悪: 両方が大テーブル + インデックスなし       │
│                                                │
│  Hash Join (ハッシュ結合)                       │
│  ① Build: 小テーブルでハッシュ表構築            │
│  ┌─────┐        ┌───────────┐                 │
│  │  B  │ ──────►│Hash Table │                 │
│  └─────┘ build  └─────┬─────┘                 │
│  ② Probe: 大テーブルでハッシュ表を照合          │
│  ┌─────┐        probe│                        │
│  │  A  │ ────────────┘                        │
│  └─────┘                                      │
│  最適: 等値結合 + 小テーブルがメモリに載る      │
│  注意: work_mem不足 → Batchesが増える → 遅い  │
│                                                │
│  Merge Join (マージ結合)                       │
│  ┌─────────┐  ┌─────────┐                     │
│  │A(sorted)│  │B(sorted)│  → 並行比較         │
│  └─────────┘  └─────────┘                     │
│  最適: 大テーブル同士 + インデックスでソート済み │
│  注意: ソートのコストが大きい場合がある         │
└────────────────────────────────────────────────┘
```

---

## 4. 統計情報

### コード例4: 統計情報の確認と更新

```sql
-- テーブルの統計情報を確認
SELECT
    attname AS column_name,
    n_distinct,           -- ユニーク値数（負数は行数に対する割合）
    null_frac,            -- NULL率（0.0〜1.0）
    avg_width,            -- 平均バイト幅
    most_common_vals,     -- 最頻値（上位N個）
    most_common_freqs,    -- 最頻値の出現率
    histogram_bounds      -- ヒストグラムの境界値
FROM pg_stats
WHERE tablename = 'employees' AND attname = 'department_id';

-- n_distinct の解釈:
--   正の値: ユニーク値の推定数（例: 10 → 10種類の値）
--   負の値: 行数に対する割合（例: -0.5 → 行数の50%がユニーク）
--   -1: 全行がユニーク

-- 統計情報の手動更新
ANALYZE employees;

-- 特定カラムの統計精度を上げる（デフォルト100、最大10000）
ALTER TABLE employees ALTER COLUMN department_id SET STATISTICS 1000;
ANALYZE employees;
-- → histogram_bounds のバケット数が増え、カーディナリティ推定が改善

-- テーブルの行数推定 vs 実際
SELECT
    relname,
    reltuples::BIGINT AS estimated_rows,    -- 推定行数（ANALYZEで更新）
    pg_stat_get_live_tuples(oid) AS actual_rows  -- 実際のlive行数
FROM pg_class
WHERE relname = 'employees';
```

### コード例5: 拡張統計（複数列の相関）

```sql
-- 拡張統計: 複数カラム間の相関を考慮した統計情報
-- PostgreSQL 10+ で利用可能

-- 問題: city='東京' AND prefecture='東京都' の選択率を
--        各カラム独立に計算すると実際より小さく見積もる
-- → 「東京に住む人は東京都」という相関を知らないため

-- 依存統計（functional dependencies）
CREATE STATISTICS stat_emp_city_pref (dependencies)
ON city, prefecture FROM addresses;
ANALYZE addresses;

-- n-distinct統計（複合カーディナリティ）
CREATE STATISTICS stat_orders_status_date (ndistinct)
ON status, DATE_TRUNC('month', order_date) FROM orders;
ANALYZE orders;

-- MCV統計（最頻値の組み合わせ）— PostgreSQL 12+
CREATE STATISTICS stat_orders_mcv (mcv)
ON status, payment_method FROM orders;
ANALYZE orders;

-- 拡張統計の効果確認
EXPLAIN SELECT * FROM addresses
WHERE city = '東京' AND prefecture = '東京都';
-- Before: rows=10 (各カラムの選択率を独立に乗算: 0.1 × 0.1 = 0.01)
-- After:  rows=1000 (相関を考慮: 東京→東京都の依存関係)
```

---

## 5. クエリリライト

### コード例6: 非効率なクエリの書き換え

```sql
-- ===== パターン1: OR → UNION ALL =====
-- NG: OR条件でインデックスが効きにくい
SELECT * FROM orders
WHERE customer_id = 42 OR product_id = 100;
-- → Bitmap OR（効率が悪い場合がある）

-- OK: UNION ALLに書き換え
SELECT * FROM orders WHERE customer_id = 42
UNION ALL
SELECT * FROM orders WHERE product_id = 100
  AND customer_id != 42;  -- 重複排除


-- ===== パターン2: IN (サブクエリ) → EXISTS =====
-- NG: IN (サブクエリ) が非効率な場合
SELECT * FROM products
WHERE id IN (SELECT product_id FROM order_items WHERE quantity > 10);

-- OK: EXISTS に書き換え（相関サブクエリとして最適化される）
SELECT * FROM products p
WHERE EXISTS (
    SELECT 1 FROM order_items oi
    WHERE oi.product_id = p.id AND oi.quantity > 10
);

-- OK: JOIN に書き換え（重複に注意）
SELECT DISTINCT p.*
FROM products p
    INNER JOIN order_items oi ON p.id = oi.product_id
WHERE oi.quantity > 10;


-- ===== パターン3: NOT IN → NOT EXISTS =====
-- NG: NOT IN はNULLの扱いが危険 + Anti Join最適化が効きにくい
SELECT * FROM products
WHERE id NOT IN (SELECT product_id FROM discontinued_products);
-- → product_id にNULLがあると結果が空になる（SQL標準の3値論理）

-- OK: NOT EXISTS（NULLセーフ + Anti Joinに最適化される）
SELECT * FROM products p
WHERE NOT EXISTS (
    SELECT 1 FROM discontinued_products dp
    WHERE dp.product_id = p.id
);
-- → Anti Join: Hash Anti Join or Merge Anti Join で効率的に処理


-- ===== パターン4: CTE（WITH句）の注意点 =====
-- PostgreSQL 12未満: CTEは最適化バリア（インライン展開されない）
-- PostgreSQL 12+: 1回だけ参照されるCTEは自動的にインライン展開

-- NG (PG11以前): CTEがSeq Scanを強制
WITH active_orders AS (
    SELECT * FROM orders WHERE status = 'active'
)
SELECT * FROM active_orders WHERE customer_id = 42;
-- → CTEが全active注文をマテリアライズしてからフィルタ

-- OK: サブクエリまたはCTE + MATERIALIZED/NOT MATERIALIZED
-- PG12+ではデフォルトでインライン展開される
WITH active_orders AS NOT MATERIALIZED (
    SELECT * FROM orders WHERE status = 'active'
)
SELECT * FROM active_orders WHERE customer_id = 42;
-- → customer_id=42 の条件がordersテーブルまでプッシュダウン


-- ===== パターン5: OFFSET の代わりにカーソルベースページング =====
-- NG: OFFSET が大きいと全行をスキャンしてからスキップ
SELECT * FROM products ORDER BY id LIMIT 20 OFFSET 100000;
-- → 100020行を読んで最初の100000行を捨てる

-- OK: WHERE id > last_seen_id でカーソルベースページング
SELECT * FROM products
WHERE id > 100000  -- 前回最後のIDを使う
ORDER BY id
LIMIT 20;
-- → Index Scanで20行だけ読む（大幅に高速）
```

### コード例7: パフォーマンス分析クエリ

```sql
-- ===== pg_stat_statements =====
-- 最も遅いクエリの特定（拡張モジュール）
-- postgresql.conf: shared_preload_libraries = 'pg_stat_statements'
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

SELECT
    calls,
    ROUND(total_exec_time::NUMERIC, 2) AS total_ms,
    ROUND(mean_exec_time::NUMERIC, 2) AS avg_ms,
    ROUND(stddev_exec_time::NUMERIC, 2) AS stddev_ms,
    rows,
    ROUND(100.0 * shared_blks_hit / NULLIF(shared_blks_hit + shared_blks_read, 0), 2)
        AS cache_hit_pct,
    LEFT(query, 120) AS query_preview
FROM pg_stat_statements
ORDER BY total_exec_time DESC
LIMIT 20;

-- キャッシュヒット率が低いクエリ（I/Oバウンド）
SELECT
    query,
    calls,
    shared_blks_hit,
    shared_blks_read,
    ROUND(100.0 * shared_blks_hit / NULLIF(shared_blks_hit + shared_blks_read, 0), 2)
        AS cache_hit_pct
FROM pg_stat_statements
WHERE calls > 100
ORDER BY cache_hit_pct ASC
LIMIT 20;


-- ===== auto_explain =====
-- 自動的にスロークエリの実行計画をログに出力
-- postgresql.conf:
-- shared_preload_libraries = 'auto_explain'
-- auto_explain.log_min_duration = '100ms'
-- auto_explain.log_analyze = on
-- auto_explain.log_buffers = on
-- auto_explain.log_format = 'json'


-- ===== 実行計画のJSON形式出力（プログラムからの解析用）=====
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT * FROM employees WHERE department_id = 10;

-- ===== 実行計画の比較（SET で挙動を変えて確認）=====
-- インデックスを無効化して比較
SET enable_indexscan = off;
SET enable_bitmapscan = off;
EXPLAIN ANALYZE SELECT * FROM employees WHERE id = 42;
-- → Seq Scan が使われる

-- 元に戻す
RESET enable_indexscan;
RESET enable_bitmapscan;
```

---

## 6. 高度な最適化テクニック

### コード例8: パラレルクエリと work_mem チューニング

```sql
-- ===== パラレルクエリの制御 =====
-- パラレルクエリのパラメータ
SHOW max_parallel_workers_per_gather;  -- デフォルト: 2
SHOW parallel_tuple_cost;              -- デフォルト: 0.1
SHOW min_parallel_table_scan_size;     -- デフォルト: 8MB

-- パラレル度を上げて大テーブルの集約を高速化
SET max_parallel_workers_per_gather = 4;
EXPLAIN ANALYZE
SELECT department_id, AVG(salary), COUNT(*)
FROM employees
GROUP BY department_id;
-- Finalize GroupAggregate
--   ->  Gather Merge  (Workers Planned: 4)
--         ->  Sort
--               ->  Partial HashAggregate
--                     ->  Parallel Seq Scan on employees

-- ===== work_mem チューニング =====
-- work_mem: ソートやハッシュ操作で使用するメモリ量
SHOW work_mem;  -- デフォルト: 4MB

-- Hash Join で Batches > 1 の場合 → work_mem不足
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id;
-- Hash Join
--   ->  Hash
--         Buckets: 65536  Batches: 4  ← Batches > 1 = ディスクに溢れている
--         Memory Usage: 4096kB

-- work_mem を増やして改善
SET work_mem = '64MB';  -- セッション単位で設定（グローバル設定は慎重に）
EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id;
-- Hash Join
--   ->  Hash
--         Buckets: 262144  Batches: 1  ← 全てメモリ内で処理
--         Memory Usage: 32768kB

-- ソートでのwork_mem影響
EXPLAIN ANALYZE
SELECT * FROM orders ORDER BY created_at DESC;
-- Sort Method: external merge  Disk: 102400kB  ← ディスクソート（遅い）
-- work_mem増加後:
-- Sort Method: quicksort  Memory: 51200kB       ← メモリソート（速い）
```

### コード例9: JITコンパイルとコスト設定

```sql
-- ===== JIT（Just-In-Time）コンパイル =====
-- PostgreSQL 11+: 複雑なクエリの式評価をネイティブコードにコンパイル
SHOW jit;                    -- on (デフォルト)
SHOW jit_above_cost;         -- 100000 (このコスト以上でJIT有効)
SHOW jit_inline_above_cost;  -- 500000
SHOW jit_optimize_above_cost; -- 500000

-- JITが有効な場合の出力
EXPLAIN ANALYZE
SELECT SUM(salary * 1.1 + bonus) FROM employees;
-- JIT:
--   Functions: 4
--   Options: Inlining true, Optimization true, Expressions true
--   Timing: Generation 1.5 ms, Inlining 10.2 ms, Optimization 50.3 ms,
--           Emission 30.1 ms, Total 92.1 ms
-- → JITのオーバーヘッドが大きい場合はjit_above_costを上げる


-- ===== コスト設定の調整 =====
-- SSD環境でのrandom_page_costの最適化
SHOW random_page_cost;  -- デフォルト: 4.0（HDD想定）
SET random_page_cost = 1.1;  -- SSD環境では1.1〜1.5が推奨
-- → Index Scanがより積極的に選択される

-- effective_cache_size の設定
SHOW effective_cache_size;  -- デフォルト: 4GB
-- サーバの物理メモリの50-75%を設定
-- → オプティマイザがIndex Scanを選びやすくなる
SET effective_cache_size = '32GB';
```

---

## スキャン方式比較表

| スキャン方式 | 適する場面 | 計算量 | I/O特性 | 並列化 |
|------------|-----------|--------|---------|:---:|
| Sequential Scan | 全行/大部分の取得 | O(N) | 順次読み | 可 |
| Index Scan | 少数行の取得 (~5%) | O(log N) | ランダム読み | 不可* |
| Index Only Scan | カバリングインデックスあり | O(log N) | インデックスのみ | 可 |
| Bitmap Index Scan | 中程度の行数 (5-30%) | O(N) | バッチ読み | 可 |
| Parallel Seq Scan | 大テーブルの全走査 | O(N/P) | 並列順次 | 可 |

*PostgreSQL 17でParallel Index Scanが導入予定

## 結合方式比較表

| 結合方式 | 適する場面 | 計算量 | メモリ使用量 | 等値/非等値 |
|---------|-----------|--------|:---:|:---:|
| Nested Loop | 小テーブル + インデックス | O(N*M) or O(N*logM) | 小 | 両方 |
| Hash Join | 等値結合、中テーブル | O(N+M) | 中（ハッシュ表） | 等値のみ |
| Merge Join | 大テーブル、ソート済み | O(NlogN+MlogM) | 小 | 等値のみ |

## コスト設定パラメータ比較表

| パラメータ | デフォルト | SSD推奨 | 影響 |
|-----------|-----------|---------|------|
| seq_page_cost | 1.0 | 1.0 | Seq Scanのコスト基準 |
| random_page_cost | 4.0 | 1.1-1.5 | Index Scanの選択率に影響 |
| effective_cache_size | 4GB | RAM 50-75% | Index Scanの有利さに影響 |
| work_mem | 4MB | 64-256MB* | ソート/ハッシュのメモリ |
| maintenance_work_mem | 64MB | 512MB-2GB | VACUUM/CREATE INDEX |

*接続数 × work_mem が搭載RAMを超えないよう注意

---

## アンチパターン

### アンチパターン1: EXPLAINなしでインデックスを追加

```sql
-- NG: 「遅いからインデックスを追加」という安易な対応
CREATE INDEX idx_guess ON orders (status);
-- 効果がない理由: statusの選択率が低い（80%が'completed'）
-- → Seq Scanの方が速いためオプティマイザがインデックスを無視

-- OK: EXPLAIN ANALYZEで原因を特定してから対策
EXPLAIN ANALYZE
SELECT * FROM orders WHERE status = 'pending';
-- → Seq Scan、Filter: Rows Removed: 95000
-- → status='pending'は全体の5% → インデックスは有効

-- さらに部分インデックスで最適化
CREATE INDEX idx_orders_pending ON orders (created_at)
WHERE status = 'pending';
```

### アンチパターン2: 推定行数と実際の行数の乖離を無視

```sql
-- 推定 rows=10 vs 実際 rows=50000 のような乖離
-- → オプティマイザが間違った実行計画を選択する原因
-- → 例: Hash Joinの方が適切なのにNested Loopを選択

-- 対策1: ANALYZEを実行して統計を更新
ANALYZE orders;

-- 対策2: 統計精度を上げる
ALTER TABLE orders ALTER COLUMN status SET STATISTICS 1000;
ANALYZE orders;

-- 対策3: 拡張統計（複数列の相関）
CREATE STATISTICS stat_orders_status_date (dependencies)
ON status, order_date FROM orders;
ANALYZE orders;

-- 対策4: 乖離の確認方法
EXPLAIN ANALYZE SELECT * FROM orders WHERE status = 'pending';
-- rows=10 (estimated) vs rows=50000 (actual) → 5000倍の乖離
-- → ANALYZEと拡張統計で改善
```

### アンチパターン3: SELECT * の安易な使用

```sql
-- NG: 不要なカラムまで取得
SELECT * FROM orders WHERE customer_id = 42;
-- → 全カラム取得 → Index Only Scanが使えない
-- → ネットワーク転送量が増大

-- OK: 必要なカラムのみ取得
SELECT id, status, total FROM orders WHERE customer_id = 42;
-- → カバリングインデックスがあればIndex Only Scan可能
-- → 転送データ量が削減
```

---

## 実践演習

### 演習1（基礎）: EXPLAIN出力の読み解き

以下のEXPLAIN ANALYZE出力を読み解き、ボトルネックを特定してください。

```sql
EXPLAIN (ANALYZE, BUFFERS)
SELECT o.id, o.total, c.name, c.email
FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE o.created_at > '2024-01-01'
  AND o.status = 'delivered'
ORDER BY o.total DESC
LIMIT 100;

-- 出力:
-- Limit  (cost=45000..45002 rows=100 width=120)
--        (actual time=2500.1..2500.3 rows=100 loops=1)
--   ->  Sort  (cost=45000..45500 rows=50000 width=120)
--              (actual time=2500.0..2500.2 rows=100 loops=1)
--         Sort Key: o.total DESC
--         Sort Method: top-N heapsort  Memory: 50kB
--         ->  Hash Join  (cost=800..44000 rows=50000 width=120)
--                        (actual time=15.0..2400.0 rows=48000 loops=1)
--               Hash Cond: (o.customer_id = c.id)
--               ->  Seq Scan on orders o  (cost=0..40000 rows=50000 width=80)
--                                         (actual time=0.1..2300.0 rows=48000 loops=1)
--                     Filter: (created_at > '2024-01-01' AND status = 'delivered')
--                     Rows Removed by Filter: 952000
--                     Buffers: shared hit=5000 read=30000
--               ->  Hash  (cost=500..500 rows=10000 width=40)
--                          (actual time=10.0..10.0 rows=10000 loops=1)
--                     ->  Seq Scan on customers c
--                     Buffers: shared hit=200
-- Planning Time: 0.5 ms
-- Execution Time: 2500.5 ms
```

<details>
<summary>模範解答</summary>

**ボトルネック分析**:

1. **最大のボトルネック**: `Seq Scan on orders` に2300ms消費
   - `Rows Removed by Filter: 952000` → 100万行中95%がフィルタで除外
   - `Buffers: shared read=30000` → 大量のディスクI/O
   - 原因: orders テーブルに適切なインデックスがない

2. **推定行数の確認**: rows=50000 (estimated) vs rows=48000 (actual) → 統計は正確

3. **Sort**: top-N heapsort Memory: 50kB → LIMIT 100 なので効率的（問題なし）

4. **Hash Join**: customers のHash構築は10ms → 問題なし

**改善策**:

```sql
-- 部分インデックス + カバリング
CREATE INDEX idx_orders_delivered_recent ON orders (total DESC)
INCLUDE (customer_id, created_at)
WHERE status = 'delivered' AND created_at > '2024-01-01';

-- 改善後: Seq Scan → Index Scan + LIMIT が直接適用
-- 2500ms → 数ms に改善
```

**改善後の実行計画**:
```
Limit  (actual time=0.05..0.5 rows=100)
  ->  Nested Loop  (actual time=0.05..0.5 rows=100)
        ->  Index Only Scan using idx_orders_delivered_recent
              (actual time=0.03..0.2 rows=100)
              Heap Fetches: 0
        ->  Index Scan using customers_pkey on customers c
              (actual time=0.002..0.002 rows=1 loops=100)
```

</details>

### 演習2（応用）: クエリリライトによる最適化

以下のクエリを書き換えて高速化してください。

```sql
-- 遅いクエリ: 各顧客の最新注文を取得
SELECT c.id, c.name,
       (SELECT MAX(o.created_at) FROM orders o WHERE o.customer_id = c.id) AS last_order,
       (SELECT SUM(o.total) FROM orders o WHERE o.customer_id = c.id) AS total_spent,
       (SELECT COUNT(*) FROM orders o WHERE o.customer_id = c.id) AS order_count
FROM customers c
WHERE c.status = 'active';
```

<details>
<summary>模範解答</summary>

```sql
-- 方法1: サブクエリをJOIN + GROUP BYに統合
SELECT
    c.id,
    c.name,
    MAX(o.created_at) AS last_order,
    COALESCE(SUM(o.total), 0) AS total_spent,
    COUNT(o.id) AS order_count
FROM customers c
LEFT JOIN orders o ON c.id = o.customer_id
WHERE c.status = 'active'
GROUP BY c.id, c.name;

-- 方法2: LATERAL JOINで効率的に取得
SELECT c.id, c.name, agg.last_order, agg.total_spent, agg.order_count
FROM customers c
LEFT JOIN LATERAL (
    SELECT
        MAX(created_at) AS last_order,
        SUM(total) AS total_spent,
        COUNT(*) AS order_count
    FROM orders
    WHERE customer_id = c.id
) agg ON TRUE
WHERE c.status = 'active';

-- 必要なインデックス
CREATE INDEX idx_orders_customer ON orders (customer_id)
INCLUDE (created_at, total);
```

**解説**: 元のクエリは顧客ごとに3つのスカラーサブクエリを実行するため、顧客数×3回のordersテーブルスキャンが発生する。方法1では1回のJOIN+GROUP BYに統合し、スキャンを1回に削減。方法2のLATERAL JOINは、顧客ごとにインデックスを使った効率的な集約を行う。

</details>

### 演習3（発展）: パフォーマンス監視ダッシュボード

pg_stat_statementsとpg_stat_user_tablesを使って、以下の情報を出力するクエリを作成してください。

1. 最もCPU時間を消費するクエリTOP10
2. キャッシュヒット率が低いテーブルTOP10
3. 推定行数と実際の行数の乖離が大きいテーブル

<details>
<summary>模範解答</summary>

```sql
-- 1. 最もCPU時間を消費するクエリTOP10
SELECT
    ROUND(total_exec_time::NUMERIC, 2) AS total_ms,
    calls,
    ROUND(mean_exec_time::NUMERIC, 2) AS avg_ms,
    rows AS total_rows,
    ROUND(100.0 * shared_blks_hit /
        NULLIF(shared_blks_hit + shared_blks_read, 0), 1) AS cache_hit_pct,
    LEFT(query, 150) AS query
FROM pg_stat_statements
WHERE calls > 10
ORDER BY total_exec_time DESC
LIMIT 10;

-- 2. キャッシュヒット率が低いテーブルTOP10
SELECT
    schemaname || '.' || relname AS table_name,
    heap_blks_hit + heap_blks_read AS total_reads,
    CASE
        WHEN heap_blks_hit + heap_blks_read > 0 THEN
            ROUND(100.0 * heap_blks_hit /
                (heap_blks_hit + heap_blks_read), 2)
        ELSE 100
    END AS cache_hit_pct,
    idx_blks_hit + idx_blks_read AS total_idx_reads,
    pg_size_pretty(pg_total_relation_size(relid)) AS total_size
FROM pg_statio_user_tables
WHERE heap_blks_hit + heap_blks_read > 1000
ORDER BY cache_hit_pct ASC
LIMIT 10;

-- 3. 推定行数と実際の行数の乖離
SELECT
    relname AS table_name,
    reltuples::BIGINT AS estimated_rows,
    n_live_tup AS actual_live_rows,
    CASE
        WHEN reltuples > 0 THEN
            ROUND(ABS(reltuples - n_live_tup) / reltuples * 100, 1)
        ELSE 0
    END AS deviation_pct,
    last_analyze,
    last_autoanalyze
FROM pg_stat_user_tables s
JOIN pg_class c ON s.relid = c.oid
WHERE n_live_tup > 1000
ORDER BY deviation_pct DESC
LIMIT 10;
```

**解説**: この3つのクエリで以下が分かる:
1. どのクエリが最もリソースを消費しているか → 最適化の優先順位
2. どのテーブルのデータがキャッシュに載っていないか → shared_buffers の増加検討
3. どのテーブルの統計が古いか → ANALYZEの実行が必要

</details>


---

## トラブルシューティング

### よくあるエラーと解決策

| エラー | 原因 | 解決策 |
|--------|------|--------|
| 初期化エラー | 設定ファイルの不備 | 設定ファイルのパスと形式を確認 |
| タイムアウト | ネットワーク遅延/リソース不足 | タイムアウト値の調整、リトライ処理の追加 |
| メモリ不足 | データ量の増大 | バッチ処理の導入、ページネーションの実装 |
| 権限エラー | アクセス権限の不足 | 実行ユーザーの権限確認、設定の見直し |
| データ不整合 | 並行処理の競合 | ロック機構の導入、トランザクション管理 |

### デバッグの手順

1. **エラーメッセージの確認**: スタックトレースを読み、発生箇所を特定する
2. **再現手順の確立**: 最小限のコードでエラーを再現する
3. **仮説の立案**: 考えられる原因をリストアップする
4. **段階的な検証**: ログ出力やデバッガを使って仮説を検証する
5. **修正と回帰テスト**: 修正後、関連する箇所のテストも実行する

```python
# デバッグ用ユーティリティ
import logging
import traceback
from functools import wraps

# ロガーの設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """関数の入出力をログ出力するデコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"呼び出し: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"戻り値: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"例外発生: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """データ処理（デバッグ対象）"""
    if not items:
        raise ValueError("空のデータ")
    return [item * 2 for item in items]
```

### パフォーマンス問題の診断

パフォーマンス問題が発生した場合の診断手順:

1. **ボトルネックの特定**: プロファイリングツールで計測
2. **メモリ使用量の確認**: メモリリークの有無をチェック
3. **I/O待ちの確認**: ディスクやネットワークI/Oの状況を確認
4. **同時接続数の確認**: コネクションプールの状態を確認

| 問題の種類 | 診断ツール | 対策 |
|-----------|-----------|------|
| CPU負荷 | cProfile, py-spy | アルゴリズム改善、並列化 |
| メモリリーク | tracemalloc, objgraph | 参照の適切な解放 |
| I/Oボトルネック | strace, iostat | 非同期I/O、キャッシュ |
| DB遅延 | EXPLAIN, slow query log | インデックス、クエリ最適化 |

---

## 設計判断ガイド

### 選択基準マトリクス

技術選択を行う際の判断基準を以下にまとめます。

| 判断基準 | 重視する場合 | 妥協できる場合 |
|---------|------------|-------------|
| パフォーマンス | リアルタイム処理、大規模データ | 管理画面、バッチ処理 |
| 保守性 | 長期運用、チーム開発 | プロトタイプ、短期プロジェクト |
| スケーラビリティ | 成長が見込まれるサービス | 社内ツール、固定ユーザー |
| セキュリティ | 個人情報、金融データ | 公開データ、社内利用 |
| 開発速度 | MVP、市場投入スピード | 品質重視、ミッションクリティカル |

### アーキテクチャパターンの選択

```
┌─────────────────────────────────────────────────┐
│              アーキテクチャ選択フロー              │
├─────────────────────────────────────────────────┤
│                                                 │
│  ① チーム規模は？                                │
│    ├─ 小規模（1-5人）→ モノリス                   │
│    └─ 大規模（10人+）→ ②へ                       │
│                                                 │
│  ② デプロイ頻度は？                               │
│    ├─ 週1回以下 → モノリス + モジュール分割         │
│    └─ 毎日/複数回 → ③へ                          │
│                                                 │
│  ③ チーム間の独立性は？                            │
│    ├─ 高い → マイクロサービス                      │
│    └─ 中程度 → モジュラーモノリス                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

### トレードオフの分析

技術的な判断には必ずトレードオフが伴います。以下の観点で分析を行いましょう:

**1. 短期 vs 長期のコスト**
- 短期的に速い方法が長期的には技術的負債になることがある
- 逆に、過剰な設計は短期的なコストが高く、プロジェクトの遅延を招く

**2. 一貫性 vs 柔軟性**
- 統一された技術スタックは学習コストが低い
- 多様な技術の採用は適材適所が可能だが、運用コストが増加

**3. 抽象化のレベル**
- 高い抽象化は再利用性が高いが、デバッグが困難になる場合がある
- 低い抽象化は直感的だが、コードの重複が発生しやすい

```python
# 設計判断の記録テンプレート
class ArchitectureDecisionRecord:
    """ADR (Architecture Decision Record) の作成"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """背景と課題の記述"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """決定内容の記述"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """結果の追加"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """却下した代替案の追加"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Markdown形式で出力"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## 背景\n{self.context}\n\n"
        md += f"## 決定\n{self.decision}\n\n"
        md += "## 結果\n"
        for c in self.consequences:
            icon = "✅" if c['type'] == 'positive' else "⚠️"
            md += f"- {icon} {c['description']}\n"
        md += "\n## 却下した代替案\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## 実務での適用シナリオ

### シナリオ1: スタートアップでのMVP開発

**状況:** 限られたリソースで素早くプロダクトをリリースする必要がある

**アプローチ:**
- シンプルなアーキテクチャを選択
- 必要最小限の機能に集中
- 自動テストはクリティカルパスのみ
- モニタリングは早期から導入

**学んだ教訓:**
- 完璧を求めすぎない（YAGNI原則）
- ユーザーフィードバックを早期に取得
- 技術的負債は意識的に管理する

### シナリオ2: レガシーシステムのモダナイゼーション

**状況:** 10年以上運用されているシステムを段階的に刷新する

**アプローチ:**
- Strangler Fig パターンで段階的に移行
- 既存のテストがない場合はCharacterization Testを先に作成
- APIゲートウェイで新旧システムを共存
- データ移行は段階的に実施

| フェーズ | 作業内容 | 期間目安 | リスク |
|---------|---------|---------|--------|
| 1. 調査 | 現状分析、依存関係の把握 | 2-4週間 | 低 |
| 2. 基盤 | CI/CD構築、テスト環境 | 4-6週間 | 低 |
| 3. 移行開始 | 周辺機能から順次移行 | 3-6ヶ月 | 中 |
| 4. コア移行 | 中核機能の移行 | 6-12ヶ月 | 高 |
| 5. 完了 | 旧システム廃止 | 2-4週間 | 中 |

### シナリオ3: 大規模チームでの開発

**状況:** 50人以上のエンジニアが同一プロダクトを開発する

**アプローチ:**
- ドメイン駆動設計で境界を明確化
- チームごとにオーナーシップを設定
- 共通ライブラリはInner Source方式で管理
- APIファーストで設計し、チーム間の依存を最小化

```python
# チーム間のAPI契約定義
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """チーム間のAPI契約"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # レスポンスタイムSLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """SLA準拠の確認"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """OpenAPI形式で出力"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# 使用例
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
```

### シナリオ4: パフォーマンスクリティカルなシステム

**状況:** ミリ秒単位のレスポンスが求められるシステム

**最適化ポイント:**
1. キャッシュ戦略（L1: インメモリ、L2: Redis、L3: CDN）
2. 非同期処理の活用
3. コネクションプーリング
4. クエリ最適化とインデックス設計

| 最適化手法 | 効果 | 実装コスト | 適用場面 |
|-----------|------|-----------|---------|
| インメモリキャッシュ | 高 | 低 | 頻繁にアクセスされるデータ |
| CDN | 高 | 低 | 静的コンテンツ |
| 非同期処理 | 中 | 中 | I/O待ちが多い処理 |
| DB最適化 | 高 | 高 | クエリが遅い場合 |
| コード最適化 | 低-中 | 高 | CPU律速の場合 |

---

## チーム開発での活用

### コードレビューのチェックリスト

このトピックに関連するコードレビューで確認すべきポイント:

- [ ] 命名規則が一貫しているか
- [ ] エラーハンドリングが適切か
- [ ] テストカバレッジは十分か
- [ ] パフォーマンスへの影響はないか
- [ ] セキュリティ上の問題はないか
- [ ] ドキュメントは更新されているか

### ナレッジ共有のベストプラクティス

| 方法 | 頻度 | 対象 | 効果 |
|------|------|------|------|
| ペアプログラミング | 随時 | 複雑なタスク | 即時のフィードバック |
| テックトーク | 週1回 | チーム全体 | 知識の水平展開 |
| ADR (設計記録) | 都度 | 将来のメンバー | 意思決定の透明性 |
| 振り返り | 2週間ごと | チーム全体 | 継続的改善 |
| モブプログラミング | 月1回 | 重要な設計 | 合意形成 |

### 技術的負債の管理

```
優先度マトリクス:

        影響度 高
          │
    ┌─────┼─────┐
    │ 計画 │ 即座 │
    │ 的に │ に   │
    │ 対応 │ 対応 │
    ├─────┼─────┤
    │ 記録 │ 次の │
    │ のみ │ Sprint│
    │     │ で   │
    └─────┼─────┘
          │
        影響度 低
    発生頻度 低  発生頻度 高
```

---

## セキュリティの考慮事項

### 一般的な脆弱性と対策

| 脆弱性 | リスクレベル | 対策 | 検出方法 |
|--------|------------|------|---------|
| インジェクション攻撃 | 高 | 入力値のバリデーション・パラメータ化クエリ | SAST/DAST |
| 認証の不備 | 高 | 多要素認証・セッション管理の強化 | ペネトレーションテスト |
| 機密データの露出 | 高 | 暗号化・アクセス制御 | セキュリティ監査 |
| 設定の不備 | 中 | セキュリティヘッダー・最小権限の原則 | 構成スキャン |
| ログの不足 | 中 | 構造化ログ・監査証跡 | ログ分析 |

### セキュアコーディングのベストプラクティス

```python
# セキュアコーディング例
import hashlib
import secrets
import hmac
from typing import Optional

class SecurityUtils:
    """セキュリティユーティリティ"""

    @staticmethod
    def generate_token(length: int = 32) -> str:
        """暗号学的に安全なトークン生成"""
        return secrets.token_urlsafe(length)

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple:
        """パスワードのハッシュ化"""
        if salt is None:
            salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            iterations=100000
        )
        return hashed.hex(), salt

    @staticmethod
    def verify_password(password: str, hashed: str, salt: str) -> bool:
        """パスワードの検証"""
        new_hash, _ = SecurityUtils.hash_password(password, salt)
        return hmac.compare_digest(new_hash, hashed)

    @staticmethod
    def sanitize_input(value: str) -> str:
        """入力値のサニタイズ"""
        dangerous_chars = ['<', '>', '"', "'", '&', '\\']
        result = value
        for char in dangerous_chars:
            result = result.replace(char, '')
        return result.strip()

# 使用例
token = SecurityUtils.generate_token()
hashed, salt = SecurityUtils.hash_password("my_password")
is_valid = SecurityUtils.verify_password("my_password", hashed, salt)
```

### セキュリティチェックリスト

- [ ] 全ての入力値がバリデーションされている
- [ ] 機密情報がログに出力されていない
- [ ] HTTPS が強制されている
- [ ] CORS ポリシーが適切に設定されている
- [ ] 依存パッケージの脆弱性スキャンが実施されている
- [ ] エラーメッセージに内部情報が含まれていない

---

## マイグレーションガイド

### バージョンアップ時の注意点

| バージョン | 主な変更点 | 移行作業 | 影響範囲 |
|-----------|-----------|---------|---------|
| v1.x → v2.x | API設計の刷新 | エンドポイント変更 | 全クライアント |
| v2.x → v3.x | 認証方式の変更 | トークン形式更新 | 認証関連 |
| v3.x → v4.x | データモデル変更 | マイグレーションスクリプト実行 | DB関連 |

### 段階的移行の手順

```python
# マイグレーションスクリプトのテンプレート
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Callable

logger = logging.getLogger(__name__)

class MigrationRunner:
    """段階的マイグレーション実行エンジン"""

    def __init__(self, migration_dir: str):
        self.migration_dir = Path(migration_dir)
        self.migrations: List[Dict] = []
        self.completed: List[str] = []

    def register(self, version: str, description: str,
                 up: Callable, down: Callable):
        """マイグレーションの登録"""
        self.migrations.append({
            'version': version,
            'description': description,
            'up': up,
            'down': down,
            'registered_at': datetime.now().isoformat()
        })

    def run_up(self, target_version: str = None):
        """マイグレーションの実行（アップグレード）"""
        for migration in self.migrations:
            if migration['version'] in self.completed:
                continue
            logger.info(f"実行中: {migration['version']} - "
                       f"{migration['description']}")
            try:
                migration['up']()
                self.completed.append(migration['version'])
                logger.info(f"完了: {migration['version']}")
            except Exception as e:
                logger.error(f"失敗: {migration['version']}: {e}")
                raise
            if target_version and migration['version'] == target_version:
                break

    def run_down(self, target_version: str):
        """マイグレーションのロールバック"""
        for migration in reversed(self.migrations):
            if migration['version'] not in self.completed:
                continue
            if migration['version'] == target_version:
                break
            logger.info(f"ロールバック: {migration['version']}")
            migration['down']()
            self.completed.remove(migration['version'])

    def status(self) -> Dict:
        """マイグレーション状態の確認"""
        return {
            'total': len(self.migrations),
            'completed': len(self.completed),
            'pending': len(self.migrations) - len(self.completed),
            'versions': {
                m['version']: 'completed'
                if m['version'] in self.completed else 'pending'
                for m in self.migrations
            }
        }
```

### ロールバック計画

移行作業には必ずロールバック計画を準備してください:

1. **データのバックアップ**: 移行前に完全バックアップを取得
2. **テスト環境での検証**: 本番と同等の環境で事前検証
3. **段階的なロールアウト**: カナリアリリースで段階的に展開
4. **監視の強化**: 移行中はメトリクスの監視間隔を短縮
5. **判断基準の明確化**: ロールバックを判断する基準を事前に定義
---

## FAQ

### Q1: EXPLAINとEXPLAIN ANALYZEの違いは？

EXPLAINは実行計画を推定するだけでクエリを実行しない（安全）。EXPLAIN ANALYZEは実際にクエリを実行して実測値を表示する。UPDATE/DELETEにEXPLAIN ANALYZEを使う場合は、トランザクション内で実行してROLLBACKすること。

```sql
BEGIN;
EXPLAIN ANALYZE UPDATE orders SET status = 'cancelled' WHERE id = 42;
ROLLBACK;  -- 実際の更新は取り消される
```

### Q2: コスト値の単位は何か？

PostgreSQLのコストは抽象的な単位で、`seq_page_cost = 1.0`（シーケンシャルページ読み取り1回）を基準とする。`random_page_cost`はデフォルト4.0（SSDでは1.1-1.5に下げることが推奨）。コスト値同士の比較は有意だが、絶対値は実時間と直接対応しない。

### Q3: パラレルクエリはいつ有効か？

大テーブルのSeq Scan、大量行の集約（COUNT, SUM等）、大テーブル同士のHash Joinなどで有効。`max_parallel_workers_per_gather`（デフォルト2）で並列度を制御。小テーブルや索引アクセスではオーバーヘッドの方が大きい。テーブルサイズが`min_parallel_table_scan_size`（デフォルト8MB）未満の場合は自動的に無効化される。

### Q4: EXPLAINでloopsが大きい場合はどう対処する？

`loops=10000` のような場合、そのノードが10000回実行されている。表示されている時間は1ループあたりの平均なので、実際の合計時間は `actual time × loops` になる。対策:
- Nested Loopの内側でloopsが大きい → JOINの順序をヒントで変更（`SET join_collapse_limit`）
- 外側の結果行数を減らす（WHERE条件の追加やインデックス改善）

### Q5: テーブルが小さいのにSeq Scanが選ばれるのはなぜ？

テーブルが数ページしかない場合、Index Scan（ランダムI/O）よりSeq Scan（順次I/O）の方が速い。これは正しいオプティマイザの判断であり、問題ではない。

---

## まとめ

| 項目 | 要点 |
|------|------|
| EXPLAIN ANALYZE | 実行計画と実測値の両方を確認。最適化の唯一の出発点 |
| 推定 vs 実際 | rowsの乖離 → ANALYZEで統計更新、拡張統計の活用 |
| スキャン方式 | Seq / Index / Bitmap / Index Only の使い分け |
| 結合方式 | Nested Loop / Hash / Merge の特性理解 |
| 統計情報 | pg_stats、SET STATISTICS、拡張統計(dependencies, ndistinct, mcv) |
| クエリリライト | OR→UNION ALL、NOT IN→NOT EXISTS、CTE最適化 |
| work_mem | Hash JoinのBatches > 1やディスクソートはwork_mem不足のサイン |
| 監視 | pg_stat_statementsでスロークエリ特定、auto_explainで自動記録 |

---

## 次に読むべきガイド

- [03-indexing.md](./03-indexing.md) — インデックスの詳細設計、部分/カバリングインデックス
- [02-transactions.md](./02-transactions.md) — トランザクションとロックの影響
- [02-performance-tuning.md](../03-practical/02-performance-tuning.md) — 総合チューニング（接続プール、shared_buffers等）
- [00-normalization.md](../02-design/00-normalization.md) — スキーマ設計と性能のトレードオフ

---

## 参考文献

1. PostgreSQL Documentation — "Using EXPLAIN" https://www.postgresql.org/docs/current/using-explain.html
2. PostgreSQL Documentation — "Row Estimation Examples" https://www.postgresql.org/docs/current/row-estimation-examples.html
3. Winand, M. (2012). *SQL Performance Explained*. https://use-the-index-luke.com/
4. Citus Data — "PostgreSQL Query Optimization" https://www.citusdata.com/blog/
5. Dalibo — "EXPLAIN depesz" https://explain.depesz.com/ — EXPLAIN出力のビジュアル解析ツール
6. pgMustard — https://www.pgmustard.com/ — EXPLAIN出力の自動分析サービス
