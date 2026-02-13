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
