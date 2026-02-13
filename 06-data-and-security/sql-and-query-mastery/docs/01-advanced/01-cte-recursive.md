# CTE / 再帰クエリ — WITH句・階層データ

> CTE（Common Table Expression）はクエリを論理的なブロックに分割する名前付き一時結果セットであり、再帰CTEは木構造や階層データの探索を可能にするSQLの強力な機能である。

## 前提知識

- SQL の基本構文（SELECT, JOIN, WHERE, GROUP BY）
- サブクエリの基本概念
- テーブル結合（INNER JOIN, LEFT JOIN）の理解
- → [SQL基礎](../00-basics/01-select-basics.md) を事前に読むことを推奨

## この章で学ぶこと

1. CTEの基本構文と非再帰CTEによるクエリの構造化
2. 再帰CTEの動作原理と終了条件の設計
3. 階層データ、グラフ探索、連番生成など実践的なパターン
4. CTEの内部実行メカニズムとパフォーマンス最適化
5. DBMS別のCTE実装差異と移植性の考慮
6. 実世界のユースケースと段階的演習

---

## 1. 非再帰CTE（WITH句）

### 1.1 CTEとは何か — 背景と動機

CTEは SQL:1999 標準で導入された機能であり、複雑なクエリを論理的な名前付きブロックに分割する。サブクエリのネストが深くなりがちな場面で、可読性と保守性を大幅に向上させる。

CTEが解決する問題を具体的に見てみよう。

```
┌───────────────── サブクエリのネスト問題 ─────────────────┐
│                                                          │
│  SELECT *                                                │
│  FROM (                                                  │
│    SELECT *                                              │
│    FROM (                                                │
│      SELECT *                                            │
│      FROM (                                              │
│        SELECT ... FROM table1                            │
│        WHERE ...                                         │
│      ) AS sub1                                           │
│      JOIN table2 ON ...                                  │
│    ) AS sub2                                             │
│    WHERE ...                                             │
│  ) AS sub3                                               │
│  ORDER BY ...;                                           │
│                                                          │
│  問題点:                                                  │
│  ・ネストが深く読みづらい                                  │
│  ・同じサブクエリを2箇所で使えない（コピペが必要）          │
│  ・デバッグ時に部分実行が困難                               │
│                                                          │
│  CTEによる解決:                                           │
│  WITH sub1 AS (SELECT ... FROM table1 WHERE ...)         │
│     , sub2 AS (SELECT ... FROM sub1 JOIN table2 ON ...)  │
│     , sub3 AS (SELECT ... FROM sub2 WHERE ...)           │
│  SELECT * FROM sub3 ORDER BY ...;                        │
│                                                          │
│  利点:                                                    │
│  ・フラットな構造で上から下へ読める                         │
│  ・同じCTEを複数回参照可能                                 │
│  ・各CTEを個別にテスト・デバッグ可能                        │
└──────────────────────────────────────────────────────────┘
```

### 1.2 コード例1: 基本的なCTE

```sql
-- CTEの基本構文
WITH department_stats AS (
    SELECT
        department_id,
        COUNT(*) AS emp_count,
        AVG(salary) AS avg_salary,
        MAX(salary) AS max_salary
    FROM employees
    GROUP BY department_id
)
SELECT
    d.name AS department,
    ds.emp_count,
    ds.avg_salary,
    ds.max_salary
FROM department_stats ds
    INNER JOIN departments d ON d.id = ds.department_id
WHERE ds.avg_salary > 500000
ORDER BY ds.avg_salary DESC;

-- 複数CTEの連鎖
WITH
active_employees AS (
    SELECT * FROM employees WHERE status = 'active'
),
dept_summary AS (
    SELECT
        department_id,
        COUNT(*) AS cnt,
        AVG(salary) AS avg_sal
    FROM active_employees
    GROUP BY department_id
),
company_avg AS (
    SELECT AVG(avg_sal) AS overall_avg FROM dept_summary
)
SELECT
    d.name,
    ds.cnt,
    ds.avg_sal,
    ca.overall_avg,
    ds.avg_sal - ca.overall_avg AS diff
FROM dept_summary ds
    CROSS JOIN company_avg ca
    INNER JOIN departments d ON d.id = ds.department_id
ORDER BY diff DESC;
```

### 1.3 CTEの実行フロー

```
┌─────────────── CTE の実行フロー ───────────────────┐
│                                                     │
│  WITH                                               │
│    cte_1 AS (SELECT ...)   ← (1) 最初に評価          │
│    cte_2 AS (SELECT ... FROM cte_1)  ← (2) 次に評価 │
│    cte_3 AS (SELECT ... FROM cte_2)  ← (3) 次に評価 │
│  SELECT ... FROM cte_3     ← (4) メインクエリ実行    │
│                                                     │
│  注意:                                              │
│  - 各CTEは一度だけ定義、複数回参照可能               │
│  - 前方のCTEのみ参照可能（後方参照は不可）           │
│  - PostgreSQL 12+: インライン展開される場合あり      │
│  - MATERIALIZED / NOT MATERIALIZED で制御可能       │
└─────────────────────────────────────────────────────┘
```

### 1.4 CTEの内部実行メカニズム

DBMSがCTEを処理する方法は大きく2つに分かれる。この違いを理解することがパフォーマンス最適化の鍵となる。

```
┌──────────── CTE の内部実装方式 ──────────────────┐
│                                                   │
│  方式1: Materialization（実体化）                  │
│  ┌─────────────────────────────────────────────┐ │
│  │ CTEの結果を一時的なワークテーブルに格納       │ │
│  │ → メインクエリはワークテーブルを参照          │ │
│  │                                               │ │
│  │ 利点: 複数回参照時に再計算を回避             │ │
│  │ 欠点: 外側のWHERE条件がCTE内に伝播しない     │ │
│  │       （プッシュダウンされない）               │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  方式2: Inlining（インライン展開）                 │
│  ┌─────────────────────────────────────────────┐ │
│  │ CTEをサブクエリとして展開し、全体を最適化     │ │
│  │ → オプティマイザが統合的に最適な計画を生成    │ │
│  │                                               │ │
│  │ 利点: WHERE条件のプッシュダウンが可能         │ │
│  │       インデックス活用の最適化                 │ │
│  │ 欠点: 複数回参照時に再計算のコスト            │ │
│  └─────────────────────────────────────────────┘ │
│                                                   │
│  DBMS別のデフォルト動作:                          │
│  ・PostgreSQL 11以前: 常にMaterialization          │
│  ・PostgreSQL 12+: オプティマイザが自動判断        │
│  ・MySQL 8.0: オプティマイザが自動判断             │
│  ・SQL Server: 常にInlining                        │
│  ・Oracle: 自動判断（INLINE/MATERIALIZEヒント）   │
└───────────────────────────────────────────────────┘
```

### 1.5 コード例2: MATERIALIZED / NOT MATERIALIZEDの使い分け

```sql
-- PostgreSQL 12+: 明示的な実体化制御

-- (1) 明示的に実体化（複数回参照するCTEに有効）
WITH MATERIALIZED heavy_computation AS (
    SELECT
        customer_id,
        SUM(amount) AS total_spent,
        COUNT(DISTINCT product_id) AS unique_products,
        AVG(amount) AS avg_order
    FROM orders
    WHERE order_date >= CURRENT_DATE - INTERVAL '1 year'
    GROUP BY customer_id
)
SELECT hc.*, c.name, c.email
FROM heavy_computation hc
    JOIN customers c ON c.id = hc.customer_id
WHERE hc.total_spent > 100000
UNION ALL
SELECT hc.*, c.name, c.email
FROM heavy_computation hc  -- 2回目の参照でも再計算不要
    JOIN customers c ON c.id = hc.customer_id
WHERE hc.unique_products > 50;

-- (2) インライン展開を強制（WHERE条件のプッシュダウンを期待）
WITH NOT MATERIALIZED filtered_orders AS (
    SELECT * FROM orders WHERE status = 'completed'
)
SELECT * FROM filtered_orders
WHERE customer_id = 42;
-- → WHERE status = 'completed' AND customer_id = 42 として
--   customer_idのインデックスが活用される

-- (3) 実行計画の確認方法
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
WITH department_stats AS (
    SELECT department_id, AVG(salary) AS avg_sal
    FROM employees
    GROUP BY department_id
)
SELECT * FROM department_stats WHERE avg_sal > 500000;
```

### 1.6 CTE内でのDML操作（PostgreSQL拡張）

PostgreSQLでは、CTE内でINSERT/UPDATE/DELETEを実行し、その結果をRETURNINGで後続のクエリに渡せる。これは「書き込みCTE」や「Data-Modifying CTE」と呼ばれる。

```sql
-- 例: 古い注文をアーカイブしつつ削除し、削除件数をログに記録
WITH deleted_orders AS (
    DELETE FROM orders
    WHERE order_date < CURRENT_DATE - INTERVAL '5 years'
    RETURNING *
),
archived AS (
    INSERT INTO orders_archive
    SELECT * FROM deleted_orders
    RETURNING order_id
)
SELECT COUNT(*) AS archived_count FROM archived;

-- 例: UPSERTの結果をCTEで活用
WITH upserted AS (
    INSERT INTO product_inventory (product_id, quantity)
    VALUES (101, 50)
    ON CONFLICT (product_id)
    DO UPDATE SET quantity = product_inventory.quantity + EXCLUDED.quantity
    RETURNING product_id, quantity
)
SELECT p.name, u.quantity
FROM upserted u JOIN products p ON p.id = u.product_id;
```

**注意**: Data-Modifying CTEはPostgreSQL固有の拡張であり、SQL標準には含まれない。MySQL、SQL Server、Oracleでは使用できない。

---

## 2. 再帰CTE

### 2.1 再帰CTEの基本構造

```sql
-- 再帰CTEの構造
WITH RECURSIVE cte_name AS (
    -- ベースケース（非再帰項）: 初期行を生成
    SELECT initial_columns
    FROM initial_table
    WHERE initial_condition

    UNION ALL  -- または UNION（重複排除）

    -- 再帰ケース（再帰項）: 前回の結果を使って次の行を生成
    SELECT next_columns
    FROM cte_name  -- 自己参照
        JOIN other_table ON ...
    WHERE termination_condition  -- 終了条件
)
SELECT * FROM cte_name;
```

### 2.2 再帰CTEの実行モデル — 内部動作の詳細

再帰CTEの実行はイテレーティブ（反復的）なプロセスである。内部では「ワーキングテーブル」と「中間テーブル」という2つの一時的なテーブルが使われる。

```
┌──────────── 再帰CTE の詳細実行ステップ ──────────────┐
│                                                       │
│  初期化:                                              │
│  ┌─────────────────────────────────────────────────┐ │
│  │ 1. ベースケースを実行                             │ │
│  │ 2. 結果を「ワーキングテーブル(WT)」に格納         │ │
│  │ 3. 同じ結果を「最終結果テーブル(RT)」にも追加     │ │
│  └─────────────────────────────────────────────────┘ │
│       │                                               │
│  反復ループ (WTが空になるまで繰り返す):                │
│  ┌─────────────────────────────────────────────────┐ │
│  │ 1. 再帰項を実行（WTの行を入力として使用）         │ │
│  │ 2. 結果を「中間テーブル(IT)」に格納               │ │
│  │ 3. ITが空 → ループ終了                            │ │
│  │ 4. ITが非空:                                      │ │
│  │    a. ITの行をRTに追加                            │ │
│  │    b. WTの内容をITで置き換え                      │ │
│  │    c. ITをクリア                                  │ │
│  │    d. ステップ1に戻る                             │ │
│  └─────────────────────────────────────────────────┘ │
│       │                                               │
│  終了:                                                │
│  ┌─────────────────────────────────────────────────┐ │
│  │ RTの全行が最終結果として返される                   │ │
│  └─────────────────────────────────────────────────┘ │
│                                                       │
│  メモリ管理:                                          │
│  ・WTは各反復で前回の結果のみ保持（全履歴は不要）     │
│  ・RTは全反復の結果を蓄積（最終結果の全行）           │
│  ・UNION ALL: RTへの追加は無条件                      │
│  ・UNION: RTに既存の行と重複する行は追加しない        │
└───────────────────────────────────────────────────────┘
```

### 2.3 具体例で見るステップバイステップ実行

```sql
-- 1から5までの連番生成で内部動作を追跡
WITH RECURSIVE nums AS (
    SELECT 1 AS n        -- ベースケース
    UNION ALL
    SELECT n + 1         -- 再帰ケース
    FROM nums
    WHERE n < 5          -- 終了条件
)
SELECT n FROM nums;
```

```
┌─── ステップバイステップ実行トレース ──────────────┐
│                                                    │
│  反復0 (ベースケース):                             │
│    WT = {1}     RT = {1}                          │
│                                                    │
│  反復1:                                            │
│    入力: WT = {1}                                  │
│    再帰項実行: 1 + 1 = 2  (n=1 < 5 → OK)          │
│    IT = {2}                                        │
│    WT ← {2}    RT = {1, 2}                        │
│                                                    │
│  反復2:                                            │
│    入力: WT = {2}                                  │
│    再帰項実行: 2 + 1 = 3  (n=2 < 5 → OK)          │
│    IT = {3}                                        │
│    WT ← {3}    RT = {1, 2, 3}                     │
│                                                    │
│  反復3:                                            │
│    入力: WT = {3}                                  │
│    再帰項実行: 3 + 1 = 4  (n=3 < 5 → OK)          │
│    IT = {4}                                        │
│    WT ← {4}    RT = {1, 2, 3, 4}                  │
│                                                    │
│  反復4:                                            │
│    入力: WT = {4}                                  │
│    再帰項実行: 4 + 1 = 5  (n=4 < 5 → OK)          │
│    IT = {5}                                        │
│    WT ← {5}    RT = {1, 2, 3, 4, 5}               │
│                                                    │
│  反復5:                                            │
│    入力: WT = {5}                                  │
│    再帰項実行: 5 + 1 = 6  (n=5 < 5 → NG! 除外)    │
│    IT = {} (空)                                    │
│    → ループ終了                                    │
│                                                    │
│  最終結果: RT = {1, 2, 3, 4, 5}                    │
└────────────────────────────────────────────────────┘
```

### 2.4 UNION ALL vs UNION の違いと使い分け

```sql
-- UNION ALL: 重複を許可（高速、一般的な選択）
WITH RECURSIVE tree AS (
    SELECT id, parent_id, name FROM nodes WHERE id = 1
    UNION ALL
    SELECT n.id, n.parent_id, n.name
    FROM nodes n JOIN tree t ON n.parent_id = t.id
)
SELECT * FROM tree;

-- UNION: 重複を排除（循環グラフで自然な終了条件になる）
-- ただし行全体の比較コストが高い
WITH RECURSIVE reachable AS (
    SELECT node_id FROM edges WHERE source = 'A'
    UNION  -- 既に到達済みのノードは追加しない → 自然に終了
    SELECT e.node_id
    FROM edges e JOIN reachable r ON e.source = r.node_id
)
SELECT * FROM reachable;
```

```
┌──── UNION ALL vs UNION 比較 ────────────────────┐
│                                                   │
│  UNION ALL:                                       │
│  ・重複チェックなし → 高速                        │
│  ・循環グラフでは無限ループの危険                  │
│  ・明示的な終了条件が必須                          │
│  ・使用場面: 木構造（循環なし）、連番生成          │
│                                                   │
│  UNION:                                           │
│  ・重複チェックあり → 低速（ハッシュ/ソート必要）  │
│  ・既出行をスキップ → 循環の自然な防止             │
│  ・使用場面: 有向グラフの到達可能性分析            │
│                                                   │
│  パフォーマンス目安（10万行の結果セット）:         │
│  ・UNION ALL: ~50ms                               │
│  ・UNION:     ~500ms (10倍程度遅い)               │
└───────────────────────────────────────────────────┘
```

---

## 3. 再帰CTEの実践パターン

### 3.1 コード例3: 組織階層（上司-部下関係）

```sql
-- 組織ツリーの探索
CREATE TABLE org_chart (
    id         INTEGER PRIMARY KEY,
    name       VARCHAR(100),
    manager_id INTEGER REFERENCES org_chart(id),
    title      VARCHAR(100)
);

-- サンプルデータ
INSERT INTO org_chart VALUES
    (1, '田中太郎', NULL, 'CEO'),
    (2, '鈴木花子', 1, 'CTO'),
    (3, '佐藤次郎', 1, 'CFO'),
    (4, '高橋美咲', 2, 'VP Engineering'),
    (5, '伊藤健一', 2, 'VP Product'),
    (6, '渡辺真理', 4, 'Sr. Engineer'),
    (7, '山本大輔', 4, 'Sr. Engineer'),
    (8, '中村優子', 6, 'Engineer');

-- 特定の社員から上位の全管理職を取得（ボトムアップ）
WITH RECURSIVE management_chain AS (
    -- ベースケース: 起点の社員
    SELECT id, name, manager_id, title, 0 AS depth
    FROM org_chart
    WHERE id = 42

    UNION ALL

    -- 再帰ケース: 上司を辿る
    SELECT o.id, o.name, o.manager_id, o.title, mc.depth + 1
    FROM org_chart o
        INNER JOIN management_chain mc ON o.id = mc.manager_id
)
SELECT
    REPEAT('  ', depth) || name AS hierarchy,
    title,
    depth
FROM management_chain
ORDER BY depth;

-- 部門長から配下の全社員を取得（トップダウン）
WITH RECURSIVE subordinates AS (
    SELECT id, name, manager_id, title, 0 AS depth,
           ARRAY[name] AS path
    FROM org_chart
    WHERE id = 1  -- CEO

    UNION ALL

    SELECT o.id, o.name, o.manager_id, o.title, s.depth + 1,
           s.path || o.name
    FROM org_chart o
        INNER JOIN subordinates s ON o.manager_id = s.id
    WHERE s.depth < 10  -- 無限再帰防止
)
SELECT
    REPEAT('  ', depth) || name AS tree,
    title,
    array_to_string(path, ' > ') AS full_path
FROM subordinates
ORDER BY path;
```

出力イメージ:

```
tree                    | title          | full_path
------------------------+----------------+----------------------------------
田中太郎               | CEO            | 田中太郎
  鈴木花子             | CTO            | 田中太郎 > 鈴木花子
    高橋美咲           | VP Engineering | 田中太郎 > 鈴木花子 > 高橋美咲
      渡辺真理         | Sr. Engineer   | 田中太郎 > ... > 渡辺真理
        中村優子       | Engineer       | 田中太郎 > ... > 中村優子
      山本大輔         | Sr. Engineer   | 田中太郎 > ... > 山本大輔
    伊藤健一           | VP Product     | 田中太郎 > 鈴木花子 > 伊藤健一
  佐藤次郎             | CFO            | 田中太郎 > 佐藤次郎
```

### 3.2 コード例4: カテゴリツリーとパンくずリスト

```sql
-- カテゴリの階層構造
CREATE TABLE categories (
    id        INTEGER PRIMARY KEY,
    name      VARCHAR(100),
    parent_id INTEGER REFERENCES categories(id)
);

INSERT INTO categories VALUES
    (1, '家電', NULL),
    (2, 'パソコン', 1),
    (3, 'ノートPC', 2),
    (4, '13インチ', 3),
    (5, '15インチ', 3),
    (6, 'デスクトップPC', 2),
    (7, 'スマートフォン', 1),
    (8, 'iPhone', 7),
    (9, 'Android', 7);

-- パンくずリストの生成（ボトムアップ）
WITH RECURSIVE breadcrumb AS (
    SELECT id, name, parent_id, name AS path, 0 AS depth
    FROM categories
    WHERE id = 4  -- 現在のカテゴリ: 13インチ

    UNION ALL

    SELECT c.id, c.name, c.parent_id,
           c.name || ' > ' || b.path,
           b.depth + 1
    FROM categories c
        INNER JOIN breadcrumb b ON c.id = b.parent_id
)
SELECT path
FROM breadcrumb
WHERE parent_id IS NULL;  -- ルートまで辿った結果
-- → "家電 > パソコン > ノートPC > 13インチ"

-- 全カテゴリのパンくずリストを一括生成（トップダウン）
WITH RECURSIVE category_tree AS (
    -- ルートカテゴリ
    SELECT id, name, parent_id,
           name::TEXT AS breadcrumb,
           0 AS depth,
           ARRAY[id] AS id_path
    FROM categories
    WHERE parent_id IS NULL

    UNION ALL

    -- 子カテゴリ
    SELECT c.id, c.name, c.parent_id,
           ct.breadcrumb || ' > ' || c.name,
           ct.depth + 1,
           ct.id_path || c.id
    FROM categories c
        JOIN category_tree ct ON c.parent_id = ct.id
)
SELECT
    REPEAT('  ', depth) || name AS tree_view,
    breadcrumb,
    depth
FROM category_tree
ORDER BY id_path;

-- カテゴリとその全子孫の商品数を集計
WITH RECURSIVE category_descendants AS (
    SELECT id, id AS root_id FROM categories
    UNION ALL
    SELECT c.id, cd.root_id
    FROM categories c
        JOIN category_descendants cd ON c.parent_id = cd.id
)
SELECT
    cat.name,
    COUNT(DISTINCT p.id) AS product_count
FROM category_descendants cd
    JOIN categories cat ON cat.id = cd.root_id
    LEFT JOIN products p ON p.category_id = cd.id
GROUP BY cat.id, cat.name
ORDER BY product_count DESC;
```

### 3.3 コード例5: 連番生成とカレンダー

```sql
-- 1〜100の連番を生成
WITH RECURSIVE numbers AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n < 100
)
SELECT n FROM numbers;

-- 日付シーケンスの生成（カレンダー）
WITH RECURSIVE calendar AS (
    SELECT DATE '2024-01-01' AS dt
    UNION ALL
    SELECT dt + INTERVAL '1 day'
    FROM calendar
    WHERE dt < '2024-12-31'
)
SELECT
    dt,
    EXTRACT(DOW FROM dt) AS day_of_week,
    TO_CHAR(dt, 'YYYY-MM') AS month
FROM calendar;

-- ※ PostgreSQLではgenerate_series()の方が効率的
SELECT generate_series('2024-01-01'::DATE, '2024-12-31'::DATE, '1 day') AS dt;

-- 実用例: 日付ごとの売上を0埋めで表示（欠損日も含める）
WITH RECURSIVE date_range AS (
    SELECT DATE '2024-01-01' AS dt
    UNION ALL
    SELECT dt + INTERVAL '1 day' FROM date_range WHERE dt < '2024-01-31'
)
SELECT
    dr.dt,
    COALESCE(SUM(o.amount), 0) AS daily_revenue,
    COUNT(o.id) AS order_count
FROM date_range dr
    LEFT JOIN orders o ON o.order_date = dr.dt
GROUP BY dr.dt
ORDER BY dr.dt;

-- 応用: 時間帯別スロット生成（予約システム）
WITH RECURSIVE time_slots AS (
    SELECT TIME '09:00' AS slot_start, TIME '09:30' AS slot_end
    UNION ALL
    SELECT
        slot_start + INTERVAL '30 minutes',
        slot_end + INTERVAL '30 minutes'
    FROM time_slots
    WHERE slot_start < TIME '17:00'
)
SELECT
    ts.slot_start,
    ts.slot_end,
    CASE WHEN r.id IS NOT NULL THEN '予約済' ELSE '空き' END AS status
FROM time_slots ts
    LEFT JOIN reservations r
        ON r.start_time <= ts.slot_start AND r.end_time > ts.slot_start
ORDER BY ts.slot_start;
```

### 3.4 コード例6: グラフの最短経路

```sql
-- グラフ構造（路線図）
CREATE TABLE routes (
    from_station VARCHAR(50),
    to_station   VARCHAR(50),
    distance     INTEGER,
    line_name    VARCHAR(50)
);

INSERT INTO routes VALUES
    ('東京', '品川', 6, '東海道線'),
    ('品川', '横浜', 22, '東海道線'),
    ('東京', '上野', 3, '山手線'),
    ('上野', '大宮', 26, '京浜東北線'),
    ('東京', '新宿', 10, '中央線'),
    ('新宿', '池袋', 5, '山手線'),
    ('池袋', '大宮', 30, '埼京線');

-- 東京から各駅への最短経路
WITH RECURSIVE shortest_path AS (
    SELECT
        from_station,
        to_station,
        distance,
        ARRAY[from_station, to_station] AS path,
        ARRAY[line_name] AS lines,
        1 AS hops
    FROM routes
    WHERE from_station = '東京'

    UNION ALL

    SELECT
        sp.from_station,
        r.to_station,
        sp.distance + r.distance,
        sp.path || r.to_station,
        sp.lines || r.line_name,
        sp.hops + 1
    FROM shortest_path sp
        INNER JOIN routes r ON sp.to_station = r.from_station
    WHERE NOT r.to_station = ANY(sp.path)  -- 循環防止
      AND sp.hops < 10                     -- 深さ制限
)
SELECT DISTINCT ON (to_station)
    to_station,
    distance,
    array_to_string(path, ' → ') AS route,
    array_to_string(lines, ', ') AS via_lines,
    hops
FROM shortest_path
ORDER BY to_station, distance;
```

### 3.5 コード例7: BOM（部品表）展開

製造業で頻出するBOM（Bill of Materials）の展開は、再帰CTEの代表的なユースケースである。

```sql
-- 部品表テーブル
CREATE TABLE bom (
    parent_part_id  INTEGER,
    child_part_id   INTEGER,
    quantity        DECIMAL(10,2),
    PRIMARY KEY (parent_part_id, child_part_id)
);

CREATE TABLE parts (
    id    INTEGER PRIMARY KEY,
    name  VARCHAR(100),
    unit_cost DECIMAL(10,2)
);

INSERT INTO parts VALUES
    (1, '自転車', 0),
    (2, 'フレーム', 15000),
    (3, '前輪', 5000),
    (4, '後輪', 5000),
    (5, 'タイヤ', 2000),
    (6, 'リム', 1500),
    (7, 'スポーク', 50),
    (8, 'ハブ', 800);

INSERT INTO bom VALUES
    (1, 2, 1),    -- 自転車 = フレーム x1
    (1, 3, 1),    -- 自転車 = 前輪 x1
    (1, 4, 1),    -- 自転車 = 後輪 x1
    (3, 5, 1),    -- 前輪 = タイヤ x1
    (3, 6, 1),    -- 前輪 = リム x1
    (3, 7, 36),   -- 前輪 = スポーク x36
    (3, 8, 1),    -- 前輪 = ハブ x1
    (4, 5, 1),    -- 後輪 = タイヤ x1
    (4, 6, 1),    -- 後輪 = リム x1
    (4, 7, 36),   -- 後輪 = スポーク x36
    (4, 8, 1);    -- 後輪 = ハブ x1

-- BOM展開: 自転車に必要な全部品と数量・コスト
WITH RECURSIVE bom_explosion AS (
    -- ベースケース: トップレベル製品
    SELECT
        b.parent_part_id,
        b.child_part_id,
        b.quantity,
        b.quantity AS total_quantity,
        1 AS level,
        ARRAY[b.parent_part_id, b.child_part_id] AS path
    FROM bom b
    WHERE b.parent_part_id = 1

    UNION ALL

    -- 再帰ケース: 子部品をさらに展開
    SELECT
        b.parent_part_id,
        b.child_part_id,
        b.quantity,
        be.total_quantity * b.quantity,  -- 累積数量
        be.level + 1,
        be.path || b.child_part_id
    FROM bom b
        JOIN bom_explosion be ON b.parent_part_id = be.child_part_id
    WHERE be.level < 10
)
SELECT
    REPEAT('  ', be.level - 1) || p.name AS part_tree,
    be.total_quantity,
    p.unit_cost,
    be.total_quantity * p.unit_cost AS total_cost,
    be.level
FROM bom_explosion be
    JOIN parts p ON p.id = be.child_part_id
ORDER BY be.path;
```

### 3.6 コード例8: 文字列の再帰的分割（パーサー）

```sql
-- カンマ区切り文字列を行に分割（再帰CTEによる実装）
WITH RECURSIVE split_string AS (
    SELECT
        1 AS idx,
        CASE
            WHEN POSITION(',' IN 'apple,banana,cherry,date') > 0
            THEN LEFT('apple,banana,cherry,date', POSITION(',' IN 'apple,banana,cherry,date') - 1)
            ELSE 'apple,banana,cherry,date'
        END AS token,
        CASE
            WHEN POSITION(',' IN 'apple,banana,cherry,date') > 0
            THEN SUBSTRING('apple,banana,cherry,date' FROM POSITION(',' IN 'apple,banana,cherry,date') + 1)
            ELSE ''
        END AS remainder

    UNION ALL

    SELECT
        idx + 1,
        CASE
            WHEN POSITION(',' IN remainder) > 0
            THEN LEFT(remainder, POSITION(',' IN remainder) - 1)
            ELSE remainder
        END,
        CASE
            WHEN POSITION(',' IN remainder) > 0
            THEN SUBSTRING(remainder FROM POSITION(',' IN remainder) + 1)
            ELSE ''
        END
    FROM split_string
    WHERE remainder <> ''
)
SELECT idx, token FROM split_string;
-- 結果:
-- idx | token
-- ----+--------
--   1 | apple
--   2 | banana
--   3 | cherry
--   4 | date

-- ※ PostgreSQL/MySQL 8.0+ ではstring_to_table()やregexp_split_to_table()を使う方が効率的
SELECT unnest(string_to_array('apple,banana,cherry,date', ',')) AS token;
```

### 3.7 コード例9: フィボナッチ数列と数学的漸化式

```sql
-- フィボナッチ数列の生成
WITH RECURSIVE fibonacci AS (
    SELECT 1 AS n, 1::BIGINT AS fib, 0::BIGINT AS prev_fib

    UNION ALL

    SELECT
        n + 1,
        fib + prev_fib,  -- F(n) = F(n-1) + F(n-2)
        fib
    FROM fibonacci
    WHERE n < 50
)
SELECT n, fib FROM fibonacci;

-- 階乗の計算
WITH RECURSIVE factorial AS (
    SELECT 1 AS n, 1::BIGINT AS fact
    UNION ALL
    SELECT n + 1, (n + 1) * fact
    FROM factorial
    WHERE n < 20
)
SELECT n, fact FROM factorial;

-- 複利計算（年利5%、30年）
WITH RECURSIVE compound_interest AS (
    SELECT
        0 AS year,
        1000000.00::DECIMAL(15,2) AS principal,
        0.00::DECIMAL(15,2) AS interest_earned
    UNION ALL
    SELECT
        year + 1,
        (principal * 1.05)::DECIMAL(15,2),
        (principal * 1.05 - 1000000)::DECIMAL(15,2)
    FROM compound_interest
    WHERE year < 30
)
SELECT year, principal, interest_earned
FROM compound_interest;
```

---

## 4. DBMS別のCTE対応状況と構文差異

### 4.1 DBMS間の互換性比較表

| 機能 | PostgreSQL | MySQL 8.0+ | SQL Server | Oracle | SQLite |
|------|-----------|------------|------------|--------|--------|
| 非再帰CTE | 8.4+ | 8.0+ | 2005+ | 9i R2+ | 3.8.3+ |
| 再帰CTE | 8.4+ | 8.0+ | 2005+ | 11g R2+ | 3.8.3+ |
| RECURSIVE キーワード | 必須 | 必須 | 不要 | 不要 | 必須 |
| MATERIALIZED ヒント | 12+ | 8.0+ | N/A | INLINE ヒント | N/A |
| Data-Modifying CTE | 対応 | 非対応 | 対応 | 非対応 | 非対応 |
| 最大再帰深度デフォルト | 無制限 | 1000 | 100 | 無制限 | 1000 |
| 深度制御方法 | statement_timeout | cte_max_recursion_depth | MAXRECURSION | 自前 | 自前 |
| CYCLE検出 (SQL:2011) | 14+ | 非対応 | 非対応 | 非対応 | 非対応 |
| SEARCH句 (SQL:2011) | 14+ | 非対応 | 非対応 | 非対応 | 非対応 |

### 4.2 DBMS別の構文例

```sql
-- PostgreSQL: WITH RECURSIVE 必須
WITH RECURSIVE tree AS (
    SELECT id, parent_id, name, 0 AS depth FROM nodes WHERE parent_id IS NULL
    UNION ALL
    SELECT n.id, n.parent_id, n.name, t.depth + 1
    FROM nodes n JOIN tree t ON n.parent_id = t.id
)
SELECT * FROM tree;

-- SQL Server: RECURSIVE キーワード不要、MAXRECURSION で制御
WITH tree AS (
    SELECT id, parent_id, name, 0 AS depth FROM nodes WHERE parent_id IS NULL
    UNION ALL
    SELECT n.id, n.parent_id, n.name, t.depth + 1
    FROM nodes n JOIN tree t ON n.parent_id = t.id
)
SELECT * FROM tree
OPTION (MAXRECURSION 200);  -- デフォルト100を引き上げ

-- MySQL 8.0: WITH RECURSIVE 必須、最大深度はシステム変数
SET SESSION cte_max_recursion_depth = 5000;
WITH RECURSIVE tree AS (
    SELECT id, parent_id, name, 0 AS depth FROM nodes WHERE parent_id IS NULL
    UNION ALL
    SELECT n.id, n.parent_id, n.name, t.depth + 1
    FROM nodes n JOIN tree t ON n.parent_id = t.id
)
SELECT * FROM tree;

-- Oracle: RECURSIVE不要、CONNECT BY も使える（レガシー）
-- CTE方式（推奨）
WITH tree (id, parent_id, name, depth) AS (
    SELECT id, parent_id, name, 0 FROM nodes WHERE parent_id IS NULL
    UNION ALL
    SELECT n.id, n.parent_id, n.name, t.depth + 1
    FROM nodes n JOIN tree t ON n.parent_id = t.id
)
SELECT * FROM tree;

-- CONNECT BY方式（Oracle固有、レガシー）
SELECT
    id, parent_id, name,
    LEVEL - 1 AS depth,
    SYS_CONNECT_BY_PATH(name, '/') AS path
FROM nodes
START WITH parent_id IS NULL
CONNECT BY PRIOR id = parent_id
ORDER SIBLINGS BY name;
```

### 4.3 PostgreSQL 14+ の CYCLE / SEARCH 句

PostgreSQL 14で SQL:2011 標準の CYCLE 句と SEARCH 句が実装された。これにより、循環検出と探索順序の制御が宣言的に記述できる。

```sql
-- CYCLE句: 循環検出を宣言的に記述
WITH RECURSIVE graph_search AS (
    SELECT id, linked_id, name FROM graph WHERE id = 1
    UNION ALL
    SELECT g.id, g.linked_id, g.name
    FROM graph g JOIN graph_search gs ON g.id = gs.linked_id
)
CYCLE id SET is_cycle USING path  -- 循環検出
SELECT * FROM graph_search WHERE NOT is_cycle;

-- SEARCH句: 探索順序の制御
-- 深さ優先探索（DFS）
WITH RECURSIVE tree AS (
    SELECT id, parent_id, name FROM nodes WHERE parent_id IS NULL
    UNION ALL
    SELECT n.id, n.parent_id, n.name
    FROM nodes n JOIN tree t ON n.parent_id = t.id
)
SEARCH DEPTH FIRST BY id SET ordercol
SELECT * FROM tree ORDER BY ordercol;

-- 幅優先探索（BFS）
WITH RECURSIVE tree AS (
    SELECT id, parent_id, name FROM nodes WHERE parent_id IS NULL
    UNION ALL
    SELECT n.id, n.parent_id, n.name
    FROM nodes n JOIN tree t ON n.parent_id = t.id
)
SEARCH BREADTH FIRST BY id SET ordercol
SELECT * FROM tree ORDER BY ordercol;
```

---

## 5. CTE vs サブクエリ vs ビュー vs 一時テーブル 比較表

| 特徴 | CTE (WITH) | サブクエリ | ビュー | 一時テーブル |
|------|-----------|-----------|--------|-------------|
| 再利用性 | クエリ内で複数回 | 1回のみ | 全クエリで使用可能 | セッション中 |
| 再帰 | 可能 | 不可能 | 不可能 | 手動ループで代替 |
| 永続性 | クエリ実行中のみ | クエリ実行中のみ | 永続 | セッション中 |
| 可読性 | 高い | 低い（ネストが深い） | 高い | 高い |
| パフォーマンス | インライン展開可 | インライン展開 | 都度展開 | インデックス作成可 |
| 更新可能 | 不可（※PG拡張あり） | 不可 | 条件付き可能 | 可能 |
| CREATE不要 | はい | はい | いいえ | いいえ |
| インデックス | 不可 | 不可 | 基底テーブル依存 | 作成可能 |
| 統計情報 | なし | なし | 基底テーブル依存 | ANALYZE可能 |

### 使い分け指針

```
┌──── CTE/サブクエリ/ビュー/一時テーブル 選択フローチャート ────┐
│                                                                │
│  Q1: 再帰的な探索が必要か？                                    │
│  → Yes: 再帰CTE                                               │
│  → No: Q2へ                                                    │
│                                                                │
│  Q2: 同じ結果セットを複数のクエリで使うか？                     │
│  → Yes:                                                        │
│    Q3: パフォーマンスが重要で、インデックスが欲しいか？          │
│    → Yes: 一時テーブル                                         │
│    → No: ビュー（頻繁に使うならマテビューも検討）              │
│  → No: Q4へ                                                    │
│                                                                │
│  Q4: 同じクエリ内で2回以上参照するか？                          │
│  → Yes: CTE（MATERIALIZEDヒント検討）                         │
│  → No: Q5へ                                                    │
│                                                                │
│  Q5: クエリのネストが深く、可読性に問題があるか？               │
│  → Yes: CTE                                                    │
│  → No: サブクエリ（オプティマイザに最大の自由度）               │
└────────────────────────────────────────────────────────────────┘
```

## 再帰CTE の注意点比較表

| 項目 | 推奨 | 非推奨 |
|------|------|--------|
| 終了条件 | WHERE depth < 100 | なし（無限ループ） |
| 循環検出 | ARRAY + ANY で経路追跡 | 検出なし |
| 結合方法 | UNION ALL（高速） | UNION（重複排除、遅い） |
| 深さ制限 | 明示的に設定 | 暗黙のDBデフォルト |
| データ型 | ベースケースで明示的にCAST | 暗黙の型推論 |
| パス追跡 | ARRAY型カラムで経路を記録 | 経路記録なし |
| 集約関数 | CTE外で実行 | 再帰項内での集約（禁止） |
| サブクエリ | CTE外で実行 | 再帰項内でのサブクエリ（非推奨） |

---

## 6. パフォーマンス最適化

### 6.1 再帰CTEのパフォーマンス特性

```
┌──── 再帰CTEのコスト構造 ────────────────────────────┐
│                                                      │
│  全体コスト = ベースケースコスト                      │
│             + Σ(各反復のコスト)                       │
│             + 最終結果の処理コスト                    │
│                                                      │
│  各反復のコスト要因:                                  │
│  ・ワーキングテーブルのスキャン                       │
│  ・JOINの実行（インデックスが効くかが重要）          │
│  ・UNION ALL/UNIONの処理                             │
│  ・ARRAY操作（パス追跡時）                           │
│                                                      │
│  最適化のポイント:                                    │
│  ・JOINカラムにインデックスを作成                     │
│  ・ベースケースの結果セットを最小化                   │
│  ・不要なカラムを再帰項で持ち回らない                 │
│  ・深さ制限で不要な探索を打ち切る                     │
└──────────────────────────────────────────────────────┘
```

### 6.2 インデックス戦略

```sql
-- 階層データの場合: parent_id と id のペアにインデックス
CREATE INDEX idx_org_chart_manager ON org_chart (manager_id);
CREATE INDEX idx_categories_parent ON categories (parent_id);
CREATE INDEX idx_bom_parent ON bom (parent_part_id);

-- グラフデータの場合: 開始ノードと終了ノードの両方
CREATE INDEX idx_routes_from ON routes (from_station);
CREATE INDEX idx_routes_to ON routes (to_station);

-- 実行計画の確認
EXPLAIN (ANALYZE, BUFFERS)
WITH RECURSIVE subordinates AS (
    SELECT id, manager_id FROM org_chart WHERE id = 1
    UNION ALL
    SELECT o.id, o.manager_id
    FROM org_chart o JOIN subordinates s ON o.manager_id = s.id
)
SELECT * FROM subordinates;
-- CTE Scan の下に Index Scan が出ていることを確認
```

### 6.3 大規模データでの最適化テクニック

```sql
-- テクニック1: 再帰深度の動的制御
-- 必要な深さだけ探索して早期終了
WITH RECURSIVE tree AS (
    SELECT id, parent_id, name, 0 AS depth FROM nodes WHERE id = 1
    UNION ALL
    SELECT n.id, n.parent_id, n.name, t.depth + 1
    FROM nodes n JOIN tree t ON n.parent_id = t.id
    WHERE t.depth < 3  -- 3階層までに限定
)
SELECT * FROM tree;

-- テクニック2: 必要なカラムだけを持ち回る
-- NG: 全カラムを再帰で持ち回る（メモリ消費大）
WITH RECURSIVE tree AS (
    SELECT * FROM large_table WHERE id = 1  -- 全カラム
    UNION ALL
    SELECT lt.* FROM large_table lt JOIN tree t ON lt.parent_id = t.id
)
SELECT * FROM tree;

-- OK: IDだけ再帰で辿り、最後にJOINして全カラム取得
WITH RECURSIVE tree_ids AS (
    SELECT id FROM large_table WHERE id = 1  -- IDのみ
    UNION ALL
    SELECT lt.id FROM large_table lt JOIN tree_ids t ON lt.parent_id = t.id
)
SELECT lt.* FROM large_table lt JOIN tree_ids ti ON lt.id = ti.id;

-- テクニック3: Closure Table パターン（再帰CTEの代替）
-- 頻繁に階層クエリを実行する場合、事前にClosure Tableを構築
CREATE TABLE category_closure (
    ancestor_id   INTEGER,
    descendant_id INTEGER,
    depth         INTEGER,
    PRIMARY KEY (ancestor_id, descendant_id)
);

-- Closure Tableの構築（1回だけ実行）
WITH RECURSIVE tree AS (
    SELECT id AS ancestor_id, id AS descendant_id, 0 AS depth
    FROM categories
    UNION ALL
    SELECT t.ancestor_id, c.id, t.depth + 1
    FROM tree t JOIN categories c ON c.parent_id = t.descendant_id
)
INSERT INTO category_closure
SELECT * FROM tree;

-- Closure Tableを使えば再帰なしで子孫を取得
SELECT c.* FROM categories c
    JOIN category_closure cc ON c.id = cc.descendant_id
WHERE cc.ancestor_id = 1 AND cc.depth <= 3;
```

### 6.4 階層データ管理手法の比較

| 手法 | クエリ速度 | INSERT速度 | DELETE速度 | ストレージ | 実装複雑度 |
|------|-----------|-----------|-----------|-----------|-----------|
| 隣接リスト + 再帰CTE | 中 | 高速 | 高速 | 最小 | 低 |
| Closure Table | 高速 | 中（テーブル更新要） | 低速（テーブル更新要） | 大 | 中 |
| Nested Set Model | 高速 | 低速（番号振り直し） | 低速（番号振り直し） | 小 | 高 |
| Materialized Path | 高速 | 高速 | 高速 | 中 | 中 |
| ltree (PostgreSQL) | 高速 | 高速 | 高速 | 中 | 低 |

---

## 7. エッジケースと落とし穴

### 7.1 エッジケース1: ベースケースが0行を返す

```sql
-- ベースケースが該当なし → 再帰も実行されず、空の結果
WITH RECURSIVE tree AS (
    SELECT id, parent_id FROM nodes WHERE id = 99999  -- 存在しないID
    UNION ALL
    SELECT n.id, n.parent_id FROM nodes n JOIN tree t ON n.parent_id = t.id
)
SELECT * FROM tree;
-- → 0行（エラーにはならない）

-- 対策: COALESCE やデフォルト値で空結果に対応
SELECT COALESCE(
    (SELECT COUNT(*) FROM tree WHERE depth <= 5),
    0
) AS descendant_count;
```

### 7.2 エッジケース2: データ型の不一致

```sql
-- NG: ベースケースと再帰ケースでデータ型が異なる
WITH RECURSIVE path AS (
    SELECT id, name FROM nodes WHERE id = 1  -- name: VARCHAR(100)
    UNION ALL
    SELECT n.id, p.name || ' > ' || n.name   -- 結合で長さが不定
    FROM nodes n JOIN path p ON n.parent_id = p.id
)
SELECT * FROM path;
-- → 深い階層で文字列がVARCHAR(100)を超える可能性

-- OK: 明示的にCAST
WITH RECURSIVE path AS (
    SELECT id, name::TEXT AS path_string FROM nodes WHERE id = 1
    UNION ALL
    SELECT n.id, (p.path_string || ' > ' || n.name)::TEXT
    FROM nodes n JOIN path p ON n.parent_id = p.id
)
SELECT * FROM path;
```

### 7.3 エッジケース3: NULL値を含む階層

```sql
-- parent_id が NULL のノードはルート
-- しかし、NULLの比較はIS NULLを使う必要がある
-- NG: NULL = NULL は FALSE なので結合に失敗
SELECT * FROM nodes n1 JOIN nodes n2 ON n1.parent_id = n2.id;
-- parent_id が NULL の行は結合されない（正しい動作だが注意）

-- ルートノードの検出
WITH RECURSIVE tree AS (
    SELECT id, parent_id, name, 0 AS depth
    FROM nodes
    WHERE parent_id IS NULL  -- ルートの条件

    UNION ALL

    SELECT n.id, n.parent_id, n.name, t.depth + 1
    FROM nodes n JOIN tree t ON n.parent_id = t.id
    WHERE t.depth < 100
)
SELECT * FROM tree;

-- 孤児ノード（親が存在しないが parent_id が NULL でもない）の検出
SELECT n.*
FROM nodes n
    LEFT JOIN nodes parent ON n.parent_id = parent.id
WHERE n.parent_id IS NOT NULL AND parent.id IS NULL;
```

### 7.4 エッジケース4: 複数のルートノード

```sql
-- 組織に複数のルート（例: グループ会社の各CEO）がある場合
WITH RECURSIVE full_tree AS (
    -- 全ルートから開始
    SELECT id, name, manager_id, 0 AS depth, id AS root_id
    FROM org_chart
    WHERE manager_id IS NULL

    UNION ALL

    SELECT o.id, o.name, o.manager_id, ft.depth + 1, ft.root_id
    FROM org_chart o JOIN full_tree ft ON o.manager_id = ft.id
)
SELECT
    root_id,
    REPEAT('  ', depth) || name AS tree,
    depth
FROM full_tree
ORDER BY root_id, depth, name;
```

---

## 8. アンチパターン

### アンチパターン1: 終了条件のない再帰CTE

```sql
-- NG: 無限再帰（DBがハングする）
WITH RECURSIVE infinite AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM infinite  -- 終了条件なし！
)
SELECT * FROM infinite;

-- OK: 終了条件を明示
WITH RECURSIVE safe AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM safe WHERE n < 1000  -- 明示的な上限
)
SELECT * FROM safe;

-- 安全策: PostgreSQLではstatement_timeoutを設定
SET statement_timeout = '5s';

-- MySQL: 最大再帰深度を設定
SET SESSION cte_max_recursion_depth = 5000;

-- SQL Server: OPTION (MAXRECURSION N) を指定
-- OPTION (MAXRECURSION 0) で無制限（危険、本番では避ける）
```

### アンチパターン2: CTEの不必要なMATERIALIZE

```sql
-- NG（PostgreSQL 11以前のデフォルト動作）: CTEが必ず実体化される
WITH expensive_cte AS (
    SELECT * FROM huge_table WHERE category = 'A'
)
SELECT * FROM expensive_cte WHERE id = 42;
-- → huge_tableのcategory='A'全行を実体化してからid=42をフィルタ

-- OK（PostgreSQL 12+）: NOT MATERIALIZEDでインライン展開を強制
WITH expensive_cte AS NOT MATERIALIZED (
    SELECT * FROM huge_table WHERE category = 'A'
)
SELECT * FROM expensive_cte WHERE id = 42;
-- → WHERE category = 'A' AND id = 42 として最適化される
```

### アンチパターン3: 再帰項内での集約関数使用

```sql
-- NG: 再帰項でSUM/COUNT/AVG等は使えない（SQLエラー）
WITH RECURSIVE running_total AS (
    SELECT id, amount, amount AS total FROM orders WHERE id = 1
    UNION ALL
    SELECT o.id, o.amount, SUM(o.amount) OVER ()  -- エラー!
    FROM orders o JOIN running_total rt ON o.id = rt.id + 1
)
SELECT * FROM running_total;

-- OK: 集約はCTEの外で行う
WITH RECURSIVE order_chain AS (
    SELECT id, amount FROM orders WHERE id = 1
    UNION ALL
    SELECT o.id, o.amount
    FROM orders o JOIN order_chain oc ON o.id = oc.id + 1
    WHERE o.id <= 100
)
SELECT id, amount, SUM(amount) OVER (ORDER BY id) AS running_total
FROM order_chain;
```

### アンチパターン4: 再帰CTEの過剰使用

```sql
-- NG: 再帰CTEで単純な連番生成（PostgreSQL）
WITH RECURSIVE nums AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM nums WHERE n < 10000
)
SELECT n FROM nums;

-- OK: generate_series()を使う（遥かに高速）
SELECT generate_series(1, 10000) AS n;

-- NG: 再帰CTEでカレンダー生成（PostgreSQL）
WITH RECURSIVE dates AS (
    SELECT CURRENT_DATE AS dt
    UNION ALL
    SELECT dt + 1 FROM dates WHERE dt < CURRENT_DATE + 365
)
SELECT dt FROM dates;

-- OK: generate_series()を使う
SELECT generate_series(
    CURRENT_DATE,
    CURRENT_DATE + INTERVAL '365 days',
    INTERVAL '1 day'
)::DATE AS dt;
```

---

## 9. 演習問題

### 演習1: 従業員の管理階層（基礎）

以下のテーブルが与えられる。

```sql
CREATE TABLE employees_ex (
    id         INTEGER PRIMARY KEY,
    name       VARCHAR(100),
    manager_id INTEGER,
    salary     INTEGER
);

INSERT INTO employees_ex VALUES
    (1, 'Alice', NULL, 1000000),
    (2, 'Bob', 1, 800000),
    (3, 'Charlie', 1, 750000),
    (4, 'David', 2, 600000),
    (5, 'Eve', 2, 650000),
    (6, 'Frank', 3, 500000),
    (7, 'Grace', 4, 450000),
    (8, 'Heidi', 4, 480000);
```

**問題**:
1. Aliceを起点として、全従業員をツリー構造で表示せよ。出力にはインデントと深さを含むこと。
2. 各マネージャーについて、直属・間接合わせた配下の人数と、配下全員の平均給与を算出せよ。
3. 任意の従業員ID（例: 7）から最上位（Alice）までの経路を「Alice > Bob > David > Grace」の形式で表示せよ。

<details>
<summary>解答例（クリックで展開）</summary>

```sql
-- 問題1: ツリー構造表示
WITH RECURSIVE org_tree AS (
    SELECT id, name, manager_id, salary, 0 AS depth, ARRAY[name] AS path
    FROM employees_ex WHERE id = 1
    UNION ALL
    SELECT e.id, e.name, e.manager_id, e.salary, ot.depth + 1, ot.path || e.name
    FROM employees_ex e JOIN org_tree ot ON e.manager_id = ot.id
)
SELECT REPEAT('  ', depth) || name AS tree, salary, depth
FROM org_tree ORDER BY path;

-- 問題2: 配下の人数と平均給与
WITH RECURSIVE all_subordinates AS (
    SELECT id AS manager_id, id AS subordinate_id FROM employees_ex
    UNION ALL
    SELECT als.manager_id, e.id
    FROM employees_ex e JOIN all_subordinates als ON e.manager_id = als.subordinate_id
)
SELECT
    m.name AS manager,
    COUNT(DISTINCT als.subordinate_id) - 1 AS subordinate_count,  -- 自身を除く
    ROUND(AVG(e.salary) FILTER (WHERE als.subordinate_id <> als.manager_id), 0) AS avg_sub_salary
FROM all_subordinates als
    JOIN employees_ex m ON m.id = als.manager_id
    JOIN employees_ex e ON e.id = als.subordinate_id
GROUP BY m.id, m.name
HAVING COUNT(DISTINCT als.subordinate_id) > 1
ORDER BY subordinate_count DESC;

-- 問題3: ボトムアップ経路
WITH RECURSIVE path_up AS (
    SELECT id, name, manager_id, name::TEXT AS chain
    FROM employees_ex WHERE id = 7
    UNION ALL
    SELECT e.id, e.name, e.manager_id, e.name || ' > ' || pu.chain
    FROM employees_ex e JOIN path_up pu ON e.id = pu.manager_id
)
SELECT chain FROM path_up WHERE manager_id IS NULL;
-- → "Alice > Bob > David > Grace"
```
</details>

### 演習2: ファイルシステムのサイズ集計（中級）

```sql
CREATE TABLE filesystem (
    id        INTEGER PRIMARY KEY,
    name      VARCHAR(255),
    parent_id INTEGER,
    is_dir    BOOLEAN,
    size_bytes BIGINT  -- ディレクトリは0、ファイルは実サイズ
);

INSERT INTO filesystem VALUES
    (1, '/', NULL, true, 0),
    (2, 'home', 1, true, 0),
    (3, 'usr', 1, true, 0),
    (4, 'alice', 2, true, 0),
    (5, 'bob', 2, true, 0),
    (6, 'readme.txt', 4, false, 1024),
    (7, 'photo.jpg', 4, false, 5242880),
    (8, 'notes.md', 5, false, 2048),
    (9, 'bin', 3, true, 0),
    (10, 'python3', 9, false, 15728640);
```

**問題**: 各ディレクトリについて、直下および再帰的な合計サイズを計算し、フルパスとともに表示せよ。

<details>
<summary>解答例（クリックで展開）</summary>

```sql
-- フルパスの構築とサイズ集計
WITH RECURSIVE dir_tree AS (
    SELECT id, name, parent_id, is_dir, size_bytes,
           '/' AS full_path, ARRAY[id] AS id_path
    FROM filesystem WHERE parent_id IS NULL

    UNION ALL

    SELECT f.id, f.name, f.parent_id, f.is_dir, f.size_bytes,
           CASE WHEN dt.full_path = '/' THEN '/' || f.name
                ELSE dt.full_path || '/' || f.name END,
           dt.id_path || f.id
    FROM filesystem f JOIN dir_tree dt ON f.parent_id = dt.id
),
dir_sizes AS (
    SELECT
        d.id AS dir_id,
        d.full_path,
        SUM(f.size_bytes) AS total_size,
        COUNT(*) FILTER (WHERE NOT f.is_dir) AS file_count
    FROM dir_tree d
        JOIN dir_tree f ON f.id_path @> ARRAY[d.id]  -- d.idがfの祖先
    WHERE d.is_dir
    GROUP BY d.id, d.full_path
)
SELECT
    full_path,
    pg_size_pretty(total_size) AS total_size_pretty,
    file_count
FROM dir_sizes
ORDER BY full_path;
```
</details>

### 演習3: グラフの全経路列挙（上級）

```sql
CREATE TABLE city_connections (
    city_from VARCHAR(50),
    city_to   VARCHAR(50),
    cost      INTEGER
);

INSERT INTO city_connections VALUES
    ('東京', '大阪', 13000),
    ('東京', '名古屋', 10000),
    ('名古屋', '大阪', 5000),
    ('大阪', '福岡', 15000),
    ('東京', '仙台', 10000),
    ('仙台', '札幌', 16000),
    ('名古屋', '福岡', 18000);
```

**問題**:
1. 東京から福岡への全経路（循環なし）を、コスト順に表示せよ。
2. 東京から他の全都市への最小コスト経路を表示せよ（ダイクストラ的アプローチ）。

<details>
<summary>解答例（クリックで展開）</summary>

```sql
-- 問題1: 東京→福岡の全経路
WITH RECURSIVE all_routes AS (
    SELECT
        city_to,
        cost,
        ARRAY['東京', city_to] AS path,
        1 AS hops
    FROM city_connections
    WHERE city_from = '東京'

    UNION ALL

    SELECT
        cc.city_to,
        ar.cost + cc.cost,
        ar.path || cc.city_to,
        ar.hops + 1
    FROM city_connections cc
        JOIN all_routes ar ON cc.city_from = ar.city_to
    WHERE NOT cc.city_to = ANY(ar.path)
      AND ar.hops < 10
)
SELECT
    array_to_string(path, ' → ') AS route,
    cost,
    hops
FROM all_routes
WHERE city_to = '福岡'
ORDER BY cost;

-- 問題2: 最小コスト経路
WITH RECURSIVE shortest AS (
    SELECT city_to, cost, ARRAY['東京', city_to] AS path
    FROM city_connections WHERE city_from = '東京'
    UNION ALL
    SELECT cc.city_to, s.cost + cc.cost, s.path || cc.city_to
    FROM city_connections cc JOIN shortest s ON cc.city_from = s.city_to
    WHERE NOT cc.city_to = ANY(s.path)
)
SELECT DISTINCT ON (city_to)
    city_to,
    cost,
    array_to_string(path, ' → ') AS route
FROM shortest
ORDER BY city_to, cost;
```
</details>

---

## 10. FAQ

### Q1: CTEは一時テーブルと同じか？

異なる。CTEはクエリ実行中にのみ存在する論理的な名前付き結果セットで、一時テーブルはセッション中に永続する物理的なテーブルである。CTEは別途CREATE/DROPが不要で、クエリの可読性向上に最適。一時テーブルはインデックスの作成や統計情報の取得が可能で、大量データの中間結果を何度も参照する場合に有利。

### Q2: 再帰CTEの最大再帰深度は？

PostgreSQLではデフォルトで無制限（ワーキングメモリが尽きるまで）。`SET max_recursive_iterations`や`statement_timeout`で制御できる。SQL Serverでは`OPTION (MAXRECURSION n)`で指定（デフォルト100）。MySQL 8.0では`cte_max_recursion_depth`システム変数（デフォルト1000）で制御。

### Q3: 再帰CTEとアプリケーション側のループどちらが良いか？

データベース内で完結する再帰CTEの方がネットワークラウンドトリップを避けられるため一般的に高速。ただし、各ステップで複雑なビジネスロジックが必要な場合はアプリケーション側のループが適切。目安として、単純な階層探索は再帰CTE、各ノードで外部APIコールが必要な場合はアプリ側ループが望ましい。

### Q4: CTEのパフォーマンスが悪い場合の診断方法は？

`EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)` で実行計画を確認する。注目すべきポイント:
- **CTE Scan** ノードが出ている場合 → CTEが実体化されている。NOT MATERIALIZEDを検討
- **WorkMem exceeded** → `work_mem` パラメータの調整が必要
- **再帰の反復回数が多い** → 深さ制限や探索範囲の見直し
- **Seq Scan on large table** → JOINカラムにインデックスを追加

### Q5: 循環（サイクル）を含むグラフで再帰CTEを安全に使うには？

3つの方法がある:
1. **ARRAY + ANY**: パスを配列で追跡し、`NOT node = ANY(path)` で既訪問ノードを除外（PostgreSQL推奨）
2. **UNION（UNION ALLではなく）**: 重複行を自動排除（低速だが簡潔）
3. **CYCLE句（PostgreSQL 14+）**: `CYCLE id SET is_cycle USING path` で宣言的に記述

### Q6: WITH RECURSIVEはどのようなクエリに使うべきでないか？

以下のケースでは再帰CTEは不適切:
- **単純な連番生成**: `generate_series()`（PostgreSQL）や数値テーブルの方が高速
- **固定深度の階層**: 自己結合を深度分だけ重ねる方がオプティマイザに最適化の余地がある
- **超大規模グラフ（100万ノード以上）**: 専用のグラフデータベース（Neo4j、Amazon Neptune）の検討を推奨
- **リアルタイム性が要求される場合**: Closure TableやMaterialized Pathの事前計算を検討

### Q7: ORM（Django、Rails、SQLAlchemy等）からCTEを使えるか？

主要なORMはCTEをサポートしている:
- **Django 4.2+**: `With`クラスと`.with_cte()`メソッド（django-cte ライブラリ）
- **SQLAlchemy**: `select().cte(recursive=True)` でネイティブサポート
- **Rails (ActiveRecord)**: Rails 7.1+ で `.with` メソッドが追加
- **Prisma**: rawクエリで対応（ネイティブサポートは限定的）

---

## 11. まとめ

| 項目 | 要点 |
|------|------|
| 非再帰CTE | クエリを論理ブロックに分割。可読性向上。複数回参照可能 |
| 再帰CTE | WITH RECURSIVE で階層・グラフデータを探索 |
| ベースケース | 再帰の起点。非再帰項として定義 |
| 再帰ケース | 自己参照して次の行を生成。終了条件必須 |
| 循環防止 | ARRAY + ANY で訪問済みノードを追跡。PostgreSQL 14+ではCYCLE句 |
| MATERIALIZED | CTEの実体化を明示制御（PostgreSQL 12+） |
| パフォーマンス | JOINカラムのインデックス、深さ制限、必要カラムの最小化が鍵 |
| 代替手法 | Closure Table、Nested Set、Materialized Path、ltree |
| DBMS差異 | RECURSIVE キーワードの要否、最大深度設定が異なる |

---

## 次に読むべきガイド

- [02-transactions.md](./02-transactions.md) — CTEを含むトランザクション管理
- [04-query-optimization.md](./04-query-optimization.md) — CTEの実行計画と最適化
- [01-schema-design.md](../02-design/01-schema-design.md) — 階層データのスキーマ設計（隣接リスト、Closure Table、Nested Set）

---

## 参考文献

1. PostgreSQL Documentation — "WITH Queries (Common Table Expressions)" https://www.postgresql.org/docs/current/queries-with.html
2. Winand, M. — "Modern SQL: WITH Clause" https://modern-sql.com/feature/with
3. Karwin, B. (2010). *SQL Antipatterns*. Chapter 3: Naive Trees. Pragmatic Bookshelf.
4. ISO/IEC 9075-2:2023 — SQL Standard Part 2: Foundation (WITH clause定義)
5. Celko, J. (2012). *Joe Celko's Trees and Hierarchies in SQL for Smarties*. Morgan Kaufmann.
6. PostgreSQL Documentation — "SEARCH and CYCLE clauses" https://www.postgresql.org/docs/current/queries-with.html#QUERIES-WITH-SEARCH
7. MySQL 8.0 Reference Manual — "WITH (Common Table Expressions)" https://dev.mysql.com/doc/refman/8.0/en/with.html
8. Microsoft SQL Server Documentation — "WITH common_table_expression" https://learn.microsoft.com/en-us/sql/t-sql/queries/with-common-table-expression-transact-sql
