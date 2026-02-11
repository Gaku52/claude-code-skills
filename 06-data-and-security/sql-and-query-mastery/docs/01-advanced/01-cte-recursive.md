# CTE / 再帰クエリ — WITH句・階層データ

> CTE（Common Table Expression）はクエリを論理的なブロックに分割する名前付き一時結果セットであり、再帰CTEは木構造や階層データの探索を可能にするSQLの強力な機能である。

## この章で学ぶこと

1. CTEの基本構文と非再帰CTEによるクエリの構造化
2. 再帰CTEの動作原理と終了条件の設計
3. 階層データ、グラフ探索、連番生成など実践的なパターン

---

## 1. 非再帰CTE（WITH句）

### コード例1: 基本的なCTE

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

### CTEの実行フロー

```
┌─────────────── CTE の実行フロー ───────────────────┐
│                                                     │
│  WITH                                               │
│    cte_1 AS (SELECT ...)   ← ① 最初に評価          │
│    cte_2 AS (SELECT ... FROM cte_1)  ← ② 次に評価 │
│    cte_3 AS (SELECT ... FROM cte_2)  ← ③ 次に評価 │
│  SELECT ... FROM cte_3     ← ④ メインクエリ実行    │
│                                                     │
│  注意:                                              │
│  - 各CTEは一度だけ定義、複数回参照可能               │
│  - 前方のCTEのみ参照可能（後方参照は不可）           │
│  - PostgreSQL 12+: インライン展開される場合あり      │
│  - MATERIALIZED / NOT MATERIALIZED で制御可能       │
└─────────────────────────────────────────────────────┘
```

---

## 2. 再帰CTE

### コード例2: 再帰CTEの基本構造

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

### 再帰CTEの実行モデル

```
┌──────────── 再帰CTE の実行ステップ ────────────┐
│                                                  │
│  ステップ0 (ベースケース)                        │
│  ┌──────────┐                                   │
│  │ 初期行   │ → ワーキングテーブルに格納         │
│  └──────────┘                                   │
│       │                                          │
│  ステップ1 (再帰1回目)                           │
│  ┌──────────┐                                   │
│  │ ステップ0│ → 再帰項に入力 → 新しい行を生成   │
│  └──────────┘                                   │
│       │                                          │
│  ステップ2 (再帰2回目)                           │
│  ┌──────────┐                                   │
│  │ ステップ1│ → 再帰項に入力 → 新しい行を生成   │
│  └──────────┘                                   │
│       │                                          │
│  ステップN (終了)                                │
│  ┌──────────┐                                   │
│  │ 空の結果 │ → 再帰終了                        │
│  └──────────┘                                   │
│                                                  │
│  最終結果 = 全ステップの行のUNION ALL             │
└──────────────────────────────────────────────────┘
```

---

## 3. 再帰CTEの実践パターン

### コード例3: 組織階層（上司-部下関係）

```sql
-- 組織ツリーの探索
CREATE TABLE org_chart (
    id         INTEGER PRIMARY KEY,
    name       VARCHAR(100),
    manager_id INTEGER REFERENCES org_chart(id),
    title      VARCHAR(100)
);

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

### コード例4: カテゴリツリーとパンくずリスト

```sql
-- カテゴリの階層構造
CREATE TABLE categories (
    id        INTEGER PRIMARY KEY,
    name      VARCHAR(100),
    parent_id INTEGER REFERENCES categories(id)
);

-- パンくずリストの生成
WITH RECURSIVE breadcrumb AS (
    SELECT id, name, parent_id, name AS path, 0 AS depth
    FROM categories
    WHERE id = 15  -- 現在のカテゴリ

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
```

### コード例5: 連番生成とカレンダー

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
```

### コード例6: グラフの最短経路

```sql
-- グラフ構造（路線図）
CREATE TABLE routes (
    from_station VARCHAR(50),
    to_station   VARCHAR(50),
    distance     INTEGER
);

-- 東京から各駅への最短経路
WITH RECURSIVE shortest_path AS (
    SELECT
        from_station,
        to_station,
        distance,
        ARRAY[from_station, to_station] AS path,
        1 AS hops
    FROM routes
    WHERE from_station = '東京'

    UNION ALL

    SELECT
        sp.from_station,
        r.to_station,
        sp.distance + r.distance,
        sp.path || r.to_station,
        sp.hops + 1
    FROM shortest_path sp
        INNER JOIN routes r ON sp.to_station = r.from_station
    WHERE NOT r.to_station = ANY(sp.path)  -- 循環防止
      AND sp.hops < 10                     -- 深さ制限
)
SELECT DISTINCT ON (to_station)
    to_station,
    distance,
    array_to_string(path, ' → ') AS route
FROM shortest_path
ORDER BY to_station, distance;
```

---

## CTE vs サブクエリ vs ビュー 比較表

| 特徴 | CTE (WITH) | サブクエリ | ビュー |
|------|-----------|-----------|--------|
| 再利用性 | クエリ内で複数回 | 1回のみ | 全クエリで使用可能 |
| 再帰 | 可能 | 不可能 | 不可能 |
| 永続性 | クエリ実行中のみ | クエリ実行中のみ | 永続 |
| 可読性 | 高い | 低い（ネストが深い） | 高い |
| パフォーマンス | インライン展開可 | インライン展開 | 都度展開 |
| 更新可能 | 不可 | 不可 | 条件付き可能 |

## 再帰CTE の注意点比較表

| 項目 | 推奨 | 非推奨 |
|------|------|--------|
| 終了条件 | WHERE depth < 100 | なし（無限ループ） |
| 循環検出 | ARRAY + ANY で経路追跡 | 検出なし |
| 結合方法 | UNION ALL（高速） | UNION（重複排除、遅い） |
| 深さ制限 | 明示的に設定 | 暗黙のDBデフォルト |
| データ型 | ベースケースで明示的にCAST | 暗黙の型推論 |

---

## アンチパターン

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

---

## FAQ

### Q1: CTEは一時テーブルと同じか？

異なる。CTEはクエリ実行中にのみ存在する論理的な名前付き結果セットで、一時テーブルはセッション中に永続する物理的なテーブルである。CTEは別途CREATE/DROPが不要で、クエリの可読性向上に最適。

### Q2: 再帰CTEの最大再帰深度は？

PostgreSQLではデフォルトで無制限（ワーキングメモリが尽きるまで）。`SET max_recursive_iterations`や`statement_timeout`で制御できる。SQL Serverでは`OPTION (MAXRECURSION n)`で指定（デフォルト100）。

### Q3: 再帰CTEとアプリケーション側のループどちらが良いか？

データベース内で完結する再帰CTEの方がネットワークラウンドトリップを避けられるため一般的に高速。ただし、各ステップで複雑なビジネスロジックが必要な場合はアプリケーション側のループが適切。

---

## まとめ

| 項目 | 要点 |
|------|------|
| 非再帰CTE | クエリを論理ブロックに分割。可読性向上 |
| 再帰CTE | WITH RECURSIVE で階層・グラフデータを探索 |
| ベースケース | 再帰の起点。非再帰項として定義 |
| 再帰ケース | 自己参照して次の行を生成。終了条件必須 |
| 循環防止 | ARRAY + ANY で訪問済みノードを追跡 |
| MATERIALIZED | CTEの実体化を明示制御（PostgreSQL 12+） |

---

## 次に読むべきガイド

- [02-transactions.md](./02-transactions.md) — CTEを含むトランザクション管理
- [04-query-optimization.md](./04-query-optimization.md) — CTEの実行計画と最適化
- [01-schema-design.md](../02-design/01-schema-design.md) — 階層データのスキーマ設計

---

## 参考文献

1. PostgreSQL Documentation — "WITH Queries (Common Table Expressions)" https://www.postgresql.org/docs/current/queries-with.html
2. Winand, M. — "Modern SQL: WITH Clause" https://modern-sql.com/feature/with
3. Karwin, B. (2010). *SQL Antipatterns*. Chapter 3: Naive Trees. Pragmatic Bookshelf.
