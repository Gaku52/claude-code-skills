# JOIN — INNER / LEFT / RIGHT / FULL / CROSS

> JOINはリレーショナルデータベースの核心機能であり、複数テーブルに分散したデータを結合条件に基づいて一つの結果セットにまとめる操作である。

## この章で学ぶこと

1. 全JOIN種別（INNER, LEFT, RIGHT, FULL, CROSS）の動作原理と使い分け
2. 結合条件の設計とパフォーマンスへの影響
3. 自己結合、複数テーブル結合など実践的なJOINパターン

---

## 1. JOINの全体像

```
┌──────────────────── JOIN種別の分類 ─────────────────────┐
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │              結合（JOIN）                         │   │
│  ├──────────────┬──────────────────────────────────┤   │
│  │  内部結合     │  外部結合                         │   │
│  │  INNER JOIN  │  LEFT / RIGHT / FULL OUTER JOIN  │   │
│  ├──────────────┼──────────────────────────────────┤   │
│  │  交差結合     │  その他                           │   │
│  │  CROSS JOIN  │  NATURAL JOIN / LATERAL JOIN     │   │
│  └──────────────┴──────────────────────────────────┘   │
│                                                         │
│  INNER JOIN  : 両方に存在する行のみ                      │
│  LEFT JOIN   : 左テーブルの全行 + 右の一致行             │
│  RIGHT JOIN  : 右テーブルの全行 + 左の一致行             │
│  FULL JOIN   : 両テーブルの全行                          │
│  CROSS JOIN  : 全組み合わせ（直積）                      │
└─────────────────────────────────────────────────────────┘
```

### サンプルデータ

```sql
-- 以降の例で使用するテーブル
CREATE TABLE departments (
    id   INTEGER PRIMARY KEY,
    name VARCHAR(50)
);

CREATE TABLE employees (
    id            INTEGER PRIMARY KEY,
    name          VARCHAR(100),
    department_id INTEGER REFERENCES departments(id)
);

INSERT INTO departments VALUES (1, '営業'), (2, '開発'), (3, '人事');
INSERT INTO employees VALUES
    (101, '田中', 1),
    (102, '鈴木', 2),
    (103, '佐藤', 1),
    (104, '高橋', NULL);  -- 部署未所属
```

---

## 2. INNER JOIN

### コード例1: INNER JOIN

```sql
-- 両方のテーブルに一致する行のみ返す
SELECT e.name AS employee, d.name AS department
FROM employees e
    INNER JOIN departments d ON e.department_id = d.id;

-- 結果:
-- employee | department
-- ---------+-----------
-- 田中     | 営業
-- 鈴木     | 開発
-- 佐藤     | 営業
--
-- ※ 高橋（department_id=NULL）は除外
-- ※ 人事（社員なし）も除外
```

---

## 3. LEFT JOIN

### コード例2: LEFT JOIN (LEFT OUTER JOIN)

```sql
-- 左テーブル（employees）の全行を保持
SELECT e.name AS employee, d.name AS department
FROM employees e
    LEFT JOIN departments d ON e.department_id = d.id;

-- 結果:
-- employee | department
-- ---------+-----------
-- 田中     | 営業
-- 鈴木     | 開発
-- 佐藤     | 営業
-- 高橋     | NULL        ← 一致なしでもNULLで表示

-- LEFT JOINで「一致しない行」だけ取得
SELECT e.name
FROM employees e
    LEFT JOIN departments d ON e.department_id = d.id
WHERE d.id IS NULL;
-- → 高橋（部署未所属の社員）
```

---

## 4. RIGHT JOIN / FULL JOIN

### コード例3: RIGHT JOINとFULL JOIN

```sql
-- RIGHT JOIN: 右テーブル（departments）の全行を保持
SELECT e.name AS employee, d.name AS department
FROM employees e
    RIGHT JOIN departments d ON e.department_id = d.id;

-- 結果:
-- employee | department
-- ---------+-----------
-- 田中     | 営業
-- 佐藤     | 営業
-- 鈴木     | 開発
-- NULL     | 人事        ← 社員がいない部署もNULLで表示

-- FULL OUTER JOIN: 両方の全行を保持
SELECT e.name AS employee, d.name AS department
FROM employees e
    FULL OUTER JOIN departments d ON e.department_id = d.id;

-- 結果:
-- employee | department
-- ---------+-----------
-- 田中     | 営業
-- 佐藤     | 営業
-- 鈴木     | 開発
-- 高橋     | NULL        ← 左のみ
-- NULL     | 人事        ← 右のみ
```

---

## 5. CROSS JOIN

### コード例4: CROSS JOINと実用例

```sql
-- CROSS JOIN: 全組み合わせ（直積）
-- 3社員 × 3部署 = 9行
SELECT e.name, d.name
FROM employees e
    CROSS JOIN departments d;

-- 実用例: カレンダーテーブルの生成
SELECT
    y.year,
    m.month
FROM generate_series(2020, 2025) AS y(year)
    CROSS JOIN generate_series(1, 12) AS m(month)
ORDER BY y.year, m.month;

-- 実用例: 全商品×全店舗の在庫マトリクス
SELECT
    p.name AS product,
    s.name AS store,
    COALESCE(i.quantity, 0) AS stock
FROM products p
    CROSS JOIN stores s
    LEFT JOIN inventory i ON i.product_id = p.id AND i.store_id = s.id;
```

### JOIN種別のベン図

```
  INNER JOIN          LEFT JOIN           RIGHT JOIN          FULL JOIN
  (共通部分)           (左全体+共通)       (共通+右全体)        (全体)

  ┌───┐ ┌───┐       ┌───┐ ┌───┐       ┌───┐ ┌───┐       ┌───┐ ┌───┐
  │   │█│   │       │███│█│   │       │   │█│███│       │███│█│███│
  │ A │█│ B │       │█A█│█│ B │       │ A │█│█B█│       │█A█│█│█B█│
  │   │█│   │       │███│█│   │       │   │█│███│       │███│█│███│
  └───┘ └───┘       └───┘ └───┘       └───┘ └───┘       └───┘ └───┘
  █ = 結果に含む    █ = 結果に含む    █ = 結果に含む    █ = 結果に含む
```

---

## 6. 実践的なJOINパターン

### コード例5: 自己結合（Self Join）

```sql
-- 社員テーブルで上司と部下の関係を表現
CREATE TABLE staff (
    id         INTEGER PRIMARY KEY,
    name       VARCHAR(100),
    manager_id INTEGER REFERENCES staff(id)
);

-- 自己結合で上司名を取得
SELECT
    s.name AS employee,
    m.name AS manager
FROM staff s
    LEFT JOIN staff m ON s.manager_id = m.id;

-- 同じ部署の社員ペアを列挙（自己結合）
SELECT
    e1.name AS employee_1,
    e2.name AS employee_2
FROM employees e1
    INNER JOIN employees e2
        ON e1.department_id = e2.department_id
        AND e1.id < e2.id;  -- 重複ペア防止
```

### コード例6: 複数テーブルの結合

```sql
-- 3テーブル結合: 注文 → 注文明細 → 商品
SELECT
    o.id AS order_id,
    o.order_date,
    p.name AS product_name,
    oi.quantity,
    oi.unit_price,
    oi.quantity * oi.unit_price AS subtotal
FROM orders o
    INNER JOIN order_items oi ON o.id = oi.order_id
    INNER JOIN products p ON oi.product_id = p.id
WHERE o.customer_id = 42
ORDER BY o.order_date DESC, p.name;
```

---

## JOIN種別比較表

| JOIN種別 | 結果行数 | NULL行 | 主な用途 |
|---------|---------|--------|---------|
| INNER JOIN | 両方に一致する行のみ | なし | 関連データの結合 |
| LEFT JOIN | 左テーブル全行 | 右側にNULL | オプショナルな関連 |
| RIGHT JOIN | 右テーブル全行 | 左側にNULL | LEFT JOINの逆（稀） |
| FULL OUTER JOIN | 両テーブル全行 | 両側にNULL | 差分検出 |
| CROSS JOIN | 左×右の全組み合わせ | なし | マトリクス生成 |
| NATURAL JOIN | 同名列で自動結合 | なし | 非推奨（暗黙結合） |

## ON句 vs USING句 vs WHERE句 比較表

| 方式 | 構文例 | 柔軟性 | 可読性 |
|------|--------|--------|--------|
| ON句 | `ON a.id = b.a_id` | 高い（複合条件可） | 明示的 |
| USING句 | `USING (id)` | 低い（同名列のみ） | 簡潔 |
| WHERE句 | `WHERE a.id = b.a_id` | 高い | 結合条件と混在 |
| NATURAL | 暗黙 | 最低（制御不能） | 危険 |

---

## アンチパターン

### アンチパターン1: 暗黙的結合（カンマ結合）

```sql
-- NG: 旧式のカンマ結合（暗黙的CROSS JOIN + WHERE）
SELECT e.name, d.name
FROM employees e, departments d
WHERE e.department_id = d.id;

-- 問題点:
-- 1. WHERE句を忘れるとCROSS JOINになる
-- 2. 結合条件とフィルタ条件が混在して読みにくい
-- 3. OUTER JOINが表現できない

-- OK: 明示的なJOIN構文
SELECT e.name, d.name
FROM employees e
    INNER JOIN departments d ON e.department_id = d.id;
```

### アンチパターン2: N+1クエリ問題

```sql
-- NG: ループ内でJOINすべきクエリを個別実行
-- アプリケーション側の疑似コード:
-- for dept in get_all_departments():
--     employees = query("SELECT * FROM employees WHERE dept_id = ?", dept.id)
--     # 部署数 N に対して N+1 回のクエリ

-- OK: JOINで1回のクエリに
SELECT d.name AS department, e.name AS employee
FROM departments d
    LEFT JOIN employees e ON d.id = e.department_id
ORDER BY d.name, e.name;
```

---

## FAQ

### Q1: LEFT JOINとINNER JOINのどちらを使うべきか？

「結合先にデータが存在しない場合も結果に含めたいか？」が判断基準。例えば「部署未所属の社員も表示したい」ならLEFT JOIN、「部署に所属する社員だけ表示したい」ならINNER JOINを使う。

### Q2: JOINの順序はパフォーマンスに影響するか？

理論的にはクエリオプティマイザが最適な結合順序を選択するため、書く順序は影響しない。ただし、複雑なクエリやオプティマイザの限界がある場合、ヒント句（`/*+ LEADING(...) */`等）で制御することがある。

### Q3: NATURAL JOINはなぜ非推奨か？

NATURAL JOINは同名の全列で自動結合するため、テーブルに列を追加しただけで結合条件が変わり、予期しない結果を返す危険がある。常にON句で明示的に結合条件を指定すべき。

---

## まとめ

| 項目 | 要点 |
|------|------|
| INNER JOIN | 両テーブルの一致行のみ。最も基本的 |
| LEFT JOIN | 左テーブル全行保持。実務で最頻出 |
| RIGHT JOIN | LEFT JOINの逆。可読性のためLEFTに書き換え推奨 |
| FULL OUTER JOIN | 両テーブル全行保持。差分検出に有用 |
| CROSS JOIN | 直積。マトリクス生成用 |
| 結合条件 | ON句で明示指定。NATURAL JOINは避ける |
| パフォーマンス | 結合列にインデックスを設定 |

---

## 次に読むべきガイド

- [03-aggregation.md](./03-aggregation.md) — GROUP BYと集約関数
- [04-subqueries.md](./04-subqueries.md) — サブクエリの活用
- [03-indexing.md](../01-advanced/03-indexing.md) — JOIN性能を左右するインデックス

---

## 参考文献

1. PostgreSQL Documentation — "Joins Between Tables" https://www.postgresql.org/docs/current/tutorial-join.html
2. Garcia-Molina, H., Ullman, J.D., & Widom, J. (2008). *Database Systems: The Complete Book*. Pearson.
3. Molinaro, A. (2005). *SQL Cookbook*. O'Reilly Media.
