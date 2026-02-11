# SQL概要 — 歴史・RDBMS・方言

> SQLはリレーショナルデータベースを操作するための宣言型言語であり、1970年代の誕生以来、データ管理の世界標準として君臨し続けている。

## この章で学ぶこと

1. SQLの歴史的経緯とリレーショナルモデルの理論的背景
2. 主要RDBMSの特徴と選定基準
3. SQL方言の違いと移植性を意識した書き方

---

## 1. SQLの歴史

### 1.1 リレーショナルモデルの誕生

1970年、IBM研究所のEdgar F. Coddが論文「A Relational Model of Data for Large Shared Data Banks」を発表した。これがリレーショナルデータベースの理論的基盤となった。

```
┌──────────────────────────────────────────────────────┐
│                  SQLの歴史年表                         │
├──────────┬───────────────────────────────────────────┤
│   1970   │ E.F. Codd がリレーショナルモデル発表        │
│   1974   │ IBM が SEQUEL (後のSQL) を開発             │
│   1979   │ Oracle V2 (初の商用RDBMS) リリース         │
│   1986   │ SQL-86 (ANSI初の標準化)                    │
│   1992   │ SQL-92 (大幅拡張、現在の基盤)              │
│   1999   │ SQL:1999 (再帰クエリ、トリガー)            │
│   2003   │ SQL:2003 (ウィンドウ関数、XML)             │
│   2011   │ SQL:2011 (テンポラルデータ)                │
│   2016   │ SQL:2016 (JSON、行パターン認識)            │
│   2023   │ SQL:2023 (プロパティグラフクエリ)           │
└──────────┴───────────────────────────────────────────┘
```

### 1.2 「SQL」の読み方

正式には「エス・キュー・エル」だが、歴史的経緯から「シークェル（sequel）」と呼ぶ人も多い。ISO/ANSI標準では「SQL」が正式名称である。

---

## 2. リレーショナルモデルの基礎概念

### コード例1: リレーション（テーブル）の基本

```sql
-- リレーション = テーブル
-- タプル     = 行（レコード）
-- 属性       = 列（カラム）

CREATE TABLE employees (
    employee_id   INTEGER PRIMARY KEY,    -- 属性（主キー）
    name          VARCHAR(100) NOT NULL,  -- 属性
    department_id INTEGER,                -- 属性（外部キー）
    salary        DECIMAL(10, 2),         -- 属性
    hired_date    DATE                    -- 属性
);

-- 1タプル（行）を挿入
INSERT INTO employees (employee_id, name, department_id, salary, hired_date)
VALUES (1, '田中太郎', 10, 450000.00, '2020-04-01');
```

### コード例2: 集合演算の基本

```sql
-- リレーショナル代数の集合演算をSQLで表現

-- 和（UNION）: 2つの結果セットを結合
SELECT name FROM employees_tokyo
UNION
SELECT name FROM employees_osaka;

-- 差（EXCEPT）: 片方にのみ存在
SELECT name FROM employees_tokyo
EXCEPT
SELECT name FROM employees_osaka;

-- 積（INTERSECT）: 両方に存在
SELECT name FROM employees_tokyo
INTERSECT
SELECT name FROM employees_osaka;
```

### リレーショナルモデルの構造

```
┌─────────────────── リレーション (employees) ───────────────────┐
│                                                                │
│  ┌─────────┬──────────┬────────────┬──────────┬────────────┐  │
│  │ emp_id  │  name    │ dept_id    │ salary   │ hired_date │  │ ← 属性（カラム）
│  ├─────────┼──────────┼────────────┼──────────┼────────────┤  │
│  │    1    │ 田中太郎 │     10     │ 450000   │ 2020-04-01 │  │ ← タプル（行）
│  │    2    │ 鈴木花子 │     20     │ 520000   │ 2019-07-15 │  │ ← タプル（行）
│  │    3    │ 佐藤次郎 │     10     │ 380000   │ 2021-01-10 │  │ ← タプル（行）
│  └─────────┴──────────┴────────────┴──────────┴────────────┘  │
│                                                                │
│  主キー: employee_id（各タプルを一意に識別）                    │
│  外部キー: department_id → departments(id)                     │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. 主要RDBMS

### コード例3: PostgreSQLの特徴的な機能

```sql
-- PostgreSQL: 高度なデータ型と拡張性
CREATE TABLE products (
    id       SERIAL PRIMARY KEY,
    name     TEXT NOT NULL,
    tags     TEXT[],                    -- 配列型
    metadata JSONB,                     -- JSONBバイナリ型
    price    NUMRANGE,                  -- 範囲型
    search   TSVECTOR                   -- 全文検索用
);

-- 配列の検索
SELECT * FROM products WHERE 'organic' = ANY(tags);

-- JSONB内の検索
SELECT * FROM products WHERE metadata @> '{"category": "food"}';
```

### コード例4: MySQLの特徴的な機能

```sql
-- MySQL: 広い普及率とシンプルな運用
CREATE TABLE articles (
    id         BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    title      VARCHAR(255) NOT NULL,
    body       LONGTEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- MySQLのフルテキストインデックス
    FULLTEXT INDEX ft_body (body)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- フルテキスト検索
SELECT * FROM articles
WHERE MATCH(body) AGAINST('データベース 設計' IN BOOLEAN MODE);
```

### コード例5: SQLiteの特徴的な機能

```sql
-- SQLite: 組み込み型、サーバー不要
-- ファイル1つがデータベース全体

-- 動的型付け（型アフィニティ）
CREATE TABLE settings (
    key   TEXT PRIMARY KEY,
    value ANY              -- どんな型でも格納可能
);

INSERT INTO settings VALUES ('max_retry', 3);
INSERT INTO settings VALUES ('api_url', 'https://example.com');
INSERT INTO settings VALUES ('enabled', TRUE);
```

### 主要RDBMS比較表

| 特徴 | PostgreSQL | MySQL | SQLite | SQL Server | Oracle |
|------|-----------|-------|--------|------------|--------|
| ライセンス | BSD | GPL/商用 | パブリックドメイン | 商用 | 商用 |
| 最大DB容量 | 無制限 | 256TB | 281TB | 524PB | 無制限 |
| JSON対応 | JSONB（高速） | JSON | JSON1拡張 | JSON | JSON |
| 全文検索 | 組み込み | 組み込み | FTS5拡張 | 組み込み | Oracle Text |
| レプリケーション | ストリーミング | グループ | なし | Always On | Data Guard |
| 拡張性 | 非常に高い | 中程度 | 低い | 高い | 高い |
| 学習コスト | 中程度 | 低い | 非常に低い | 中程度 | 高い |
| 主な用途 | 汎用 | Web | 組み込み/モバイル | エンタープライズ | エンタープライズ |

---

## 4. SQL方言の違い

### 方言比較表

| 操作 | 標準SQL / PostgreSQL | MySQL | SQLite | SQL Server |
|------|---------------------|-------|--------|------------|
| 文字列結合 | `'A' \|\| 'B'` | `CONCAT('A','B')` | `'A' \|\| 'B'` | `'A' + 'B'` |
| LIMIT | `LIMIT 10` | `LIMIT 10` | `LIMIT 10` | `TOP 10` / `FETCH` |
| UPSERT | `ON CONFLICT DO UPDATE` | `ON DUPLICATE KEY UPDATE` | `ON CONFLICT DO UPDATE` | `MERGE` |
| 自動連番 | `SERIAL` / `GENERATED` | `AUTO_INCREMENT` | `AUTOINCREMENT` | `IDENTITY` |
| 日付取得 | `CURRENT_TIMESTAMP` | `NOW()` | `datetime('now')` | `GETDATE()` |
| BOOLEAN | `TRUE` / `FALSE` | `1` / `0` | `1` / `0` | `1` / `0` |
| IF文 | `CASE WHEN` | `IF()` / `CASE` | `CASE WHEN` | `IIF()` / `CASE` |

---

## 5. SQLの分類

```
┌────────────────────────── SQL言語の分類 ──────────────────────────┐
│                                                                   │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────┐ │
│  │  DDL    │  │  DML    │  │  DCL    │  │  TCL    │  │  DQL  │ │
│  │         │  │         │  │         │  │         │  │       │ │
│  │ CREATE  │  │ INSERT  │  │ GRANT   │  │ BEGIN   │  │SELECT │ │
│  │ ALTER   │  │ UPDATE  │  │ REVOKE  │  │ COMMIT  │  │       │ │
│  │ DROP    │  │ DELETE  │  │         │  │ROLLBACK │  │       │ │
│  │ TRUNCATE│  │ MERGE   │  │         │  │SAVEPOINT│  │       │ │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └───────┘ │
│                                                                   │
│  DDL = Data Definition Language    （データ定義言語）              │
│  DML = Data Manipulation Language  （データ操作言語）              │
│  DCL = Data Control Language       （データ制御言語）              │
│  TCL = Transaction Control Language（トランザクション制御言語）    │
│  DQL = Data Query Language         （データ問合せ言語）            │
└───────────────────────────────────────────────────────────────────┘
```

---

## アンチパターン

### アンチパターン1: SQL方言に依存した設計

```sql
-- NG: MySQL固有の構文に依存
SELECT * FROM users LIMIT 10, 20;  -- offset, limit の順序がMySQL固有

-- OK: 標準SQLに近い書き方
SELECT * FROM users
ORDER BY id
OFFSET 10 ROWS
FETCH NEXT 20 ROWS ONLY;  -- SQL:2008標準
```

**問題点**: 特定のRDBMSにロックインされ、将来の移行コストが膨大になる。

### アンチパターン2: RDBMSの選定を後回しにする

```
プロジェクト初期に「とりあえずSQLite」で開発 → 本番で突然PostgreSQLに変更
→ SQLiteにない機能（並行書き込み、権限管理、JSON演算子）を多用していて
  大規模な書き直しが発生
```

**対策**: 開発初期から本番と同じRDBMSを使用する。Docker等で環境構築を容易にする。

---

## FAQ

### Q1: SQLは「古い技術」ではないのか？

SQLは1970年代に生まれたが、SQL:2023まで継続的に拡張されている。JSON対応、グラフクエリ、時系列データなど現代的な機能が追加され続けており、むしろ適用範囲は広がっている。NoSQLブームの後、多くのNoSQLデータベースがSQLライクなクエリ言語を採用した（CQLやN1QLなど）事実が、SQLの設計の優秀さを証明している。

### Q2: どのRDBMSを選べばよいか？

判断基準は以下の通り:
- **個人/小規模プロジェクト**: SQLite（ゼロ設定）
- **Webアプリケーション（汎用）**: PostgreSQL（機能の豊富さ、拡張性）
- **既存のLAMP環境**: MySQL / MariaDB（エコシステムの充実）
- **エンタープライズ / .NET環境**: SQL Server
- **大規模ミッションクリティカル**: Oracle（サポート品質）

### Q3: 標準SQLだけ学べばどのRDBMSでも使えるか？

基本的なCRUD操作（SELECT, INSERT, UPDATE, DELETE）とJOINは標準SQLで書ける。ただし、日付関数、文字列関数、ページネーション構文などは方言差が大きい。標準SQLを基盤としつつ、使用するRDBMS固有の機能も把握することが実務では必要。

---

## まとめ

| 項目 | 要点 |
|------|------|
| SQLの本質 | リレーショナルモデルに基づく宣言型データ操作言語 |
| 標準規格 | SQL:2023が最新。SQL-92が広く実装された基盤 |
| 主要RDBMS | PostgreSQL（汎用）、MySQL（Web）、SQLite（組み込み） |
| SQL分類 | DDL / DML / DCL / TCL / DQL の5分類 |
| 方言対策 | 標準SQLを基盤に、RDBMS固有機能は意識的に分離 |
| 選定基準 | 規模、チームスキル、既存インフラ、ライセンス費用 |

---

## 次に読むべきガイド

- [01-crud-operations.md](./01-crud-operations.md) — CRUD操作の詳細
- [02-joins.md](./02-joins.md) — JOINの全種類と使い分け
- [00-normalization.md](../02-design/00-normalization.md) — 正規化の理論

---

## 参考文献

1. Codd, E.F. (1970). "A Relational Model of Data for Large Shared Data Banks". *Communications of the ACM*, 13(6), 377-387.
2. ISO/IEC 9075:2023 — Information technology — Database languages — SQL (最新SQL標準規格)
3. PostgreSQL Documentation — https://www.postgresql.org/docs/current/
4. Date, C.J. (2019). *SQL and Relational Theory: How to Write Accurate SQL Code*. O'Reilly Media.
