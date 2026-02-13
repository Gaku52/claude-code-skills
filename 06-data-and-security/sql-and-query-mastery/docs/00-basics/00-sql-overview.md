# SQL概要 — 歴史・RDBMS・方言・リレーショナルモデル

> SQLはリレーショナルデータベースを操作するための宣言型言語であり、1970年代の誕生以来、データ管理の世界標準として君臨し続けている。本章ではSQLの歴史的背景からリレーショナルモデルの理論、主要RDBMSの特徴と選定基準、SQL方言の違い、そしてSQLの分類体系までを体系的に学ぶ。

---

## この章で学ぶこと

1. SQLの歴史的経緯とリレーショナルモデルの理論的背景を理解する
2. 主要RDBMSの特徴・アーキテクチャ・選定基準を把握する
3. SQL方言の違いと移植性を意識した書き方を身につける
4. SQLの分類体系（DDL/DML/DCL/TCL/DQL）を正確に区別できるようになる

---

## 前提知識

- コンピュータサイエンスの基礎（データ構造の概念）
- テキストエディタやターミナルの基本操作
- [security-fundamentals 00-basics](../../security-fundamentals/docs/00-basics/) — データ保護の基本概念（推奨）

> 本章はSQL学習の入口であり、プログラミング経験がなくても読み進められるように構成している。

---

## 1. SQLの歴史

### 1.1 リレーショナルモデルの誕生

1970年、IBM研究所のEdgar F. Coddが論文「A Relational Model of Data for Large Shared Data Banks」を発表した。この論文がリレーショナルデータベースの理論的基盤となり、以後50年以上にわたってデータ管理の主流パラダイムとして君臨し続けている。

Coddの画期的な点は、データの物理的格納方法とデータの論理的操作を完全に分離した点にある。それまでのデータベースシステム（階層型、ネットワーク型）では、データの物理的な格納構造を知らなければデータにアクセスできなかった。Coddのリレーショナルモデルは「データは表（リレーション）として論理的に表現し、集合演算で操作する」という宣言的アプローチを提唱した。

### 1.2 SQLの歴史年表

```
┌──────────────────────────────────────────────────────────────┐
│                     SQLの歴史年表                              │
├──────────┬───────────────────────────────────────────────────┤
│   1970   │ E.F. Codd がリレーショナルモデル発表                │
│   1974   │ IBM が SEQUEL (後のSQL) を開発 — System R プロジェクト│
│   1977   │ Larry Ellison が Software Development Labs 設立     │
│   1979   │ Oracle V2 (初の商用RDBMS) リリース                  │
│   1983   │ IBM DB2 リリース                                    │
│   1986   │ SQL-86 (ANSI初の標準化)                             │
│   1989   │ SQL-89 (整合性制約の追加)                           │
│   1992   │ SQL-92 (大幅拡張、現在の基盤、JOIN構文の標準化)     │
│   1995   │ MySQL 1.0 リリース                                  │
│   1996   │ PostgreSQL 6.0 リリース (Ingres後継)                │
│   1999   │ SQL:1999 (再帰クエリ、トリガー、オブジェクト指向拡張)│
│   2000   │ SQLite 1.0 リリース                                 │
│   2003   │ SQL:2003 (ウィンドウ関数、XML、MERGE文)             │
│   2006   │ SQL:2006 (XQuery統合)                               │
│   2008   │ SQL:2008 (TRUNCATE、FETCH FIRST)                   │
│   2011   │ SQL:2011 (テンポラルデータ、期間述語)               │
│   2016   │ SQL:2016 (JSON、行パターン認識、多態テーブル関数)   │
│   2023   │ SQL:2023 (プロパティグラフクエリ、SQL/PGQ)          │
└──────────┴───────────────────────────────────────────────────┘
```

### 1.3 「SQL」の読み方と名称の由来

SQLの前身は「SEQUEL」（Structured English Query Language）であり、IBM System Rプロジェクトで開発された。商標上の問題から「SQL」に改名されたが、歴史的経緯から「シークェル（sequel）」と呼ぶ人も多い。ISO/ANSI標準では「SQL」が正式名称であり、読み方は「エス・キュー・エル」が公式である。

なお、「Structured Query Language」の略とされることが多いが、これは後付けの解釈であり、現在のISO標準では「SQL」は何の略語でもなく、それ自体が正式名称として定義されている。

### 1.4 宣言型言語としてのSQL

SQLは手続き型言語（C、Java、Python等）とは根本的に異なる**宣言型言語**である。

```
┌──────────────── 手続き型 vs 宣言型 ────────────────┐
│                                                      │
│  手続き型（HOW — どうやるか）                         │
│  ┌────────────────────────────────────────────┐    │
│  │ 1. ファイルを開く                            │    │
│  │ 2. 1行ずつ読み込む                           │    │
│  │ 3. 条件に合う行を配列に追加する               │    │
│  │ 4. 配列を並べ替える                           │    │
│  │ 5. 先頭10件を取り出す                         │    │
│  └────────────────────────────────────────────┘    │
│                                                      │
│  宣言型（WHAT — 何が欲しいか）                       │
│  ┌────────────────────────────────────────────┐    │
│  │ SELECT * FROM users                         │    │
│  │ WHERE age >= 20                              │    │
│  │ ORDER BY name                                │    │
│  │ LIMIT 10;                                    │    │
│  │                                              │    │
│  │ → 「どうやって」取得するかはDBエンジンに任せる │    │
│  └────────────────────────────────────────────┘    │
│                                                      │
│  宣言型のメリット:                                    │
│  - 実装の詳細を知らなくてよい                         │
│  - オプティマイザが最適な実行計画を自動選択           │
│  - データ量やインデックスの変化に自動対応             │
└──────────────────────────────────────────────────────┘
```

---

## 2. リレーショナルモデルの基礎概念

### 2.1 数学的基盤

リレーショナルモデルは集合論と一階述語論理に基づいている。

- **ドメイン（定義域）**: ある属性が取りうる値の集合。例えば「年齢」のドメインは0〜150の整数
- **リレーション（関係）**: ドメインの直積の部分集合。実装上はテーブルに対応
- **タプル（組）**: リレーションの要素。実装上は行（レコード）に対応
- **属性（アトリビュート）**: リレーションの各列。実装上はカラムに対応
- **候補キー**: タプルを一意に特定できる属性の最小集合
- **主キー（Primary Key）**: 候補キーの中から選ばれた代表
- **外部キー（Foreign Key）**: 他のリレーションの主キーを参照する属性

### コード例1: リレーション（テーブル）の基本

```sql
-- リレーション = テーブル
-- タプル     = 行（レコード）
-- 属性       = 列（カラム）
-- ドメイン   = 列が取りうる値の範囲（データ型 + 制約）

CREATE TABLE employees (
    employee_id   INTEGER PRIMARY KEY,          -- 主キー（候補キーから選択）
    name          VARCHAR(100) NOT NULL,        -- 属性（NOT NULL制約 = ドメインからNULLを除外）
    email         VARCHAR(255) UNIQUE NOT NULL, -- 属性（UNIQUE制約 = 候補キー）
    department_id INTEGER,                      -- 外部キー（参照整合性）
    salary        DECIMAL(10, 2)
        CHECK (salary >= 0),                    -- CHECK制約 = ドメインの制限
    hired_date    DATE NOT NULL,                -- 属性
    FOREIGN KEY (department_id) REFERENCES departments(id)
);

-- 1タプル（行）を挿入
INSERT INTO employees (employee_id, name, email, department_id, salary, hired_date)
VALUES (1, '田中太郎', 'tanaka@example.com', 10, 450000.00, '2020-04-01');

-- リレーショナル代数の「選択」演算 = WHERE句
SELECT * FROM employees WHERE salary > 400000;

-- リレーショナル代数の「射影」演算 = SELECTで列を指定
SELECT name, salary FROM employees;
```

### コード例2: 集合演算の基本

```sql
-- リレーショナル代数の集合演算をSQLで表現

-- 準備: 2つの互換なリレーションを作成
CREATE TABLE employees_tokyo (
    name VARCHAR(100),
    role VARCHAR(50)
);

CREATE TABLE employees_osaka (
    name VARCHAR(100),
    role VARCHAR(50)
);

INSERT INTO employees_tokyo VALUES ('田中', '開発'), ('鈴木', '営業'), ('佐藤', '開発');
INSERT INTO employees_osaka VALUES ('鈴木', '営業'), ('高橋', '企画'), ('山田', '開発');

-- 和（UNION）: 2つの結果セットを結合（重複排除）
SELECT name, role FROM employees_tokyo
UNION
SELECT name, role FROM employees_osaka;
-- → 田中, 鈴木, 佐藤, 高橋, 山田 （鈴木は1回だけ）

-- UNION ALL: 重複を排除しない（高速）
SELECT name, role FROM employees_tokyo
UNION ALL
SELECT name, role FROM employees_osaka;
-- → 田中, 鈴木, 佐藤, 鈴木, 高橋, 山田 （鈴木が2回）

-- 差（EXCEPT）: 左にのみ存在する行
SELECT name, role FROM employees_tokyo
EXCEPT
SELECT name, role FROM employees_osaka;
-- → 田中, 佐藤 （東京だけにいる社員）

-- 積（INTERSECT）: 両方に存在する行
SELECT name, role FROM employees_tokyo
INTERSECT
SELECT name, role FROM employees_osaka;
-- → 鈴木 （両方にいる社員）
```

### 2.2 リレーショナルモデルの構造図

```
┌──────────────────── リレーション (employees) ─────────────────────┐
│                                                                    │
│  ┌─────────┬──────────┬───────────┬────────────┬────────────┐    │
│  │ emp_id  │  name    │  email    │ dept_id    │ hired_date │    │
│  ├─────────┼──────────┼───────────┼────────────┼────────────┤    │ ← スキーマ
│  │    1    │ 田中太郎 │ tanaka@.. │     10     │ 2020-04-01 │    │ ← タプル1
│  │    2    │ 鈴木花子 │ suzuki@.. │     20     │ 2019-07-15 │    │ ← タプル2
│  │    3    │ 佐藤次郎 │ sato@..   │     10     │ 2021-01-10 │    │ ← タプル3
│  └─────────┴──────────┴───────────┴────────────┴────────────┘    │
│                                                                    │
│  スキーマ: employees(emp_id: INT, name: VARCHAR, ...)              │
│  度数（degree）: 属性の数 = 5                                       │
│  基数（cardinality）: タプルの数 = 3                                │
│  主キー: employee_id（各タプルを一意に識別）                        │
│  外部キー: department_id → departments(id)                         │
│  候補キー: {employee_id}, {email}                                  │
└────────────────────────────────────────────────────────────────────┘
```

### 2.3 正規化の概要

正規化はリレーショナルモデルの重要な設計原則であり、データの冗長性を排除し、更新異常を防ぐ手法である。

```sql
-- 非正規化状態（アンチパターン）
-- 1つのテーブルに全情報を詰め込む
CREATE TABLE orders_denormalized (
    order_id      INTEGER,
    customer_name VARCHAR(100),     -- 顧客名が注文ごとに重複
    customer_email VARCHAR(255),    -- メールも重複
    product_name  VARCHAR(100),     -- 商品名も重複
    product_price DECIMAL(10, 2),   -- 価格も重複
    quantity      INTEGER,
    order_date    DATE
);

-- 正規化後（第3正規形）
-- 各エンティティを独立したテーブルに分離
CREATE TABLE customers (
    id    INTEGER PRIMARY KEY,
    name  VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE products (
    id    INTEGER PRIMARY KEY,
    name  VARCHAR(100) NOT NULL,
    price DECIMAL(10, 2) NOT NULL CHECK (price >= 0)
);

CREATE TABLE orders (
    id          INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id),
    order_date  DATE NOT NULL DEFAULT CURRENT_DATE
);

CREATE TABLE order_items (
    id         INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES orders(id),
    product_id INTEGER NOT NULL REFERENCES products(id),
    quantity   INTEGER NOT NULL CHECK (quantity > 0),
    unit_price DECIMAL(10, 2) NOT NULL  -- 注文時点の価格を保存
);
```

### 2.4 ACID特性

リレーショナルデータベースの信頼性を支える4つの性質。

```
┌──────────────────── ACID特性 ────────────────────┐
│                                                    │
│  A = Atomicity（原子性）                           │
│  ┌─────────────────────────────────────────┐     │
│  │ トランザクションは「全て成功」か             │     │
│  │ 「全て失敗（ロールバック）」のどちらか       │     │
│  │ 例: 送金 = 引き出し + 入金を不可分に         │     │
│  └─────────────────────────────────────────┘     │
│                                                    │
│  C = Consistency（一貫性）                         │
│  ┌─────────────────────────────────────────┐     │
│  │ トランザクション前後でデータの整合性が保たれる│     │
│  │ 例: 外部キー制約、CHECK制約を常に満たす      │     │
│  └─────────────────────────────────────────┘     │
│                                                    │
│  I = Isolation（分離性）                           │
│  ┌─────────────────────────────────────────┐     │
│  │ 並行トランザクションが互いに干渉しない       │     │
│  │ 例: 2人が同時に同じ口座を操作しても正しい結果│     │
│  └─────────────────────────────────────────┘     │
│                                                    │
│  D = Durability（永続性）                          │
│  ┌─────────────────────────────────────────┐     │
│  │ COMMITされたデータはシステム障害後も失われない│     │
│  │ 例: WAL（Write-Ahead Logging）による保証     │     │
│  └─────────────────────────────────────────┘     │
└────────────────────────────────────────────────────┘
```

---

## 3. 主要RDBMS

### 3.1 PostgreSQLの特徴と内部アーキテクチャ

PostgreSQLは「世界で最も先進的なオープンソースリレーショナルデータベース」を標榜し、SQL標準への準拠度が高く、拡張性に優れる。

**内部アーキテクチャの特徴:**
- **MVCC（Multi-Version Concurrency Control）**: 行の各バージョンを保持し、読み取りと書き込みが互いにブロックしない
- **WAL（Write-Ahead Logging）**: 変更をまずログに書き込み、クラッシュ後もデータを復元可能にする
- **プロセスベースアーキテクチャ**: 各接続に独立したプロセスを割り当て

### コード例3: PostgreSQLの特徴的な機能

```sql
-- PostgreSQL: 高度なデータ型と拡張性

-- 配列型: 1カラムに複数値を格納
CREATE TABLE products (
    id       SERIAL PRIMARY KEY,
    name     TEXT NOT NULL,
    tags     TEXT[],                    -- 配列型
    metadata JSONB,                     -- JSONBバイナリ型（GINインデックス対応）
    price    NUMRANGE,                  -- 範囲型（価格帯を表現）
    search   TSVECTOR                   -- 全文検索用ベクトル
);

-- サンプルデータ挿入
INSERT INTO products (name, tags, metadata, price, search) VALUES (
    'オーガニック味噌',
    ARRAY['organic', 'japanese', 'fermented'],
    '{"category": "food", "origin": "Japan", "weight_g": 500}',
    NUMRANGE(300, 500),
    to_tsvector('japanese', 'オーガニック 味噌 国産 大豆')
);

-- 配列の検索: ANY演算子
SELECT * FROM products WHERE 'organic' = ANY(tags);

-- 配列の包含: @>演算子
SELECT * FROM products WHERE tags @> ARRAY['organic', 'japanese'];

-- JSONB内の検索: @>演算子（包含）
SELECT * FROM products WHERE metadata @> '{"category": "food"}';

-- JSONB内の値を取得: ->>演算子
SELECT name, metadata->>'origin' AS origin FROM products;

-- 範囲型の検索: @>演算子（値が範囲内か）
SELECT * FROM products WHERE price @> 350;

-- 全文検索: @@演算子
SELECT * FROM products WHERE search @@ to_tsquery('japanese', 'オーガニック & 味噌');

-- テーブル継承（PostgreSQL固有）
CREATE TABLE employees_2024 () INHERITS (employees);

-- GENERATED COLUMNS（計算列）
CREATE TABLE orders_v2 (
    id         SERIAL PRIMARY KEY,
    quantity   INTEGER NOT NULL,
    unit_price DECIMAL(10, 2) NOT NULL,
    total      DECIMAL(10, 2) GENERATED ALWAYS AS (quantity * unit_price) STORED
);

-- LATERAL JOIN: FROM句での相関サブクエリ
SELECT d.name, top3.*
FROM departments d
CROSS JOIN LATERAL (
    SELECT e.name, e.salary
    FROM employees e
    WHERE e.department_id = d.id
    ORDER BY e.salary DESC
    LIMIT 3
) top3;
```

### 3.2 MySQLの特徴

MySQLは世界で最も広く使われているオープンソースRDBMSの一つであり、特にWebアプリケーション（LAMP/LEMPスタック）との親和性が高い。

**内部アーキテクチャの特徴:**
- **プラガブルストレージエンジン**: InnoDB、MyISAM等を用途に応じて選択可能
- **スレッドベースアーキテクチャ**: 各接続にスレッドを割り当て（PostgreSQLはプロセス）
- **InnoDB**: MVCC、外部キー、トランザクションを完全サポートするデフォルトエンジン

### コード例4: MySQLの特徴的な機能

```sql
-- MySQL: 広い普及率とシンプルな運用

-- AUTO_INCREMENTとエンジン指定
CREATE TABLE articles (
    id         BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    title      VARCHAR(255) NOT NULL,
    body       LONGTEXT,
    status     ENUM('draft', 'published', 'archived') DEFAULT 'draft',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    -- MySQLのフルテキストインデックス
    FULLTEXT INDEX ft_body (body)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- フルテキスト検索（ブーリアンモード）
SELECT * FROM articles
WHERE MATCH(body) AGAINST('データベース +設計 -NoSQL' IN BOOLEAN MODE);

-- ON DUPLICATE KEY UPDATE（UPSERT）
INSERT INTO user_settings (user_id, setting_key, setting_value)
VALUES (1, 'theme', 'dark')
ON DUPLICATE KEY UPDATE
    setting_value = VALUES(setting_value),
    updated_at = NOW();

-- REPLACE INTO（存在すれば削除→挿入）
REPLACE INTO cache_table (cache_key, cache_value, expires_at)
VALUES ('user:1:profile', '{"name":"田中"}', DATE_ADD(NOW(), INTERVAL 1 HOUR));

-- JSON型（MySQL 5.7+）
CREATE TABLE events (
    id   BIGINT AUTO_INCREMENT PRIMARY KEY,
    data JSON NOT NULL,
    -- 仮想生成列 + インデックス
    event_type VARCHAR(50) GENERATED ALWAYS AS (data->>'$.type') VIRTUAL,
    INDEX idx_event_type (event_type)
);

INSERT INTO events (data) VALUES ('{"type": "login", "user_id": 42, "ip": "192.168.1.1"}');
SELECT * FROM events WHERE data->>'$.type' = 'login';

-- Window Functions（MySQL 8.0+）
SELECT
    name,
    department_id,
    salary,
    RANK() OVER (PARTITION BY department_id ORDER BY salary DESC) AS dept_rank
FROM employees;
```

### 3.3 SQLiteの特徴

SQLiteは「サーバーレス」のRDBMSであり、データベース全体が単一ファイルに格納される。モバイルアプリ、組み込みシステム、テスト環境、デスクトップアプリケーションに広く使われている。

### コード例5: SQLiteの特徴的な機能

```sql
-- SQLite: 組み込み型、サーバー不要
-- ファイル1つがデータベース全体（またはインメモリ :memory:）

-- 動的型付け（型アフィニティ）
-- SQLiteは列に宣言された型を「ヒント」として使い、実際にはどんな値でも格納可能
CREATE TABLE settings (
    key   TEXT PRIMARY KEY,
    value ANY              -- どんな型でも格納可能
);

INSERT INTO settings VALUES ('max_retry', 3);          -- INTEGER
INSERT INTO settings VALUES ('api_url', 'https://example.com');  -- TEXT
INSERT INTO settings VALUES ('enabled', 1);            -- INTEGER (SQLiteにBOOLEAN型はない)
INSERT INTO settings VALUES ('ratio', 3.14);           -- REAL

-- typeof()で実際の型を確認
SELECT key, value, typeof(value) FROM settings;
-- max_retry | 3    | integer
-- api_url   | https://example.com | text
-- enabled   | 1    | integer
-- ratio     | 3.14 | real

-- JSON拡張（SQLite 3.38.0+ で組み込み）
CREATE TABLE logs (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    data TEXT NOT NULL  -- JSON文字列として格納
);

INSERT INTO logs (data) VALUES ('{"level": "error", "message": "接続失敗", "code": 500}');

SELECT
    json_extract(data, '$.level') AS level,
    json_extract(data, '$.message') AS message
FROM logs
WHERE json_extract(data, '$.level') = 'error';

-- UPSERT（SQLite 3.24.0+）
INSERT INTO settings (key, value)
VALUES ('max_retry', 5)
ON CONFLICT(key) DO UPDATE SET value = excluded.value;

-- WAL（Write-Ahead Logging）モードで並行読み取り性能を向上
-- PRAGMA journal_mode=WAL;
```

### コード例6: SQL Serverの特徴的な機能

```sql
-- SQL Server: エンタープライズ向け、.NETとの統合が深い

-- IDENTITY列（自動連番）
CREATE TABLE orders (
    order_id   INT IDENTITY(1,1) PRIMARY KEY,  -- 1から1ずつ増加
    order_date DATETIME2 DEFAULT SYSDATETIME(),
    customer_id INT NOT NULL,
    total_amount DECIMAL(12, 2)
);

-- TOP N（SQL Serverのページネーション）
SELECT TOP 10 * FROM orders ORDER BY order_date DESC;

-- OFFSET-FETCH（SQL:2008標準に準拠）
SELECT * FROM orders
ORDER BY order_date DESC
OFFSET 20 ROWS
FETCH NEXT 10 ROWS ONLY;

-- MERGE文（UPSERT + DELETE を1文で）
MERGE INTO user_settings AS target
USING (VALUES (1, 'theme', 'dark')) AS source (user_id, setting_key, setting_value)
ON target.user_id = source.user_id AND target.setting_key = source.setting_key
WHEN MATCHED THEN
    UPDATE SET setting_value = source.setting_value
WHEN NOT MATCHED THEN
    INSERT (user_id, setting_key, setting_value)
    VALUES (source.user_id, source.setting_key, source.setting_value);

-- STRING_AGG（文字列集約、SQL Server 2017+）
SELECT
    department_id,
    STRING_AGG(name, ', ') WITHIN GROUP (ORDER BY name) AS members
FROM employees
GROUP BY department_id;

-- IIF（条件式の簡略記法）
SELECT name, salary,
    IIF(salary >= 500000, '高給', '一般') AS grade
FROM employees;
```

### 3.4 主要RDBMS比較表

| 特徴 | PostgreSQL | MySQL | SQLite | SQL Server | Oracle |
|------|-----------|-------|--------|------------|--------|
| ライセンス | PostgreSQL License (BSD系) | GPL v2/商用 | パブリックドメイン | 商用 (Express版は無料) | 商用 |
| 最大DB容量 | 無制限 | 256TB (InnoDB) | 281TB | 524PB | 無制限 |
| 最大行サイズ | 1.6TB | 65,535バイト | 1GB | 8,060バイト | 制限なし |
| 同時接続数 | 数千（設定依存） | 数千 | 単一書き込み | 32,767 | 設定依存 |
| MVCC | あり | あり (InnoDB) | WALモードで近似 | あり | あり |
| JSON対応 | JSONB（GINインデックス、高速） | JSON（5.7+） | JSON1拡張 | JSON (2016+) | JSON (21c+) |
| 全文検索 | 組み込み（多言語対応） | 組み込み（InnoDB/MyISAM） | FTS5拡張 | 組み込み | Oracle Text |
| レプリケーション | ストリーミング/ロジカル | グループ/非同期 | なし | Always On AG | Data Guard |
| パーティショニング | 宣言的 (10+) | ネイティブ | なし | ネイティブ | ネイティブ |
| 拡張性 | 非常に高い（Extension） | 中程度（Plugin） | 低い | 高い | 高い |
| 学習コスト | 中程度 | 低い | 非常に低い | 中程度 | 高い |
| 主な用途 | 汎用/地理情報/分析 | Web/LAMP | 組み込み/モバイル | エンタープライズ/.NET | ミッションクリティカル |

### 3.5 RDBMS内部アーキテクチャの共通構造

```
┌──────────────── RDBMS の内部アーキテクチャ ────────────────┐
│                                                             │
│  クライアント                                               │
│      │                                                      │
│      ▼                                                      │
│  ┌───────────────────────┐                                 │
│  │   接続マネージャ       │  接続プーリング、認証             │
│  └───────────┬───────────┘                                 │
│              ▼                                              │
│  ┌───────────────────────┐                                 │
│  │   パーサー             │  SQL文の構文解析 → パースツリー  │
│  └───────────┬───────────┘                                 │
│              ▼                                              │
│  ┌───────────────────────┐                                 │
│  │   オプティマイザ       │  統計情報に基づく実行計画選択     │
│  │   (クエリプランナー)   │  コストベース最適化              │
│  └───────────┬───────────┘                                 │
│              ▼                                              │
│  ┌───────────────────────┐                                 │
│  │   エグゼキュータ       │  実行計画に従いデータを取得      │
│  └───────────┬───────────┘                                 │
│              ▼                                              │
│  ┌───────────────────────┐                                 │
│  │   ストレージエンジン   │  バッファプール、ディスクI/O     │
│  │   + トランザクション   │  WAL、MVCC、ロック管理           │
│  │   マネージャ           │                                  │
│  └───────────────────────┘                                 │
│                                                             │
│  SQL文のライフサイクル:                                      │
│  文字列 → パース → 最適化 → 実行 → 結果返却                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. SQL方言の違い

SQL標準は広範だが、各RDBMSの実装範囲と独自拡張は大きく異なる。移植性の高いコードを書くには、方言差を正確に把握する必要がある。

### 4.1 方言比較表（主要操作）

| 操作 | 標準SQL / PostgreSQL | MySQL | SQLite | SQL Server |
|------|---------------------|-------|--------|------------|
| 文字列結合 | `'A' \|\| 'B'` | `CONCAT('A','B')` | `'A' \|\| 'B'` | `'A' + 'B'` / `CONCAT` |
| LIMIT | `LIMIT 10` | `LIMIT 10` | `LIMIT 10` | `TOP 10` / `FETCH NEXT 10 ROWS ONLY` |
| UPSERT | `ON CONFLICT DO UPDATE` | `ON DUPLICATE KEY UPDATE` | `ON CONFLICT DO UPDATE` | `MERGE` |
| 自動連番 | `SERIAL` / `GENERATED` | `AUTO_INCREMENT` | `AUTOINCREMENT` | `IDENTITY` |
| 日付取得 | `CURRENT_TIMESTAMP` | `NOW()` / `CURRENT_TIMESTAMP` | `datetime('now')` | `GETDATE()` / `SYSDATETIME()` |
| BOOLEAN | `TRUE` / `FALSE` | `TRUE` / `FALSE` (=1/0) | `1` / `0` | `BIT` (1/0) |
| IF式 | `CASE WHEN` | `IF()` / `CASE` | `CASE WHEN` / `IIF` | `IIF()` / `CASE` |
| 日付差分 | `age()` / `-` 演算子 | `DATEDIFF()` | `julianday()` | `DATEDIFF()` |
| 正規表現 | `~` / `~*` | `REGEXP` | なし (拡張で可) | なし (CLR) |
| 配列型 | `TEXT[]` | なし (JSON代替) | なし | なし |
| CTE | `WITH` / `WITH RECURSIVE` | `WITH RECURSIVE` (8.0+) | `WITH RECURSIVE` (3.8.3+) | `WITH` |
| ウィンドウ関数 | 全対応 | 8.0+ | 3.25.0+ | 全対応 |

### 4.2 方言比較表（日付操作）

| 操作 | PostgreSQL | MySQL | SQLite | SQL Server |
|------|-----------|-------|--------|------------|
| 現在日時 | `NOW()` / `CURRENT_TIMESTAMP` | `NOW()` | `datetime('now')` | `GETDATE()` |
| 日付加算 | `date + INTERVAL '1 day'` | `DATE_ADD(date, INTERVAL 1 DAY)` | `datetime(date, '+1 day')` | `DATEADD(DAY, 1, date)` |
| 日付差分 | `date1 - date2` | `DATEDIFF(date1, date2)` | `julianday(d1) - julianday(d2)` | `DATEDIFF(DAY, d2, d1)` |
| 年の取得 | `EXTRACT(YEAR FROM date)` | `YEAR(date)` | `strftime('%Y', date)` | `YEAR(date)` |
| 月初日 | `DATE_TRUNC('month', date)` | `DATE_FORMAT(date, '%Y-%m-01')` | `date(date, 'start of month')` | `DATEFROMPARTS(YEAR(d), MONTH(d), 1)` |
| 書式変換 | `TO_CHAR(date, 'YYYY-MM-DD')` | `DATE_FORMAT(date, '%Y-%m-%d')` | `strftime('%Y-%m-%d', date)` | `FORMAT(date, 'yyyy-MM-dd')` |

### コード例7: 移植性の高いSQL

```sql
-- 方言差を最小化する書き方のパターン

-- (1) ページネーション: OFFSET-FETCH（SQL:2008標準）を使う
-- PostgreSQL, SQL Server, SQLite (3.35.0+) で動作
SELECT employee_id, name, salary
FROM employees
ORDER BY salary DESC
OFFSET 20 ROWS
FETCH NEXT 10 ROWS ONLY;

-- (2) CASE式は全RDBMSで使用可能
SELECT name, salary,
    CASE
        WHEN salary >= 600000 THEN '高給'
        WHEN salary >= 400000 THEN '中給'
        ELSE '標準'
    END AS salary_grade
FROM employees;

-- (3) COALESCE: NULL代替（全RDBMSで使用可能）
SELECT name, COALESCE(phone, email, '連絡先不明') AS contact
FROM employees;

-- (4) 日付リテラルの標準記法
SELECT * FROM orders
WHERE order_date >= DATE '2024-01-01'
  AND order_date <  DATE '2025-01-01';

-- (5) CAST式は全RDBMSで使用可能
SELECT CAST(price AS INTEGER) AS rounded_price FROM products;

-- (6) EXISTS/NOT EXISTS は全RDBMSで同一動作
SELECT * FROM customers c
WHERE EXISTS (
    SELECT 1 FROM orders o WHERE o.customer_id = c.id
);
```

### コード例8: 方言差の吸収パターン（アプリケーション設計）

```sql
-- 方言差をアプリケーション層で吸収する設計パターン

-- パターン1: ビューで方言を隠蔽
-- PostgreSQL用
CREATE VIEW v_monthly_sales AS
SELECT
    DATE_TRUNC('month', sale_date) AS month,
    SUM(amount) AS total
FROM sales
GROUP BY DATE_TRUNC('month', sale_date);

-- MySQL用（同じ論理だが関数が異なる）
-- CREATE VIEW v_monthly_sales AS
-- SELECT
--     DATE_FORMAT(sale_date, '%Y-%m-01') AS month,
--     SUM(amount) AS total
-- FROM sales
-- GROUP BY DATE_FORMAT(sale_date, '%Y-%m-01');

-- パターン2: 条件付きINSERT（方言差が大きい）
-- 標準的なアプローチ: アプリケーション側でIF-ELSE
-- PostgreSQL: INSERT ... ON CONFLICT DO UPDATE
-- MySQL:      INSERT ... ON DUPLICATE KEY UPDATE
-- SQL Server: MERGE
-- SQLite:     INSERT ... ON CONFLICT DO UPDATE

-- パターン3: ストアドプロシージャで方言を吸収
-- PostgreSQLの例
CREATE OR REPLACE FUNCTION upsert_setting(
    p_user_id INTEGER,
    p_key TEXT,
    p_value TEXT
) RETURNS VOID AS $$
BEGIN
    INSERT INTO user_settings (user_id, key, value)
    VALUES (p_user_id, p_key, p_value)
    ON CONFLICT (user_id, key)
    DO UPDATE SET value = EXCLUDED.value, updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- 呼び出し側はRDBMSを意識しない
SELECT upsert_setting(1, 'theme', 'dark');
```

---

## 5. SQLの分類

SQLは操作の性質に応じて5つのカテゴリに分類される。

```
┌──────────────────────── SQL言語の分類 ──────────────────────────┐
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   DDL    │  │   DML    │  │   DCL    │  │   TCL    │       │
│  │          │  │          │  │          │  │          │       │
│  │ CREATE   │  │ INSERT   │  │ GRANT    │  │ BEGIN    │       │
│  │ ALTER    │  │ UPDATE   │  │ REVOKE   │  │ COMMIT   │       │
│  │ DROP     │  │ DELETE   │  │          │  │ ROLLBACK │       │
│  │ TRUNCATE │  │ MERGE    │  │          │  │ SAVEPOINT│       │
│  │ RENAME   │  │ SELECT   │  │          │  │          │       │
│  │ COMMENT  │  │          │  │          │  │          │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│                                                                  │
│  DDL = Data Definition Language    （データ定義言語）             │
│  DML = Data Manipulation Language  （データ操作言語）             │
│  DCL = Data Control Language       （データ制御言語）             │
│  TCL = Transaction Control Language（トランザクション制御言語）   │
│                                                                  │
│  ※ DQL (Data Query Language) としてSELECTを独立分類する場合もある │
│  ※ 分類はRDBMSや教科書により若干異なる                            │
└──────────────────────────────────────────────────────────────────┘
```

### 5.1 各分類の詳細

### コード例9: DDL（Data Definition Language）

```sql
-- DDL: データベースオブジェクトの構造を定義・変更・削除

-- CREATE: オブジェクトの作成
CREATE TABLE departments (
    id   INTEGER PRIMARY KEY,
    name VARCHAR(50) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_employees_dept ON employees(department_id);

CREATE VIEW v_active_employees AS
SELECT * FROM employees WHERE status = 'active';

-- ALTER: オブジェクトの変更
ALTER TABLE employees ADD COLUMN phone VARCHAR(20);
ALTER TABLE employees ALTER COLUMN name SET NOT NULL;
ALTER TABLE employees DROP COLUMN phone;
ALTER TABLE employees RENAME COLUMN name TO full_name;

-- DROP: オブジェクトの削除
DROP TABLE IF EXISTS temp_data;
DROP INDEX IF EXISTS idx_old;
DROP VIEW IF EXISTS v_old_view;

-- TRUNCATE: テーブルの全行削除（DDL操作、高速）
TRUNCATE TABLE log_entries;

-- COMMENT: オブジェクトにコメントを付与（PostgreSQL）
COMMENT ON TABLE employees IS '従業員マスタテーブル';
COMMENT ON COLUMN employees.salary IS '月額基本給（円）';
```

### コード例10: DCL（Data Control Language）とTCL

```sql
-- DCL: データアクセス権限の制御

-- GRANT: 権限の付与
GRANT SELECT, INSERT ON employees TO app_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO admin_user;
GRANT USAGE ON SEQUENCE employees_id_seq TO app_user;

-- REVOKE: 権限の剥奪
REVOKE DELETE ON employees FROM app_user;
REVOKE ALL PRIVILEGES ON employees FROM public;

-- ROLEの作成と管理
CREATE ROLE readonly_user LOGIN PASSWORD 'secure_password';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;

-- TCL: トランザクションの制御

-- 基本的なトランザクション
BEGIN;
    UPDATE accounts SET balance = balance - 10000 WHERE id = 1;
    UPDATE accounts SET balance = balance + 10000 WHERE id = 2;
COMMIT;

-- SAVEPOINT: 部分的なロールバック
BEGIN;
    INSERT INTO orders (customer_id, total) VALUES (1, 5000);
    SAVEPOINT before_items;

    INSERT INTO order_items (order_id, product_id) VALUES (100, 1);
    -- エラーが発生した場合、SAVEPOINTまで戻す
    -- ROLLBACK TO SAVEPOINT before_items;

    INSERT INTO order_items (order_id, product_id) VALUES (100, 2);
COMMIT;
```

### 5.2 SQL分類の詳細比較表

| 分類 | 目的 | 主要コマンド | ロールバック可否 | 暗黙COMMIT |
|------|------|-------------|----------------|-----------|
| DDL | 構造定義 | CREATE, ALTER, DROP, TRUNCATE | DB依存 | 多くのRDBMSでYes |
| DML | データ操作 | INSERT, UPDATE, DELETE, SELECT | Yes | No |
| DCL | 権限制御 | GRANT, REVOKE | DB依存 | 多くのRDBMSでYes |
| TCL | トランザクション | BEGIN, COMMIT, ROLLBACK, SAVEPOINT | N/A | N/A |

---

## 6. SQL学習のロードマップ

```
┌──────────────── SQL学習のロードマップ ────────────────┐
│                                                        │
│  Level 1: 基礎（本章 + 00-basics）                     │
│  ┌──────────────────────────────────────────┐         │
│  │ SQL概要 → CRUD → JOIN → 集約 → サブクエリ │         │
│  └──────────────────────────────────────────┘         │
│           │                                            │
│           ▼                                            │
│  Level 2: 応用（01-advanced）                          │
│  ┌──────────────────────────────────────────┐         │
│  │ ウィンドウ関数 → CTE → トランザクション     │         │
│  │ → インデックス → クエリ最適化              │         │
│  └──────────────────────────────────────────┘         │
│           │                                            │
│           ▼                                            │
│  Level 3: 設計（02-design）                            │
│  ┌──────────────────────────────────────────┐         │
│  │ 正規化 → スキーマ設計 → 制約 → ER図        │         │
│  └──────────────────────────────────────────┘         │
│           │                                            │
│           ▼                                            │
│  Level 4: 実践（03-practical）                         │
│  ┌──────────────────────────────────────────┐         │
│  │ マイグレーション → バックアップ → 監視      │         │
│  │ → セキュリティ → パフォーマンスチューニング │         │
│  └──────────────────────────────────────────┘         │
└────────────────────────────────────────────────────────┘
```

---

## アンチパターン

### アンチパターン1: SQL方言に依存した設計

```sql
-- NG: MySQL固有の構文に依存
SELECT * FROM users LIMIT 10, 20;  -- offset, limitの順序がMySQL固有
-- さらにMySQLのGROUP BY拡張に依存
SELECT department_id, name, MAX(salary)
FROM employees
GROUP BY department_id;
-- → MySQLではnameが不定値で返る（ONLY_FULL_GROUP_BY無効時）
-- → PostgreSQLでは即エラー

-- OK: 標準SQLに近い書き方
SELECT * FROM users
ORDER BY id
OFFSET 10 ROWS
FETCH NEXT 20 ROWS ONLY;  -- SQL:2008標準

-- OK: GROUP BYの正しい使用
SELECT department_id, MIN(name), MAX(salary)
FROM employees
GROUP BY department_id;
```

**問題点**: 特定のRDBMSにロックインされ、将来の移行コストが膨大になる。また、MySQLのGROUP BY拡張は非決定的な結果を返すため、バグの温床となる。

**WHY**: 方言依存のコードは「動く」が「正しくない」ことがある。標準SQLを基盤にすることで、コードの移植性だけでなく正確性も確保できる。

### アンチパターン2: RDBMSの選定を後回しにする

```
プロジェクト初期に「とりあえずSQLite」で開発 → 本番で突然PostgreSQLに変更
→ SQLiteにない機能を多用していて大規模な書き直しが発生

具体的に困る例:
┌──────────────────────────────────────────────────────┐
│ SQLiteにない/異なる機能          │ 移行時の問題       │
├──────────────────────────────────┼────────────────────┤
│ 並行書き込み（単一ライター）      │ 本番で競合エラー   │
│ ALTER TABLE制約（変更が限定的）   │ マイグレーション困難│
│ 権限管理（なし）                 │ セキュリティ未設計  │
│ ENUM型（なし）                   │ バリデーション漏れ  │
│ ストアドプロシージャ（なし）      │ ロジック再実装     │
│ DATE/TIME型（文字列で代替）       │ 日付計算の全面修正 │
└──────────────────────────────────┴────────────────────┘
```

**対策**:
- 開発初期から本番と同じRDBMSを使用する
- Docker / docker-compose で環境構築を容易にする
- テスト環境もSQLite以外を使う（docker-compose.test.yml等）

### アンチパターン3: NULLの3値論理を無視する

```sql
-- NG: NULLとの比較は常にUNKNOWN
SELECT * FROM employees WHERE department_id = NULL;
-- → 結果は常に0行！（NULL = NULL は UNKNOWN）

SELECT * FROM employees WHERE department_id != 10;
-- → department_id が NULL の行は含まれない！

-- OK: IS NULL / IS NOT NULL を使用
SELECT * FROM employees WHERE department_id IS NULL;

-- OK: NULLを考慮した条件
SELECT * FROM employees
WHERE department_id != 10 OR department_id IS NULL;

-- NULLの3値論理:
-- TRUE AND NULL    = NULL (UNKNOWN)
-- FALSE AND NULL   = FALSE
-- TRUE OR NULL     = TRUE
-- FALSE OR NULL    = NULL (UNKNOWN)
-- NOT NULL         = NULL (UNKNOWN)
```

**WHY**: SQLは2値論理（TRUE/FALSE）ではなく3値論理（TRUE/FALSE/UNKNOWN）を採用している。NULLとの演算結果は常にUNKNOWNであり、WHERE句でUNKNOWNは「条件を満たさない」として扱われる。

---

## 実践演習

### 演習1（基礎）: RDBMSの特徴を整理する

以下の要件に対して最適なRDBMSを選び、その理由を述べよ。

1. 個人ブログアプリ（月間PV 1,000以下、予算ゼロ）
2. ECサイト（同時接続 100、決済処理あり、将来の拡張性重視）
3. IoTデバイスの組み込みデータストア（メモリ制約あり、サーバー接続不可）
4. 大企業の基幹システム（SLA 99.99%、24時間サポート必須）
5. 地理情報システム（GIS）で位置データを頻繁に扱う

<details>
<summary>模範解答</summary>

1. **SQLite** — サーバー不要で設定ゼロ。ファイル1つで完結し、小規模サイトには十分な性能。VPS不要ならさらにコスト削減。ただしWordPress等のCMSを使うならMySQL/MariaDB。

2. **PostgreSQL** — ACID準拠のトランザクションで決済処理の信頼性を確保。JSONB、パーティショニング、ロジカルレプリケーション等の拡張機能で将来の成長にも対応。オープンソースで商用利用にも制限なし。

3. **SQLite** — サーバープロセスが不要で、ライブラリとしてアプリケーションに組み込める。フットプリントが小さく（約700KB）、設定ファイルも不要。Android/iOS標準のデータベースエンジン。

4. **Oracle Database** または **SQL Server** — 24時間有人サポート、高可用性構成（Oracle RAC / SQL Server Always On）、包括的な監視ツール、SLA保証を提供。コストは高いが、ダウンタイムのビジネスインパクトが大きい場合は正当化される。

5. **PostgreSQL + PostGIS** — PostGIS拡張により、空間インデックス（GiST/SP-GiST）、地理演算関数（ST_Distance、ST_Contains等）、座標系変換を標準サポート。オープンソースでGIS分野のデファクトスタンダード。

</details>

### 演習2（応用）: SQL方言の移植性を検証する

以下のMySQL固有のSQLを、PostgreSQLと標準SQLに書き換えよ。

```sql
-- MySQL版
SELECT
    id,
    IF(status = 1, 'active', 'inactive') AS status_label,
    DATE_FORMAT(created_at, '%Y年%m月%d日') AS formatted_date,
    GROUP_CONCAT(tag SEPARATOR ', ') AS tags
FROM articles
WHERE MATCH(body) AGAINST('データベース' IN BOOLEAN MODE)
GROUP BY id
LIMIT 5, 10;
```

<details>
<summary>模範解答</summary>

```sql
-- PostgreSQL版
SELECT
    id,
    CASE WHEN status = 1 THEN 'active' ELSE 'inactive' END AS status_label,
    TO_CHAR(created_at, 'YYYY"年"MM"月"DD"日"') AS formatted_date,
    STRING_AGG(tag, ', ') AS tags
FROM articles
WHERE search_vector @@ to_tsquery('japanese', 'データベース')
    -- 注: search_vectorはTSVECTOR型のカラム（事前にGINインデックス作成が必要）
GROUP BY id
ORDER BY id  -- OFFSET-FETCHにはORDER BYが必要
OFFSET 5 ROWS
FETCH NEXT 10 ROWS ONLY;

-- 標準SQL版（全文検索は標準SQLに含まれないため、LIKE代替）
SELECT
    id,
    CASE WHEN status = 1 THEN 'active' ELSE 'inactive' END AS status_label,
    -- 標準SQLの日付書式変換はRDBMS依存が大きいため、アプリ層で処理推奨
    CAST(created_at AS DATE) AS sale_date,
    -- STRING_AGGは SQL:2023 で標準化された（LISTAGG はOracleの先行実装）
    LISTAGG(tag, ', ') WITHIN GROUP (ORDER BY tag) AS tags
FROM articles
WHERE body LIKE '%データベース%'  -- 全文検索の代替（性能は劣る）
GROUP BY id
ORDER BY id
OFFSET 5 ROWS
FETCH NEXT 10 ROWS ONLY;
```

**移植のポイント:**
- `IF()` → `CASE WHEN ... THEN ... ELSE ... END`（標準SQL）
- `DATE_FORMAT()` → `TO_CHAR()`（PostgreSQL）/ アプリ層での変換
- `GROUP_CONCAT()` → `STRING_AGG()`（PostgreSQL）/ `LISTAGG()`（Oracle/標準）
- `MATCH ... AGAINST` → `@@` + `to_tsquery()`（PostgreSQL）/ `LIKE`（汎用）
- `LIMIT offset, count` → `OFFSET ... ROWS FETCH NEXT ... ROWS ONLY`（標準SQL）

</details>

### 演習3（発展）: リレーショナルモデルの原則に基づいた設計

以下の非正規化されたスプレッドシートデータを、第3正規形のテーブル設計に変換せよ。CREATE TABLE文、制約、サンプルデータのINSERT文、および「全注文の詳細を表示するSELECT文」を記述すること。

```
注文ID | 注文日      | 顧客名   | 顧客メール        | 商品名    | 単価  | 数量 | 配送先住所
1      | 2024-01-15 | 田中太郎 | tanaka@mail.com  | ノートPC  | 80000 | 1    | 東京都新宿区1-1
1      | 2024-01-15 | 田中太郎 | tanaka@mail.com  | マウス    | 3000  | 2    | 東京都新宿区1-1
2      | 2024-01-16 | 鈴木花子 | suzuki@mail.com  | キーボード | 5000  | 1    | 大阪府大阪市2-2
```

<details>
<summary>模範解答</summary>

```sql
-- 第3正規形への分解

-- 1. 顧客テーブル（顧客情報の重複を排除）
CREATE TABLE customers (
    id    SERIAL PRIMARY KEY,
    name  VARCHAR(100) NOT NULL,
    email VARCHAR(255) NOT NULL UNIQUE
);

-- 2. 商品テーブル（商品情報の重複を排除）
CREATE TABLE products (
    id    SERIAL PRIMARY KEY,
    name  VARCHAR(100) NOT NULL,
    price DECIMAL(10, 2) NOT NULL CHECK (price >= 0)
);

-- 3. 注文テーブル（注文ヘッダー）
CREATE TABLE orders (
    id              SERIAL PRIMARY KEY,
    customer_id     INTEGER NOT NULL REFERENCES customers(id),
    order_date      DATE NOT NULL DEFAULT CURRENT_DATE,
    shipping_address TEXT NOT NULL
);

-- 4. 注文明細テーブル（注文と商品の多対多関係）
CREATE TABLE order_items (
    id         SERIAL PRIMARY KEY,
    order_id   INTEGER NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    product_id INTEGER NOT NULL REFERENCES products(id),
    unit_price DECIMAL(10, 2) NOT NULL CHECK (unit_price >= 0),
        -- 注文時点の価格を保存（商品マスタの価格変更の影響を受けない）
    quantity   INTEGER NOT NULL CHECK (quantity > 0),
    UNIQUE (order_id, product_id)  -- 同一注文内での商品重複を防止
);

-- サンプルデータ
INSERT INTO customers (id, name, email) VALUES
    (1, '田中太郎', 'tanaka@mail.com'),
    (2, '鈴木花子', 'suzuki@mail.com');

INSERT INTO products (id, name, price) VALUES
    (1, 'ノートPC', 80000),
    (2, 'マウス', 3000),
    (3, 'キーボード', 5000);

INSERT INTO orders (id, customer_id, order_date, shipping_address) VALUES
    (1, 1, '2024-01-15', '東京都新宿区1-1'),
    (2, 2, '2024-01-16', '大阪府大阪市2-2');

INSERT INTO order_items (order_id, product_id, unit_price, quantity) VALUES
    (1, 1, 80000, 1),  -- 注文1: ノートPC x 1
    (1, 2, 3000, 2),   -- 注文1: マウス x 2
    (2, 3, 5000, 1);   -- 注文2: キーボード x 1

-- 全注文の詳細を表示するクエリ
SELECT
    o.id AS order_id,
    o.order_date,
    c.name AS customer_name,
    c.email AS customer_email,
    p.name AS product_name,
    oi.unit_price,
    oi.quantity,
    oi.unit_price * oi.quantity AS subtotal,
    o.shipping_address
FROM orders o
    INNER JOIN customers c ON o.customer_id = c.id
    INNER JOIN order_items oi ON o.id = oi.order_id
    INNER JOIN products p ON oi.product_id = p.id
ORDER BY o.id, p.name;

-- 注文ごとの合計金額
SELECT
    o.id AS order_id,
    c.name AS customer_name,
    SUM(oi.unit_price * oi.quantity) AS total_amount
FROM orders o
    INNER JOIN customers c ON o.customer_id = c.id
    INNER JOIN order_items oi ON o.id = oi.order_id
GROUP BY o.id, c.name
ORDER BY o.id;
```

**設計のポイント:**
- `unit_price`を`order_items`に持たせることで、商品マスタの価格変更が過去の注文に影響しない
- `ON DELETE CASCADE`で注文削除時に明細も自動削除
- `UNIQUE (order_id, product_id)`で同一商品の重複明細を防止
- `CHECK`制約でドメイン（価格≥0、数量>0）を保証

</details>

---

## FAQ

### Q1: SQLは「古い技術」ではないのか？

SQLは1970年代に生まれたが、SQL:2023まで継続的に拡張されている。JSON対応、グラフクエリ（SQL/PGQ）、時系列データ、行パターン認識など現代的な機能が追加され続けており、むしろ適用範囲は広がっている。

NoSQLブームの後、多くのNoSQLデータベースがSQLライクなクエリ言語を採用した（CassandraのCQL、CouchbaseのN1QL、Google BigQueryのSQL方言など）事実が、SQLの設計の優秀さを証明している。さらに、NewSQLと呼ばれる分散データベース（CockroachDB、TiDB、YugabyteDB等）もSQLインターフェースを採用しており、SQLは分散システムにおいても標準的なアクセス言語となっている。

### Q2: どのRDBMSを選べばよいか？

判断基準は以下の通り:

| 条件 | 推奨RDBMS | 理由 |
|------|----------|------|
| 個人/小規模プロジェクト | SQLite | ゼロ設定、ファイル1つで完結 |
| Webアプリ（汎用） | PostgreSQL | 機能の豊富さ、拡張性、標準SQL準拠 |
| 既存のLAMP環境 | MySQL / MariaDB | エコシステムの充実、運用実績 |
| .NET / Azure環境 | SQL Server | Visual Studio/Azureとの統合 |
| 大規模ミッションクリティカル | Oracle | サポート品質、RAC、実績 |
| GIS / 地理情報 | PostgreSQL + PostGIS | 空間データ処理のデファクト |
| 分析 / DWH | PostgreSQL / BigQuery | 分析関数の充実 |

### Q3: 標準SQLだけ学べばどのRDBMSでも使えるか？

基本的なCRUD操作（SELECT, INSERT, UPDATE, DELETE）とJOINは標準SQLで書ける。ただし、以下の領域は方言差が大きい:

- **日付/時刻関数**: `DATE_TRUNC` vs `DATE_FORMAT` vs `strftime` vs `DATEPART`
- **文字列関数**: `||` vs `CONCAT` vs `+`
- **ページネーション**: `LIMIT` vs `TOP` vs `FETCH FIRST`
- **UPSERT**: `ON CONFLICT` vs `ON DUPLICATE KEY` vs `MERGE`
- **全文検索**: `@@` / `to_tsquery` vs `MATCH ... AGAINST` vs FTS5
- **ストアドプロシージャ**: PL/pgSQL vs MySQL Stored Procedures vs T-SQL

標準SQLを基盤としつつ、使用するRDBMS固有の機能も把握することが実務では必要。ORMを使う場合でも、生成されるSQLを理解できることが重要である。

### Q4: SQLとNoSQLはどう使い分けるか？

| 基準 | SQL (RDBMS) | NoSQL |
|------|------------|-------|
| データ構造 | 固定スキーマ、正規化 | 柔軟/スキーマレス |
| 整合性 | ACID（強い整合性） | BASE（結果整合性が多い） |
| スケーリング | 垂直スケーリングが主 | 水平スケーリングが得意 |
| クエリ | 複雑なJOIN、集約 | キーバリュー、ドキュメント検索 |
| 適する場面 | 業務データ、会計、在庫 | ログ、キャッシュ、IoT、SNS |

実際のプロダクションでは両方を併用するケースが多い（例: PostgreSQL + Redis + Elasticsearch）。

### Q5: SQL学習の効率的な方法は？

1. **まず手を動かす**: SQLite（設定不要）かDockerでPostgreSQLを起動し、実際にクエリを書く
2. **順序を守る**: SELECT → WHERE → JOIN → GROUP BY → サブクエリ → ウィンドウ関数の順で学ぶ
3. **実データで練習**: 公開データセット（Kaggle、data.go.jp等）を取り込んで分析する
4. **EXPLAINを使う**: 実行計画を確認する習慣をつける
5. **1日1問**: LeetCode SQL、HackerRank SQL等で毎日練習する

---

## まとめ

| 項目 | 要点 |
|------|------|
| SQLの本質 | リレーショナルモデルに基づく宣言型データ操作言語。「何が欲しいか」を記述する |
| 理論的基盤 | 集合論と一階述語論理。Coddの12の規則 |
| 標準規格 | SQL:2023が最新。SQL-92が広く実装された基盤。継続的に拡張中 |
| 主要RDBMS | PostgreSQL（汎用）、MySQL（Web）、SQLite（組み込み）、SQL Server/Oracle（エンタープライズ） |
| ACID特性 | 原子性・一貫性・分離性・永続性。トランザクションの信頼性を保証 |
| SQL分類 | DDL（定義）/ DML（操作）/ DCL（制御）/ TCL（トランザクション）の4分類 |
| 方言対策 | 標準SQLを基盤に、RDBMS固有機能は意識的に分離。ビューやプロシージャで抽象化 |
| NULL | 3値論理（TRUE/FALSE/UNKNOWN）。`= NULL`ではなく`IS NULL`を使う |
| 選定基準 | 規模、チームスキル、既存インフラ、ライセンス費用、将来の拡張性 |

---

## 次に読むべきガイド

- [01-crud-operations.md](./01-crud-operations.md) — SELECT/INSERT/UPDATE/DELETEの詳細と安全な実行パターン
- [02-joins.md](./02-joins.md) — JOINの全種類と使い分け
- [03-aggregation.md](./03-aggregation.md) — GROUP BYと集約関数による分析
- [00-normalization.md](../02-design/00-normalization.md) — 正規化の理論と実践
- [../../security-fundamentals/docs/00-basics/](../../security-fundamentals/docs/00-basics/) — セキュリティ基礎（SQLインジェクション防止を含む）

---

## 参考文献

1. Codd, E.F. (1970). "A Relational Model of Data for Large Shared Data Banks". *Communications of the ACM*, 13(6), 377-387.
2. ISO/IEC 9075:2023 — Information technology — Database languages — SQL（最新SQL標準規格）
3. PostgreSQL Documentation — https://www.postgresql.org/docs/current/
4. Date, C.J. (2019). *SQL and Relational Theory: How to Write Accurate SQL Code*. 3rd Edition. O'Reilly Media.
5. Karwin, B. (2010). *SQL Antipatterns: Avoiding the Pitfalls of Database Programming*. Pragmatic Bookshelf.
6. MySQL Reference Manual — https://dev.mysql.com/doc/refman/8.0/en/
7. SQLite Documentation — https://www.sqlite.org/docs.html
8. Celko, J. (2010). *Joe Celko's SQL for Smarties: Advanced SQL Programming*. 4th Edition. Morgan Kaufmann.
