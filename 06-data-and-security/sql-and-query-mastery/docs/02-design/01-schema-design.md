# スキーマ設計 — ER図・制約・パーティション

> スキーマ設計はデータベースの骨格を決定する作業であり、テーブル構造、制約、リレーションシップの設計品質がアプリケーション全体の信頼性とパフォーマンスを左右する。

## この章で学ぶこと

1. ER図の読み書きとリレーションシップの種類（1:1, 1:N, M:N）
2. 制約（PRIMARY KEY, FOREIGN KEY, UNIQUE, CHECK, NOT NULL）の適切な使い方
3. パーティショニングによる大規模テーブルの管理戦略
4. 主キー戦略（SERIAL, UUID, ULID）の比較と選定基準
5. テーブル設計のベストプラクティスと共通パターン

## 前提知識

- SQLの基本構文（CREATE TABLE、ALTER TABLE）
- [00-normalization.md](./00-normalization.md) の正規化理論
- リレーショナルモデルの基礎概念

---

## 1. ER図とリレーションシップ

### 1.1 ER図の基本記法

```
┌─────────────── ER図 (Entity-Relationship Diagram) ───────────────┐
│                                                                   │
│  ┌───────────────┐     1:N     ┌──────────────┐                 │
│  │  departments  │────────────│  employees   │                  │
│  ├───────────────┤             ├──────────────┤                 │
│  │ PK id         │             │ PK id        │                 │
│  │    name       │             │ FK dept_id   │──┐              │
│  │    location   │             │    name      │  │              │
│  └───────────────┘             │    salary    │  │              │
│                                 │    email     │  │              │
│                                 └──────────────┘  │              │
│                                                    │ M:N         │
│  ┌───────────────┐     1:1     ┌──────────────┐  │              │
│  │  user_profiles│────────────│  users       │  │              │
│  ├───────────────┤             ├──────────────┤  │              │
│  │ PK/FK user_id │             │ PK id        │  │              │
│  │    bio        │             │    username  │  │              │
│  │    avatar_url │             │    password  │  │              │
│  └───────────────┘             └──────────────┘  │              │
│                                                    │              │
│  ┌───────────────┐             ┌──────────────┐  │              │
│  │  projects     │             │ emp_projects │◀─┘              │
│  ├───────────────┤     M:N     ├──────────────┤                 │
│  │ PK id         │────────────│ PK emp_id    │                 │
│  │    name       │             │ PK proj_id   │                 │
│  │    deadline   │             │    role      │                 │
│  └───────────────┘             └──────────────┘                 │
│                                 中間テーブル                      │
└───────────────────────────────────────────────────────────────────┘
```

### 1.2 ER図の主要な記法比較

```
┌──────── ER図の記法スタイル ──────────────────┐
│                                                │
│  Chen記法（学術的）:                           │
│  ┌──────┐    ◇     ┌──────┐                  │
│  │ 社員 │───<所属>───│ 部署 │                  │
│  └──────┘    ◇     └──────┘                  │
│  エンティティ=□ 関連=◇ 属性=○                │
│                                                │
│  IE記法（Information Engineering）:            │
│  ┌──────┐ ──||──< ┌──────┐                   │
│  │ 部署 │          │ 社員 │                   │
│  └──────┘          └──────┘                   │
│  ||=1 <=多 O=0(オプション)                    │
│                                                │
│  UML記法:                                      │
│  ┌──────┐  1..1     0..* ┌──────┐            │
│  │ 部署 │────────────────│ 社員 │            │
│  └──────┘                └──────┘            │
│  多重度を数値で表現                            │
│                                                │
│  実務では IE記法（カラスの足）が最も一般的     │
└────────────────────────────────────────────────┘
```

### コード例1: リレーションシップの実装

```sql
-- 1:1 リレーション
-- 実装パターン: 共有主キー（FK = PK）
CREATE TABLE users (
    id       SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email    VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE user_profiles (
    user_id    INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    bio        TEXT,
    avatar_url VARCHAR(500),
    birthdate  DATE,
    -- user_id が PK かつ FK → 1:1 を強制
    -- ON DELETE CASCADE → ユーザー削除時にプロフィールも削除
    CONSTRAINT chk_birthdate CHECK (birthdate <= CURRENT_DATE)
);

-- 1:1 リレーション: 代替パターン（UNIQUE FK）
CREATE TABLE user_settings (
    id       SERIAL PRIMARY KEY,
    user_id  INTEGER UNIQUE NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    -- UNIQUE制約でuser_idの重複を防止 → 1:1を実現
    theme    VARCHAR(20) DEFAULT 'light',
    language VARCHAR(10) DEFAULT 'ja',
    notifications_enabled BOOLEAN DEFAULT TRUE
);

-- 1:N リレーション
CREATE TABLE departments (
    id       SERIAL PRIMARY KEY,
    name     VARCHAR(100) NOT NULL,
    location VARCHAR(100),
    -- 部署コードのユニーク制約
    code     VARCHAR(10) UNIQUE NOT NULL
);

CREATE TABLE employees (
    id            SERIAL PRIMARY KEY,
    name          VARCHAR(100) NOT NULL,
    department_id INTEGER REFERENCES departments(id) ON DELETE SET NULL,
    salary        DECIMAL(10, 2) CHECK (salary >= 0),
    hired_date    DATE NOT NULL DEFAULT CURRENT_DATE,
    email         VARCHAR(255) UNIQUE NOT NULL,
    -- 部分インデックス: アクティブな社員のメール一意性
    is_active     BOOLEAN NOT NULL DEFAULT TRUE
);

-- 部分ユニークインデックス（PostgreSQL）
CREATE UNIQUE INDEX idx_active_employees_email
    ON employees (email) WHERE is_active = TRUE;

-- M:N リレーション（中間テーブル）
CREATE TABLE projects (
    id       SERIAL PRIMARY KEY,
    name     VARCHAR(200) NOT NULL,
    deadline DATE,
    status   VARCHAR(20) NOT NULL DEFAULT 'planning'
             CHECK (status IN ('planning', 'active', 'completed', 'cancelled'))
);

CREATE TABLE employee_projects (
    employee_id INTEGER REFERENCES employees(id) ON DELETE CASCADE,
    project_id  INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    role        VARCHAR(50) DEFAULT 'member',
    joined_at   DATE NOT NULL DEFAULT CURRENT_DATE,
    left_at     DATE,
    PRIMARY KEY (employee_id, project_id),
    -- 期間制約: 退出日は参加日以降
    CONSTRAINT chk_dates CHECK (left_at IS NULL OR left_at >= joined_at)
);

-- 自己参照リレーション（上司-部下）
CREATE TABLE employees_hierarchy (
    id         SERIAL PRIMARY KEY,
    name       VARCHAR(100) NOT NULL,
    manager_id INTEGER REFERENCES employees_hierarchy(id) ON DELETE SET NULL,
    level      INTEGER NOT NULL DEFAULT 0
);

-- 自己参照のツリー構造を確認するクエリ
WITH RECURSIVE hierarchy AS (
    SELECT id, name, manager_id, 0 AS depth
    FROM employees_hierarchy WHERE manager_id IS NULL
    UNION ALL
    SELECT e.id, e.name, e.manager_id, h.depth + 1
    FROM employees_hierarchy e
    JOIN hierarchy h ON e.manager_id = h.id
)
SELECT REPEAT('  ', depth) || name AS org_chart FROM hierarchy ORDER BY depth;
```

### 1.3 リレーションシップのカーディナリティ設計

```
┌──── カーディナリティの設計判断フロー ──────────┐
│                                                  │
│  Q: エンティティAとBの関係は？                   │
│  │                                              │
│  ├── Aの1レコードに対してBは何レコード？         │
│  │   ├── 常に1つ → 1:1（統合も検討）            │
│  │   ├── 0または1つ → 1:0..1（別テーブルに分離）│
│  │   └── 複数可能 → 1:N or M:N                 │
│  │                                              │
│  ├── BからAへの逆方向は？                       │
│  │   ├── Bの1レコードはAの1レコードのみ → 1:N   │
│  │   └── Bの1レコードは複数のAに関連 → M:N      │
│  │                                              │
│  └── M:Nの場合、中間テーブルに属性はある？      │
│      ├── ある → 属性付き中間テーブル             │
│      └── ない → 純粋な結合テーブル              │
│                                                  │
│  1:1 の場合の追加判断:                           │
│  ├── 両方とも必須 → 統合を検討                  │
│  ├── 片方がオプション → 別テーブルに分離        │
│  └── アクセスパターンが異なる → 別テーブルに分離│
└──────────────────────────────────────────────────┘
```

---

## 2. 制約

### 2.1 制約の内部動作

```
┌──────── 制約とインデックスの関係 ──────────────┐
│                                                  │
│  PRIMARY KEY:                                    │
│  → 自動的にUNIQUEインデックスが作成される        │
│  → NOT NULL が暗黙的に適用される                │
│  → PostgreSQL: B-Treeインデックス               │
│  → InnoDB: クラスタードインデックス              │
│                                                  │
│  UNIQUE:                                         │
│  → 自動的にUNIQUEインデックスが作成される        │
│  → NULLは許容（複数のNULLが可能）              │
│  → PostgreSQL: NULLsは一意性チェック対象外      │
│  → SQL Server: NULL は1つだけ許容（デフォルト） │
│                                                  │
│  FOREIGN KEY:                                    │
│  → 自動的にはインデックスが作成されない          │
│  → 手動でのインデックス作成を強く推奨           │
│  → INSERT/UPDATE時に参照先の存在を確認          │
│  → DELETE時に参照元の存在を確認                 │
│                                                  │
│  CHECK:                                          │
│  → インデックスは作成されない                    │
│  → INSERT/UPDATE時に条件を評価                  │
│  → PostgreSQLでは関数呼び出しも可能             │
└──────────────────────────────────────────────────┘
```

### コード例2: 各種制約の活用

```sql
-- NOT NULL: NULL禁止
-- UNIQUE: 重複禁止
-- CHECK: 値の範囲制約
-- DEFAULT: デフォルト値
-- REFERENCES: 外部キー制約

CREATE TABLE orders (
    id              SERIAL PRIMARY KEY,
    customer_id     INTEGER NOT NULL REFERENCES customers(id),
    order_number    VARCHAR(20) UNIQUE NOT NULL,
    status          VARCHAR(20) NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'confirmed', 'shipped',
                                      'delivered', 'cancelled')),
    total_amount    DECIMAL(12, 2) NOT NULL CHECK (total_amount >= 0),
    discount_rate   DECIMAL(3, 2) DEFAULT 0.00
                    CHECK (discount_rate BETWEEN 0 AND 1),
    order_date      DATE NOT NULL DEFAULT CURRENT_DATE,
    shipped_date    DATE,
    delivered_date  DATE,

    -- テーブルレベルの制約（複数カラムにまたがる制約）
    CONSTRAINT chk_dates CHECK (
        shipped_date IS NULL OR shipped_date >= order_date
    ),
    CONSTRAINT chk_delivery CHECK (
        delivered_date IS NULL OR delivered_date >= shipped_date
    ),
    -- 状態遷移の制約（簡易版）
    CONSTRAINT chk_status_dates CHECK (
        (status = 'shipped' AND shipped_date IS NOT NULL)
        OR (status != 'shipped')
    )
);

-- 排他制約（PostgreSQL: 期間の重複防止）
CREATE EXTENSION IF NOT EXISTS btree_gist;  -- 排他制約に必要

CREATE TABLE reservations (
    id        SERIAL PRIMARY KEY,
    room_id   INTEGER NOT NULL,
    guest     VARCHAR(100),
    period    DATERANGE NOT NULL,
    EXCLUDE USING GIST (
        room_id WITH =,
        period WITH &&  -- 同じ部屋の期間重複を禁止
    )
);

-- 排他制約の動作確認
INSERT INTO reservations (room_id, guest, period)
VALUES (101, '田中', '[2024-03-01, 2024-03-05)');

-- 重複する予約は拒否される
INSERT INTO reservations (room_id, guest, period)
VALUES (101, '鈴木', '[2024-03-03, 2024-03-07)');
-- → ERROR: conflicting key value violates exclusion constraint

-- 異なる部屋なら予約可能
INSERT INTO reservations (room_id, guest, period)
VALUES (102, '鈴木', '[2024-03-03, 2024-03-07)');
-- → 成功

-- 複合ユニーク制約の活用
CREATE TABLE subscriptions (
    id          SERIAL PRIMARY KEY,
    user_id     INTEGER NOT NULL REFERENCES users(id),
    plan_id     INTEGER NOT NULL REFERENCES plans(id),
    started_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at    TIMESTAMPTZ,
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    -- 同一ユーザーは同一プランのアクティブサブスクリプションを1つだけ
    CONSTRAINT uq_active_subscription UNIQUE (user_id, plan_id, is_active)
    -- 注意: is_active=FALSE の重複は許容される
);

-- 条件付きユニーク（PostgreSQL: 部分ユニークインデックス）
CREATE UNIQUE INDEX idx_active_subscription
    ON subscriptions (user_id, plan_id)
    WHERE is_active = TRUE;
-- → アクティブなサブスクリプションのみユニーク制約
```

### コード例3: 外部キーの参照アクション

```sql
-- ON DELETE / ON UPDATE のオプション
CREATE TABLE order_items (
    id         SERIAL PRIMARY KEY,
    order_id   INTEGER NOT NULL
               REFERENCES orders(id)
               ON DELETE CASCADE      -- 親削除時: 子も削除
               ON UPDATE CASCADE,     -- 親更新時: 子も更新
    product_id INTEGER NOT NULL
               REFERENCES products(id)
               ON DELETE RESTRICT,    -- 親削除時: エラー（子あれば削除拒否）
    quantity   INTEGER NOT NULL CHECK (quantity > 0),
    unit_price DECIMAL(10, 2) NOT NULL
);

-- 参照アクション一覧:
-- CASCADE    : 親に追従（削除/更新）
-- RESTRICT   : 子がある場合は親の操作を拒否（即時チェック）
-- NO ACTION  : 子がある場合は親の操作を拒否（遅延チェック可）※デフォルト
-- SET NULL   : 子のFK列をNULLに設定
-- SET DEFAULT: 子のFK列をDEFAULT値に設定
```

### 2.2 参照アクションの使い分け

```
┌──── 参照アクションの選定ガイド ──────────────┐
│                                                │
│  CASCADE:                                      │
│  ├── 使う: 親子が一体的（注文-注文明細）      │
│  ├── 使う: 所有関係（ユーザー-プロフィール）  │
│  └── 注意: 大量削除時のパフォーマンス影響     │
│                                                │
│  RESTRICT / NO ACTION:                         │
│  ├── 使う: 参照整合性を厳格に維持したい場合   │
│  ├── 使う: 誤って親を削除することを防止       │
│  └── 違い: RESTRICTは即時、NO ACTIONは遅延可 │
│                                                │
│  SET NULL:                                     │
│  ├── 使う: 親がなくなっても子は存在可能       │
│  ├── 例: 社員-部署（部署消滅でも社員は残る）  │
│  └── 前提: FK列がNULLable                     │
│                                                │
│  SET DEFAULT:                                  │
│  ├── 使う: デフォルト値に戻したい場合         │
│  ├── 例: カテゴリ削除時に「未分類」に設定     │
│  └── 前提: DEFAULT値が有効な参照先            │
└────────────────────────────────────────────────┘
```

### コード例4: 制約の遅延チェック（Deferred Constraints）

```sql
-- 制約の遅延チェック: トランザクション終了時に評価
CREATE TABLE categories (
    id        SERIAL PRIMARY KEY,
    name      VARCHAR(100) NOT NULL,
    parent_id INTEGER REFERENCES categories(id)
              DEFERRABLE INITIALLY DEFERRED
);

-- 循環参照を避けつつ、相互参照を可能にする
BEGIN;
INSERT INTO categories (id, name, parent_id) VALUES (1, 'Root', NULL);
INSERT INTO categories (id, name, parent_id) VALUES (2, 'Child', 1);
-- parent_id=1は既に存在するので問題ない

-- 遅延制約なら、以下も可能:
-- INSERT INTO categories VALUES (10, 'A', 20);  -- 20はまだない
-- INSERT INTO categories VALUES (20, 'B', 10);  -- 10は上で挿入済み
-- COMMIT; -- ここで初めてFK制約がチェックされる
COMMIT;
```

---

## 3. パーティショニング

### 3.1 パーティショニングの判断基準

```
┌──── パーティショニングの導入判断 ──────────────┐
│                                                  │
│  導入すべき条件:                                 │
│  ├── テーブルサイズが数百GB以上                  │
│  ├── 数億行を超えるテーブル                      │
│  ├── 時系列データで古いデータの削除/アーカイブ   │
│  │   が頻繁に発生する                           │
│  ├── クエリが特定の範囲にほぼ限定される          │
│  └── VACUUMやANALYZEに時間がかかりすぎる        │
│                                                  │
│  導入すべきでない条件:                           │
│  ├── テーブルサイズが数GB以下                    │
│  ├── クエリが全パーティションにまたがる          │
│  ├── パーティションキーの選定が困難              │
│  └── 運用コスト（パーティション管理）が見合わない│
│                                                  │
│  パーティション数の目安:                         │
│  ├── 10-100 パーティション: 最適                │
│  ├── 100-1000: 注意が必要（計画時間増大）       │
│  └── 1000+: 非推奨（パフォーマンス低下）        │
└──────────────────────────────────────────────────┘
```

### コード例5: テーブルパーティショニング

```sql
-- レンジパーティション（日付ベース）
CREATE TABLE access_logs (
    id         BIGSERIAL,
    user_id    INTEGER,
    action     VARCHAR(50),
    ip_address INET,
    created_at TIMESTAMP NOT NULL,
    -- パーティションテーブルの場合、PKにパーティションキーを含める必要がある
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- 月別パーティション作成
CREATE TABLE access_logs_2024_01
    PARTITION OF access_logs
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE access_logs_2024_02
    PARTITION OF access_logs
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- デフォルトパーティション（範囲外のデータを受け入れ）
CREATE TABLE access_logs_default
    PARTITION OF access_logs DEFAULT;

-- パーティションの自動作成スクリプト（PostgreSQL: 関数で自動化）
CREATE OR REPLACE FUNCTION create_monthly_partition(
    table_name TEXT,
    year INTEGER,
    month INTEGER
) RETURNS VOID AS $$
DECLARE
    partition_name TEXT;
    start_date DATE;
    end_date DATE;
BEGIN
    partition_name := format('%s_%s_%s', table_name, year,
                            LPAD(month::TEXT, 2, '0'));
    start_date := make_date(year, month, 1);
    end_date := start_date + INTERVAL '1 month';

    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS %I PARTITION OF %I
         FOR VALUES FROM (%L) TO (%L)',
        partition_name, table_name, start_date, end_date
    );
END;
$$ LANGUAGE plpgsql;

-- 使用例: 2024年の12ヶ月分を一括作成
SELECT create_monthly_partition('access_logs', 2024, m)
FROM generate_series(1, 12) AS m;

-- リストパーティション（地域ベース）
CREATE TABLE sales (
    id     SERIAL,
    region VARCHAR(20) NOT NULL,
    amount DECIMAL(10,2),
    date   DATE,
    PRIMARY KEY (id, region)
) PARTITION BY LIST (region);

CREATE TABLE sales_japan PARTITION OF sales
    FOR VALUES IN ('tokyo', 'osaka', 'nagoya');
CREATE TABLE sales_asia PARTITION OF sales
    FOR VALUES IN ('seoul', 'taipei', 'singapore');
CREATE TABLE sales_default PARTITION OF sales DEFAULT;

-- ハッシュパーティション（均等分散）
CREATE TABLE events (
    id      BIGSERIAL,
    user_id INTEGER NOT NULL,
    data    JSONB,
    PRIMARY KEY (id, user_id)
) PARTITION BY HASH (user_id);

CREATE TABLE events_0 PARTITION OF events FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE events_1 PARTITION OF events FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE events_2 PARTITION OF events FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE events_3 PARTITION OF events FOR VALUES WITH (MODULUS 4, REMAINDER 3);

-- パーティションプルーニングの確認
EXPLAIN ANALYZE
SELECT * FROM access_logs WHERE created_at >= '2024-03-01' AND created_at < '2024-04-01';
-- → access_logs_2024_03 のみがスキャンされる
```

### パーティショニングの構造

```
┌──────── パーティショニングの種類 ────────────┐
│                                               │
│  レンジ (RANGE)                               │
│  ┌──────┬──────┬──────┬──────┐               │
│  │ 1月  │ 2月  │ 3月  │ ...  │               │
│  └──────┴──────┴──────┴──────┘               │
│  日付・数値の範囲で分割                       │
│  用途: 時系列データ、ログ、取引履歴           │
│                                               │
│  リスト (LIST)                                │
│  ┌──────┬──────┬──────┐                      │
│  │ 日本 │ 韓国 │ 台湾 │                      │
│  └──────┴──────┴──────┘                      │
│  離散的な値で分割                             │
│  用途: 地域別、カテゴリ別、ステータス別       │
│                                               │
│  ハッシュ (HASH)                              │
│  ┌──────┬──────┬──────┬──────┐               │
│  │ mod0 │ mod1 │ mod2 │ mod3 │               │
│  └──────┴──────┴──────┴──────┘               │
│  ハッシュ値で均等分散                         │
│  用途: 均等分散が必要な場合、特定の条件なし   │
│                                               │
│  ※ パーティションプルーニング:                 │
│    クエリ条件に該当しないパーティションを       │
│    自動的にスキップ → 大幅な性能向上           │
└───────────────────────────────────────────────┘
```

### 3.2 パーティションの運用管理

```sql
-- 古いパーティションの削除（DROP: 瞬時に実行）
DROP TABLE access_logs_2023_01;

-- パーティションのデタッチ（テーブルは残す）
ALTER TABLE access_logs DETACH PARTITION access_logs_2023_01;

-- デタッチしたテーブルをアーカイブテーブルに名前変更
ALTER TABLE access_logs_2023_01 RENAME TO access_logs_archive_2023_01;

-- パーティションの統計情報確認
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) AS total_size,
    n_live_tup AS row_count
FROM pg_stat_user_tables
WHERE tablename LIKE 'access_logs_%'
ORDER BY tablename;
```

---

## 4. 主キー戦略

### コード例6: 主キーの設計パターン

```sql
-- 1. 自然キー: ビジネス上の意味を持つキー
CREATE TABLE countries (
    code CHAR(2) PRIMARY KEY,   -- ISO 3166-1 alpha-2
    name VARCHAR(100) NOT NULL
);

-- 2. サロゲートキー: 人工的な連番ID
CREATE TABLE products (
    id   SERIAL PRIMARY KEY,    -- サロゲートキー
    sku  VARCHAR(20) UNIQUE NOT NULL,  -- 自然キー（ビジネス識別子）
    name VARCHAR(200) NOT NULL
);

-- SERIAL vs IDENTITY の違い（PostgreSQL 10+）
CREATE TABLE products_v2 (
    -- IDENTITY列（SQL標準、推奨）
    id   INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name VARCHAR(200) NOT NULL
);
-- GENERATED ALWAYS: 手動でのID指定を拒否（安全）
-- GENERATED BY DEFAULT: 手動指定も許可

-- 3. UUID: 分散システム向け
CREATE TABLE events (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL,
    payload    JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- UUID v7: 時系列ソート可能（PostgreSQL 17+またはアプリ生成）
-- UUIDv7のフォーマット:
-- ┌─────────────────────┬──────┬───────────────────┐
-- │  48bit timestamp    │ ver  │  74bit random      │
-- └─────────────────────┴──────┴───────────────────┘
-- → タイムスタンプベースでソート可能、B-Treeに有利

-- 4. ULID: 時系列ソート可能なUUID代替
-- アプリケーション側で生成するのが一般的
-- PostgreSQLでは拡張を使用するか、テキストで格納
CREATE TABLE activities (
    id         CHAR(26) PRIMARY KEY,  -- ULID（Base32エンコード、26文字）
    action     VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 5. 複合キー: 中間テーブルで使用
CREATE TABLE user_roles (
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    role_id INTEGER REFERENCES roles(id) ON DELETE CASCADE,
    granted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, role_id)
);
```

### 主キー戦略の選定フロー

```
┌──── 主キー戦略の選定 ──────────────────────────┐
│                                                  │
│  Q: 分散システム or マイクロサービス？            │
│  │                                              │
│  ├── Yes                                        │
│  │   Q: 時系列ソートが必要？                    │
│  │   ├── Yes → UUID v7 or ULID                 │
│  │   └── No  → UUID v4                         │
│  │                                              │
│  └── No (単一DB)                                │
│      Q: 外部に公開するID？                      │
│      ├── Yes → UUID（推測困難）                 │
│      └── No                                     │
│          Q: パフォーマンス最優先？               │
│          ├── Yes → SERIAL/IDENTITY（最も高速）  │
│          └── No  → SERIAL/IDENTITY              │
│                                                  │
│  例外:                                           │
│  ├── ISO規格コード → 自然キー（国コード等）     │
│  ├── 中間テーブル → 複合キー                    │
│  └── イベントソーシング → UUID v7 / ULID         │
└──────────────────────────────────────────────────┘
```

---

## 5. テーブル設計のベストプラクティス

### コード例7: 推奨されるテーブル構造

```sql
-- 推奨されるテーブル構造
CREATE TABLE articles (
    -- 主キー
    id          BIGSERIAL PRIMARY KEY,

    -- ビジネスデータ
    title       VARCHAR(500) NOT NULL,
    slug        VARCHAR(500) UNIQUE NOT NULL,
    body        TEXT NOT NULL,
    excerpt     VARCHAR(1000),
    status      VARCHAR(20) NOT NULL DEFAULT 'draft'
                CHECK (status IN ('draft', 'published', 'archived')),

    -- リレーション
    author_id   INTEGER NOT NULL REFERENCES users(id),
    category_id INTEGER REFERENCES categories(id) ON DELETE SET NULL,

    -- メタデータ
    tags        TEXT[] DEFAULT '{}',
    metadata    JSONB DEFAULT '{}',

    -- 監査列（全テーブル共通）
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at  TIMESTAMPTZ,  -- 論理削除用

    -- インデックス
    CONSTRAINT articles_slug_format CHECK (slug ~ '^[a-z0-9-]+$')
);

-- 更新日時の自動更新トリガー
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_articles_updated_at
    BEFORE UPDATE ON articles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- 論理削除のソフトデリートパターン
-- 削除済みレコードを除外するビュー
CREATE VIEW active_articles AS
SELECT * FROM articles WHERE deleted_at IS NULL;

-- 論理削除関数
CREATE OR REPLACE FUNCTION soft_delete_article(article_id BIGINT)
RETURNS VOID AS $$
BEGIN
    UPDATE articles SET deleted_at = NOW() WHERE id = article_id;
END;
$$ LANGUAGE plpgsql;

-- インデックス設計
CREATE INDEX idx_articles_author ON articles (author_id);
CREATE INDEX idx_articles_category ON articles (category_id);
CREATE INDEX idx_articles_status ON articles (status) WHERE deleted_at IS NULL;
CREATE INDEX idx_articles_created ON articles (created_at DESC);
CREATE INDEX idx_articles_tags ON articles USING GIN (tags);
CREATE INDEX idx_articles_metadata ON articles USING GIN (metadata);
```

### コード例8: 共通パターン — 監査テーブル

```sql
-- 監査テーブル（変更履歴の記録）
CREATE TABLE audit_log (
    id          BIGSERIAL PRIMARY KEY,
    table_name  VARCHAR(100) NOT NULL,
    record_id   TEXT NOT NULL,
    action      VARCHAR(10) NOT NULL CHECK (action IN ('INSERT', 'UPDATE', 'DELETE')),
    old_data    JSONB,
    new_data    JSONB,
    changed_by  INTEGER REFERENCES users(id),
    changed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ip_address  INET
) PARTITION BY RANGE (changed_at);

-- 汎用監査トリガー
CREATE OR REPLACE FUNCTION audit_trigger_func()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (table_name, record_id, action, new_data, changed_by)
        VALUES (TG_TABLE_NAME, NEW.id::TEXT, 'INSERT', row_to_json(NEW)::JSONB,
                current_setting('app.current_user_id', TRUE)::INTEGER);
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (table_name, record_id, action, old_data, new_data, changed_by)
        VALUES (TG_TABLE_NAME, NEW.id::TEXT, 'UPDATE',
                row_to_json(OLD)::JSONB, row_to_json(NEW)::JSONB,
                current_setting('app.current_user_id', TRUE)::INTEGER);
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (table_name, record_id, action, old_data, changed_by)
        VALUES (TG_TABLE_NAME, OLD.id::TEXT, 'DELETE', row_to_json(OLD)::JSONB,
                current_setting('app.current_user_id', TRUE)::INTEGER);
        RETURN OLD;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- articlesテーブルに監査を適用
CREATE TRIGGER trg_articles_audit
    AFTER INSERT OR UPDATE OR DELETE ON articles
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_func();
```

---

## リレーションシップ種別比較表

| 種別 | 実装方法 | 例 | FK配置 | インデックス |
|------|---------|-----|--------|------------|
| 1:1 | 共有主キー or UNIQUE FK | ユーザー - プロフィール | 子テーブル | PK（自動） |
| 1:N | 子テーブルにFK | 部署 - 社員 | 子テーブル（N側） | FK列に手動作成 |
| M:N | 中間テーブル | 社員 - プロジェクト | 中間テーブル | 複合PK + 個別FK |
| 自己参照 | 同テーブルにFK | 社員 - 上司 | 同テーブル | FK列に手動作成 |
| ポリモーフィック | 型識別カラム + FK | コメント - 各種エンティティ | 注意が必要 | 型+IDの複合 |

## 主キー戦略比較表

| 方式 | サイズ | 長所 | 短所 | 適する場面 | B-Tree効率 |
|------|--------|------|------|-----------|-----------|
| SERIAL/IDENTITY | 4/8 byte | シンプル、高速、省メモリ | 分散非対応、推測可能 | 単一DB | 最高 |
| UUID v4 | 16 byte | 分散生成可能、推測困難 | ソート不可、インデックス断片化 | 分散システム | 低 |
| UUID v7 | 16 byte | 時系列ソート、分散対応 | PostgreSQL 17+必要 | イベント系 | 高 |
| ULID | 16 byte | 時系列ソート、分散対応 | DB非標準 | イベントソーシング | 高 |
| 自然キー | 可変 | ビジネス意味あり | 変更リスク、長い | ISO規格コード | 中 |
| 複合キー | 可変 | 正規化に忠実 | JOINが複雑 | 中間テーブル | 中 |

## RDBMS間のパーティション機能比較表

| 機能 | PostgreSQL | MySQL (InnoDB) | Oracle | SQL Server |
|------|-----------|----------------|--------|------------|
| RANGE | ✓ (10+) | ✓ | ✓ | ✓ |
| LIST | ✓ (10+) | ✓ | ✓ | ✗（CHECK制約で代替） |
| HASH | ✓ (11+) | ✓ | ✓ | ✗ |
| サブパーティション | ✓ (手動) | ✓ | ✓ | ✗ |
| DEFAULT パーティション | ✓ (11+) | ✗ | ✗ | ✗ |
| 自動パーティション | ✗（pg_partman） | ✗ | ✓ (Interval) | ✓ (Sliding Window) |
| パーティションプルーニング | ✓ | ✓ | ✓ | ✓ |
| DETACH CONCURRENTLY | ✓ (14+) | ✗ | ✗ | ✗ |
| グローバルインデックス | ✗ | ✗ | ✓ | ✓ |

---

## アンチパターン

### アンチパターン1: ポリモーフィック関連の不適切な実装

```sql
-- NG: 型に応じてFK先が変わる（外部キー制約が使えない）
CREATE TABLE comments (
    id              SERIAL PRIMARY KEY,
    commentable_type VARCHAR(50),   -- 'article' or 'product' or 'video'
    commentable_id   INTEGER,       -- FK先が不明
    body            TEXT
);
-- → commentable_id に REFERENCES を付けられない
-- → 参照整合性がアプリケーション層に依存

-- OK: 専用の中間テーブルを使う
CREATE TABLE comments (
    id   SERIAL PRIMARY KEY,
    body TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE article_comments (
    comment_id INTEGER PRIMARY KEY REFERENCES comments(id) ON DELETE CASCADE,
    article_id INTEGER NOT NULL REFERENCES articles(id) ON DELETE CASCADE
);
CREATE TABLE product_comments (
    comment_id INTEGER PRIMARY KEY REFERENCES comments(id) ON DELETE CASCADE,
    product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE
);

-- OK: 排他的リレーション（PostgreSQL CHECK制約）
CREATE TABLE comments_v2 (
    id          SERIAL PRIMARY KEY,
    body        TEXT NOT NULL,
    article_id  INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    product_id  INTEGER REFERENCES products(id) ON DELETE CASCADE,
    video_id    INTEGER REFERENCES videos(id) ON DELETE CASCADE,
    -- 正確に1つだけがNOT NULL
    CONSTRAINT chk_exclusive CHECK (
        (article_id IS NOT NULL)::INTEGER +
        (product_id IS NOT NULL)::INTEGER +
        (video_id IS NOT NULL)::INTEGER = 1
    )
);
```

### アンチパターン2: 制約の不足

```sql
-- NG: 制約なしのテーブル
CREATE TABLE users (
    id    SERIAL,
    email TEXT,
    age   INTEGER
);
-- → 重複メール、NULL、負の年齢、不正な形式が全て許容される

-- OK: 適切な制約を設定
CREATE TABLE users (
    id    SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE
          CHECK (email ~* '^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$'),
    age   INTEGER CHECK (age BETWEEN 0 AND 200)
);
```

### アンチパターン3: インデックスなしのFK

```sql
-- NG: FK列にインデックスがない
CREATE TABLE orders (
    id          SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id)
    -- customer_id にインデックスがない
    -- → JOINやDELETEが遅い
);

-- OK: FK列にインデックスを作成
CREATE TABLE orders (
    id          SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id)
);
CREATE INDEX idx_orders_customer ON orders (customer_id);
-- → customers.idのDELETE時にorders側のインデックスで高速確認
-- → JOINもインデックス使用で高速化
```

---

## エッジケース

### エッジケース1: 循環参照

```sql
-- 循環参照の例: 部署に「現在のリーダー」を設定したい
-- employees → departments (所属) と departments → employees (リーダー)
CREATE TABLE departments (
    id        SERIAL PRIMARY KEY,
    name      VARCHAR(100) NOT NULL,
    leader_id INTEGER  -- 後でFKを追加
);

CREATE TABLE employees (
    id      SERIAL PRIMARY KEY,
    name    VARCHAR(100) NOT NULL,
    dept_id INTEGER REFERENCES departments(id)
);

-- 循環FK
ALTER TABLE departments ADD CONSTRAINT fk_leader
    FOREIGN KEY (leader_id) REFERENCES employees(id)
    DEFERRABLE INITIALLY DEFERRED;

-- 挿入時は遅延制約で対応
BEGIN;
INSERT INTO departments (id, name) VALUES (1, '開発部');
INSERT INTO employees (id, name, dept_id) VALUES (1, '田中', 1);
UPDATE departments SET leader_id = 1 WHERE id = 1;
COMMIT;
```

### エッジケース2: 大量バルクロード時の制約一時無効化

```sql
-- 大量データロード時のFK制約を一時的に無効化
-- PostgreSQL:
ALTER TABLE order_items DISABLE TRIGGER ALL;

-- データロード
COPY order_items FROM '/data/order_items.csv' CSV HEADER;

-- 制約を再有効化
ALTER TABLE order_items ENABLE TRIGGER ALL;

-- データ整合性の手動検証
SELECT oi.id FROM order_items oi
LEFT JOIN orders o ON oi.order_id = o.id
WHERE o.id IS NULL;
-- → 0行であることを確認
```

### エッジケース3: マルチテナントのスキーマ設計

```sql
-- 方式1: 行レベル分離（1テーブルに全テナント）
CREATE TABLE tenant_users (
    id        SERIAL PRIMARY KEY,
    tenant_id INTEGER NOT NULL REFERENCES tenants(id),
    name      VARCHAR(100) NOT NULL,
    email     VARCHAR(255) NOT NULL,
    UNIQUE (tenant_id, email)  -- テナント内でのメール一意性
);

-- RLSで自動フィルタリング
ALTER TABLE tenant_users ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON tenant_users
    USING (tenant_id = current_setting('app.tenant_id')::INTEGER);

-- 方式2: スキーマ分離（テナントごとにスキーマ）
CREATE SCHEMA tenant_001;
CREATE TABLE tenant_001.users (...);
-- メリット: 完全な分離、テナント削除が容易
-- デメリット: テナント数が多いとスキーマ管理が困難
```

---

## 演習

### 演習1（基礎）: ECサイトのスキーマ設計

以下の要件を満たすERDとCREATE TABLE文を作成せよ。

**要件**:
- 顧客、商品、注文、注文明細
- 顧客は複数の配送先住所を持てる
- 商品はカテゴリに属する（カテゴリは階層構造）
- 注文ステータスの遷移管理
- 全テーブルに監査列

### 演習2（応用）: パーティションの設計

1億行のアクセスログテーブルに対して、月次パーティションを設計せよ。古いデータ（1年以上）のアーカイブ戦略も含めること。

### 演習3（発展）: マルチテナントスキーマ

SaaS アプリケーションのマルチテナントスキーマを3つの方式（行レベル、スキーマレベル、DB レベル）で設計し、トレードオフを分析せよ。

---

## FAQ

### Q1: サロゲートキーと自然キーどちらを主キーにすべきか？

多くの場合サロゲートキー（SERIAL/UUID）が推奨される。自然キーは変更リスクがあり、複合キーはJOINを複雑化する。ただし、ISO国コード（JP, US）のような安定した自然キーは直接使用しても問題ない。

### Q2: パーティショニングはいつ導入すべきか？

テーブルサイズが数百GB以上、または数億行を超える場合に検討する。パーティショニングによりパーティションプルーニング（不要なパーティションのスキップ）が効き、古いデータのアーカイブ/削除もパーティション単位で高速に行える。

### Q3: 外部キー制約はパフォーマンスに影響するか？

INSERT/UPDATE/DELETE時にFK先の存在確認が発生するため若干のオーバーヘッドがある。ただし、データ整合性の保証というメリットが圧倒的に大きい。大量バルクロード時のみ一時的に無効化する手法がある。FK列へのインデックスも忘れずに作成すること。

### Q4: 論理削除と物理削除のどちらを使うべきか？

監査要件やデータ復元の必要性がある場合は論理削除（`deleted_at`カラム）を使用する。ただし、全クエリに `WHERE deleted_at IS NULL` を付ける必要があり、インデックス設計にも影響する。要件がなければ物理削除がシンプル。

### Q5: JSONB列はいつ使うべきか？

スキーマが頻繁に変わる属性（商品の可変仕様、ユーザー設定等）に適している。ただし、リレーションの代替として使うべきではない。GINインデックスで効率的に検索可能だが、JOIN対象のデータはリレーショナルに設計すべき。

---

## トラブルシューティング

| 問題 | 原因 | 対処法 |
|------|------|--------|
| FK制約違反でINSERT失敗 | 参照先が存在しない | INSERT順序の確認、遅延制約の検討 |
| パーティションに入らない | 範囲外のデータ | DEFAULTパーティションの追加 |
| UUIDのINSERTが遅い | インデックス断片化 | UUID v7/ULIDへの移行検討 |
| UNIQUE制約違反 | 重複データ | ON CONFLICT (UPSERT) の使用 |
| CHECK制約が複雑すぎる | 条件の過剰設定 | アプリケーション層でのバリデーション併用 |
| 外部キーのDELETEが遅い | FK列のインデックス不足 | FK列にインデックス作成 |

---

## セキュリティに関する考察

```sql
-- Row Level Security（RLS）の活用
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;

-- 一般ユーザー: 自分の注文のみ閲覧可能
CREATE POLICY orders_user_policy ON orders
    FOR SELECT
    USING (customer_id = current_setting('app.current_user_id')::INTEGER);

-- 管理者: 全注文閲覧可能
CREATE POLICY orders_admin_policy ON orders
    FOR ALL
    USING (current_setting('app.current_role') = 'admin');

-- テーブル権限の最小化
GRANT SELECT, INSERT, UPDATE ON orders TO app_user;
-- DELETE権限は付与しない → 論理削除のみ
REVOKE DELETE ON orders FROM app_user;
```

---

## まとめ

| 項目 | 要点 |
|------|------|
| ER図 | テーブル間の関係を視覚化。設計の出発点。IE記法が標準的 |
| リレーション | 1:1, 1:N, M:N を正しく実装。自己参照やポリモーフィックも |
| 制約 | NOT NULL, UNIQUE, CHECK, FK で整合性保証。排他制約も活用 |
| 参照アクション | CASCADE, RESTRICT, SET NULL を適切に選択 |
| 主キー | SERIAL（単一DB）/ UUID v7（分散）が推奨 |
| パーティション | 大規模テーブルをRANGE/LIST/HASHで分割。運用管理も重要 |
| 監査列 | created_at, updated_at は全テーブルに。監査トリガーで変更履歴 |
| 論理削除 | deleted_at カラム + ビュー/RLS で管理 |

---

## 次に読むべきガイド

- [02-migration.md](./02-migration.md) — スキーマ変更のマイグレーション
- [03-data-modeling.md](./03-data-modeling.md) — 分析向けデータモデリング
- [00-normalization.md](./00-normalization.md) — 正規化の理論

---

## 参考文献

1. PostgreSQL Documentation — "Table Partitioning" https://www.postgresql.org/docs/current/ddl-partitioning.html
2. Fowler, M. (2002). *Patterns of Enterprise Application Architecture*. Addison-Wesley.
3. Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly Media. Chapter 2: Data Models and Query Languages.
4. PostgreSQL Documentation — "Row Security Policies" https://www.postgresql.org/docs/current/ddl-rowsecurity.html
5. Percona Blog — "UUID vs Auto-Increment" https://www.percona.com/blog/
6. Date, C.J. (2019). *Database Design and Relational Theory*. O'Reilly Media.
