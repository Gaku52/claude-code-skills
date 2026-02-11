# スキーマ設計 — ER図・制約・パーティション

> スキーマ設計はデータベースの骨格を決定する作業であり、テーブル構造、制約、リレーションシップの設計品質がアプリケーション全体の信頼性とパフォーマンスを左右する。

## この章で学ぶこと

1. ER図の読み書きとリレーションシップの種類（1:1, 1:N, M:N）
2. 制約（PRIMARY KEY, FOREIGN KEY, UNIQUE, CHECK, NOT NULL）の適切な使い方
3. パーティショニングによる大規模テーブルの管理戦略

---

## 1. ER図とリレーションシップ

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

### コード例1: リレーションシップの実装

```sql
-- 1:1 リレーション
CREATE TABLE users (
    id       SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email    VARCHAR(255) UNIQUE NOT NULL
);

CREATE TABLE user_profiles (
    user_id    INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    bio        TEXT,
    avatar_url VARCHAR(500),
    birthdate  DATE
);

-- 1:N リレーション
CREATE TABLE departments (
    id       SERIAL PRIMARY KEY,
    name     VARCHAR(100) NOT NULL,
    location VARCHAR(100)
);

CREATE TABLE employees (
    id            SERIAL PRIMARY KEY,
    name          VARCHAR(100) NOT NULL,
    department_id INTEGER REFERENCES departments(id) ON DELETE SET NULL,
    salary        DECIMAL(10, 2) CHECK (salary >= 0),
    hired_date    DATE NOT NULL DEFAULT CURRENT_DATE
);

-- M:N リレーション（中間テーブル）
CREATE TABLE projects (
    id       SERIAL PRIMARY KEY,
    name     VARCHAR(200) NOT NULL,
    deadline DATE
);

CREATE TABLE employee_projects (
    employee_id INTEGER REFERENCES employees(id) ON DELETE CASCADE,
    project_id  INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    role        VARCHAR(50) DEFAULT 'member',
    joined_at   DATE NOT NULL DEFAULT CURRENT_DATE,
    PRIMARY KEY (employee_id, project_id)
);
```

---

## 2. 制約

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

    -- テーブルレベルの制約
    CONSTRAINT chk_dates CHECK (
        shipped_date IS NULL OR shipped_date >= order_date
    ),
    CONSTRAINT chk_delivery CHECK (
        delivered_date IS NULL OR delivered_date >= shipped_date
    )
);

-- 排他制約（PostgreSQL: 期間の重複防止）
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

---

## 3. パーティショニング

### コード例4: テーブルパーティショニング

```sql
-- レンジパーティション（日付ベース）
CREATE TABLE access_logs (
    id         BIGSERIAL,
    user_id    INTEGER,
    action     VARCHAR(50),
    ip_address INET,
    created_at TIMESTAMP NOT NULL
) PARTITION BY RANGE (created_at);

-- 月別パーティション作成
CREATE TABLE access_logs_2024_01
    PARTITION OF access_logs
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE access_logs_2024_02
    PARTITION OF access_logs
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- リストパーティション（地域ベース）
CREATE TABLE sales (
    id     SERIAL,
    region VARCHAR(20) NOT NULL,
    amount DECIMAL(10,2),
    date   DATE
) PARTITION BY LIST (region);

CREATE TABLE sales_japan PARTITION OF sales
    FOR VALUES IN ('tokyo', 'osaka', 'nagoya');
CREATE TABLE sales_asia PARTITION OF sales
    FOR VALUES IN ('seoul', 'taipei', 'singapore');

-- ハッシュパーティション（均等分散）
CREATE TABLE events (
    id      BIGSERIAL,
    user_id INTEGER NOT NULL,
    data    JSONB
) PARTITION BY HASH (user_id);

CREATE TABLE events_0 PARTITION OF events FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE events_1 PARTITION OF events FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE events_2 PARTITION OF events FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE events_3 PARTITION OF events FOR VALUES WITH (MODULUS 4, REMAINDER 3);
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
│                                               │
│  リスト (LIST)                                │
│  ┌──────┬──────┬──────┐                      │
│  │ 日本 │ 韓国 │ 台湾 │                      │
│  └──────┴──────┴──────┘                      │
│  離散的な値で分割                             │
│                                               │
│  ハッシュ (HASH)                              │
│  ┌──────┬──────┬──────┬──────┐               │
│  │ mod0 │ mod1 │ mod2 │ mod3 │               │
│  └──────┴──────┴──────┴──────┘               │
│  ハッシュ値で均等分散                         │
│                                               │
│  ※ パーティションプルーニング:                 │
│    クエリ条件に該当しないパーティションを       │
│    自動的にスキップ → 大幅な性能向上           │
└───────────────────────────────────────────────┘
```

### コード例5: 主キーとサロゲートキーの設計

```sql
-- 自然キー: ビジネス上の意味を持つキー
CREATE TABLE countries (
    code CHAR(2) PRIMARY KEY,   -- ISO 3166-1 alpha-2
    name VARCHAR(100) NOT NULL
);

-- サロゲートキー: 人工的な連番ID
CREATE TABLE products (
    id   SERIAL PRIMARY KEY,    -- サロゲートキー
    sku  VARCHAR(20) UNIQUE NOT NULL,  -- 自然キー（ビジネス識別子）
    name VARCHAR(200) NOT NULL
);

-- UUID: 分散システム向け
CREATE TABLE events (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_type VARCHAR(50) NOT NULL,
    payload    JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ULID: 時系列ソート可能なUUID代替
-- アプリケーション側で生成するのが一般的
```

### コード例6: テーブル設計のベストプラクティス

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
```

---

## リレーションシップ種別比較表

| 種別 | 実装方法 | 例 | FK配置 |
|------|---------|-----|--------|
| 1:1 | 共有主キー or UNIQUE FK | ユーザー - プロフィール | 子テーブル |
| 1:N | 子テーブルにFK | 部署 - 社員 | 子テーブル（N側） |
| M:N | 中間テーブル | 社員 - プロジェクト | 中間テーブル |
| 自己参照 | 同テーブルにFK | 社員 - 上司 | 同テーブル |
| ポリモーフィック | 型識別カラム + FK | コメント - 各種エンティティ | 注意が必要 |

## 主キー戦略比較表

| 方式 | 長所 | 短所 | 適する場面 |
|------|------|------|-----------|
| SERIAL/IDENTITY | シンプル、高速、省メモリ | 分散非対応、推測可能 | 単一DB |
| UUID v4 | 分散生成可能、推測困難 | 16byte、ソート不可 | 分散システム |
| ULID | 時系列ソート、分散対応 | 16byte | イベントソーシング |
| 自然キー | ビジネス意味あり | 変更リスク、長い | ISO規格コード |
| 複合キー | 正規化に忠実 | JOINが複雑 | 中間テーブル |

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

-- OK: 専用の中間テーブルを使う
CREATE TABLE article_comments (
    comment_id INTEGER PRIMARY KEY REFERENCES comments(id),
    article_id INTEGER NOT NULL REFERENCES articles(id)
);
CREATE TABLE product_comments (
    comment_id INTEGER PRIMARY KEY REFERENCES comments(id),
    product_id INTEGER NOT NULL REFERENCES products(id)
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

---

## FAQ

### Q1: サロゲートキーと自然キーどちらを主キーにすべきか？

多くの場合サロゲートキー（SERIAL/UUID）が推奨される。自然キーは変更リスクがあり、複合キーはJOINを複雑化する。ただし、ISO国コード（JP, US）のような安定した自然キーは直接使用しても問題ない。

### Q2: パーティショニングはいつ導入すべきか？

テーブルサイズが数百GB以上、または数億行を超える場合に検討する。パーティショニングによりパーティションプルーニング（不要なパーティションのスキップ）が効き、古いデータのアーカイブ/削除もパーティション単位で高速に行える。

### Q3: 外部キー制約はパフォーマンスに影響するか？

INSERT/UPDATE/DELETE時にFK先の存在確認が発生するため若干のオーバーヘッドがある。ただし、データ整合性の保証というメリットが圧倒的に大きい。大量バルクロード時のみ一時的に無効化する手法がある。

---

## まとめ

| 項目 | 要点 |
|------|------|
| ER図 | テーブル間の関係を視覚化。設計の出発点 |
| リレーション | 1:1, 1:N, M:N を正しく実装 |
| 制約 | NOT NULL, UNIQUE, CHECK, FK で整合性保証 |
| 主キー | SERIAL（単一DB）/ UUID（分散）が一般的 |
| パーティション | 大規模テーブルをRANGE/LIST/HASHで分割 |
| 監査列 | created_at, updated_at は全テーブルに |

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
