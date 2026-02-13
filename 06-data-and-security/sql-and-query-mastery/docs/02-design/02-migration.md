# マイグレーション

> データベーススキーマの変更をバージョン管理し、ゼロダウンタイムで安全にデプロイする手法を実践的に習得する。本章では、マイグレーションの理論的基盤から、大規模プロダクション環境で求められるゼロダウンタイム手法、ロック回避戦略、ロールバック設計までを網羅的に解説する。

## 前提知識

- [01-schema-design.md](./01-schema-design.md) — スキーマ設計の基礎
- [03-indexing.md](../01-advanced/03-indexing.md) — インデックスの理解
- [02-transactions.md](../01-advanced/02-transactions.md) — トランザクションとロックの基礎
- DDL（Data Definition Language）の基本構文

## この章で学ぶこと

1. **マイグレーションの基本** — バージョン管理、ロールバック戦略、ツール選定
2. **ゼロダウンタイム手法** — Expand-Contract パターン、オンライン DDL、段階的移行
3. **危険な操作の回避** — ロック問題、大規模データ移行、後方互換性の確保
4. **CI/CD統合** — マイグレーションの自動化、lint、テスト戦略
5. **複数環境の管理** — 開発/ステージング/本番の整合性
6. **RDBMS別の注意点** — PostgreSQL, MySQL, SQL Server それぞれの特性

---

## 1. マイグレーションの基本概念

### なぜマイグレーションが必要か

データベーススキーマはアプリケーションの進化とともに変更される。マイグレーションシステムなしでは以下の問題が発生する。

```
マイグレーションなしの世界（アンチパターン）
=============================================

問題1: 環境間の不整合
  開発者A: ALTER TABLE users ADD COLUMN phone VARCHAR(20);
  開発者B: ALTER TABLE users ADD COLUMN phone VARCHAR(15);  -- 型が違う！
  本番:    phone列が存在しない  -- 適用漏れ

問題2: ロールバック不能
  DBA: ALTER TABLE orders DROP COLUMN old_status;
  → 「やっぱり戻して」→ データ消失、復旧不能

問題3: 適用順序の管理不能
  migration_1: ADD COLUMN status
  migration_2: CREATE INDEX ON status
  → migration_2が先に実行されると失敗

マイグレーションシステムによる解決:
  ✓ バージョン番号で適用順序を保証
  ✓ UP/DOWNスクリプトでロールバック可能
  ✓ schema_migrationsテーブルで適用状態を管理
  ✓ 全環境で同じスクリプトを使用
```

### マイグレーションのライフサイクル

```
マイグレーションのライフサイクル
=================================

v1 (現在)          v2 (目標)
+-----------+      +-----------+
| users     |      | users     |
|  id       |  --> |  id       |
|  name     |      |  name     |
|  email    |      |  email    |
+-----------+      |  phone    |  <-- 追加
                   |  status   |  <-- 追加
                   +-----------+

マイグレーションファイル:
  20260211_001_add_phone_to_users.sql
  20260211_002_add_status_to_users.sql

各ファイルに UP (適用) と DOWN (ロールバック) を記述

適用フロー:
  ┌──────────┐
  │ 未適用   │ → migrate up → │ 適用済み │
  │ pending  │                 │ applied  │
  └──────────┘                 └──────────┘
                                    │
                               migrate down
                                    │
                                    ▼
                               ┌──────────┐
                               │ ロール   │
                               │ バック済 │
                               └──────────┘
```

### コード例 1: マイグレーションツールの比較と使用

```sql
-- === Flyway 形式 ===
-- ファイル命名: V{version}__{description}.sql
-- V2__add_phone_to_users.sql
ALTER TABLE users ADD COLUMN phone VARCHAR(20);

-- ロールバック用（Flyway Pro/Enterprise のみ）
-- U2__add_phone_to_users.sql
ALTER TABLE users DROP COLUMN phone;

-- === golang-migrate 形式 ===
-- 000002_add_phone.up.sql
ALTER TABLE users ADD COLUMN phone VARCHAR(20);

-- 000002_add_phone.down.sql
ALTER TABLE users DROP COLUMN phone;

-- === Liquibase 形式 (XML) ===
-- changelog-2.0.xml
-- <changeSet id="2" author="developer">
--   <addColumn tableName="users">
--     <column name="phone" type="VARCHAR(20)"/>
--   </addColumn>
--   <rollback>
--     <dropColumn tableName="users" columnName="phone"/>
--   </rollback>
-- </changeSet>
```

```bash
# golang-migrate の使用
# マイグレーション作成
migrate create -ext sql -dir ./migrations -seq add_phone_to_users

# 全マイグレーション適用
migrate -path ./migrations -database "postgres://user:pass@localhost/mydb" up

# 1つロールバック
migrate -path ./migrations -database "postgres://user:pass@localhost/mydb" down 1

# 特定バージョンまで適用
migrate -path ./migrations -database "postgres://user:pass@localhost/mydb" goto 5

# 現在のバージョン確認
migrate -path ./migrations -database "postgres://user:pass@localhost/mydb" version

# Flyway
flyway -url=jdbc:postgresql://localhost/mydb migrate
flyway -url=jdbc:postgresql://localhost/mydb info
flyway -url=jdbc:postgresql://localhost/mydb validate
flyway -url=jdbc:postgresql://localhost/mydb repair  # メタデータの修復
```

### コード例 2: Prisma によるマイグレーション

```prisma
// schema.prisma
model User {
  id        Int      @id @default(autoincrement())
  email     String   @unique
  name      String
  phone     String?  // 新規追加
  status    String   @default("active")  // 新規追加
  createdAt DateTime @default(now())
  orders    Order[]
}

model Order {
  id        Int      @id @default(autoincrement())
  userId    Int
  user      User     @relation(fields: [userId], references: [id])
  total     Decimal
  status    String   @default("pending")
  createdAt DateTime @default(now())
}
```

```bash
# マイグレーション生成（開発環境）
npx prisma migrate dev --name add_phone_and_status
# → prisma/migrations/20260211_add_phone_and_status/migration.sql が生成される

# 本番適用（CI/CD パイプライン内）
npx prisma migrate deploy

# ステータス確認
npx prisma migrate status

# マイグレーションのリセット（開発環境のみ）
npx prisma migrate reset

# スキーマの差分確認
npx prisma migrate diff \
  --from-schema-datamodel prisma/schema.prisma \
  --to-schema-datasource prisma/schema.prisma
```

### マイグレーションツール比較表

| ツール | 言語/エコシステム | 方式 | ロールバック | 宣言的 | 特徴 |
|--------|------------------|------|------------|--------|------|
| Flyway | Java/JVM | SQL/Java | Pro版のみ | × | エンタープライズ実績豊富 |
| Liquibase | Java/JVM | XML/YAML/SQL | ○ | ○ | 多形式対応、差分検出 |
| golang-migrate | Go | SQL | ○ | × | シンプル、軽量 |
| Prisma Migrate | TypeScript | 自動生成SQL | × | ○ | ORM統合、型安全 |
| Alembic | Python | Python/SQL | ○ | × | SQLAlchemy統合 |
| Atlas | Go | HCL/SQL | ○ | ○ | 宣言的+命令的両対応 |
| Sqitch | Perl | SQL | ○ | × | 依存関係ベース |
| Knex.js | JavaScript | JavaScript | ○ | × | Node.js統合 |

### 命令的 vs 宣言的マイグレーション

```
命令的マイグレーション（従来型）
================================
開発者が「どう変更するか」を記述

  V1: CREATE TABLE users (id INT, name VARCHAR(100));
  V2: ALTER TABLE users ADD COLUMN email VARCHAR(255);
  V3: CREATE INDEX idx_users_email ON users(email);

  利点: 変更の順序と内容を完全に制御
  欠点: 人手で書くためエラーリスク

宣言的マイグレーション（最新型）
================================
開発者が「最終的にどうなるべきか」を記述
ツールが差分を自動計算

  schema.hcl:
    table "users" {
      column "id" { type = int }
      column "name" { type = varchar(100) }
      column "email" { type = varchar(255) }
      index "idx_users_email" { columns = [column.email] }
    }

  ツール: diff → ALTER TABLE ADD COLUMN email ...
                → CREATE INDEX idx_users_email ...

  利点: 宣言的で読みやすい、差分の自動計算
  欠点: 複雑な移行（データ変換等）は表現困難
```

---

## 2. ゼロダウンタイムマイグレーション

### Expand-Contract パターン

```
Expand-Contract パターン
==========================

Phase 1: Expand（拡張）
  - 新カラム/テーブルを追加
  - 古い形式と新しい形式の両方をサポート
  - アプリは新旧両方に書き込み

Phase 2: Migrate（移行）
  - バックグラウンドで既存データを変換
  - 新しい形式へのアクセスに段階的に切替

Phase 3: Contract（縮退）
  - 古いカラム/テーブルを削除
  - 新しい形式のみをサポート

Timeline:
  Expand       Migrate      Contract
  [+col]       [data]       [-col]
  |------------|------------|----------|
  v1 + v2      v2           v2 only

  ← アプリv1互換 →← アプリv2のみ →

各フェーズの安全な移行:
  Phase 1: マイグレーション実行 → アプリv2デプロイ
  Phase 2: バックフィルジョブ実行（非同期）
  Phase 3: 旧カラム削除マイグレーション
  ※ 各フェーズ間に十分な監視期間を設ける
```

### コード例 3: カラムリネームのゼロダウンタイム手法

```sql
-- [NG] 直接リネーム --> ダウンタイム発生
ALTER TABLE users RENAME COLUMN name TO full_name;
-- --> 既存アプリが "name" を参照してエラー

-- [OK] Expand-Contract パターン（3フェーズ）

-- ===== Phase 1: Expand（新カラム追加 + トリガー） =====
-- マイグレーション: 20260211_001_expand_user_name.sql

ALTER TABLE users ADD COLUMN full_name VARCHAR(255);

-- 既存データをコピー
UPDATE users SET full_name = name WHERE full_name IS NULL;

-- 双方向同期トリガー
CREATE OR REPLACE FUNCTION sync_user_name() RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    IF NEW.full_name IS NULL THEN
      NEW.full_name := NEW.name;
    ELSIF NEW.name IS NULL THEN
      NEW.name := NEW.full_name;
    END IF;
  ELSIF TG_OP = 'UPDATE' THEN
    IF NEW.full_name IS DISTINCT FROM OLD.full_name THEN
      NEW.name := NEW.full_name;
    ELSIF NEW.name IS DISTINCT FROM OLD.name THEN
      NEW.full_name := NEW.name;
    END IF;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_sync_user_name
BEFORE INSERT OR UPDATE ON users
FOR EACH ROW EXECUTE FUNCTION sync_user_name();

-- ===== Phase 2: アプリデプロイ =====
-- アプリを full_name を使うように変更してデプロイ
-- name と full_name の両方を読み書きする互換コードをデプロイ

-- ===== Phase 3: Contract（旧カラム・トリガー削除） =====
-- マイグレーション: 20260218_001_contract_user_name.sql
-- （1週間以上の監視期間を経てから実行）

DROP TRIGGER trg_sync_user_name ON users;
DROP FUNCTION sync_user_name();
ALTER TABLE users DROP COLUMN name;
```

### コード例 4: テーブル分割のゼロダウンタイム手法

```sql
-- ユーザーテーブルからプロフィール情報を分離する例

-- Phase 1: 新テーブル作成 + トリガーで同期
CREATE TABLE user_profiles (
    user_id     INTEGER PRIMARY KEY REFERENCES users(id),
    bio         TEXT,
    avatar_url  VARCHAR(500),
    website     VARCHAR(500),
    created_at  TIMESTAMP DEFAULT NOW(),
    updated_at  TIMESTAMP DEFAULT NOW()
);

-- 既存データの移行
INSERT INTO user_profiles (user_id, bio, avatar_url, website)
SELECT id, bio, avatar_url, website
FROM users
WHERE bio IS NOT NULL OR avatar_url IS NOT NULL;

-- 書き込みの同期トリガー
CREATE OR REPLACE FUNCTION sync_user_profile() RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN
    INSERT INTO user_profiles (user_id, bio, avatar_url, website, updated_at)
    VALUES (NEW.id, NEW.bio, NEW.avatar_url, NEW.website, NOW())
    ON CONFLICT (user_id) DO UPDATE SET
      bio = EXCLUDED.bio,
      avatar_url = EXCLUDED.avatar_url,
      website = EXCLUDED.website,
      updated_at = NOW();
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_sync_profile
AFTER INSERT OR UPDATE ON users
FOR EACH ROW EXECUTE FUNCTION sync_user_profile();

-- Phase 2: アプリを新テーブルに切替
-- Phase 3: 旧カラムとトリガーを削除
DROP TRIGGER trg_sync_profile ON users;
DROP FUNCTION sync_user_profile();
ALTER TABLE users DROP COLUMN bio;
ALTER TABLE users DROP COLUMN avatar_url;
ALTER TABLE users DROP COLUMN website;
```

---

## 3. 危険な操作と安全な代替

### 危険な DDL 操作比較表

| 操作 | 危険度 | ロック種別 | ロック時間 | 安全な代替 |
|---|---|---|---|---|
| `ADD COLUMN` (デフォルトなし) | 低 | AccessExclusiveLock | 瞬時 | そのまま使用可 |
| `ADD COLUMN DEFAULT x` (PG11+) | 低 | AccessExclusiveLock | 瞬時 | そのまま使用可 |
| `ADD COLUMN DEFAULT x` (PG10以前) | 高 | AccessExclusiveLock | 全行書換 | 追加後に UPDATE |
| `DROP COLUMN` | 中 | AccessExclusiveLock | 瞬時（論理削除） | Contract フェーズで実施 |
| `ALTER TYPE` (型変更) | 高 | AccessExclusiveLock | 全行書換 | 新カラム + バックフィル |
| `SET NOT NULL` | 高 | AccessExclusiveLock | 全行スキャン | CHECK制約→VALIDATE→NOT NULL |
| `CREATE INDEX` | 高 | ShareLock | テーブルサイズ依存 | `CONCURRENTLY` |
| `ADD CONSTRAINT` (FK) | 高 | ShareRowExclusiveLock | 全行検証 | `NOT VALID` + `VALIDATE` |
| `RENAME COLUMN` | 高 | AccessExclusiveLock | 瞬時だがアプリ互換性破壊 | Expand-Contract |
| `RENAME TABLE` | 高 | AccessExclusiveLock | 瞬時だがアプリ互換性破壊 | ビュー経由の移行 |
| `DROP TABLE` | 致命的 | AccessExclusiveLock | 瞬時 | RENAME → 監視 → DROP |

### PostgreSQLのロック種別

```
PostgreSQL ロック種別と競合マトリクス
======================================

ロック種別（軽い順）:
  1. AccessShareLock        ← SELECT
  2. RowShareLock           ← SELECT FOR UPDATE
  3. RowExclusiveLock       ← INSERT/UPDATE/DELETE
  4. ShareUpdateExclusiveLock ← VACUUM, VALIDATE CONSTRAINT
  5. ShareLock              ← CREATE INDEX
  6. ShareRowExclusiveLock  ← CREATE TRIGGER, FK追加
  7. ExclusiveLock          ← REFRESH MATERIALIZED VIEW CONCURRENTLY
  8. AccessExclusiveLock    ← ALTER TABLE, DROP TABLE

競合の例:
  AccessExclusiveLock は全操作をブロック
  → ALTER TABLE 実行中は SELECT すら待たされる
  → つまり「瞬時」でも、ロック取得待ちで長時間ブロックする可能性

対策:
  SET lock_timeout = '5s';  -- ロック取得を5秒で諦める
  ALTER TABLE users ADD COLUMN phone VARCHAR(20);
  RESET lock_timeout;
```

### コード例 5: 安全なインデックス追加

```sql
-- [NG] テーブルロックでダウンタイム
CREATE INDEX idx_orders_email ON orders (email);
-- ShareLock: INSERT/UPDATE/DELETEがブロックされる

-- [OK] ロックなし（CONCURRENTLY）
CREATE INDEX CONCURRENTLY idx_orders_email ON orders (email);
-- 注意事項:
-- 1. トランザクション内では使用不可
-- 2. 構築時間は約2-3倍
-- 3. 失敗するとINVALIDインデックスが残る
-- 4. テーブルが2回スキャンされる

-- INVALID インデックスの確認と対処
SELECT indexrelid::regclass, indisvalid
FROM pg_index
WHERE NOT indisvalid;

-- INVALIDインデックスの再構築
REINDEX INDEX CONCURRENTLY idx_orders_email;
-- または削除して再作成
DROP INDEX CONCURRENTLY idx_orders_email;
CREATE INDEX CONCURRENTLY idx_orders_email ON orders (email);
```

### コード例 6: 安全な NOT NULL 制約の追加

```sql
-- [NG] 直接NOT NULLを設定 → 全行スキャン + AccessExclusiveLock
ALTER TABLE users ALTER COLUMN email SET NOT NULL;
-- 1000万行: 数秒〜数十秒のロック

-- [OK] 3段階で安全に追加
-- Step 1: CHECK 制約を NOT VALID で追加（瞬時、ロック最小）
SET lock_timeout = '5s';
ALTER TABLE users
ADD CONSTRAINT chk_users_email_not_null
CHECK (email IS NOT NULL) NOT VALID;
-- → 新しい行のみチェック、既存行は未検証

-- Step 2: 既存データの検証（ShareUpdateExclusiveLock のみ）
-- SELECT/INSERT/UPDATE/DELETEは並行実行可能
ALTER TABLE users VALIDATE CONSTRAINT chk_users_email_not_null;
-- → 全行を検証するが、弱いロックのみ

-- Step 3: NOT NULL に昇格（PostgreSQL 12+は自動認識）
-- PostgreSQL 12+: CHECK制約が存在すれば瞬時に設定可能
ALTER TABLE users ALTER COLUMN email SET NOT NULL;
ALTER TABLE users DROP CONSTRAINT chk_users_email_not_null;

-- 安全な外部キー制約の追加
-- Step 1: NOT VALID で追加
ALTER TABLE orders
ADD CONSTRAINT fk_orders_user_id
FOREIGN KEY (user_id) REFERENCES users(id) NOT VALID;

-- Step 2: 既存データの検証
ALTER TABLE orders VALIDATE CONSTRAINT fk_orders_user_id;
```

### コード例 7: 大規模データのバックフィル

```sql
-- [NG] 一括 UPDATE --> 長時間ロック + WAL 肥大化 + VACUUM負荷
UPDATE orders SET status = 'active' WHERE status IS NULL;
-- 1000万行の場合:
-- - 数分~数十分のロック
-- - WALが数GB生成される
-- - VACUUM が必要になる
-- - レプリカの遅延が発生する

-- [OK] バッチ処理で段階的に更新
DO $$
DECLARE
  batch_size INT := 10000;
  total_updated INT := 0;
  rows_affected INT;
BEGIN
  LOOP
    UPDATE orders
    SET status = 'active'
    WHERE id IN (
      SELECT id FROM orders
      WHERE status IS NULL
      LIMIT batch_size
      FOR UPDATE SKIP LOCKED  -- 他トランザクションと競合回避
    );

    GET DIAGNOSTICS rows_affected = ROW_COUNT;
    total_updated := total_updated + rows_affected;

    RAISE NOTICE 'Updated: % (total: %)', rows_affected, total_updated;

    EXIT WHEN rows_affected = 0;

    PERFORM pg_sleep(0.1);  -- 負荷調整（レプリカ遅延を防ぐ）
    COMMIT;
  END LOOP;
END $$;

-- [推奨] 主キー範囲ベースのバッチ処理（より予測可能）
DO $$
DECLARE
  batch_size INT := 10000;
  min_id INT;
  max_id INT;
  current_id INT;
BEGIN
  SELECT MIN(id), MAX(id) INTO min_id, max_id FROM orders WHERE status IS NULL;

  current_id := min_id;
  WHILE current_id <= max_id LOOP
    UPDATE orders
    SET status = 'active'
    WHERE id >= current_id
      AND id < current_id + batch_size
      AND status IS NULL;

    current_id := current_id + batch_size;

    RAISE NOTICE 'Progress: %/%', current_id - min_id, max_id - min_id;
    COMMIT;
    PERFORM pg_sleep(0.05);
  END LOOP;
END $$;
```

### バックフィルの進行状況モニタリング

```sql
-- 進捗確認クエリ（別セッションから実行）
SELECT
    COUNT(*) FILTER (WHERE status IS NOT NULL) AS completed,
    COUNT(*) FILTER (WHERE status IS NULL) AS remaining,
    COUNT(*) AS total,
    ROUND(
        COUNT(*) FILTER (WHERE status IS NOT NULL)::NUMERIC / COUNT(*) * 100, 1
    ) AS progress_pct
FROM orders;

-- WAL生成量の確認
SELECT pg_wal_lsn_diff(pg_current_wal_lsn(), '0/0') AS wal_bytes;

-- レプリカ遅延の確認
SELECT
    client_addr,
    state,
    pg_wal_lsn_diff(sent_lsn, replay_lsn) AS replay_lag_bytes,
    replay_lag
FROM pg_stat_replication;
```

---

## 4. MySQL 固有の注意点

### MySQL のオンラインDDL

```
MySQL オンライン DDL の挙動
=============================

MySQL 8.0 のALGORITHM:
  INSTANT   : メタデータ変更のみ（瞬時）
  INPLACE   : テーブルコピーなし（並行DML可能）
  COPY      : テーブル全体をコピー（DMLブロック）

操作別の対応:
  ADD COLUMN (末尾)     → INSTANT (MySQL 8.0.12+)
  ADD COLUMN (途中)     → INPLACE or COPY
  DROP COLUMN           → INPLACE (再構築あり)
  MODIFY COLUMN (型変更) → COPY（テーブルロック）
  ADD INDEX             → INPLACE（並行DML可能）
  DROP INDEX            → INPLACE
  RENAME COLUMN         → INSTANT (MySQL 8.0.28+)

注意:
  INPLACEでもメタデータロック取得時に一瞬ブロックする
  長時間トランザクションがあるとメタデータロック待ちになる
```

### コード例 8: MySQL でのマイグレーション

```sql
-- MySQL: ALGORITHM指定
ALTER TABLE users
ADD COLUMN phone VARCHAR(20),
ALGORITHM=INSTANT;  -- 瞬時（MySQL 8.0.12+）

-- MySQL: pt-online-schema-change（Percona Tool）
-- 大規模テーブルのスキーマ変更に推奨
-- 内部的に:
-- 1. 新しいテーブルを作成
-- 2. トリガーで書き込みを同期
-- 3. データをバッチコピー
-- 4. テーブルを切り替え（RENAME）

-- bash:
-- pt-online-schema-change \
--   --alter "ADD COLUMN phone VARCHAR(20)" \
--   --execute \
--   D=mydb,t=users

-- gh-ost（GitHubのツール）
-- トリガーなしでオンラインスキーマ変更
-- bash:
-- gh-ost \
--   --alter="ADD COLUMN phone VARCHAR(20)" \
--   --database=mydb \
--   --table=users \
--   --execute
```

---

## 5. マイグレーション CI/CD

```
CI/CD パイプラインでのマイグレーション
========================================

1. PR 作成
   │
   ▼
2. マイグレーション lint
   - SQL 構文チェック
   - 危険な操作の検出
   - ロールバック可能性の確認
   - スキーマの整合性チェック
   │
   ▼
3. テスト環境での適用テスト
   - 空DBに全マイグレーション適用
   - 本番のスキーマダンプとの差分確認
   │
   ▼
4. ステージング適用
   - 本番同等のデータ量でテスト
   - 適用時間の計測
   - ロールバックのテスト
   │
   ▼
5. レビュー承認
   - DBA/チームリードの承認
   - 適用計画の確認
   │
   ▼
6. 本番適用
   - Blue/Green または Rolling
   - 監視ダッシュボード確認
   - ロールバック手順の準備
   │
   ▼
7. 事後監視
   - エラーレート確認
   - クエリパフォーマンス確認
   - レプリカ遅延確認
```

### コード例 9: マイグレーション lint ツール

```bash
# squawk: PostgreSQL マイグレーション lint
npm install -g squawk-cli

# 危険な操作を検出
squawk migrations/V3__add_index.sql

# 出力例:
# migrations/V3__add_index.sql:1:1
#   warning: prefer-create-index-concurrently
#   CREATE INDEX on a large table without CONCURRENTLY
#   can lock the table for a long time.
#
#   Instead:
#   CREATE INDEX CONCURRENTLY idx_orders_email ON orders (email);

# squawk の設定ファイル (.squawk.toml)
# [general]
# excluded_rules = []
#
# [custom_rules]
# ban_drop_column = true  # DROP COLUMNを禁止
```

```yaml
# GitHub Actions での自動 lint
# .github/workflows/migration-check.yml
name: Migration Check
on:
  pull_request:
    paths:
      - 'migrations/**'

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install squawk
        run: npm install -g squawk-cli

      - name: Get changed migration files
        id: changes
        run: |
          echo "files=$(git diff --name-only origin/main -- 'migrations/*.sql' | tr '\n' ' ')" >> $GITHUB_OUTPUT

      - name: Run squawk
        run: squawk ${{ steps.changes.outputs.files }}

  test-apply:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_DB: test
          POSTGRES_PASSWORD: test
        ports:
          - 5432:5432
    steps:
      - uses: actions/checkout@v4

      - name: Apply all migrations
        run: |
          migrate -path ./migrations \
            -database "postgres://postgres:test@localhost:5432/test?sslmode=disable" \
            up

      - name: Verify schema
        run: |
          pg_dump -s postgres://postgres:test@localhost:5432/test > /tmp/schema.sql
          diff expected_schema.sql /tmp/schema.sql
```

### コード例 10: マイグレーション適用スクリプト

```bash
#!/bin/bash
# deploy_migration.sh - 安全なマイグレーション適用スクリプト

set -euo pipefail

DATABASE_URL="${DATABASE_URL:?DATABASE_URL is required}"
MIGRATIONS_PATH="${MIGRATIONS_PATH:-./migrations}"
LOCK_TIMEOUT="${LOCK_TIMEOUT:-5s}"
STATEMENT_TIMEOUT="${STATEMENT_TIMEOUT:-30s}"

echo "=== マイグレーション開始 ==="
echo "Database: ${DATABASE_URL%%@*}@..."
echo "Path: ${MIGRATIONS_PATH}"

# 1. 現在のバージョンを確認
CURRENT_VERSION=$(migrate -path "${MIGRATIONS_PATH}" -database "${DATABASE_URL}" version 2>&1 || true)
echo "現在のバージョン: ${CURRENT_VERSION}"

# 2. ドライラン（適用するマイグレーション一覧）
echo ""
echo "=== 適用予定のマイグレーション ==="
migrate -path "${MIGRATIONS_PATH}" -database "${DATABASE_URL}" up -dry-run 2>&1 || true

# 3. 確認
read -p "続行しますか？ (y/N): " confirm
if [[ "${confirm}" != "y" ]]; then
    echo "中止しました"
    exit 0
fi

# 4. タイムアウト設定を適用
psql "${DATABASE_URL}" -c "ALTER DATABASE $(psql "${DATABASE_URL}" -t -c 'SELECT current_database()') SET lock_timeout = '${LOCK_TIMEOUT}';"
psql "${DATABASE_URL}" -c "ALTER DATABASE $(psql "${DATABASE_URL}" -t -c 'SELECT current_database()') SET statement_timeout = '${STATEMENT_TIMEOUT}';"

# 5. マイグレーション適用
echo ""
echo "=== マイグレーション適用中 ==="
if migrate -path "${MIGRATIONS_PATH}" -database "${DATABASE_URL}" up; then
    echo "✓ マイグレーション成功"
else
    echo "✗ マイグレーション失敗"
    echo "ロールバックを検討してください:"
    echo "  migrate -path ${MIGRATIONS_PATH} -database \"\${DATABASE_URL}\" down 1"
    exit 1
fi

# 6. タイムアウト設定をリセット
psql "${DATABASE_URL}" -c "ALTER DATABASE $(psql "${DATABASE_URL}" -t -c 'SELECT current_database()') RESET lock_timeout;"
psql "${DATABASE_URL}" -c "ALTER DATABASE $(psql "${DATABASE_URL}" -t -c 'SELECT current_database()') RESET statement_timeout;"

# 7. 新バージョンの確認
NEW_VERSION=$(migrate -path "${MIGRATIONS_PATH}" -database "${DATABASE_URL}" version 2>&1 || true)
echo ""
echo "=== マイグレーション完了 ==="
echo "バージョン: ${CURRENT_VERSION} → ${NEW_VERSION}"
```

---

## 6. ロールバック戦略

### ロールバックの種類

```
ロールバック戦略の比較
========================

1. DOWN マイグレーション（逆実行）
   [適用]  ALTER TABLE users ADD COLUMN phone VARCHAR(20);
   [戻し]  ALTER TABLE users DROP COLUMN phone;
   ○ 最もシンプル
   ✗ データ損失あり（カラム削除でデータ消える）

2. 前方修正（Forward Fix）
   → ロールバックせず、修正マイグレーションを追加
   [V3] ALTER TABLE users ADD COLUMN phone VARCHAR(20);
   [V4] ALTER TABLE users ALTER COLUMN phone TYPE VARCHAR(30);
   ○ データ損失なし
   ○ 本番で最も推奨
   ✗ 緊急時に時間がかかる

3. バックアップ復元
   → DBバックアップから復元
   ○ 確実に戻る
   ✗ ダウンタイムが長い
   ✗ マイグレーション以降のデータが失われる

4. ポイントインタイムリカバリ（PITR）
   → WALを使って特定時点に復元
   ○ 任意の時点に復元可能
   ✗ 設定が複雑
   ✗ 復元に時間がかかる

推奨: 通常は前方修正、致命的な場合のみDOWNマイグレーション
```

### コード例 11: 安全なロールバック設計

```sql
-- UP マイグレーション
-- 20260211_003_add_orders_status.up.sql
BEGIN;

-- ロック待ちのタイムアウト設定
SET lock_timeout = '5s';

ALTER TABLE orders ADD COLUMN status_new VARCHAR(20);

-- デフォルト値の設定（新しい行のみ）
ALTER TABLE orders ALTER COLUMN status_new SET DEFAULT 'pending';

-- マイグレーションバージョンの記録
INSERT INTO schema_migrations (version, description, applied_at)
VALUES ('20260211_003', 'add_orders_status', NOW());

COMMIT;

-- DOWN マイグレーション
-- 20260211_003_add_orders_status.down.sql
BEGIN;

SET lock_timeout = '5s';

ALTER TABLE orders DROP COLUMN IF EXISTS status_new;

DELETE FROM schema_migrations WHERE version = '20260211_003';

COMMIT;
```

### コード例 12: 不可逆マイグレーションの安全策

```sql
-- テーブル削除は直接行わず、リネームで段階的に実施
-- Phase 1: リネーム（即座にロールバック可能）
ALTER TABLE legacy_data RENAME TO _deprecated_legacy_data_20260211;

-- Phase 2: 1-2週間の監視期間
-- アプリケーションエラーがないことを確認

-- Phase 3: バックアップ後に削除
-- pg_dump -t _deprecated_legacy_data_20260211 > backup.sql
DROP TABLE IF EXISTS _deprecated_legacy_data_20260211;

-- カラム削除も同様に段階的に
-- Phase 1: カラムを使用していないことを確認
SELECT count(*) FROM pg_stat_user_tables
WHERE relname = 'users';

-- Phase 2: アプリケーションログで確認（1週間）
-- Phase 3: 削除
ALTER TABLE users DROP COLUMN IF EXISTS old_column;
```

---

## 7. 複数環境の管理

### 環境別マイグレーション戦略

```
環境別のマイグレーション戦略
==============================

開発環境:
  - migrate reset が可能
  - シードデータ投入
  - スキーマ変更のテスト
  └── migrate up → テスト → migrate down → 修正 → migrate up

ステージング環境:
  - 本番と同等のデータ量（匿名化済み）
  - マイグレーション時間の計測
  - ロールバックのテスト
  └── バックアップ → migrate up → テスト → (問題あれば) 復元

本番環境:
  - 段階的適用（カナリアデプロイ）
  - ロック時間の最小化
  - 監視付きで実行
  └── スナップショット → lock_timeout設定 → migrate up → 監視

環境間の整合性チェック:
  pg_dump -s production > prod_schema.sql
  pg_dump -s staging > staging_schema.sql
  diff prod_schema.sql staging_schema.sql  -- 差分がないことを確認
```

### コード例 13: 環境別マイグレーション設定

```yaml
# database.yml (Rails風の設定例)
development:
  adapter: postgresql
  database: myapp_dev
  pool: 5
  timeout: 5000
  migration_options:
    lock_timeout: "30s"
    statement_timeout: "5min"

staging:
  adapter: postgresql
  database: myapp_staging
  pool: 10
  migration_options:
    lock_timeout: "10s"
    statement_timeout: "2min"

production:
  adapter: postgresql
  database: myapp_prod
  pool: 25
  migration_options:
    lock_timeout: "5s"
    statement_timeout: "30s"
    concurrent_index: true
    batch_backfill: true
    batch_size: 10000
    batch_sleep: 0.1
```

---

## 8. 高度なマイグレーションパターン

### コード例 14: enum型の安全な変更

```sql
-- PostgreSQL の ENUM 型にはALTER TYPEの制約がある

-- [OK] 値の追加（安全）
ALTER TYPE order_status ADD VALUE 'cancelled';
ALTER TYPE order_status ADD VALUE 'refunded' AFTER 'shipped';

-- [NG] 値の削除/リネーム → 直接は不可能
-- 安全な代替手順:

-- 1. 新しいENUM型を作成
CREATE TYPE order_status_v2 AS ENUM (
    'pending', 'confirmed', 'shipped', 'delivered', 'cancelled', 'refunded'
);

-- 2. カラムの型を変更
ALTER TABLE orders
    ALTER COLUMN status TYPE order_status_v2
    USING status::text::order_status_v2;

-- 3. 旧ENUM型を削除
DROP TYPE order_status;

-- 4. 新ENUM型をリネーム
ALTER TYPE order_status_v2 RENAME TO order_status;
```

### コード例 15: パーティションテーブルへの移行

```sql
-- 既存の大規模テーブルをパーティション化する
-- 注意: PostgreSQLでは既存テーブルを直接パーティション化できない

-- Phase 1: パーティションテーブルを作成
CREATE TABLE orders_partitioned (
    id          SERIAL,
    customer_id INTEGER NOT NULL,
    order_date  DATE NOT NULL,
    total       DECIMAL(10, 2),
    status      VARCHAR(20),
    created_at  TIMESTAMP DEFAULT NOW()
) PARTITION BY RANGE (order_date);

-- 月次パーティションを作成
CREATE TABLE orders_y2024m01 PARTITION OF orders_partitioned
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
CREATE TABLE orders_y2024m02 PARTITION OF orders_partitioned
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
-- ... 他の月も同様

-- デフォルトパーティション（範囲外データの受け皿）
CREATE TABLE orders_default PARTITION OF orders_partitioned DEFAULT;

-- Phase 2: データ移行（バッチで）
INSERT INTO orders_partitioned
SELECT * FROM orders
WHERE order_date >= '2024-01-01' AND order_date < '2024-02-01';
-- バッチごとにCOMMIT

-- Phase 3: ビューで透過的にアクセス
CREATE VIEW orders_v AS
SELECT * FROM orders_partitioned
UNION ALL
SELECT * FROM orders WHERE order_date < '2024-01-01';

-- Phase 4: アプリを新テーブルに切替
-- Phase 5: 旧テーブルをアーカイブ/削除
```

---

## エッジケース

### エッジケース1: 長時間トランザクションとのデッドロック

```sql
-- 問題: 長時間トランザクションがある状態でALTER TABLEを実行すると
-- ロック待ちのカスケードが発生する

-- セッション1（アプリ）: 長時間トランザクション
BEGIN;
SELECT * FROM users WHERE id = 1;  -- AccessShareLock取得
-- ... 10分間放置 ...

-- セッション2（マイグレーション）: ALTER TABLE
ALTER TABLE users ADD COLUMN phone VARCHAR(20);
-- → AccessExclusiveLock を要求
-- → セッション1のAccessShareLockを待つ

-- セッション3-N（アプリ）: 新しいSELECT
SELECT * FROM users WHERE id = 2;
-- → AccessShareLock を要求
-- → セッション2のAccessExclusiveLockを待つ
-- → 全アプリがブロックされる！

-- 対策: lock_timeout を設定
SET lock_timeout = '5s';
ALTER TABLE users ADD COLUMN phone VARCHAR(20);
-- 5秒以内にロック取得できなければエラー → リトライ
RESET lock_timeout;
```

### エッジケース2: マイグレーションの途中失敗

```sql
-- トランザクション内で複数操作を実行する場合
-- 途中で失敗すると全体がロールバックされる

BEGIN;
ALTER TABLE users ADD COLUMN phone VARCHAR(20);
ALTER TABLE users ADD COLUMN fax VARCHAR(20);
CREATE INDEX CONCURRENTLY idx_users_phone ON users(phone);
-- → エラー: CREATE INDEX CONCURRENTLYはトランザクション内で使用不可
ROLLBACK;

-- 対策: CONCURRENTLYはトランザクション外で実行
-- migration_part1.sql (トランザクション内)
BEGIN;
ALTER TABLE users ADD COLUMN phone VARCHAR(20);
ALTER TABLE users ADD COLUMN fax VARCHAR(20);
COMMIT;

-- migration_part2.sql (トランザクション外)
CREATE INDEX CONCURRENTLY idx_users_phone ON users(phone);
```

### エッジケース3: レプリケーション遅延

```sql
-- 問題: DDL実行後、レプリカに反映されるまでの遅延
-- レプリカを読み取りに使用している場合、スキーマ不整合が発生

-- 対策1: レプリカ遅延の確認
SELECT
    client_addr,
    pg_wal_lsn_diff(sent_lsn, replay_lsn) AS lag_bytes,
    replay_lag
FROM pg_stat_replication;

-- 対策2: マイグレーション後にレプリカの同期を待つ
-- アプリケーション側:
-- 1. マイグレーション実行
-- 2. レプリカ遅延が0になるまで待機
-- 3. アプリデプロイ
```

---

## セキュリティに関する注意事項

### 1. マイグレーション実行権限の管理

```sql
-- 専用のマイグレーションユーザーを作成
CREATE ROLE migration_user WITH LOGIN PASSWORD 'secure_password';

-- 必要最小限の権限を付与
GRANT CONNECT ON DATABASE mydb TO migration_user;
GRANT CREATE ON SCHEMA public TO migration_user;
GRANT ALL ON ALL TABLES IN SCHEMA public TO migration_user;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO migration_user;

-- DDL権限（PostgreSQL）
ALTER ROLE migration_user CREATEDB;  -- 必要な場合のみ

-- アプリケーションユーザーとは分離
-- アプリユーザーにはDDL権限を付与しない
CREATE ROLE app_user WITH LOGIN PASSWORD 'app_password';
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;
```

### 2. マイグレーションファイルの監査

```sql
-- マイグレーション履歴テーブルの設計
CREATE TABLE migration_audit (
    id           SERIAL PRIMARY KEY,
    version      VARCHAR(50) NOT NULL,
    description  VARCHAR(255),
    applied_by   VARCHAR(100) DEFAULT current_user,
    applied_at   TIMESTAMP DEFAULT NOW(),
    execution_ms INTEGER,
    checksum     VARCHAR(64),  -- ファイルのSHA-256
    rollback_sql TEXT          -- ロールバック用SQLを保存
);
```

---

## アンチパターン

### 1. マイグレーションとアプリデプロイの同時実行

**問題**: 新しいカラムを参照するアプリをデプロイすると同時にマイグレーションを実行すると、マイグレーション完了前のリクエストがエラーになる。

**対策**: マイグレーションは常にアプリデプロイの前に実行する。Expand-Contract パターンで後方互換性を維持し、「マイグレーション → デプロイ → クリーンアップ」の3段階で進める。

### 2. 手動でのスキーマ変更

**問題**: DBA が直接 ALTER TABLE を実行すると、マイグレーション履歴との不整合が発生し、以降の自動マイグレーションが失敗する。

**対策**: すべてのスキーマ変更はマイグレーションファイルを通じて行う。緊急時の手動変更もマイグレーションファイルとして記録し、履歴を正す。

### 3. ロールバックスクリプトなしのマイグレーション

**問題**: DOWNマイグレーションがないと、問題発生時にロールバックできない。バックアップ復元が唯一の手段になる。

**対策**: すべてのマイグレーションにDOWNスクリプトを用意する。不可逆な変更（DROP TABLE等）の場合は、DOWNスクリプトにRECREATEを記述するか、明示的に「不可逆」とコメントする。

### 4. 大量のマイグレーションを一度に適用

**問題**: 50個のマイグレーションを一度に本番に適用すると、途中で失敗した場合の切り分けが困難。

**対策**: 大規模変更は複数のリリースに分割し、各リリースで少数のマイグレーションを適用する。依存関係のあるマイグレーションはグループ化して管理する。

---

## 演習問題

### 演習1（基礎）: マイグレーションファイルの作成

以下のスキーマ変更に対する UP/DOWN マイグレーションを作成せよ。

1. `products` テーブルに `description TEXT` カラムを追加
2. `orders` テーブルに `shipped_at TIMESTAMP` カラムを追加し、デフォルト値を NULL とする
3. `users` テーブルの `email` カラムにユニーク制約を追加

<details>
<summary>解答例</summary>

```sql
-- 1. products に description を追加
-- UP:
ALTER TABLE products ADD COLUMN description TEXT;
-- DOWN:
ALTER TABLE products DROP COLUMN description;

-- 2. orders に shipped_at を追加
-- UP:
ALTER TABLE orders ADD COLUMN shipped_at TIMESTAMP;
-- DOWN:
ALTER TABLE orders DROP COLUMN shipped_at;

-- 3. users.email にユニーク制約（安全版）
-- UP:
CREATE UNIQUE INDEX CONCURRENTLY idx_users_email_unique ON users(email);
ALTER TABLE users ADD CONSTRAINT uq_users_email UNIQUE USING INDEX idx_users_email_unique;
-- DOWN:
ALTER TABLE users DROP CONSTRAINT uq_users_email;
```

</details>

### 演習2（応用）: ゼロダウンタイムマイグレーション設計

以下のシナリオに対するゼロダウンタイムマイグレーション計画を設計せよ。

- `orders` テーブルの `status` カラムを VARCHAR(20) から ENUM 型に変更する
- 現在のstatus値: 'pending', 'paid', 'shipped', 'delivered'
- 1000万件のレコードが存在する
- ダウンタイムは許容しない

<details>
<summary>解答例</summary>

```sql
-- Phase 1: Expand（新カラム追加）
CREATE TYPE order_status AS ENUM ('pending', 'paid', 'shipped', 'delivered');
ALTER TABLE orders ADD COLUMN status_v2 order_status;

-- トリガーで同期
CREATE OR REPLACE FUNCTION sync_order_status() RETURNS TRIGGER AS $$
BEGIN
  IF NEW.status_v2 IS NULL AND NEW.status IS NOT NULL THEN
    NEW.status_v2 := NEW.status::order_status;
  ELSIF NEW.status IS NULL AND NEW.status_v2 IS NOT NULL THEN
    NEW.status := NEW.status_v2::text;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER trg_sync_status BEFORE INSERT OR UPDATE ON orders
FOR EACH ROW EXECUTE FUNCTION sync_order_status();

-- Phase 2: バックフィル（バッチ更新）
-- バッチで既存データを移行（上記バッチ処理パターン参照）

-- Phase 3: アプリを status_v2 に切替
-- Phase 4: Contract（旧カラム削除）
DROP TRIGGER trg_sync_status ON orders;
DROP FUNCTION sync_order_status();
ALTER TABLE orders DROP COLUMN status;
ALTER TABLE orders RENAME COLUMN status_v2 TO status;
```

</details>

### 演習3（発展）: マイグレーション障害対応

以下の障害シナリオに対する対応手順を記述せよ。

- 本番環境で `CREATE INDEX CONCURRENTLY` が途中で失敗し、INVALID インデックスが残った
- 同時に、バックフィルジョブが実行中で50%完了の状態
- アプリケーションは正常に動作中

<details>
<summary>解答例</summary>

```sql
-- 1. INVALIDインデックスの確認
SELECT indexrelid::regclass, indisvalid
FROM pg_index WHERE NOT indisvalid;

-- 2. INVALIDインデックスの削除（CONCURRENTLYで安全に）
DROP INDEX CONCURRENTLY idx_failing_index;

-- 3. バックフィルジョブの状態確認
-- 進捗を確認（上記モニタリングクエリ使用）
-- ジョブが正常に実行中ならそのまま続行

-- 4. インデックスの再作成（バックフィル完了後）
CREATE INDEX CONCURRENTLY idx_new_index ON table(column);

-- 5. 結果の検証
SELECT indexrelid::regclass, indisvalid
FROM pg_index WHERE indexrelid = 'idx_new_index'::regclass;
```

</details>

---

## FAQ

### Q1: マイグレーションツールはどれを選ぶべきですか？

**A**: プロジェクトの技術スタックに合わせて選択します:
- **Flyway/Liquibase**: Java エコシステム。エンタープライズ向け
- **golang-migrate**: Go プロジェクト。シンプルで汎用
- **Prisma Migrate**: TypeScript/Node.js。ORM 統合
- **Alembic**: Python (SQLAlchemy)。柔軟なスクリプト対応
- **Atlas**: 宣言的スキーマ管理。最新のアプローチ
- **Knex.js**: JavaScript。Express/Fastify プロジェクトに

### Q2: テーブルの型変更（VARCHAR -> TEXT等）を安全に行うには？

**A**: 直接の `ALTER TYPE` はテーブルロックが発生します。安全な手順:
1. 新しい型のカラムを追加
2. トリガーで双方向同期
3. バックフィルで既存データをコピー
4. アプリを新カラムに切替
5. 旧カラムを削除

ただし、PostgreSQLでは VARCHAR(N) → VARCHAR(M) (M > N) の拡大はメタデータ変更のみで瞬時に完了します。TEXT への変更も同様です。

### Q3: マイグレーションの適用時間をどう見積もりますか？

**A**: ステージング環境で本番同等のデータ量を使って計測します。目安として:
- `ADD COLUMN` (デフォルトなし): ミリ秒
- `ADD COLUMN DEFAULT` (PG11+): ミリ秒
- `CREATE INDEX CONCURRENTLY`: テーブルサイズの約2-3倍の時間（1億行で数分〜十数分）
- バックフィル UPDATE: 行数 / バッチサイズ * (実行時間 + スリープ)
- `ALTER TABLE ALTER TYPE`: テーブルの全行を書き換えるため、テーブルサイズに比例

### Q4: マイグレーションの命名規則は？

**A**: 以下の形式が一般的です:
- **タイムスタンプベース**: `20260211143025_add_phone_to_users.sql`（推奨、競合しにくい）
- **連番ベース**: `000042_add_phone_to_users.sql`（シンプルだがブランチ間で競合する）
- **セマンティック**: `V2.1.0__add_phone_to_users.sql`（Flyway形式）

命名のベストプラクティス:
- 動詞で始める: `add_`, `create_`, `remove_`, `rename_`, `modify_`
- テーブル名を含める
- 目的が分かる名前にする

### Q5: データベースの中身を変更するマイグレーション（DML）は含めるべきか？

**A**: 2つの考え方があります:

1. **DMLを含める**: マスターデータの投入やデータ変換など、スキーマ変更と密接に関連するDMLはマイグレーションに含める
2. **DMLは別管理**: シードデータは別スクリプトで管理し、マイグレーションはDDLのみにする

推奨: スキーマ変更に伴うデータ変換はマイグレーションに含め、初期データ投入はシードスクリプトとして分離する。

---

## トラブルシューティング

### 問題1: マイグレーションが「dirty」状態になった

```bash
# golang-migrate: dirty状態の解消
# 1. 現在の状態を確認
migrate -path ./migrations -database "$DB_URL" version
# → 出力: 5 (dirty)

# 2. dirty フラグをクリア（手動で修正済みの場合）
migrate -path ./migrations -database "$DB_URL" force 5

# 3. 修正後に再適用
migrate -path ./migrations -database "$DB_URL" up
```

### 問題2: lock_timeoutで失敗する

```sql
-- 原因: 長時間トランザクションがロックを保持
-- 1. ロック保持しているクエリを確認
SELECT
    pid,
    usename,
    state,
    query_start,
    NOW() - query_start AS duration,
    LEFT(query, 100) AS query
FROM pg_stat_activity
WHERE state != 'idle'
  AND query_start < NOW() - INTERVAL '1 minute'
ORDER BY duration DESC;

-- 2. 長時間トランザクションが完了するのを待つか、キャンセル
SELECT pg_cancel_backend(pid);  -- クエリのキャンセル
-- SELECT pg_terminate_backend(pid);  -- セッションの強制終了（最終手段）
```

### 問題3: ステージングと本番のスキーマが一致しない

```bash
# スキーマの差分を確認
pg_dump -s $STAGING_URL > staging_schema.sql
pg_dump -s $PRODUCTION_URL > production_schema.sql
diff staging_schema.sql production_schema.sql

# もしくは migra を使用（差分SQLの自動生成）
pip install migra
migra $STAGING_URL $PRODUCTION_URL
# → ALTER TABLE ... が出力される
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| バージョン管理 | すべてのスキーマ変更をマイグレーションファイルで管理 |
| Expand-Contract | 追加 → 移行 → 削除の3段階で後方互換性を維持 |
| CONCURRENTLY | インデックス作成はロックなしの CONCURRENTLY を使用 |
| NOT VALID | 制約追加は NOT VALID + VALIDATE の2段階で |
| lock_timeout | DDL実行前にロック取得のタイムアウトを設定 |
| バッチ更新 | 大規模データ更新はバッチ処理で負荷分散 |
| ロールバック | すべてのマイグレーションに DOWN スクリプトを用意 |
| CI/CD | lint ツールで危険な操作を自動検出 |
| 環境分離 | 開発/ステージング/本番で同じスクリプトを使用 |
| 監査 | マイグレーション実行の記録と権限管理 |

## 次に読むべきガイド

- [インデックス](../01-advanced/03-indexing.md) — CONCURRENTLY の詳細と最適化
- [NoSQL 比較](../03-practical/04-nosql-comparison.md) — スキーマレス DB のマイグレーション
- [トランザクション](../01-advanced/02-transactions.md) — ロックとトランザクション管理

## 参考文献

1. **PostgreSQL 公式**: [ALTER TABLE](https://www.postgresql.org/docs/current/sql-altertable.html) — DDL 操作のロック動作の詳細
2. **PostgreSQL 公式**: [Lock Monitoring](https://wiki.postgresql.org/wiki/Lock_Monitoring) — ロック監視の手法
3. **Braintree Blog**: [Safe Operations for High Volume PostgreSQL](https://medium.com/braintree-product-technology) — ゼロダウンタイム手法
4. **squawk**: [PostgreSQL Migration Linter](https://squawkhq.com/) — マイグレーション安全性チェックツール
5. **Martin Kleppmann**: *Designing Data-Intensive Applications* — データシステム設計の包括的な解説
6. **GitHub Engineering**: [gh-ost: Online Schema Migration](https://github.com/github/gh-ost) — トリガーなしのオンラインDDL
