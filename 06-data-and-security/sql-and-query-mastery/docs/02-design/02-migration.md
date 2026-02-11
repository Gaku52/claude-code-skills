# マイグレーション

> データベーススキーマの変更をバージョン管理し、ゼロダウンタイムで安全にデプロイする手法を実践的に習得する

## この章で学ぶこと

1. **マイグレーションの基本** — バージョン管理、ロールバック戦略、ツール選定
2. **ゼロダウンタイム手法** — Expand-Contract パターン、オンライン DDL、段階的移行
3. **危険な操作の回避** — ロック問題、大規模データ移行、後方互換性の確保

---

## 1. マイグレーションの基本概念

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
```

### コード例 1: マイグレーションツールの比較と使用

```sql
-- Flyway 形式: V{version}__{description}.sql
-- V2__add_phone_to_users.sql
ALTER TABLE users ADD COLUMN phone VARCHAR(20);

-- golang-migrate 形式
-- 000002_add_phone.up.sql
ALTER TABLE users ADD COLUMN phone VARCHAR(20);

-- 000002_add_phone.down.sql
ALTER TABLE users DROP COLUMN phone;
```

```bash
# golang-migrate の使用
migrate -path ./migrations -database "postgres://user:pass@localhost/mydb" up
migrate -path ./migrations -database "postgres://user:pass@localhost/mydb" down 1

# Flyway
flyway -url=jdbc:postgresql://localhost/mydb migrate
flyway -url=jdbc:postgresql://localhost/mydb info
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
}
```

```bash
# マイグレーション生成
npx prisma migrate dev --name add_phone_and_status

# 本番適用
npx prisma migrate deploy

# ステータス確認
npx prisma migrate status
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
  Expand    Migrate   Contract
  [+col]    [data]    [-col]
  |---------|---------|---------|
  v1 + v2   v2        v2 only
```

### コード例 3: カラムリネームのゼロダウンタイム手法

```sql
-- [NG] 直接リネーム --> ダウンタイム発生
ALTER TABLE users RENAME COLUMN name TO full_name;
-- --> 既存アプリが "name" を参照してエラー

-- [OK] Expand-Contract パターン
-- Phase 1: Expand（新カラム追加 + トリガー）
ALTER TABLE users ADD COLUMN full_name VARCHAR(255);

-- 既存データをコピー
UPDATE users SET full_name = name WHERE full_name IS NULL;

-- 双方向同期トリガー
CREATE OR REPLACE FUNCTION sync_user_name() RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP = 'INSERT' OR TG_OP = 'UPDATE' THEN
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

-- Phase 2: アプリを full_name に切替（デプロイ）

-- Phase 3: Contract（旧カラム・トリガー削除）
DROP TRIGGER trg_sync_user_name ON users;
DROP FUNCTION sync_user_name();
ALTER TABLE users DROP COLUMN name;
```

---

## 3. 危険な操作と安全な代替

### 危険な DDL 操作比較表

| 操作 | 危険度 | ロック | 安全な代替 |
|---|---|---|---|
| `ALTER TABLE ADD COLUMN` (デフォルトなし) | 低 | 瞬時 | そのまま使用可 |
| `ALTER TABLE ADD COLUMN DEFAULT x` (PG11+) | 低 | 瞬時 | そのまま使用可 |
| `ALTER TABLE ADD COLUMN DEFAULT x` (PG10以前) | 高 | テーブル全行書換 | 追加後に UPDATE |
| `ALTER TABLE DROP COLUMN` | 中 | 瞬時（論理削除） | Contract フェーズで実施 |
| `ALTER TABLE ALTER TYPE` | 高 | テーブルロック | 新カラム + バックフィル |
| `CREATE INDEX` | 高 | テーブルロック | `CREATE INDEX CONCURRENTLY` |
| `ALTER TABLE ADD CONSTRAINT` | 高 | テーブルロック | `NOT VALID` + 後で `VALIDATE` |
| `ALTER TABLE RENAME COLUMN` | 高 | 瞬時だがアプリ互換性破壊 | Expand-Contract |

### コード例 4: 安全なインデックス追加

```sql
-- [NG] テーブルロックでダウンタイム
CREATE INDEX idx_orders_email ON orders (email);

-- [OK] ロックなし（CONCURRENTLY）
CREATE INDEX CONCURRENTLY idx_orders_email ON orders (email);
-- 注意: トランザクション内では使用不可
-- 注意: 構築時間は約2倍

-- [OK] 安全な NOT NULL 制約の追加
-- Step 1: CHECK 制約を NOT VALID で追加（ロックなし）
ALTER TABLE users
ADD CONSTRAINT chk_users_email_not_null
CHECK (email IS NOT NULL) NOT VALID;

-- Step 2: 既存データの検証（ShareUpdateExclusiveLock のみ）
ALTER TABLE users VALIDATE CONSTRAINT chk_users_email_not_null;

-- Step 3: NOT NULL に昇格
ALTER TABLE users ALTER COLUMN email SET NOT NULL;
ALTER TABLE users DROP CONSTRAINT chk_users_email_not_null;
```

### コード例 5: 大規模データのバックフィル

```sql
-- [NG] 一括 UPDATE --> 長時間ロック + WAL 肥大化
UPDATE orders SET status = 'active' WHERE status IS NULL;
-- 1000万行の場合、数分~数十分のロック

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
      FOR UPDATE SKIP LOCKED
    );

    GET DIAGNOSTICS rows_affected = ROW_COUNT;
    total_updated := total_updated + rows_affected;

    RAISE NOTICE 'Updated: % (total: %)', rows_affected, total_updated;

    EXIT WHEN rows_affected = 0;

    PERFORM pg_sleep(0.1);  -- 負荷調整
    COMMIT;
  END LOOP;
END $$;
```

---

## 4. マイグレーション CI/CD

```
CI/CD パイプラインでのマイグレーション
========================================

1. PR 作成
   |
   v
2. マイグレーション lint
   - SQL 構文チェック
   - 危険な操作の検出
   - ロールバック可能性の確認
   |
   v
3. ステージング適用
   - 本番同等のデータ量でテスト
   - 適用時間の計測
   |
   v
4. レビュー承認
   |
   v
5. 本番適用
   - Blue/Green または Rolling
   - 監視ダッシュボード確認
```

### コード例 6: マイグレーション lint ツール（squawk）

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
```

---

## 5. ロールバック戦略

### コード例 7: 安全なロールバック設計

```sql
-- UP マイグレーション
-- 20260211_003_add_orders_status.up.sql
BEGIN;

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

ALTER TABLE orders DROP COLUMN IF EXISTS status_new;

DELETE FROM schema_migrations WHERE version = '20260211_003';

COMMIT;
```

---

## アンチパターン

### 1. マイグレーションとアプリデプロイの同時実行

**問題**: 新しいカラムを参照するアプリをデプロイすると同時にマイグレーションを実行すると、マイグレーション完了前のリクエストがエラーになる。

**対策**: マイグレーションは常にアプリデプロイの前に実行する。Expand-Contract パターンで後方互換性を維持し、「マイグレーション → デプロイ → クリーンアップ」の3段階で進める。

### 2. 手動でのスキーマ変更

**問題**: DBA が直接 ALTER TABLE を実行すると、マイグレーション履歴との不整合が発生し、以降の自動マイグレーションが失敗する。

**対策**: すべてのスキーマ変更はマイグレーションファイルを通じて行う。緊急時の手動変更もマイグレーションファイルとして記録し、履歴を正す。

---

## FAQ

### Q1: マイグレーションツールはどれを選ぶべきですか？

**A**: プロジェクトの技術スタックに合わせて選択します:
- **Flyway/Liquibase**: Java エコシステム。エンタープライズ向け
- **golang-migrate**: Go プロジェクト。シンプルで汎用
- **Prisma Migrate**: TypeScript/Node.js。ORM 統合
- **Alembic**: Python (SQLAlchemy)。柔軟なスクリプト対応
- **Atlas**: 宣言的スキーマ管理。最新のアプローチ

### Q2: テーブルの型変更（VARCHAR -> TEXT等）を安全に行うには？

**A**: 直接の `ALTER TYPE` はテーブルロックが発生します。安全な手順:
1. 新しい型のカラムを追加
2. トリガーで双方向同期
3. バックフィルで既存データをコピー
4. アプリを新カラムに切替
5. 旧カラムを削除

### Q3: マイグレーションの適用時間をどう見積もりますか？

**A**: ステージング環境で本番同等のデータ量を使って計測します。目安として:
- `ADD COLUMN` (デフォルトなし): ミリ秒
- `ADD COLUMN DEFAULT` (PG11+): ミリ秒
- `CREATE INDEX CONCURRENTLY`: テーブルサイズの約2-3倍の時間（1億行で数分〜十数分）
- バックフィル UPDATE: 行数 / バッチサイズ * (実行時間 + スリープ)

---

## まとめ

| 項目 | 要点 |
|---|---|
| バージョン管理 | すべてのスキーマ変更をマイグレーションファイルで管理 |
| Expand-Contract | 追加 → 移行 → 削除の3段階で後方互換性を維持 |
| CONCURRENTLY | インデックス作成はロックなしの CONCURRENTLY を使用 |
| NOT VALID | 制約追加は NOT VALID + VALIDATE の2段階で |
| バッチ更新 | 大規模データ更新はバッチ処理で負荷分散 |
| ロールバック | すべてのマイグレーションに DOWN スクリプトを用意 |
| CI/CD | lint ツールで危険な操作を自動検出 |

## 次に読むべきガイド

- [インデックス](../01-advanced/03-indexing.md) — CONCURRENTLY の詳細と最適化
- [NoSQL 比較](../03-practical/04-nosql-comparison.md) — スキーマレス DB のマイグレーション

## 参考文献

1. **PostgreSQL 公式**: [ALTER TABLE](https://www.postgresql.org/docs/current/sql-altertable.html) — DDL 操作のロック動作の詳細
2. **Braintree Blog**: [Safe Operations for High Volume PostgreSQL](https://medium.com/braintree-product-technology) — ゼロダウンタイム手法
3. **squawk**: [PostgreSQL Migration Linter](https://squawkhq.com/) — マイグレーション安全性チェックツール
