# トランザクション — ACID・分離レベル・デッドロック

> トランザクションはデータベース操作の論理的な作業単位であり、ACID特性によってデータの一貫性と信頼性を保証する、データベースシステムの根幹を成す仕組みである。

## この章で学ぶこと

1. ACID特性の各要素を具体例で理解する
2. 4つの分離レベルと各レベルで発生するアノマリー
3. デッドロックの原因と回避・検出戦略

---

## 1. ACID特性

```
┌────────────────── ACID特性 ──────────────────┐
│                                               │
│  A - Atomicity（原子性）                      │
│    「全部成功」か「全部失敗」のいずれか         │
│    途中で失敗したら全ての変更を巻き戻す        │
│                                               │
│  C - Consistency（一貫性）                    │
│    トランザクション前後でデータの制約が        │
│    常に満たされる                              │
│                                               │
│  I - Isolation（分離性）                      │
│    並行するトランザクションは互いに            │
│    干渉しない（かのように見える）              │
│                                               │
│  D - Durability（耐久性）                     │
│    COMMITされたデータは障害が発生しても        │
│    失われない（WAL、ディスクに永続化）         │
└───────────────────────────────────────────────┘
```

### コード例1: トランザクションの基本

```sql
-- 銀行送金: Atomicityの典型例
BEGIN;

-- 送金元の残高を減らす
UPDATE accounts SET balance = balance - 10000
WHERE account_id = 'A001';

-- 送金先の残高を増やす
UPDATE accounts SET balance = balance + 10000
WHERE account_id = 'B002';

-- 送金元の残高が0未満でないか確認
DO $$
BEGIN
    IF (SELECT balance FROM accounts WHERE account_id = 'A001') < 0 THEN
        RAISE EXCEPTION '残高不足';
    END IF;
END $$;

-- 全て成功したらコミット
COMMIT;

-- エラーが発生した場合は自動的にROLLBACKされる
-- または明示的に ROLLBACK; を実行
```

### コード例2: SAVEPOINTによる部分ロールバック

```sql
BEGIN;

INSERT INTO orders (customer_id, total) VALUES (1, 5000);
SAVEPOINT order_created;

INSERT INTO order_items (order_id, product_id, quantity) VALUES (100, 1, 2);
SAVEPOINT items_added;

-- ポイント付与が失敗しても注文自体はコミットしたい
BEGIN
    INSERT INTO points (customer_id, amount) VALUES (1, 50);
EXCEPTION WHEN OTHERS THEN
    ROLLBACK TO SAVEPOINT items_added;
    -- ポイント付与のみ取り消し、注文は維持
END;

COMMIT;
```

---

## 2. 分離レベル

### コード例3: 分離レベルの設定

```sql
-- トランザクション単位で設定
BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;
-- ... 操作 ...
COMMIT;

-- セッション単位で設定
SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL REPEATABLE READ;

-- 現在の分離レベルを確認（PostgreSQL）
SHOW transaction_isolation;
```

### 分離レベルとアノマリーの関係

```
┌─────── 分離レベルとアノマリーの関係 ──────────────┐
│                                                    │
│  アノマリー（異常現象）:                            │
│                                                    │
│  ダーティリード: 未COMMITのデータが見える           │
│  ┌─────┐                ┌─────┐                   │
│  │ Tx1 │ UPDATE balance │ Tx2 │ SELECT balance    │
│  │     │ = 500          │     │ → 500 (未COMMIT!) │
│  │     │ ROLLBACK       │     │ ← 不正な値を使用  │
│  └─────┘                └─────┘                   │
│                                                    │
│  ノンリピータブルリード: 同じSELECTが異なる結果     │
│  ┌─────┐                ┌─────┐                   │
│  │ Tx1 │ SELECT → 1000  │ Tx2 │                   │
│  │     │                │     │ UPDATE → 500      │
│  │     │                │     │ COMMIT             │
│  │     │ SELECT → 500   │     │ ← 値が変わった!   │
│  └─────┘                └─────┘                   │
│                                                    │
│  ファントムリード: 行数が変わる                     │
│  ┌─────┐                ┌─────┐                   │
│  │ Tx1 │ COUNT → 10行   │ Tx2 │                   │
│  │     │                │     │ INSERT (1行追加)   │
│  │     │                │     │ COMMIT             │
│  │     │ COUNT → 11行   │     │ ← 行数が変わった! │
│  └─────┘                └─────┘                   │
└────────────────────────────────────────────────────┘
```

### コード例4: 各分離レベルの動作例

```sql
-- READ UNCOMMITTED（PostgreSQLでは実質READ COMMITTED）
-- ダーティリードを許可。実用ではほぼ使わない
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;

-- READ COMMITTED（PostgreSQLのデフォルト）
-- COMMITされたデータのみ見える
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
BEGIN;
SELECT balance FROM accounts WHERE id = 1;  -- 1000
-- 他のTxがUPDATE→COMMITした後...
SELECT balance FROM accounts WHERE id = 1;  -- 1200（変わる）
COMMIT;

-- REPEATABLE READ
-- トランザクション開始時のスナップショットを使用
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
BEGIN;
SELECT balance FROM accounts WHERE id = 1;  -- 1000
-- 他のTxがUPDATE→COMMITしても...
SELECT balance FROM accounts WHERE id = 1;  -- 1000（変わらない）
COMMIT;

-- SERIALIZABLE（最も厳格）
-- 直列実行と同等の結果を保証
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
BEGIN;
SELECT SUM(balance) FROM accounts;
-- 他のTxと競合する場合、
-- ERROR: could not serialize access due to concurrent update
COMMIT;
```

---

## 3. デッドロック

### コード例5: デッドロックの発生と対策

```sql
-- デッドロックの典型的なシナリオ
-- Tx1: accounts A → B の順にロック
-- Tx2: accounts B → A の順にロック

-- Tx1:
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE id = 'A';  -- Aをロック
-- ここでTx2がBをロック
UPDATE accounts SET balance = balance + 100 WHERE id = 'B';  -- Bを待機...

-- Tx2:
BEGIN;
UPDATE accounts SET balance = balance - 200 WHERE id = 'B';  -- Bをロック
-- ここでTx1がAをロック済み
UPDATE accounts SET balance = balance + 200 WHERE id = 'A';  -- Aを待機...
-- → デッドロック！

-- 対策1: 常に同じ順序でロックする
BEGIN;
-- ID順にソートしてからロック
UPDATE accounts SET balance = balance - 100 WHERE id = 'A';  -- 常にA→Bの順
UPDATE accounts SET balance = balance + 100 WHERE id = 'B';
COMMIT;

-- 対策2: SELECT FOR UPDATE で事前にロック取得
BEGIN;
SELECT * FROM accounts WHERE id IN ('A', 'B')
ORDER BY id FOR UPDATE;  -- ソート順でロック取得
UPDATE accounts SET balance = balance - 100 WHERE id = 'A';
UPDATE accounts SET balance = balance + 100 WHERE id = 'B';
COMMIT;

-- 対策3: ロックタイムアウトの設定
SET lock_timeout = '5s';  -- 5秒でロック待ちを諦める
```

### コード例6: 楽観的ロック vs 悲観的ロック

```sql
-- 悲観的ロック（Pessimistic Locking）: 先にロックを取得
BEGIN;
SELECT * FROM products WHERE id = 42 FOR UPDATE;
-- → 他のTxは待機
UPDATE products SET stock = stock - 1 WHERE id = 42;
COMMIT;

-- FOR UPDATE NOWAIT: ロック取得不可なら即エラー
SELECT * FROM products WHERE id = 42 FOR UPDATE NOWAIT;

-- FOR UPDATE SKIP LOCKED: ロック中の行をスキップ（キュー処理向け）
SELECT * FROM tasks WHERE status = 'pending'
ORDER BY created_at
LIMIT 1
FOR UPDATE SKIP LOCKED;

-- 楽観的ロック（Optimistic Locking）: バージョンで競合検出
-- アプリケーション側で実装
UPDATE products
SET stock = stock - 1,
    version = version + 1
WHERE id = 42
  AND version = 5;  -- 読み取り時のバージョンと一致するか確認
-- 影響行数が0なら競合発生 → リトライ
```

---

## 分離レベル比較表

| 分離レベル | ダーティリード | ノンリピータブルリード | ファントムリード | パフォーマンス |
|-----------|:---:|:---:|:---:|:---:|
| READ UNCOMMITTED | 発生 | 発生 | 発生 | 最速 |
| READ COMMITTED | 防止 | 発生 | 発生 | 速い |
| REPEATABLE READ | 防止 | 防止 | 発生* | 普通 |
| SERIALIZABLE | 防止 | 防止 | 防止 | 最遅 |

*PostgreSQLのREPEATABLE READはスナップショット分離を使用し、ファントムリードも防止する。

## ロック方式比較表

| 方式 | ロックタイミング | 競合検出 | スループット | 適する場面 |
|------|---------------|---------|------------|-----------|
| 悲観的ロック (FOR UPDATE) | データ読取時 | 事前防止 | 低い | 競合が頻繁な場合 |
| 楽観的ロック (version) | 更新時 | 事後検出 | 高い | 競合が稀な場合 |
| SKIP LOCKED | 読取時 | スキップ | 高い | キュー/ジョブ処理 |
| アドバイザリロック | 任意 | アプリ制御 | 柔軟 | 特殊な排他制御 |

---

## アンチパターン

### アンチパターン1: 長時間トランザクション

```sql
-- NG: トランザクション中にユーザー入力を待つ
BEGIN;
SELECT * FROM products WHERE id = 42 FOR UPDATE;
-- ... ユーザーが画面で考え中（数分間ロック保持）...
UPDATE products SET price = 1000 WHERE id = 42;
COMMIT;

-- 問題点:
-- 1. 長時間ロックで他のトランザクションがブロック
-- 2. デッドロックのリスク増大
-- 3. MVCC環境でVACUUMが遅延

-- OK: トランザクションを短く保つ
-- 読取は通常のSELECT、更新時のみ短いトランザクション
SELECT * FROM products WHERE id = 42;  -- ロックなし
-- ユーザー操作完了後
BEGIN;
UPDATE products SET price = 1000 WHERE id = 42 AND version = 5;  -- 楽観的ロック
COMMIT;
```

### アンチパターン2: 不必要に高い分離レベル

```sql
-- NG: 全てSERIALIZABLEにする
SET default_transaction_isolation = 'serializable';

-- 問題点:
-- 1. シリアライゼーション失敗でリトライが頻発
-- 2. パフォーマンス低下
-- 3. 多くの場合READ COMMITTEDで十分

-- OK: 用途に応じて分離レベルを選択
-- レポート/分析: READ COMMITTED（デフォルト）
-- 残高計算: REPEATABLE READ
-- 厳密な一貫性が必要: SERIALIZABLE
```

---

## FAQ

### Q1: AUTO COMMITとは何か？

AUTO COMMITが有効（デフォルト）の場合、各SQL文が自動的に独立したトランザクションとして実行される。明示的なBEGIN/COMMITを使わない限り、各文の完了時に自動的にCOMMITされる。バッチ処理では明示的にBEGINで囲むことで、中間状態のCOMMITを防ぐ。

### Q2: PostgreSQLのMVCCとは？

MVCC（Multi-Version Concurrency Control）は、各行の複数バージョンを保持することで、読み取りと書き込みが互いにブロックしない仕組み。SELECTはロックを取得せず、トランザクション開始時点のスナップショットを読む。これによりREADERとWRITERが共存できる。

### Q3: トランザクションがROLLBACKされるのはどんな場合か？

(1) 明示的な`ROLLBACK`文の実行、(2) 制約違反やデッドロックなどのエラー発生、(3) クライアント接続の切断、(4) `statement_timeout`の超過。PostgreSQLではエラー発生後、トランザクション内の以降の文は全て拒否される（SAVEPOINT使用時を除く）。

---

## まとめ

| 項目 | 要点 |
|------|------|
| ACID | 原子性・一貫性・分離性・耐久性の4特性 |
| デフォルト分離レベル | READ COMMITTED（PostgreSQL, Oracle） |
| MVCC | 読み書きの非ブロック化。PostgreSQLの基盤 |
| デッドロック対策 | ロック順序の統一、タイムアウト設定 |
| 悲観的ロック | FOR UPDATE。競合が多い場面で有効 |
| 楽観的ロック | versionカラム。競合が少ない場面で有効 |
| ベストプラクティス | トランザクションを短く、分離レベルは最小限 |

---

## 次に読むべきガイド

- [03-indexing.md](./03-indexing.md) — ロックとインデックスの関係
- [04-query-optimization.md](./04-query-optimization.md) — トランザクションと実行計画
- [02-performance-tuning.md](../03-practical/02-performance-tuning.md) — 接続プールとトランザクション管理

---

## 参考文献

1. PostgreSQL Documentation — "Transaction Isolation" https://www.postgresql.org/docs/current/transaction-iso.html
2. Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly Media. Chapter 7: Transactions.
3. Berenson, H. et al. (1995). "A Critique of ANSI SQL Isolation Levels". *SIGMOD Record*, 24(2).
