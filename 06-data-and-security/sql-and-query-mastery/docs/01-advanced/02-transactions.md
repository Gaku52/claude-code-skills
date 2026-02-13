# トランザクション — ACID・分離レベル・デッドロック・MVCC

> トランザクションはデータベース操作の論理的な作業単位であり、ACID特性によってデータの一貫性と信頼性を保証する、データベースシステムの根幹を成す仕組みである。本章では、ACID特性の各要素を内部実装レベルで理解し、分離レベルの選択基準、デッドロックの回避・検出戦略、そしてMVCC（多版型同時実行制御）の仕組みまでを体系的に習得する。

---

## この章で学ぶこと

1. **ACID特性の各要素を内部実装レベルで理解する** — WAL（Write-Ahead Logging）による耐久性保証、チェックポイントの仕組み
2. **4つの分離レベルと各レベルで発生するアノマリーを実例で把握する** — ダーティリード、ノンリピータブルリード、ファントムリード、シリアライゼーション異常
3. **デッドロックの原因と回避・検出戦略を実装できる** — ロック順序の統一、タイムアウト設定、楽観的ロックと悲観的ロックの使い分け
4. **MVCCの内部構造を理解し、VACUUM戦略を適切に設計できる** — xmin/xmax、スナップショット、可視性マップ

---

## 前提知識

| トピック | 内容 | 参照先 |
|---------|------|--------|
| SQL基礎 | SELECT/INSERT/UPDATE/DELETE の基本構文 | [00-basics/](../00-basics/) |
| テーブル設計 | PRIMARY KEY、FOREIGN KEY、制約 | [01-schema-design.md](../02-design/01-schema-design.md) |
| データベース接続 | psqlまたはGUIツールでの接続方法 | [00-basics/](../00-basics/) |

---

## 1. ACID特性

### なぜACIDが必要か

データベースがACID特性を持たない場合、以下のような深刻な問題が発生する。

- **銀行送金**: Aさんの口座から10万円を引き出した後、Bさんの口座への入金が失敗した場合、10万円が消失する（原子性の欠如）
- **在庫管理**: 2つの注文が同時に最後の1個を購入し、在庫が-1になる（分離性の欠如）
- **会計処理**: 借方と貸方の合計が一致しない仕訳が作成される（一貫性の欠如）
- **障害復旧**: コミット直後にサーバがクラッシュし、データが消える（耐久性の欠如）

```
┌────────────────── ACID特性 ──────────────────┐
│                                               │
│  A - Atomicity（原子性）                      │
│    「全部成功」か「全部失敗」のいずれか         │
│    途中で失敗したら全ての変更を巻き戻す        │
│    実装: UNDO ログ / WAL のロールバック機能    │
│                                               │
│  C - Consistency（一貫性）                    │
│    トランザクション前後でデータの制約が        │
│    常に満たされる                              │
│    実装: CHECK制約、FK制約、トリガー           │
│                                               │
│  I - Isolation（分離性）                      │
│    並行するトランザクションは互いに            │
│    干渉しない（かのように見える）              │
│    実装: MVCC / ロックプロトコル               │
│                                               │
│  D - Durability（耐久性）                     │
│    COMMITされたデータは障害が発生しても        │
│    失われない                                  │
│    実装: WAL（Write-Ahead Logging）           │
│         + fsync / チェックポイント             │
└───────────────────────────────────────────────┘
```

### ACID特性の内部実装

```
┌─────────────── WAL（Write-Ahead Logging）の仕組み ──────────────┐
│                                                                  │
│  ① クライアント: BEGIN; UPDATE ...; COMMIT;                     │
│                                                                  │
│  ② WALバッファ: 変更内容をまずWALログに書き込む                 │
│     ┌─────────────────────────────────────────────┐             │
│     │ LSN=100: UPDATE accounts SET balance=900    │             │
│     │ LSN=101: UPDATE accounts SET balance=1100   │             │
│     │ LSN=102: COMMIT                              │             │
│     └─────────────────────────────────────────────┘             │
│                                                                  │
│  ③ WALディスク書き込み: COMMIT時にfsyncで永続化                 │
│     → この時点で耐久性（Durability）が保証される                │
│                                                                  │
│  ④ 共有バッファ: テーブルデータは遅延書き込み（dirty page）     │
│                                                                  │
│  ⑤ チェックポイント: 定期的にdirty pageをディスクに書き出す     │
│     → チェックポイント以前のWALは不要になる                      │
│                                                                  │
│  ⑥ クラッシュリカバリ: 最後のチェックポイントから                │
│     WALを再生（リプレイ）して一貫状態に復元                      │
│                                                                  │
│  [Client] → [WAL Buffer] → [WAL Disk] ← 耐久性保証点           │
│                ↓                                                 │
│          [Shared Buffer] → [Data Disk] ← チェックポイントで書出  │
└──────────────────────────────────────────────────────────────────┘
```

### コード例1: トランザクションの基本

```sql
-- テスト用テーブルの準備
CREATE TABLE accounts (
    account_id VARCHAR(10) PRIMARY KEY,
    owner_name VARCHAR(100) NOT NULL,
    balance    DECIMAL(12, 2) NOT NULL DEFAULT 0
                CHECK (balance >= 0)
);

INSERT INTO accounts VALUES ('A001', '田中太郎', 100000);
INSERT INTO accounts VALUES ('B002', '鈴木花子', 50000);

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

-- 確認: 合計残高は変わらない（一貫性）
SELECT SUM(balance) FROM accounts;
-- 結果: 150000（送金前と同じ）
```

**なぜBEGIN/COMMITで囲むのか**: BEGIN/COMMITで囲まない場合、各UPDATE文が独立したトランザクションとして実行される。1つ目のUPDATEが成功した後、2つ目が失敗すると、10,000円が消失する。

### コード例2: SAVEPOINTによる部分ロールバック

```sql
-- SAVEPOINTを使った段階的なトランザクション制御
BEGIN;

-- 注文の作成（必須処理）
INSERT INTO orders (id, customer_id, total)
VALUES (100, 1, 5000);
SAVEPOINT order_created;

-- 注文明細の追加（必須処理）
INSERT INTO order_items (order_id, product_id, quantity, unit_price)
VALUES (100, 1, 2, 2500);
SAVEPOINT items_added;

-- ポイント付与（オプション処理 — 失敗しても注文は成立させたい）
DO $$
BEGIN
    INSERT INTO points (customer_id, amount, reason)
    VALUES (1, 50, '注文ポイント');
EXCEPTION WHEN OTHERS THEN
    -- ポイントテーブルが存在しない等のエラーをキャッチ
    RAISE NOTICE 'ポイント付与失敗: %', SQLERRM;
    ROLLBACK TO SAVEPOINT items_added;
    -- ポイント付与のみ取り消し、注文は維持
END $$;

-- 在庫の減算（必須処理）
UPDATE products SET stock = stock - 2 WHERE id = 1;

COMMIT;

-- SAVEPOINTの階層構造:
-- BEGIN
--   └── SAVEPOINT order_created
--         └── SAVEPOINT items_added
--               └── ポイント付与（失敗時はここまでロールバック）
--         └── 在庫減算
-- COMMIT
```

**SAVEPOINTの使いどころ**: 一部の処理が失敗しても全体をロールバックしたくない場合に使用する。PostgreSQLではトランザクション内でエラーが発生すると、以降のSQL文は全て拒否されるため、SAVEPOINTを使わないとCOMMITもできなくなる。

---

## 2. 分離レベル

### なぜ分離レベルが必要か

完全な分離性（SERIALIZABLE）を常に保証すると、並行性能が大幅に低下する。現実のアプリケーションでは、用途に応じて分離性の程度を選択するトレードオフが必要になる。

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
-- 結果: "read committed"（デフォルト）

-- MySQL でのデフォルト分離レベル確認
-- SELECT @@transaction_isolation;
-- 結果: "REPEATABLE-READ"（MySQLのデフォルト）

-- サーバ全体のデフォルトを変更（postgresql.conf）
-- default_transaction_isolation = 'read committed'
```

### 分離レベルとアノマリーの関係

```
┌─────── 分離レベルとアノマリーの関係 ──────────────┐
│                                                    │
│  アノマリー（異常現象）:                            │
│                                                    │
│  1. ダーティリード: 未COMMITのデータが見える        │
│  ┌─────┐                ┌─────┐                   │
│  │ Tx1 │ UPDATE balance │ Tx2 │ SELECT balance    │
│  │     │ = 500          │     │ → 500 (未COMMIT!) │
│  │     │ ROLLBACK       │     │ ← 不正な値を使用  │
│  └─────┘                └─────┘                   │
│                                                    │
│  2. ノンリピータブルリード: 同じSELECTが異なる結果  │
│  ┌─────┐                ┌─────┐                   │
│  │ Tx1 │ SELECT → 1000  │ Tx2 │                   │
│  │     │                │     │ UPDATE → 500      │
│  │     │                │     │ COMMIT             │
│  │     │ SELECT → 500   │     │ ← 値が変わった!   │
│  └─────┘                └─────┘                   │
│                                                    │
│  3. ファントムリード: 行数が変わる                  │
│  ┌─────┐                ┌─────┐                   │
│  │ Tx1 │ COUNT → 10行   │ Tx2 │                   │
│  │     │                │     │ INSERT (1行追加)   │
│  │     │                │     │ COMMIT             │
│  │     │ COUNT → 11行   │     │ ← 行数が変わった! │
│  └─────┘                └─────┘                   │
│                                                    │
│  4. シリアライゼーション異常: 直列実行では          │
│     起こりえない結果が並行実行で発生                │
│  ┌─────┐                ┌─────┐                   │
│  │ Tx1 │ x=1読取→y=1書込│ Tx2 │ y=1読取→x=1書込  │
│  │     │ 結果: x=0,y=1  │     │ 結果: x=1,y=0    │
│  │     │ 直列なら x=1,y=1 または x=0,y=0 のはず    │
│  └─────┘                └─────┘                   │
└────────────────────────────────────────────────────┘
```

### コード例4: 各分離レベルの動作例

```sql
-- ===== READ UNCOMMITTED =====
-- PostgreSQLでは実質READ COMMITTEDとして動作する
-- MySQL InnoDB では実際にダーティリードが発生する
SET TRANSACTION ISOLATION LEVEL READ UNCOMMITTED;

-- ===== READ COMMITTED（PostgreSQLのデフォルト）=====
-- 各SQL文の開始時にスナップショットを取得
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
BEGIN;
SELECT balance FROM accounts WHERE id = 1;  -- 1000

-- 他のトランザクションがここでUPDATE→COMMITした場合
SELECT balance FROM accounts WHERE id = 1;  -- 1200（変わる）
-- → 同じトランザクション内でも、最新のCOMMIT済みデータが見える
COMMIT;

-- ===== REPEATABLE READ =====
-- トランザクション開始時（最初のクエリ実行時）にスナップショットを取得
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
BEGIN;
SELECT balance FROM accounts WHERE id = 1;  -- 1000

-- 他のトランザクションがここでUPDATE→COMMITしても
SELECT balance FROM accounts WHERE id = 1;  -- 1000（変わらない）
-- → トランザクション開始時のスナップショットを一貫して見続ける

-- ただし、自身のUPDATEが他のTxと競合するとエラー
UPDATE accounts SET balance = balance + 100 WHERE id = 1;
-- ERROR: could not serialize access due to concurrent update
COMMIT;

-- ===== SERIALIZABLE（最も厳格）=====
-- 直列実行と同等の結果を保証
-- SSI (Serializable Snapshot Isolation) で実装
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;
BEGIN;
-- 同じデータに対して読み書きが競合するトランザクションがある場合、
-- どちらかがシリアライゼーション失敗エラーで中断される
SELECT SUM(balance) FROM accounts;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
COMMIT;
-- ERROR: could not serialize access due to read/write dependencies
-- → リトライが必要
```

### コード例5: SERIALIZABLE分離レベルでのリトライ実装

```sql
-- アプリケーション側でのリトライロジック（疑似コード → PL/pgSQL実装）
CREATE OR REPLACE FUNCTION transfer_with_retry(
    p_from_account VARCHAR,
    p_to_account   VARCHAR,
    p_amount       DECIMAL,
    p_max_retries  INTEGER DEFAULT 3
) RETURNS VOID AS $$
DECLARE
    v_retries INTEGER := 0;
    v_success BOOLEAN := FALSE;
BEGIN
    WHILE NOT v_success AND v_retries < p_max_retries LOOP
        BEGIN
            -- SERIALIZABLE分離レベルで実行
            SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;

            -- 残高確認
            IF (SELECT balance FROM accounts WHERE account_id = p_from_account) < p_amount THEN
                RAISE EXCEPTION '残高不足';
            END IF;

            -- 送金実行
            UPDATE accounts SET balance = balance - p_amount
            WHERE account_id = p_from_account;
            UPDATE accounts SET balance = balance + p_amount
            WHERE account_id = p_to_account;

            -- 送金履歴記録
            INSERT INTO transfer_log (from_account, to_account, amount, transferred_at)
            VALUES (p_from_account, p_to_account, p_amount, NOW());

            v_success := TRUE;

        EXCEPTION
            WHEN serialization_failure OR deadlock_detected THEN
                v_retries := v_retries + 1;
                RAISE NOTICE 'リトライ %/% (理由: %)', v_retries, p_max_retries, SQLERRM;
                -- 指数バックオフ的な待機
                PERFORM pg_sleep(0.1 * power(2, v_retries - 1));
        END;
    END LOOP;

    IF NOT v_success THEN
        RAISE EXCEPTION '最大リトライ回数超過 (%回)', p_max_retries;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- 使用例
SELECT transfer_with_retry('A001', 'B002', 10000);
```

---

## 3. デッドロック

### デッドロックの仕組み

```
┌────────────── デッドロックの構造 ──────────────────┐
│                                                     │
│  リソース依存グラフ（Wait-for Graph）:              │
│                                                     │
│  Tx1 ──(Bを待つ)──→ Tx2                            │
│   ↑                   │                            │
│   └──(Aを待つ)────────┘                            │
│                                                     │
│  循環的な待ち関係 = デッドロック                     │
│                                                     │
│  タイムライン:                                      │
│  t1: Tx1 → LOCK(A) ✓                              │
│  t2: Tx2 → LOCK(B) ✓                              │
│  t3: Tx1 → LOCK(B) → 待機（Tx2がBを保持中）       │
│  t4: Tx2 → LOCK(A) → 待機（Tx1がAを保持中）       │
│  → 相互に待ち合い → 永遠に解決しない！             │
│                                                     │
│  PostgreSQLの対策:                                  │
│  - deadlock_timeout（デフォルト1秒）経過後に        │
│    Wait-for Graphを検査                             │
│  - 循環を検出したら1つのTxを強制ROLLBACK           │
│  - ログに "Process X waits for ... blocked by       │
│    process Y" と出力                                │
└─────────────────────────────────────────────────────┘
```

### コード例6: デッドロックの発生と対策

```sql
-- ===== デッドロックの典型的なシナリオ =====
-- ターミナル1 (Tx1):
BEGIN;
UPDATE accounts SET balance = balance - 100 WHERE account_id = 'A001';  -- A001をロック
-- ここでTx2がB002をロック
UPDATE accounts SET balance = balance + 100 WHERE account_id = 'B002';  -- B002を待機...

-- ターミナル2 (Tx2):
BEGIN;
UPDATE accounts SET balance = balance - 200 WHERE account_id = 'B002';  -- B002をロック
-- Tx1がA001をロック済み
UPDATE accounts SET balance = balance + 200 WHERE account_id = 'A001';  -- A001を待機...
-- → デッドロック発生！

-- ===== 対策1: 常に同じ順序でロックする =====
-- すべてのトランザクションがID順にロックを取得すれば循環しない
BEGIN;
-- account_idの昇順でロック取得
UPDATE accounts SET balance = balance - 100 WHERE account_id = 'A001';  -- A→Bの順
UPDATE accounts SET balance = balance + 100 WHERE account_id = 'B002';
COMMIT;

-- ===== 対策2: SELECT FOR UPDATEで事前にロック取得 =====
BEGIN;
-- ORDER BYでソートしてからロック取得
SELECT * FROM accounts
WHERE account_id IN ('A001', 'B002')
ORDER BY account_id
FOR UPDATE;
-- ↑ この時点でA001, B002の両方をロック済み
-- 他のTxは待機するが、デッドロックにはならない

UPDATE accounts SET balance = balance - 100 WHERE account_id = 'A001';
UPDATE accounts SET balance = balance + 100 WHERE account_id = 'B002';
COMMIT;

-- ===== 対策3: ロックタイムアウトの設定 =====
SET lock_timeout = '5s';  -- 5秒でロック待ちを諦める
-- ERROR: canceling statement due to lock timeout

-- ===== 対策4: デッドロック検出設定の確認 =====
SHOW deadlock_timeout;
-- デフォルト: 1s（この時間待ってからデッドロック検出を実行）
```

### コード例7: 楽観的ロック vs 悲観的ロック

```sql
-- ===== 悲観的ロック（Pessimistic Locking）=====
-- 先にロックを取得してからデータを操作する
-- 適する場面: 競合が頻繁に発生する場合

-- 基本形: FOR UPDATE
BEGIN;
SELECT * FROM products WHERE id = 42 FOR UPDATE;
-- → 他のTxのFOR UPDATEは待機
UPDATE products SET stock = stock - 1 WHERE id = 42;
COMMIT;

-- NOWAIT: ロック取得不可なら即エラー
SELECT * FROM products WHERE id = 42 FOR UPDATE NOWAIT;
-- ERROR: could not obtain lock on row in relation "products"
-- → アプリ側で即座にエラーハンドリング可能

-- SKIP LOCKED: ロック中の行をスキップ（キュー処理向け）
SELECT id, task_data FROM tasks
WHERE status = 'pending'
ORDER BY created_at
LIMIT 1
FOR UPDATE SKIP LOCKED;
-- → ワーカーが並行して未処理タスクを取得
-- → 同じタスクを複数ワーカーが取得することがない

-- FOR SHARE: 読み取りロック（他TxのUPDATEは待機、SELECTは許可）
SELECT * FROM products WHERE id = 42 FOR SHARE;


-- ===== 楽観的ロック（Optimistic Locking）=====
-- ロックを取得せず、更新時に競合を検出する
-- 適する場面: 競合が稀な場合、Web APIなどのステートレス環境

-- Step 1: データ読み取り時にバージョンを記録
SELECT id, name, stock, version FROM products WHERE id = 42;
-- → stock=10, version=5

-- Step 2: 更新時にバージョンの一致を確認
UPDATE products
SET stock = stock - 1,
    version = version + 1
WHERE id = 42
  AND version = 5;  -- 読み取り時のバージョンと一致するか確認
-- → 影響行数が0なら競合発生 → アプリ側でリトライ

-- バージョンカラムの代わりにupdated_atを使う方式
UPDATE products
SET stock = stock - 1,
    updated_at = NOW()
WHERE id = 42
  AND updated_at = '2024-01-15 10:30:00';
-- → タイムスタンプの精度に注意（マイクロ秒まで比較）
```

### コード例8: アドバイザリロック

```sql
-- ===== アドバイザリロック（Advisory Lock）=====
-- テーブルや行ではなく、任意のアプリケーションレベルのロック
-- テーブルへのロック影響がないため、柔軟な排他制御が可能

-- セッションレベルのアドバイザリロック
-- 同じキーで取得を試みる他のセッションはブロックされる
SELECT pg_advisory_lock(12345);  -- ロック取得（ブロッキング）
-- ... 排他的な処理 ...
SELECT pg_advisory_unlock(12345);  -- ロック解放

-- 非ブロッキング版: ロック取得を試み、取れなければFALSEを返す
SELECT pg_try_advisory_lock(12345);  -- TRUE / FALSE

-- トランザクションレベル: COMMIT/ROLLBACK時に自動解放
BEGIN;
SELECT pg_advisory_xact_lock(12345);
-- ... 処理 ...
COMMIT;  -- 自動的にロック解放

-- 2つのキーを使う場合（テーブルID + 行ID の組み合わせなど）
SELECT pg_advisory_lock(hashtext('orders'), 42);

-- 実用例: 外部APIの重複呼び出し防止
CREATE OR REPLACE FUNCTION process_payment(p_order_id INTEGER)
RETURNS VOID AS $$
BEGIN
    -- 注文IDでアドバイザリロックを取得
    IF NOT pg_try_advisory_xact_lock(hashtext('payment'), p_order_id) THEN
        RAISE EXCEPTION '同じ注文の決済が処理中です';
    END IF;

    -- 決済処理（外部API呼び出しを含む）
    UPDATE orders SET status = 'processing' WHERE id = p_order_id;
    -- ... 外部API呼び出し ...
    UPDATE orders SET status = 'paid' WHERE id = p_order_id;
END;
$$ LANGUAGE plpgsql;
```

---

## 4. MVCC（Multi-Version Concurrency Control）

### MVCCの内部構造

PostgreSQLのMVCCは、各行に不可視なシステムカラム（`xmin`, `xmax`）を持たせることで実現される。

```
┌──────────── MVCCの内部構造（PostgreSQL）─────────────┐
│                                                       │
│  各タプル（行）のヘッダ:                               │
│  ┌────────────────────────────────────────────────┐  │
│  │ xmin  = 100  （この行を挿入したTxのID）         │  │
│  │ xmax  = 0    （この行を削除/更新したTxのID）     │  │
│  │ ctid  = (0,1)（物理的な位置: ページ0, タプル1） │  │
│  │ infomask = ...（コミット/中断フラグ等）         │  │
│  │ [データ本体]                                    │  │
│  └────────────────────────────────────────────────┘  │
│                                                       │
│  UPDATEの内部動作:                                    │
│  ① 旧タプルのxmaxを現在のTxIDに設定                  │
│  ② 新タプルを別の場所にINSERT（xmin=現在のTxID）     │
│  ③ 旧タプルのctidを新タプルの位置に更新              │
│                                                       │
│  例: Tx200がUPDATE accounts SET balance=900           │
│                                                       │
│  旧タプル:                                            │
│  xmin=100, xmax=200, ctid=(0,5) → 新タプルへ         │
│  balance=1000                                         │
│                                                       │
│  新タプル:                                            │
│  xmin=200, xmax=0, ctid=(0,5)                        │
│  balance=900                                          │
│                                                       │
│  → 旧タプルは即削除されない（他のTxが参照中かも）    │
│  → VACUUMが不要になった旧タプルを回収               │
└───────────────────────────────────────────────────────┘
```

### コード例9: MVCCの動作確認

```sql
-- xmin, xmax の確認
-- PostgreSQLでは隠しカラムとして参照可能
SELECT xmin, xmax, ctid, * FROM accounts WHERE account_id = 'A001';
-- xmin=100, xmax=0, ctid=(0,1), account_id='A001', balance=100000

-- 現在のトランザクションIDの確認
SELECT txid_current();
-- 結果: 200

-- トランザクション内でUPDATE
BEGIN;
UPDATE accounts SET balance = 90000 WHERE account_id = 'A001';

-- 別セッションから確認（READ COMMITTED）
-- → 旧タプル（balance=100000）が見える（Tx200は未コミット）

COMMIT;

-- 別セッションから再確認
-- → 新タプル（balance=90000）が見える

-- スナップショットの確認
SELECT txid_current_snapshot();
-- 結果: '200:205:200,202'
-- 意味: xmin=200, xmax=205, 実行中のTxID=[200, 202]
-- → TxID 200未満はコミット済み、200と202は実行中、
--   201, 203, 204はコミット済み、205以降は未開始
```

### コード例10: VACUUM — MVCCのゴミ回収

```sql
-- 不要なタプル（dead tuple）の確認
SELECT
    schemaname,
    relname AS table_name,
    n_live_tup,          -- 有効なタプル数
    n_dead_tup,          -- 不要なタプル数
    n_mod_since_analyze, -- 最後のANALYZE以降の変更数
    last_vacuum,         -- 最後のVACUUM実行日時
    last_autovacuum,     -- 最後のauto VACUUM実行日時
    last_analyze         -- 最後のANALYZE実行日時
FROM pg_stat_user_tables
WHERE n_dead_tup > 0
ORDER BY n_dead_tup DESC;

-- 手動VACUUM（通常はautovacuumに任せる）
VACUUM (VERBOSE) accounts;
-- INFO: "accounts": found 150 removable, 1000 nonremovable row versions
-- INFO: "accounts": removed 150 row versions

-- VACUUM FULL: テーブルを完全に書き直す（排他ロック）
-- → 通常はpg_repackを使用する（オンラインで実行可能）
VACUUM FULL accounts;

-- autovacuumのパラメータ確認
SHOW autovacuum_vacuum_threshold;       -- 50（デフォルト）
SHOW autovacuum_vacuum_scale_factor;    -- 0.2（デフォルト）
-- → dead tuples > 50 + 0.2 * n_live_tup で自動実行

-- テーブル単位でautovacuumの設定を変更
ALTER TABLE accounts SET (
    autovacuum_vacuum_threshold = 100,
    autovacuum_vacuum_scale_factor = 0.05,
    autovacuum_analyze_threshold = 50,
    autovacuum_analyze_scale_factor = 0.02
);

-- テーブル膨張率の確認
SELECT
    relname,
    pg_size_pretty(pg_total_relation_size(relid)) AS total_size,
    pg_size_pretty(pg_relation_size(relid)) AS table_size,
    ROUND(100.0 * n_dead_tup / NULLIF(n_live_tup + n_dead_tup, 0), 1) AS dead_pct
FROM pg_stat_user_tables
ORDER BY n_dead_tup DESC
LIMIT 10;
```

---

## 5. トランザクション設計パターン

### コード例11: べき等なトランザクション設計

```sql
-- ===== べき等（Idempotent）設計 =====
-- 同じ操作を複数回実行しても結果が変わらない設計
-- ネットワーク障害でリトライが発生する環境では必須

-- NG: べき等でない（重複実行で金額が倍になる）
INSERT INTO payments (order_id, amount) VALUES (42, 10000);

-- OK: べき等（重複実行しても1回分のみ）
INSERT INTO payments (order_id, amount)
VALUES (42, 10000)
ON CONFLICT (order_id) DO NOTHING;

-- OK: 冪等キーを使ったべき等設計
CREATE TABLE payments (
    id              SERIAL PRIMARY KEY,
    idempotency_key UUID UNIQUE NOT NULL,  -- クライアントが生成する一意キー
    order_id        INTEGER NOT NULL,
    amount          DECIMAL(10, 2) NOT NULL,
    status          VARCHAR(20) DEFAULT 'pending',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 冪等キーで重複を防止
INSERT INTO payments (idempotency_key, order_id, amount)
VALUES ('a1b2c3d4-e5f6-7890-abcd-ef1234567890', 42, 10000)
ON CONFLICT (idempotency_key) DO NOTHING
RETURNING *;
-- → 2回目の実行では挿入されず、RETURNINGも空
```

### コード例12: 分散トランザクション — Sagaパターン

```sql
-- ===== Sagaパターン（マイクロサービス間のトランザクション）=====
-- 各サービスのローカルトランザクション + 補償トランザクションで一貫性を維持

-- サガの状態管理テーブル
CREATE TABLE saga_instances (
    saga_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    saga_type     VARCHAR(50) NOT NULL,
    current_step  INTEGER NOT NULL DEFAULT 0,
    status        VARCHAR(20) NOT NULL DEFAULT 'running'
                  CHECK (status IN ('running', 'completed', 'compensating', 'failed')),
    payload       JSONB NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE saga_steps (
    saga_id       UUID REFERENCES saga_instances(saga_id),
    step_number   INTEGER NOT NULL,
    step_name     VARCHAR(100) NOT NULL,
    status        VARCHAR(20) NOT NULL DEFAULT 'pending',
    result        JSONB,
    executed_at   TIMESTAMPTZ,
    compensated_at TIMESTAMPTZ,
    PRIMARY KEY (saga_id, step_number)
);

-- 注文サガの例:
-- Step 1: 在庫予約      → 補償: 在庫予約取消
-- Step 2: 決済実行      → 補償: 返金処理
-- Step 3: 配送手配      → 補償: 配送キャンセル
-- Step 4: 注文確定

-- 各ステップはローカルトランザクションで実行
-- 途中で失敗した場合、完了済みステップの補償トランザクションを逆順に実行
```

---

## 分離レベル比較表

| 分離レベル | ダーティリード | ノンリピータブルリード | ファントムリード | シリアライゼーション異常 | パフォーマンス |
|-----------|:---:|:---:|:---:|:---:|:---:|
| READ UNCOMMITTED | 発生 | 発生 | 発生 | 発生 | 最速 |
| READ COMMITTED | **防止** | 発生 | 発生 | 発生 | 速い |
| REPEATABLE READ | **防止** | **防止** | 発生* | 発生 | 普通 |
| SERIALIZABLE | **防止** | **防止** | **防止** | **防止** | 最遅 |

*PostgreSQLのREPEATABLE READはSSI（Serializable Snapshot Isolation）の前段階であるSI（Snapshot Isolation）を使用し、ファントムリードも防止する。ただしwrite skew（書き込みスキュー）は防止できない。

## ロック方式比較表

| 方式 | ロックタイミング | 競合検出 | スループット | 適する場面 | リトライ要否 |
|------|:---:|:---:|:---:|-----------|:---:|
| 悲観的ロック (FOR UPDATE) | データ読取時 | 事前防止 | 低い | 競合が頻繁な場合 | 不要 |
| 楽観的ロック (version) | 更新時 | 事後検出 | 高い | 競合が稀な場合 | 必要 |
| SKIP LOCKED | 読取時 | スキップ | 高い | キュー/ジョブ処理 | 不要 |
| FOR UPDATE NOWAIT | 読取時 | 即時エラー | 高い | 短時間処理 | 必要 |
| アドバイザリロック | 任意 | アプリ制御 | 柔軟 | 外部API排他制御 | 必要 |

## RDBMS別デフォルト比較表

| RDBMS | デフォルト分離レベル | MVCC | デッドロック検出 |
|-------|:---:|:---:|:---:|
| PostgreSQL | READ COMMITTED | スナップショット分離 | Wait-for Graph (1秒後) |
| MySQL InnoDB | REPEATABLE READ | Undo Log ベース | Wait-for Graph (即時) |
| Oracle | READ COMMITTED | Undo Tablespace ベース | 即時検出 |
| SQL Server | READ COMMITTED | ロック方式 (RCSI有効時はMVCC) | Wait-for Graph (5秒) |
| SQLite | SERIALIZABLE | WAL モード時はMVCC | タイムアウト (5秒) |

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
-- 3. MVCC環境でVACUUMが遅延（長命Txがスナップショットを保持）
-- 4. コネクションプールの枯渇
-- 5. レプリケーション遅延の原因

-- OK: トランザクションを短く保つ（楽観的ロック）
-- 読取は通常のSELECT、更新時のみ短いトランザクション
SELECT id, price, version FROM products WHERE id = 42;
-- → price=800, version=5
-- ユーザー操作完了後に短いトランザクションで更新
BEGIN;
UPDATE products SET price = 1000, version = version + 1
WHERE id = 42 AND version = 5;  -- 楽観的ロック
-- 影響行数=0ならリトライ
COMMIT;
```

### アンチパターン2: 不必要に高い分離レベル

```sql
-- NG: 全てSERIALIZABLEにする
SET default_transaction_isolation = 'serializable';

-- 問題点:
-- 1. シリアライゼーション失敗でリトライが頻発（並行性低下）
-- 2. パフォーマンスが20-50%低下する場合がある
-- 3. 多くの場合READ COMMITTEDで十分
-- 4. リトライロジックを全てのトランザクションに実装する必要がある

-- OK: 用途に応じて分離レベルを選択
-- 一般的なCRUD操作: READ COMMITTED（デフォルト）
BEGIN;
UPDATE products SET stock = stock - 1 WHERE id = 42;
COMMIT;

-- 残高計算や整合性チェック: REPEATABLE READ
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SELECT SUM(balance) FROM accounts;
-- 途中で他のTxが残高を変更してもスナップショットが一貫
COMMIT;

-- 厳密な一貫性が必要な金融処理: SERIALIZABLE
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
-- ダブルスペンディング防止など
COMMIT;
```

### アンチパターン3: コミットされないトランザクション

```sql
-- NG: BEGINしたまま放置
BEGIN;
SELECT * FROM large_table;
-- ... 接続を保持したまま放置 ...
-- → VACUUMが不要タプルを回収できない
-- → テーブル膨張が進行する

-- 問題の検出
SELECT pid, state, query, xact_start,
       NOW() - xact_start AS duration
FROM pg_stat_activity
WHERE state = 'idle in transaction'
  AND xact_start < NOW() - INTERVAL '5 minutes';

-- 対策: idle_in_transaction_session_timeout の設定
SET idle_in_transaction_session_timeout = '10min';
-- → 10分間アイドルなトランザクションを自動的に切断

-- postgresql.conf に設定推奨
-- idle_in_transaction_session_timeout = '10min'
```

---

## 実践演習

### 演習1（基礎）: トランザクションとSAVEPOINTの基本操作

以下のシナリオをSQLで実装してください。

**要件**:
1. `warehouse_items` テーブルから商品A(id=1)の在庫を3個減らす
2. `shipments` テーブルに出荷レコードを挿入する
3. `notifications` テーブルに通知を挿入するが、失敗しても出荷は成立させる
4. 在庫が0未満になる場合は全体をロールバックする

```sql
-- テスト用テーブル
CREATE TABLE warehouse_items (
    id    INTEGER PRIMARY KEY,
    name  VARCHAR(100) NOT NULL,
    stock INTEGER NOT NULL CHECK (stock >= 0)
);

CREATE TABLE shipments (
    id         SERIAL PRIMARY KEY,
    item_id    INTEGER NOT NULL REFERENCES warehouse_items(id),
    quantity   INTEGER NOT NULL,
    shipped_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE notifications (
    id         SERIAL PRIMARY KEY,
    message    TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO warehouse_items VALUES (1, '商品A', 10);
```

<details>
<summary>模範解答</summary>

```sql
BEGIN;

-- Step 1: 在庫の減算
UPDATE warehouse_items SET stock = stock - 3 WHERE id = 1;

-- Step 2: 在庫チェック
DO $$
BEGIN
    IF (SELECT stock FROM warehouse_items WHERE id = 1) < 0 THEN
        RAISE EXCEPTION '在庫不足: 商品A';
    END IF;
END $$;

-- Step 3: 出荷レコードの挿入
INSERT INTO shipments (item_id, quantity)
VALUES (1, 3);
SAVEPOINT after_shipment;

-- Step 4: 通知の挿入（失敗しても出荷は成立）
DO $$
BEGIN
    INSERT INTO notifications (message)
    VALUES ('商品A: 3個出荷しました');
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE '通知の送信に失敗しましたが、出荷は成立します: %', SQLERRM;
    ROLLBACK TO SAVEPOINT after_shipment;
END $$;

COMMIT;

-- 確認
SELECT * FROM warehouse_items WHERE id = 1;
-- stock = 7
SELECT * FROM shipments WHERE item_id = 1;
-- quantity = 3
```

**解説**: SAVEPOINTを `after_shipment` の位置に設置することで、通知挿入が失敗しても出荷レコードまでの処理は維持される。PostgreSQLではトランザクション内のエラーが全てのSQL文を無効化するため、EXCEPTION ブロックでキャッチしてSAVEPOINTまでロールバックする必要がある。

</details>

### 演習2（応用）: 楽観的ロックの完全な実装

ECサイトの在庫管理で、楽観的ロックを使った購入処理を実装してください。

**要件**:
1. `products` テーブルに `version` カラムを使った楽観的ロック
2. 在庫不足時は適切なエラーメッセージ
3. バージョン競合時は最大3回リトライ
4. リトライ間には指数バックオフを入れる

```sql
CREATE TABLE products (
    id       SERIAL PRIMARY KEY,
    name     VARCHAR(200) NOT NULL,
    stock    INTEGER NOT NULL CHECK (stock >= 0),
    price    DECIMAL(10, 2) NOT NULL,
    version  INTEGER NOT NULL DEFAULT 1,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO products VALUES (1, 'ノートPC', 5, 98000, 1, NOW());
```

<details>
<summary>模範解答</summary>

```sql
CREATE OR REPLACE FUNCTION purchase_product(
    p_product_id  INTEGER,
    p_quantity    INTEGER,
    p_max_retries INTEGER DEFAULT 3
) RETURNS TABLE (
    success    BOOLEAN,
    message    TEXT,
    new_stock  INTEGER,
    new_version INTEGER
) AS $$
DECLARE
    v_current_stock   INTEGER;
    v_current_version INTEGER;
    v_rows_affected   INTEGER;
    v_retries         INTEGER := 0;
BEGIN
    LOOP
        -- Step 1: 現在のデータを読み取り（ロックなし）
        SELECT stock, version
        INTO v_current_stock, v_current_version
        FROM products
        WHERE id = p_product_id;

        -- 商品が存在しない場合
        IF NOT FOUND THEN
            RETURN QUERY SELECT FALSE, '商品が見つかりません'::TEXT, 0, 0;
            RETURN;
        END IF;

        -- 在庫不足チェック
        IF v_current_stock < p_quantity THEN
            RETURN QUERY SELECT FALSE,
                format('在庫不足: 要求=%s, 在庫=%s', p_quantity, v_current_stock)::TEXT,
                v_current_stock, v_current_version;
            RETURN;
        END IF;

        -- Step 2: 楽観的ロックでUPDATE
        UPDATE products
        SET stock = stock - p_quantity,
            version = version + 1,
            updated_at = NOW()
        WHERE id = p_product_id
          AND version = v_current_version;

        GET DIAGNOSTICS v_rows_affected = ROW_COUNT;

        -- 更新成功
        IF v_rows_affected = 1 THEN
            RETURN QUERY SELECT TRUE,
                '購入成功'::TEXT,
                v_current_stock - p_quantity,
                v_current_version + 1;
            RETURN;
        END IF;

        -- バージョン競合 → リトライ
        v_retries := v_retries + 1;
        IF v_retries >= p_max_retries THEN
            RETURN QUERY SELECT FALSE,
                format('最大リトライ回数(%s)を超過しました', p_max_retries)::TEXT,
                0, 0;
            RETURN;
        END IF;

        -- 指数バックオフ（0.1秒, 0.2秒, 0.4秒...）
        PERFORM pg_sleep(0.1 * power(2, v_retries - 1));
        RAISE NOTICE 'バージョン競合 → リトライ %/%', v_retries, p_max_retries;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- 使用例
SELECT * FROM purchase_product(1, 2);
-- success=true, message='購入成功', new_stock=3, new_version=2

-- 在庫不足テスト
SELECT * FROM purchase_product(1, 100);
-- success=false, message='在庫不足: 要求=100, 在庫=3'
```

**解説**: 楽観的ロックでは、SELECTとUPDATEの間に他のトランザクションがデータを変更した場合、`WHERE version = v_current_version` の条件に一致せず `ROW_COUNT = 0` になる。この場合にリトライを行う。指数バックオフにより、高負荷時の過剰なリトライを防ぐ。

</details>

### 演習3（発展）: 分離レベルの動作検証

2つのターミナル（セッション）を使い、以下の分離レベルの違いを実際に検証してください。

**課題**:
1. READ COMMITTED で「ノンリピータブルリード」が発生することを確認
2. REPEATABLE READ で「ノンリピータブルリード」が防止されることを確認
3. REPEATABLE READ で更新競合エラーが発生することを確認
4. SERIALIZABLEで「write skew」が検出されることを確認

```sql
-- 準備
CREATE TABLE test_accounts (
    id      INTEGER PRIMARY KEY,
    balance INTEGER NOT NULL,
    type    VARCHAR(20) NOT NULL
);
INSERT INTO test_accounts VALUES (1, 1000, 'checking');
INSERT INTO test_accounts VALUES (2, 2000, 'savings');
```

<details>
<summary>模範解答</summary>

```sql
-- ===== 検証1: READ COMMITTED でのノンリピータブルリード =====

-- ターミナル1:
BEGIN TRANSACTION ISOLATION LEVEL READ COMMITTED;
SELECT balance FROM test_accounts WHERE id = 1;
-- 結果: 1000

-- ターミナル2:
BEGIN;
UPDATE test_accounts SET balance = 1500 WHERE id = 1;
COMMIT;

-- ターミナル1（続き）:
SELECT balance FROM test_accounts WHERE id = 1;
-- 結果: 1500 ← 値が変わった！（ノンリピータブルリード）
COMMIT;


-- ===== 検証2: REPEATABLE READ での防止 =====

-- ターミナル1:
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SELECT balance FROM test_accounts WHERE id = 1;
-- 結果: 1500

-- ターミナル2:
BEGIN;
UPDATE test_accounts SET balance = 2000 WHERE id = 1;
COMMIT;

-- ターミナル1（続き）:
SELECT balance FROM test_accounts WHERE id = 1;
-- 結果: 1500 ← 値が変わらない！（ノンリピータブルリード防止）
COMMIT;

-- ターミナル1で再度確認:
SELECT balance FROM test_accounts WHERE id = 1;
-- 結果: 2000 ← COMMIT後は最新値が見える


-- ===== 検証3: REPEATABLE READ での更新競合 =====

-- 準備
UPDATE test_accounts SET balance = 1000 WHERE id = 1;

-- ターミナル1:
BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ;
SELECT balance FROM test_accounts WHERE id = 1;  -- 1000

-- ターミナル2:
BEGIN;
UPDATE test_accounts SET balance = 1500 WHERE id = 1;
COMMIT;

-- ターミナル1（続き）:
UPDATE test_accounts SET balance = balance + 100 WHERE id = 1;
-- ERROR: could not serialize access due to concurrent update
ROLLBACK;


-- ===== 検証4: SERIALIZABLE でのwrite skew検出 =====

-- 準備: 医師のオンコール当番（最低1人は勤務必須）
UPDATE test_accounts SET balance = 1000 WHERE id = 1;
UPDATE test_accounts SET balance = 2000 WHERE id = 2;

-- ルール: id=1とid=2の合計が0より大きくなければならない

-- ターミナル1:
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
-- 合計を確認: 3000 > 0、OK
SELECT SUM(balance) FROM test_accounts;
-- id=1の残高を0にする
UPDATE test_accounts SET balance = 0 WHERE id = 1;

-- ターミナル2（Tx1がCOMMIT前に実行）:
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
-- 合計を確認: 3000 > 0、OK（Tx1の変更は見えない）
SELECT SUM(balance) FROM test_accounts;
-- id=2の残高を0にする
UPDATE test_accounts SET balance = 0 WHERE id = 2;

-- ターミナル1:
COMMIT;  -- 成功

-- ターミナル2:
COMMIT;
-- ERROR: could not serialize access due to read/write dependencies
-- → write skewが検出され、Tx2がロールバックされる
-- → 合計が0になることを防止！
```

**解説**: Write skewは、2つのトランザクションが互いに異なる行を読み取り、条件を満たすことを確認してから異なる行を更新する場合に発生する。READ COMMITTEDやREPEATABLE READでは検出できず、SERIALIZABLEでのみ防止できる。これがSERIALIZABLEが必要になる代表的なケースである。

</details>

---

## FAQ

### Q1: AUTO COMMITとは何か？

AUTO COMMITが有効（PostgreSQLのデフォルト）の場合、各SQL文が自動的に独立したトランザクションとして実行される。明示的な`BEGIN`/`COMMIT`を使わない限り、各文の完了時に自動的にCOMMITされる。

**注意点**:
- `psql`ではデフォルトでAUTO COMMITが有効（`\set AUTOCOMMIT on`）
- `\set AUTOCOMMIT off` にすると全てのSQLがトランザクション内で実行される
- バッチ処理では明示的にBEGINで囲むことで、中間状態のCOMMITを防ぐ
- アプリケーションフレームワーク（Django、Rails等）は通常リクエスト単位でトランザクションを管理する

### Q2: PostgreSQLのMVCCとロックベースの違いは何か？

**MVCC（Multi-Version Concurrency Control）**:
- 各行の複数バージョンを保持し、読み取りと書き込みが互いにブロックしない
- SELECTはロックを取得せず、トランザクション開始時点のスナップショットを読む
- **利点**: READERとWRITERが共存。読み取りが遅延しない
- **欠点**: 不要バージョンの蓄積（VACUUMが必要）、テーブル膨張

**ロックベース（SQL Serverのデフォルト等）**:
- 共有ロック（読み取り）と排他ロック（書き込み）で制御
- WRITERがいるとREADERも待機する
- **利点**: 実装がシンプル、VACUUMが不要
- **欠点**: 読み取りと書き込みが競合、デッドロックが多い

### Q3: トランザクションがROLLBACKされるのはどんな場合か？

(1) 明示的な`ROLLBACK`文の実行
(2) 制約違反（CHECK, FK, UNIQUE等）やデッドロックなどのエラー発生
(3) クライアント接続の切断
(4) `statement_timeout`や`idle_in_transaction_session_timeout`の超過
(5) `serialization_failure`（SERIALIZABLEまたはREPEATABLE READでの競合）
(6) `lock_timeout`の超過
(7) ディスク容量不足やOOM（Out of Memory）

PostgreSQLではエラー発生後、トランザクション内の以降の文は全て拒否される（`ERROR: current transaction is aborted, commands ignored until end of transaction block`）。SAVEPOINTを使用している場合のみ、部分的な復旧が可能。

### Q4: 2フェーズコミット（2PC）とは何か？

2PC（Two-Phase Commit）は、複数のデータベースやリソースマネージャにまたがるトランザクションの原子性を保証するプロトコル。

**Phase 1（Prepare）**: コーディネータが各参加者に「準備完了か？」と確認。各参加者はデータを永続化し「YES」または「NO」を応答。
**Phase 2（Commit/Abort）**: 全参加者がYESなら「COMMIT」、1つでもNOなら「ABORT」を送信。

PostgreSQLでは`PREPARE TRANSACTION`で実装できるが、分散トランザクションのオーバーヘッドは大きい。現代のマイクロサービスアーキテクチャではSagaパターンを推奨する。

### Q5: トランザクションログ（WAL）の設定はどうすべきか？

```sql
-- WAL関連の主要設定
SHOW wal_level;                 -- replica（デフォルト）
SHOW max_wal_size;              -- 1GB（デフォルト）
SHOW min_wal_size;              -- 80MB（デフォルト）
SHOW checkpoint_timeout;        -- 5min（デフォルト）
SHOW synchronous_commit;        -- on（デフォルト）

-- パフォーマンス重視の場合（データ損失リスクあり）
-- synchronous_commit = off
-- → COMMITがWAL書き込みを待たない（最大wal_writer_delay分のデータ損失リスク）
-- → 大量のINSERTバッチ処理で有効
```

---

## まとめ

| 項目 | 要点 |
|------|------|
| ACID | 原子性・一貫性・分離性・耐久性の4特性。WALが基盤 |
| デフォルト分離レベル | PostgreSQL/Oracle: READ COMMITTED、MySQL: REPEATABLE READ |
| MVCC | 読み書きの非ブロック化。各行にxmin/xmaxを持たせて実現 |
| VACUUM | MVCCの不要バージョンを回収。autovacuumの設定が重要 |
| デッドロック対策 | ロック順序の統一、タイムアウト設定、NOWAIT/SKIP LOCKED |
| 悲観的ロック | FOR UPDATE。競合が多い場面で有効。ブロッキング |
| 楽観的ロック | versionカラム。競合が少ない場面で有効。リトライ必要 |
| アドバイザリロック | テーブル/行に影響しないアプリケーションレベルのロック |
| べき等設計 | 重複実行しても結果が変わらない設計。リトライ環境で必須 |
| Sagaパターン | マイクロサービス間のトランザクション。補償トランザクションで一貫性維持 |
| ベストプラクティス | トランザクションを短く、分離レベルは最小限、リトライロジック必須 |

---

## 次に読むべきガイド

- [03-indexing.md](./03-indexing.md) — ロックとインデックスの関係、FOR UPDATEとインデックスの相互作用
- [04-query-optimization.md](./04-query-optimization.md) — トランザクションと実行計画、ロックの影響
- [02-performance-tuning.md](../03-practical/02-performance-tuning.md) — 接続プールとトランザクション管理、VACUUM戦略
- [00-normalization.md](../02-design/00-normalization.md) — 正規化とトランザクション設計の関係
- [security-fundamentals/](../../security-fundamentals/) — SQLインジェクション対策とトランザクションセキュリティ

---

## 参考文献

1. PostgreSQL Documentation — "Transaction Isolation" https://www.postgresql.org/docs/current/transaction-iso.html
2. PostgreSQL Documentation — "Concurrency Control" https://www.postgresql.org/docs/current/mvcc.html
3. Kleppmann, M. (2017). *Designing Data-Intensive Applications*. O'Reilly Media. Chapter 7: Transactions.
4. Berenson, H. et al. (1995). "A Critique of ANSI SQL Isolation Levels". *SIGMOD Record*, 24(2).
5. Fekete, A. et al. (2005). "Making Snapshot Isolation Serializable". *ACM Transactions on Database Systems*, 30(2).
6. PostgreSQL Wiki — "Lock Monitoring" https://wiki.postgresql.org/wiki/Lock_Monitoring
