# スレッドと並行性

> スレッドは「軽量プロセス」であり、同一プロセス内でメモリを共有しながら並行実行される実行単位。

## この章で学ぶこと

- [ ] スレッドとプロセスの違いを説明できる
- [ ] ユーザースレッドとカーネルスレッドの違いを理解する
- [ ] 競合状態とその対策を知る

---

## 1. スレッドの基本

```
プロセス vs スレッド:

  プロセス:                    スレッド:
  ┌──────────────┐            ┌──────────────┐
  │ 独立メモリ空間│            │ メモリ共有    │
  │ ┌────┐       │            │ ┌────┐┌────┐ │
  │ │実行│       │            │ │Th1 ││Th2 │ │
  │ │    │       │            │ │    ││    │ │
  │ └────┘       │            │ └────┘└────┘ │
  │ スタック      │            │ 各自のスタック│
  │ ヒープ       │            │ 共有ヒープ   │
  │ データ       │            │ 共有データ   │
  │ コード       │            │ 共有コード   │
  └──────────────┘            └──────────────┘

  スレッドが共有するもの:
  - コードセクション
  - データセクション（グローバル変数）
  - ヒープ
  - 開いているファイル
  - シグナルハンドラ

  スレッド固有のもの:
  - スタック（ローカル変数）
  - プログラムカウンタ
  - レジスタ
  - スレッドID

  利点:
  1. 生成コスト: fork()の10〜100倍速い
  2. コンテキストスイッチ: TLBフラッシュ不要（同一アドレス空間）
  3. 通信: メモリ共有で高速（IPC不要）
  4. リソース効率: メモリ使用量が少ない

  欠点:
  1. 共有メモリ → 競合状態のリスク
  2. 1スレッドのクラッシュ → プロセス全体が死ぬ
  3. デバッグが困難
```

---

## 2. スレッドモデル

```
1. ユーザーレベルスレッド（N:1）:
   カーネルはスレッドを認識しない

   ユーザー空間: [Th1][Th2][Th3]
                     ↓
   カーネル:    [1プロセス]

   利点: 切替が超高速（syscall不要）
   欠点: 1スレッドがブロック→全スレッドがブロック
         マルチコアを活用できない

2. カーネルレベルスレッド（1:1）:
   各スレッドをカーネルが管理

   ユーザー空間: [Th1][Th2][Th3]
                  ↓    ↓    ↓
   カーネル:    [KT1][KT2][KT3]

   利点: マルチコア活用、独立ブロック
   欠点: 切替にsyscallが必要
   採用: Linux (NPTL), Windows, macOS

3. ハイブリッド（M:N）:
   M個のユーザースレッドをN個のカーネルスレッドにマッピング

   ユーザー空間: [Th1][Th2][Th3][Th4]
                  ↓  ↗ ↓    ↓  ↗
   カーネル:    [KT1]  [KT2]

   利点: 柔軟、効率的
   欠点: 実装が複雑
   採用: Go (goroutine), Erlang (process)

Goのgoroutine（M:Nモデルの成功例）:
  数千〜数百万のgoroutineを少数のOSスレッドで実行
  → スタックは2KB（OSスレッドは1〜8MB）
  → Goランタイムがスケジューリング
```

---

## 3. 同期と競合状態

```
競合状態（Race Condition）:

  2つのスレッドが同時に共有変数を更新

  counter = 0

  Thread A:          Thread B:
  read counter (0)
                     read counter (0)
  counter + 1 = 1
                     counter + 1 = 1
  write counter (1)
                     write counter (1)

  期待値: 2  実際: 1  ← バグ！

  クリティカルセクション:
  共有リソースにアクセスするコード領域
  → 一度に1つのスレッドのみ実行を許可する必要

同期プリミティブ:

  1. Mutex（相互排除）:
     ┌────────────────────────────────┐
     │ lock()                         │
     │   // クリティカルセクション     │
     │   // 1スレッドのみ実行可能     │
     │ unlock()                       │
     └────────────────────────────────┘

  2. セマフォ:
     Mutexの一般化（N個のスレッドが同時アクセス可能）
     P(wait): カウンタ--, 0なら待機
     V(signal): カウンタ++

  3. 条件変数（Condition Variable）:
     特定の条件が成立するまで待機
     wait(): ロック解放+待機
     signal(): 1つのスレッドを起床
     broadcast(): 全スレッドを起床

  4. 読み書きロック（RWLock）:
     読み取り: 複数同時OK
     書き込み: 排他的（1スレッドのみ）
     → 読み取りが多いシステムで有効

デッドロック:
  Thread A: lock(X) → lock(Y) を待つ
  Thread B: lock(Y) → lock(X) を待つ
  → 互いに相手を待ち続けて永久に停止

  デッドロックの4条件（Coffman条件）:
  1. 相互排除: リソースは排他的
  2. 保持と待機: 保持しながら他を待つ
  3. 非横取り: 強制解放不可
  4. 循環待ち: 循環的な待ち関係

  → 1つでも崩せばデッドロック回避
  → 実践的には「ロック順序の統一」が最も有効
```

```python
import threading

counter = 0
lock = threading.Lock()

def increment():
    global counter
    for _ in range(100000):
        with lock:  # Mutexで保護
            counter += 1

t1 = threading.Thread(target=increment)
t2 = threading.Thread(target=increment)
t1.start(); t2.start()
t1.join(); t2.join()
print(counter)  # 200000（lock がないと < 200000）
```

---

## 4. 現代の並行性モデル

```
1. async/await（協調的マルチタスク）:
   シングルスレッドで非同期I/Oを効率的に処理
   → JavaScript, Python asyncio, Rust tokio, Swift concurrency

2. アクターモデル:
   各アクターが独立した状態を持ち、メッセージで通信
   → 共有メモリなし → 競合状態なし
   → Erlang/Elixir, Akka (Scala/Java)

3. CSP（Communicating Sequential Processes）:
   チャネルを介してメッセージを送受信
   → Go (goroutine + channel)
   「メモリを共有して通信するな、通信してメモリを共有せよ」

4. STM（Software Transactional Memory）:
   データベースのトランザクションのようにメモリを操作
   → Haskell, Clojure
```

---

## 実践演習

### 演習1: [基礎] — スレッドの生成と競合

```python
# 以下のコードをlockなしとlockありで実行し、結果を比較せよ
import threading

counter = 0

def worker():
    global counter
    for _ in range(1000000):
        counter += 1  # ← 保護なし

threads = [threading.Thread(target=worker) for _ in range(4)]
for t in threads: t.start()
for t in threads: t.join()
print(f"Expected: 4000000, Got: {counter}")
```

### 演習2: [応用] — デッドロックの再現と回避

```python
# デッドロックを意図的に発生させ、ロック順序の統一で回避せよ
import threading

lock_a = threading.Lock()
lock_b = threading.Lock()

def worker1():
    lock_a.acquire()
    # time.sleep(0.1)  # デッドロックを確実にする
    lock_b.acquire()
    print("Worker1 done")
    lock_b.release()
    lock_a.release()

def worker2():
    lock_b.acquire()  # ← lock_a を先に取るよう修正すれば回避
    lock_a.acquire()
    print("Worker2 done")
    lock_a.release()
    lock_b.release()
```

---

## FAQ

### Q1: GIL（Global Interpreter Lock）とは？

CPython（Python標準実装）には、同時に1つのスレッドしかPythonバイトコードを実行できない制約がある。CPU密度の高い処理ではスレッドでは並列化できない（→ multiprocessingを使う）。I/O待ちの間はGILが解放されるため、I/O密度の高い処理ではスレッドでも効果がある。

### Q2: スレッドとコルーチンの違いは？

スレッドはOSがスケジューリング（プリエンプティブ）。コルーチンはプログラマが明示的にyield/awaitで切替（協調的）。コルーチンはスレッドより遥かに軽量（スタックサイズ、生成コスト）で、100万個のコルーチンも実用的。

### Q3: マルチスレッドとマルチプロセスの使い分けは？

- **マルチスレッド**: メモリ共有が必要、I/O密度が高い、軽量な並行性が欲しい場合
- **マルチプロセス**: CPU密度が高い計算、隔離が必要（セキュリティ）、GILの制約を回避したい場合（Python）

---

## まとめ

| 概念 | ポイント |
|------|---------|
| スレッド | プロセス内の軽量実行単位。メモリ共有 |
| モデル | 1:1(Linux), M:N(Go goroutine) |
| 競合状態 | 共有リソースの同時アクセスで発生 |
| 同期 | Mutex, セマフォ, 条件変数, RWLock |
| デッドロック | 循環待ち。ロック順序の統一で回避 |

---

## 次に読むべきガイド
→ [[02-scheduling.md]] — CPUスケジューリング

---

## 参考文献
1. Silberschatz, A. et al. "Operating System Concepts." 10th Ed, Ch.4, 2018.
2. Herlihy, M. & Shavit, N. "The Art of Multiprocessor Programming." 2nd Ed, 2020.
