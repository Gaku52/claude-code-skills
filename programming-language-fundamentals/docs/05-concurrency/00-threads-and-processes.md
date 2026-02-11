# スレッドとプロセス

> 並行処理は「複数の処理を同時に進める」技術。プロセスとスレッドの違いを理解することが、並行プログラミングの出発点。

## この章で学ぶこと

- [ ] プロセスとスレッドの違いを理解する
- [ ] 各言語のスレッドモデルを把握する
- [ ] 同期プリミティブの基本を理解する

---

## 1. 並行（Concurrency）vs 並列（Parallelism）

```
並行（Concurrency）:
  「複数のタスクを管理する構造」
  1つのCPUコアでもタスクを切り替えて実現可能

  Task A: ──▓▓──────▓▓──────▓▓──
  Task B: ────▓▓──────▓▓──────▓▓

並列（Parallelism）:
  「複数のタスクを物理的に同時に実行」
  複数のCPUコアが必要

  Core 1: ──▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓──  Task A
  Core 2: ──▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓──  Task B

  Rob Pike: "Concurrency is about dealing with lots of things at once.
             Parallelism is about doing lots of things at once."
```

---

## 2. プロセス vs スレッド

```
┌──────────────┬──────────────────┬──────────────────┐
│              │ プロセス          │ スレッド          │
├──────────────┼──────────────────┼──────────────────┤
│ メモリ空間    │ 独立（隔離）     │ 共有             │
│ 作成コスト    │ 高い（ms単位）   │ 低い（μs単位）   │
│ 通信          │ IPC（パイプ等）  │ 共有メモリ       │
│ 安全性        │ 高い（隔離）     │ 低い（競合リスク）│
│ オーバーヘッド│ 大きい           │ 小さい           │
│ 用途          │ 独立したプログラム│ 1プログラム内の並行│
└──────────────┴──────────────────┴──────────────────┘

プロセス:
  ┌─────────────┐  ┌─────────────┐
  │ Process A   │  │ Process B   │
  │ ┌─────────┐ │  │ ┌─────────┐ │
  │ │ メモリ   │ │  │ │ メモリ   │ │
  │ │ コード   │ │  │ │ コード   │ │
  │ │ データ   │ │  │ │ データ   │ │
  │ └─────────┘ │  │ └─────────┘ │
  └─────────────┘  └─────────────┘
  ← 完全に隔離 →

スレッド:
  ┌───────────────────────────┐
  │ Process                   │
  │ ┌─────────────────────┐   │
  │ │ 共有メモリ・コード    │   │
  │ └─────────────────────┘   │
  │ ┌─────┐ ┌─────┐ ┌─────┐ │
  │ │ T1  │ │ T2  │ │ T3  │ │  ← スレッドごとにスタック
  │ │stack│ │stack│ │stack│ │
  │ └─────┘ └─────┘ └─────┘ │
  └───────────────────────────┘
```

---

## 3. スレッドの基本操作

```python
# Python: threading モジュール
import threading

def worker(name):
    print(f"Thread {name} started")
    # 処理...
    print(f"Thread {name} finished")

threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(f"Worker-{i}",))
    threads.append(t)
    t.start()

for t in threads:
    t.join()  # 全スレッドの完了を待つ

# 注意: Python の GIL（Global Interpreter Lock）
# → CPython では1度に1スレッドしか Python コードを実行できない
# → CPU密集型処理には multiprocessing を使う
# → I/O密集型処理には threading が有効
```

```rust
// Rust: std::thread（OSスレッド）
use std::thread;

let handle = thread::spawn(|| {
    println!("Hello from thread!");
    42  // スレッドの戻り値
});

let result = handle.join().unwrap();  // → 42

// データの共有（Arc + Mutex）
use std::sync::{Arc, Mutex};

let counter = Arc::new(Mutex::new(0));
let mut handles = vec![];

for _ in 0..10 {
    let counter = Arc::clone(&counter);
    let handle = thread::spawn(move || {
        let mut num = counter.lock().unwrap();
        *num += 1;
    });
    handles.push(handle);
}

for handle in handles {
    handle.join().unwrap();
}
println!("Result: {}", *counter.lock().unwrap());  // → 10
```

```go
// Go: goroutine（軽量スレッド）
func worker(id int, wg *sync.WaitGroup) {
    defer wg.Done()
    fmt.Printf("Worker %d started\n", id)
    time.Sleep(time.Second)
    fmt.Printf("Worker %d done\n", id)
}

var wg sync.WaitGroup
for i := 0; i < 5; i++ {
    wg.Add(1)
    go worker(i, &wg)  // goroutine を起動（goキーワード）
}
wg.Wait()  // 全goroutineの完了を待つ

// goroutine は OSスレッドより遥かに軽量
// → 数百万の goroutine を同時に実行可能
// → Go ランタイムが M:N スケジューリング
```

---

## 4. 同期プリミティブ

```
Mutex（排他制御）:
  1つのスレッドだけがリソースにアクセスできる

  Thread 1: lock() → [使用中] → unlock()
  Thread 2: lock() → [待機...] → [使用中] → unlock()

RWLock（読み書きロック）:
  読み取りは複数同時、書き込みは排他的

  Reader 1: read_lock() → [読取] → unlock()
  Reader 2: read_lock() → [読取] → unlock()  ← 同時にOK
  Writer:   write_lock() → [待機...] → [書込] → unlock()

Semaphore（セマフォ）:
  同時アクセス数を制限
  例: 最大5つの同時DB接続

Condition Variable（条件変数）:
  特定の条件が満たされるまでスレッドを待機させる
```

---

## 5. データ競合と回避策

```
データ競合（Data Race）:
  2つ以上のスレッドが同じメモリに同時アクセスし、
  少なくとも1つが書き込みで、同期がない

  Thread 1: read(x)=0 → write(x)=1
  Thread 2: read(x)=0 → write(x)=1
  結果: x=1（期待は x=2）

回避策:
  1. Mutex でアクセスを排他制御
  2. アトミック操作を使う
  3. データを共有しない（メッセージパッシング）
  4. 不変データを使う（関数型アプローチ）
  5. 言語の型システムで防止（Rust）
```

```rust
// Rust: コンパイル時にデータ競合を防止
let mut data = vec![1, 2, 3];

// ❌ コンパイルエラー: 可変参照を複数スレッドに渡せない
// thread::spawn(|| { data.push(4); });
// thread::spawn(|| { data.push(5); });

// ✅ Arc<Mutex<T>> で安全に共有
let data = Arc::new(Mutex::new(vec![1, 2, 3]));
let d1 = Arc::clone(&data);
let d2 = Arc::clone(&data);

thread::spawn(move || { d1.lock().unwrap().push(4); });
thread::spawn(move || { d2.lock().unwrap().push(5); });
```

---

## まとめ

| 概念 | 説明 | 代表言語 |
|------|------|---------|
| OSスレッド | カーネル管理、重い | C, Java, Rust |
| 軽量スレッド | ランタイム管理、軽い | Go(goroutine), Erlang |
| Mutex | 排他制御 | 全言語 |
| Arc/共有所有権 | スレッド間でデータ共有 | Rust |
| GIL | Python の制約 | CPython |

---

## 次に読むべきガイド
→ [[01-async-await.md]] — async/await

---

## 参考文献
1. Herlihy, M. & Shavit, N. "The Art of Multiprocessor Programming." 2nd Ed, 2020.
2. Klabnik, S. & Nichols, C. "The Rust Programming Language." Ch.16, 2023.
