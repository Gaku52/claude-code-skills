# スレッドとプロセス

> 並行処理は「複数の処理を同時に進める」技術。プロセスとスレッドの違いを理解することが、並行プログラミングの出発点。

## この章で学ぶこと

- [ ] プロセスとスレッドの違いを理解する
- [ ] 各言語のスレッドモデルを把握する
- [ ] 同期プリミティブの基本を理解する
- [ ] データ競合の原因と回避策を習得する
- [ ] デッドロック・ライブロックの検出と防止を理解する
- [ ] スレッドプールとタスクベースの並行処理を理解する
- [ ] OS レベルのプロセス管理を実践的に理解する

---

## 1. 並行（Concurrency）vs 並列（Parallelism）

### 1.1 基本概念

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

### 1.2 並行と並列の関係

```
┌─────────────────────────────────────────────────────────────┐
│                   並行性と並列性の関係                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  並行だが並列でない:                                         │
│    シングルコアCPUでの複数タスクの切り替え                    │
│    例: Node.js のイベントループ                              │
│                                                              │
│  並列だが並行でない:                                         │
│    同じ処理を複数データに同時適用（SIMD）                     │
│    例: GPU の行列演算                                        │
│                                                              │
│  並行かつ並列:                                               │
│    複数コアで複数タスクを同時管理                             │
│    例: マルチスレッド Web サーバー                           │
│                                                              │
│  どちらでもない:                                             │
│    シングルスレッドの逐次処理                                │
│    例: 単純なスクリプト                                      │
│                                                              │
│            ┌────────────────────────┐                        │
│            │      並行性             │                        │
│            │  ┌─────────────────┐   │                        │
│            │  │  並行 + 並列     │   │                        │
│            │  │  (マルチスレッド) │   │                        │
│  ┌────────┤  └─────────────────┘   │                        │
│  │ 並列性  │  並行のみ              │                        │
│  │ (SIMD) │  (イベントループ)      │                        │
│  └────────┴────────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 なぜ並行処理が必要なのか

```
1. レスポンス性の向上:
   - UI がフリーズしない（重い処理をバックグラウンドで実行）
   - Web サーバーが複数のリクエストを同時に処理

2. スループットの向上:
   - I/O 待ち時間を有効活用（I/Oバウンド処理）
   - マルチコア CPU を効率的に利用（CPUバウンド処理）

3. リソースの効率的利用:
   - ネットワーク I/O の待ち時間に他の処理を実行
   - ディスク I/O と計算を重複実行

I/Oバウンド vs CPUバウンド:
  I/Oバウンド: ネットワーク、ディスク、DB 等の待ち時間が支配的
    → 並行処理（async/await、スレッド、goroutine）が有効
    → 待ち時間中に他の処理を進められる

  CPUバウンド: CPU 演算が支配的
    → 並列処理（マルチプロセス、マルチスレッド）が有効
    → 物理的に複数のCPUコアを使う必要がある
```

---

## 2. プロセス vs スレッド

### 2.1 基本的な違い

```
┌──────────────┬──────────────────┬──────────────────┐
│              │ プロセス          │ スレッド          │
├──────────────┼──────────────────┼──────────────────┤
│ メモリ空間    │ 独立（隔離）     │ 共有             │
│ 作成コスト    │ 高い（ms単位）   │ 低い（μs単位）   │
│ 通信          │ IPC（パイプ等）  │ 共有メモリ       │
│ 安全性        │ 高い（隔離）     │ 低い（競合リスク）│
│ オーバーヘッド│ 大きい           │ 小さい           │
│ コンテキスト  │ 全レジスタ+ページ │ レジスタ+スタック│
│ スイッチ      │ テーブル切替     │ のみ             │
│ 障害の影響    │ 他プロセスに影響  │ 同一プロセス内   │
│              │ しない           │ 全スレッドに影響  │
│ 用途          │ 独立したプログラム│ 1プログラム内の並行│
└──────────────┴──────────────────┴──────────────────┘
```

### 2.2 メモリモデルの詳細

```
プロセス:
  ┌─────────────┐  ┌─────────────┐
  │ Process A   │  │ Process B   │
  │ ┌─────────┐ │  │ ┌─────────┐ │
  │ │ コード   │ │  │ │ コード   │ │
  │ │ データ   │ │  │ │ データ   │ │
  │ │ ヒープ   │ │  │ │ ヒープ   │ │
  │ │ スタック  │ │  │ │ スタック  │ │
  │ │ FD テーブル│ │  │ │ FD テーブル│ │
  │ └─────────┘ │  │ └─────────┘ │
  └─────────────┘  └─────────────┘
  ← 完全に隔離（仮想アドレス空間が別）→

スレッド:
  ┌──────────────────────────────────┐
  │ Process                          │
  │ ┌────────────────────────────┐   │
  │ │ 共有: コード、データ、ヒープ │   │
  │ │       FD テーブル、シグナル  │   │
  │ └────────────────────────────┘   │
  │ ┌────────┐ ┌────────┐ ┌────────┐│
  │ │Thread 1│ │Thread 2│ │Thread 3││
  │ │スタック │ │スタック │ │スタック ││
  │ │レジスタ │ │レジスタ │ │レジスタ ││
  │ │TLS     │ │TLS     │ │TLS     ││
  │ └────────┘ └────────┘ └────────┘│
  └──────────────────────────────────┘
  ※ TLS = Thread Local Storage
```

### 2.3 プロセスの詳細

```
プロセスのライフサイクル:

  ┌──────┐    fork()    ┌──────┐
  │ 生成  │ ──────────→ │ 準備  │
  └──────┘              └──┬───┘
                           │ スケジュール
                           ▼
  ┌──────┐   タイムアウト  ┌──────┐
  │ 待機  │ ←──────────── │ 実行  │
  └──┬───┘   / I/O完了    └──┬───┘
     │       / シグナル       │ exit()
     │ I/O完了                ▼
     └────────────────→ ┌──────┐
                        │ 終了  │
                        └──────┘

プロセスの状態:
  - 新規 (New):      作成直後
  - 準備 (Ready):    CPU 割当待ち
  - 実行 (Running):  CPU で実行中
  - 待機 (Waiting):  I/O やイベント待ち
  - 終了 (Terminated): 実行完了

プロセス制御ブロック (PCB):
  - プロセスID (PID)
  - プロセスの状態
  - プログラムカウンタ
  - CPU レジスタの内容
  - メモリ管理情報（ページテーブル）
  - I/O 状態情報（オープンファイル）
  - アカウンティング情報（CPU使用時間等）
```

### 2.4 スレッドモデルの種類

```
1:1 モデル（ネイティブスレッド）:
  ユーザースレッド 1つ ↔ カーネルスレッド 1つ
  言語: C, Java, Rust, C++
  利点: 真の並列実行、カーネルスケジューラを使用
  欠点: スレッド作成コストが比較的高い

N:1 モデル（グリーンスレッド）:
  ユーザースレッド N個 ↔ カーネルスレッド 1つ
  言語: 初期のJava(Green Threads), Ruby(Fiber)
  利点: スレッド作成コストが非常に低い
  欠点: マルチコアを活用できない

M:N モデル（ハイブリッド）:
  ユーザースレッド M個 ↔ カーネルスレッド N個
  言語: Go(goroutine), Erlang(process), Java(Virtual Threads 21+)
  利点: 軽量かつマルチコア活用可能
  欠点: スケジューラの実装が複雑

┌────────────────────────────────────────────────────────┐
│             M:N モデル（Go の例）                       │
│                                                        │
│  goroutine  goroutine  goroutine  goroutine  goroutine │
│     G1         G2         G3         G4         G5     │
│     │          │          │          │          │      │
│     └────┬─────┘          └────┬─────┘          │      │
│          │                     │                │      │
│      ┌───┴───┐             ┌───┴───┐        ┌───┴───┐ │
│      │  P1   │             │  P2   │        │  P3   │ │
│      │(論理)  │             │(論理)  │        │(論理)  │ │
│      └───┬───┘             └───┬───┘        └───┬───┘ │
│          │                     │                │      │
│      ┌───┴───┐             ┌───┴───┐        ┌───┴───┐ │
│      │  M1   │             │  M2   │        │  M3   │ │
│      │(OS)   │             │(OS)   │        │(OS)   │ │
│      └───────┘             └───────┘        └───────┘ │
│                                                        │
│  G = Goroutine, P = Processor(論理), M = Machine(OS)   │
└────────────────────────────────────────────────────────┘
```

---

## 3. スレッドの基本操作

### 3.1 Python のスレッド

```python
# Python: threading モジュール
import threading
import time

def worker(name, delay):
    print(f"Thread {name} started")
    time.sleep(delay)  # I/O を模擬
    print(f"Thread {name} finished after {delay}s")

# スレッドの作成と実行
threads = []
for i in range(5):
    t = threading.Thread(target=worker, args=(f"Worker-{i}", i * 0.5))
    threads.append(t)
    t.start()

for t in threads:
    t.join()  # 全スレッドの完了を待つ

print("All threads completed")

# 注意: Python の GIL（Global Interpreter Lock）
# → CPython では1度に1スレッドしか Python コードを実行できない
# → CPU密集型処理には multiprocessing を使う
# → I/O密集型処理には threading が有効

# デーモンスレッド
daemon_thread = threading.Thread(target=worker, args=("Daemon", 10), daemon=True)
daemon_thread.start()
# メインスレッドが終了するとデーモンスレッドも自動終了

# スレッドプール（concurrent.futures）
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_url(url):
    """URL からデータを取得する（模擬）"""
    time.sleep(1)  # ネットワーク遅延を模擬
    return f"Data from {url}"

urls = [
    "https://api.example.com/users",
    "https://api.example.com/posts",
    "https://api.example.com/comments",
    "https://api.example.com/albums",
    "https://api.example.com/photos",
]

# 最大3スレッドのプールで並行実行
with ThreadPoolExecutor(max_workers=3) as executor:
    # submit: 個別にタスクを投入
    futures = {executor.submit(fetch_url, url): url for url in urls}

    for future in as_completed(futures):
        url = futures[future]
        try:
            data = future.result()
            print(f"{url}: {data}")
        except Exception as e:
            print(f"{url} generated an exception: {e}")

# map: 全タスクを一括投入
with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(fetch_url, urls))
    for url, data in zip(urls, results):
        print(f"{url}: {data}")
```

### 3.2 Python のマルチプロセス

```python
# Python: multiprocessing（GIL を回避する CPU 密集型処理向け）
from multiprocessing import Process, Pool, Queue, Value, Array
import os

def cpu_intensive_task(n):
    """CPU密集型の計算"""
    result = sum(i * i for i in range(n))
    return result

# Process の直接使用
def worker_process(name):
    pid = os.getpid()
    print(f"Process {name} (PID: {pid}) started")
    result = cpu_intensive_task(10_000_000)
    print(f"Process {name} result: {result}")

processes = []
for i in range(4):
    p = Process(target=worker_process, args=(f"P-{i}",))
    processes.append(p)
    p.start()

for p in processes:
    p.join()

# プロセスプール
with Pool(processes=4) as pool:
    # map: 同期的に全タスクを実行
    results = pool.map(cpu_intensive_task, [1_000_000] * 8)
    print(f"Results: {results}")

    # apply_async: 非同期にタスクを投入
    async_results = [
        pool.apply_async(cpu_intensive_task, (1_000_000,))
        for _ in range(8)
    ]
    results = [r.get(timeout=30) for r in async_results]
    print(f"Async results: {results}")

# プロセス間通信: Queue
def producer(queue):
    for i in range(10):
        queue.put(f"item-{i}")
    queue.put(None)  # 終了シグナル

def consumer(queue):
    while True:
        item = queue.get()
        if item is None:
            break
        print(f"Consumed: {item}")

q = Queue()
prod = Process(target=producer, args=(q,))
cons = Process(target=consumer, args=(q,))
prod.start()
cons.start()
prod.join()
cons.join()

# 共有メモリ: Value, Array
counter = Value('i', 0)  # 'i' = int型
shared_array = Array('d', [0.0] * 10)  # 'd' = double型

def increment(counter, lock):
    for _ in range(100000):
        with lock:
            counter.value += 1

lock = multiprocessing.Lock()
ps = [Process(target=increment, args=(counter, lock)) for _ in range(4)]
for p in ps:
    p.start()
for p in ps:
    p.join()
print(f"Counter: {counter.value}")  # → 400000
```

### 3.3 Rust のスレッド

```rust
// Rust: std::thread（OSスレッド）
use std::thread;
use std::time::Duration;

fn main() {
    // スレッドの生成
    let handle = thread::spawn(|| {
        println!("Hello from thread! ID: {:?}", thread::current().id());
        thread::sleep(Duration::from_millis(100));
        42  // スレッドの戻り値
    });

    let result = handle.join().unwrap();  // → 42
    println!("Thread returned: {}", result);

    // 複数スレッドの生成
    let handles: Vec<_> = (0..5)
        .map(|i| {
            thread::spawn(move || {
                println!("Thread {} started", i);
                thread::sleep(Duration::from_millis(100 * i as u64));
                println!("Thread {} finished", i);
                i * i
            })
        })
        .collect();

    let results: Vec<_> = handles
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    println!("Results: {:?}", results);  // [0, 1, 4, 9, 16]

    // スレッドビルダー（名前とスタックサイズの設定）
    let builder = thread::Builder::new()
        .name("custom-thread".into())
        .stack_size(32 * 1024);  // 32KB スタック

    let handle = builder.spawn(|| {
        let name = thread::current().name().unwrap_or("unnamed").to_string();
        println!("Running in thread: {}", name);
    }).unwrap();

    handle.join().unwrap();
}

// データの共有（Arc + Mutex）
use std::sync::{Arc, Mutex};

fn shared_counter_example() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
            // MutexGuard がドロップされるとロック解放
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
    println!("Result: {}", *counter.lock().unwrap());  // → 10
}

// RwLock（読み書きロック）
use std::sync::RwLock;

fn rwlock_example() {
    let config = Arc::new(RwLock::new(HashMap::new()));

    // 複数の読み取りスレッド
    let readers: Vec<_> = (0..5)
        .map(|i| {
            let config = Arc::clone(&config);
            thread::spawn(move || {
                let data = config.read().unwrap();
                println!("Reader {}: {:?}", i, *data);
            })
        })
        .collect();

    // 1つの書き込みスレッド
    {
        let config = Arc::clone(&config);
        thread::spawn(move || {
            let mut data = config.write().unwrap();
            data.insert("key", "value");
            println!("Writer: inserted data");
        })
        .join()
        .unwrap();
    }

    for r in readers {
        r.join().unwrap();
    }
}

// Atomic 操作（ロックフリー）
use std::sync::atomic::{AtomicI64, Ordering};

fn atomic_example() {
    let counter = Arc::new(AtomicI64::new(0));
    let handles: Vec<_> = (0..10)
        .map(|_| {
            let counter = Arc::clone(&counter);
            thread::spawn(move || {
                for _ in 0..1000 {
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }
    println!("Atomic counter: {}", counter.load(Ordering::Relaxed)); // → 10000
}
```

### 3.4 Go の goroutine

```go
package main

import (
    "fmt"
    "runtime"
    "sync"
    "sync/atomic"
    "time"
)

// Go: goroutine（軽量スレッド）
func worker(id int, wg *sync.WaitGroup) {
    defer wg.Done()
    fmt.Printf("Worker %d started (goroutine)\n", id)
    time.Sleep(time.Second)
    fmt.Printf("Worker %d done\n", id)
}

func main() {
    // 使用するCPUコア数を設定
    runtime.GOMAXPROCS(runtime.NumCPU())

    var wg sync.WaitGroup
    for i := 0; i < 5; i++ {
        wg.Add(1)
        go worker(i, &wg) // goroutine を起動（goキーワード）
    }
    wg.Wait() // 全goroutineの完了を待つ

    // goroutine は OSスレッドより遥かに軽量
    // → 数百万の goroutine を同時に実行可能
    // → Go ランタイムが M:N スケジューリング
    // → 1 goroutine のスタックは 2KB から開始（動的に成長）

    // 大量のgoroutineを生成する例
    var wg2 sync.WaitGroup
    count := int64(0)
    for i := 0; i < 100000; i++ {
        wg2.Add(1)
        go func() {
            defer wg2.Done()
            atomic.AddInt64(&count, 1)
        }()
    }
    wg2.Wait()
    fmt.Printf("Count: %d\n", count) // → 100000
}

// Mutex を使ったスレッドセーフなカウンター
type SafeCounter struct {
    mu sync.Mutex
    v  map[string]int
}

func (c *SafeCounter) Inc(key string) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.v[key]++
}

func (c *SafeCounter) Value(key string) int {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.v[key]
}

// RWMutex を使ったキャッシュ
type Cache struct {
    mu   sync.RWMutex
    data map[string]interface{}
}

func (c *Cache) Get(key string) (interface{}, bool) {
    c.mu.RLock()         // 読み取りロック（複数同時OK）
    defer c.mu.RUnlock()
    val, ok := c.data[key]
    return val, ok
}

func (c *Cache) Set(key string, value interface{}) {
    c.mu.Lock()          // 書き込みロック（排他）
    defer c.mu.Unlock()
    c.data[key] = value
}

// sync.Once: 一度だけ実行を保証
var (
    instance *Database
    once     sync.Once
)

type Database struct {
    // ...
}

func GetDB() *Database {
    once.Do(func() {
        instance = &Database{}
        fmt.Println("Database initialized")
    })
    return instance
}
```

### 3.5 Java のスレッド

```java
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

// Java: Thread クラスと Runnable インターフェース
public class ThreadExample {

    // 方法1: Thread クラスを継承
    static class MyThread extends Thread {
        @Override
        public void run() {
            System.out.println("Thread: " + getName() + " running");
        }
    }

    // 方法2: Runnable インターフェースを実装
    static class MyRunnable implements Runnable {
        @Override
        public void run() {
            System.out.println("Runnable running in: " + Thread.currentThread().getName());
        }
    }

    public static void main(String[] args) throws Exception {
        // Thread クラス
        MyThread t1 = new MyThread();
        t1.start();
        t1.join();

        // Runnable（ラムダ式）
        Thread t2 = new Thread(() -> {
            System.out.println("Lambda thread running");
        });
        t2.start();
        t2.join();

        // ExecutorService（スレッドプール）
        ExecutorService executor = Executors.newFixedThreadPool(4);

        // submit: Future を返す
        Future<Integer> future = executor.submit(() -> {
            Thread.sleep(1000);
            return 42;
        });
        System.out.println("Result: " + future.get()); // → 42

        // invokeAll: 全タスクの完了を待つ
        var tasks = List.of(
            (Callable<String>) () -> { Thread.sleep(100); return "A"; },
            (Callable<String>) () -> { Thread.sleep(200); return "B"; },
            (Callable<String>) () -> { Thread.sleep(300); return "C"; }
        );
        var futures = executor.invokeAll(tasks);
        for (var f : futures) {
            System.out.println(f.get());
        }

        executor.shutdown();
        executor.awaitTermination(10, TimeUnit.SECONDS);

        // Java 21+: Virtual Threads（Project Loom）
        // 軽量な仮想スレッド（goroutine に類似）
        try (var vExecutor = Executors.newVirtualThreadPerTaskExecutor()) {
            for (int i = 0; i < 100000; i++) {
                final int id = i;
                vExecutor.submit(() -> {
                    // 各仮想スレッドで実行
                    Thread.sleep(Duration.ofMillis(100));
                    return "VThread-" + id;
                });
            }
        }
    }
}

// AtomicInteger: ロックフリーなカウンター
class AtomicCounter {
    private final AtomicInteger count = new AtomicInteger(0);

    public void increment() {
        count.incrementAndGet();
    }

    public int get() {
        return count.get();
    }

    // CAS（Compare-And-Swap）操作
    public boolean compareAndSet(int expected, int newValue) {
        return count.compareAndSet(expected, newValue);
    }
}
```

### 3.6 C++ のスレッド

```cpp
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <future>
#include <vector>
#include <iostream>

// C++11 以降の std::thread
void basic_thread_example() {
    // スレッド生成
    std::thread t([] {
        std::cout << "Hello from thread " << std::this_thread::get_id() << std::endl;
    });
    t.join(); // 完了を待つ

    // detach: スレッドを切り離す（デーモンスレッド化）
    std::thread daemon([] {
        // バックグラウンド処理...
    });
    daemon.detach();
}

// Mutex による排他制御
std::mutex mtx;
int shared_counter = 0;

void increment_with_mutex() {
    for (int i = 0; i < 100000; i++) {
        std::lock_guard<std::mutex> lock(mtx); // RAII ロック
        shared_counter++;
    } // lock_guard のデストラクタでアンロック
}

// shared_mutex（読み書きロック、C++17）
std::shared_mutex rw_mutex;
std::map<std::string, std::string> cache;

std::string read_cache(const std::string& key) {
    std::shared_lock lock(rw_mutex); // 読み取りロック
    auto it = cache.find(key);
    return it != cache.end() ? it->second : "";
}

void write_cache(const std::string& key, const std::string& value) {
    std::unique_lock lock(rw_mutex); // 書き込みロック
    cache[key] = value;
}

// std::async と std::future
void async_example() {
    // 非同期タスクの実行
    auto future = std::async(std::launch::async, [] {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        return 42;
    });

    // 他の処理を行う...
    std::cout << "Waiting for result..." << std::endl;

    int result = future.get(); // 結果を取得（ブロッキング）
    std::cout << "Result: " << result << std::endl;
}

// atomic 操作
std::atomic<int> atomic_counter{0};

void atomic_increment() {
    for (int i = 0; i < 100000; i++) {
        atomic_counter.fetch_add(1, std::memory_order_relaxed);
    }
}
```

---

## 4. 同期プリミティブ

### 4.1 基本的な同期プリミティブ

```
Mutex（排他制御 / Mutual Exclusion）:
  1つのスレッドだけがリソースにアクセスできる

  Thread 1: lock() → [使用中] → unlock()
  Thread 2: lock() → [待機...] → [使用中] → unlock()

  用途: 共有リソースの排他的アクセス
  注意: ロックを忘れるとデータ競合、解放を忘れるとデッドロック

RWLock（読み書きロック）:
  読み取りは複数同時、書き込みは排他的

  Reader 1: read_lock() → [読取] → unlock()
  Reader 2: read_lock() → [読取] → unlock()  ← 同時にOK
  Writer:   write_lock() → [待機...] → [書込] → unlock()

  用途: 読み取りが多く書き込みが少ないデータ（設定情報、キャッシュ等）

Semaphore（セマフォ）:
  同時アクセス数を制限するカウンター
  acquire() でカウンタ減算（0 なら待機）
  release() でカウンタ加算

  例: 最大5つの同時DB接続
  Semaphore(5):
    Thread 1: acquire() → [接続中] → release()
    Thread 2: acquire() → [接続中] → release()
    ...
    Thread 6: acquire() → [待機...] → （スロットが空くまで）

Condition Variable（条件変数）:
  特定の条件が満たされるまでスレッドを待機させる
  wait(): 条件が満たされるまで待機
  notify_one(): 1つの待機中スレッドを起床
  notify_all(): 全待機中スレッドを起床

  用途: Producer-Consumer パターン、イベント待ち

Barrier（バリア）:
  指定数のスレッドが全員到達するまで全員が待機

  Thread 1: ──処理── barrier.wait() ──処理──
  Thread 2: ──処理── barrier.wait() ──処理──
  Thread 3: ──処理── barrier.wait() ──処理──
                     ↑ 全員が到達するまで待機
```

### 4.2 TypeScript / JavaScript での同期

```typescript
// JavaScript はシングルスレッドだが、並行処理の同期は必要

// Semaphore の実装
class Semaphore {
    private queue: Array<() => void> = [];
    private count: number;

    constructor(maxConcurrency: number) {
        this.count = maxConcurrency;
    }

    async acquire(): Promise<void> {
        if (this.count > 0) {
            this.count--;
            return;
        }
        return new Promise<void>((resolve) => {
            this.queue.push(resolve);
        });
    }

    release(): void {
        const next = this.queue.shift();
        if (next) {
            next();
        } else {
            this.count++;
        }
    }
}

// 使用例: API 呼び出しの並行数制限
async function fetchWithLimit(urls: string[], maxConcurrency: number) {
    const semaphore = new Semaphore(maxConcurrency);
    const results = await Promise.all(
        urls.map(async (url) => {
            await semaphore.acquire();
            try {
                const response = await fetch(url);
                return response.json();
            } finally {
                semaphore.release();
            }
        })
    );
    return results;
}

// Mutex の実装（async/await ベース）
class AsyncMutex {
    private locked = false;
    private queue: Array<() => void> = [];

    async lock(): Promise<() => void> {
        if (!this.locked) {
            this.locked = true;
            return () => this.unlock();
        }
        return new Promise<() => void>((resolve) => {
            this.queue.push(() => {
                resolve(() => this.unlock());
            });
        });
    }

    private unlock(): void {
        const next = this.queue.shift();
        if (next) {
            next();
        } else {
            this.locked = false;
        }
    }
}

// Web Workers（ブラウザでの並列処理）
// main.js
const worker = new Worker("worker.js");
worker.postMessage({ data: [1, 2, 3, 4, 5], operation: "sum" });
worker.onmessage = (event) => {
    console.log("Result from worker:", event.data);
};

// worker.js
// self.onmessage = (event) => {
//     const { data, operation } = event.data;
//     if (operation === "sum") {
//         const result = data.reduce((a, b) => a + b, 0);
//         self.postMessage(result);
//     }
// };

// SharedArrayBuffer と Atomics（共有メモリ）
const buffer = new SharedArrayBuffer(1024);
const view = new Int32Array(buffer);

// メインスレッド
Atomics.store(view, 0, 42);
Atomics.notify(view, 0);

// ワーカースレッド
// Atomics.wait(view, 0, 0); // 値が 0 でなくなるまで待機
// const value = Atomics.load(view, 0); // → 42
```

### 4.3 条件変数の実践

```python
import threading
from collections import deque

# Producer-Consumer パターン（条件変数使用）
class BoundedBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.lock = threading.Lock()
        self.not_full = threading.Condition(self.lock)
        self.not_empty = threading.Condition(self.lock)

    def produce(self, item):
        with self.not_full:
            while len(self.buffer) >= self.capacity:
                self.not_full.wait()  # バッファが空くまで待機
            self.buffer.append(item)
            self.not_empty.notify()   # 消費者を起床

    def consume(self):
        with self.not_empty:
            while len(self.buffer) == 0:
                self.not_empty.wait()  # アイテムが来るまで待機
            item = self.buffer.popleft()
            self.not_full.notify()     # 生産者を起床
            return item

# 使用例
buffer = BoundedBuffer(capacity=5)

def producer():
    for i in range(20):
        buffer.produce(f"item-{i}")
        print(f"Produced: item-{i}")

def consumer(name):
    for _ in range(10):
        item = buffer.consume()
        print(f"Consumer {name} consumed: {item}")

t1 = threading.Thread(target=producer)
t2 = threading.Thread(target=consumer, args=("A",))
t3 = threading.Thread(target=consumer, args=("B",))

t1.start(); t2.start(); t3.start()
t1.join(); t2.join(); t3.join()
```

---

## 5. データ競合と回避策

### 5.1 データ競合の原理

```
データ競合（Data Race）:
  2つ以上のスレッドが同じメモリに同時アクセスし、
  少なくとも1つが書き込みで、同期がない

  Thread 1: read(x)=0 → write(x)=1
  Thread 2: read(x)=0 → write(x)=1
  結果: x=1（期待は x=2）

具体的な競合シナリオ:

  Time →
  Thread 1:  read x (=0)  →  compute 0+1  →  write x (=1)
  Thread 2:       read x (=0)  →  compute 0+1  →  write x (=1)

  期待: x = 2
  実際: x = 1（Thread 2 が Thread 1 の書き込みを上書き）

  これが「TOCTOU（Time of Check to Time of Use）」問題

回避策:
  1. Mutex でアクセスを排他制御
  2. アトミック操作を使う
  3. データを共有しない（メッセージパッシング）
  4. 不変データを使う（関数型アプローチ）
  5. 言語の型システムで防止（Rust）
```

### 5.2 Rust のコンパイル時データ競合防止

```rust
// Rust: コンパイル時にデータ競合を防止
use std::thread;
use std::sync::{Arc, Mutex};

fn main() {
    let mut data = vec![1, 2, 3];

    // コンパイルエラー: 可変参照を複数スレッドに渡せない
    // thread::spawn(|| { data.push(4); });
    // thread::spawn(|| { data.push(5); });
    // エラー: `data` does not implement `Send`
    // → Rust の所有権システムがデータ競合をコンパイル時に防止

    // Arc<Mutex<T>> で安全に共有
    let data = Arc::new(Mutex::new(vec![1, 2, 3]));
    let d1 = Arc::clone(&data);
    let d2 = Arc::clone(&data);

    let h1 = thread::spawn(move || {
        d1.lock().unwrap().push(4);
    });
    let h2 = thread::spawn(move || {
        d2.lock().unwrap().push(5);
    });

    h1.join().unwrap();
    h2.join().unwrap();
    println!("{:?}", data.lock().unwrap()); // [1, 2, 3, 4, 5] or [1, 2, 3, 5, 4]

    // Send と Sync トレイト
    // Send: 値を別のスレッドに移動できる
    // Sync: 参照を別のスレッドと共有できる（&T が Send）
    //
    // ほとんどの型は Send + Sync
    // Rc<T> は Send でない（スレッドセーフでないため）
    // → Arc<T> を使う（アトミック参照カウント）
    //
    // Cell<T>, RefCell<T> は Sync でない
    // → Mutex<T>, RwLock<T> を使う
}
```

### 5.3 よくある並行処理のバグ

```
1. TOCTOU（Time of Check to Time of Use）:
   if (file.exists()) {     // チェック時点
       file.read();          // 使用時点 ← この間に別スレッドが削除する可能性
   }

2. ABA 問題:
   スレッド1: read A → (一時停止)
   スレッド2: A → B → A に変更
   スレッド1: read A → 「変わってない」と誤判断
   → CAS (Compare-And-Swap) では検出できない

3. メモリの可視性問題:
   各CPUコアにはキャッシュがある
   スレッド1 がメモリに書き込んでも、
   スレッド2 のキャッシュには反映されていない場合がある
   → メモリバリア / volatile / Atomic 操作で解決

4. 偽りの共有（False Sharing）:
   異なるスレッドが同じキャッシュライン上の異なる変数にアクセス
   → キャッシュの無効化が頻発してパフォーマンス低下
   → パディングでキャッシュラインを分離
```

---

## 6. デッドロックとその回避

### 6.1 デッドロックの条件

```
デッドロック: 2つ以上のスレッドが互いにロックの解放を待ち続ける状態

  Thread 1: lock(A) → lock(B) を待つ → 永遠に待つ
  Thread 2: lock(B) → lock(A) を待つ → 永遠に待つ

  ┌──────────┐  lock(B)を待つ  ┌──────────┐
  │ Thread 1 │ ─────────────→ │ Thread 2 │
  │ (A保持)  │ ←───────────── │ (B保持)  │
  └──────────┘  lock(A)を待つ  └──────────┘

デッドロックの4条件（Coffman条件）:
  1. 相互排除: リソースは一度に1スレッドのみ使用
  2. 保持と待機: リソースを保持したまま別のリソースを待つ
  3. 横取り不可: スレッドからリソースを強制的に奪えない
  4. 循環待機: スレッドの待機が循環する

→ 1つでも条件を壊せばデッドロックは発生しない
```

### 6.2 デッドロックの回避策

```python
import threading

# デッドロックの例
lock_a = threading.Lock()
lock_b = threading.Lock()

# BAD: デッドロックの可能性
def thread1_bad():
    lock_a.acquire()
    # time.sleep(0.001)  # タイミングによりデッドロック
    lock_b.acquire()
    # 処理...
    lock_b.release()
    lock_a.release()

def thread2_bad():
    lock_b.acquire()  # 逆順でロック取得
    lock_a.acquire()
    # 処理...
    lock_a.release()
    lock_b.release()

# 回避策1: ロック順序の統一
def thread1_good():
    lock_a.acquire()  # 常に A → B の順
    lock_b.acquire()
    # 処理...
    lock_b.release()
    lock_a.release()

def thread2_good():
    lock_a.acquire()  # 常に A → B の順（統一）
    lock_b.acquire()
    # 処理...
    lock_b.release()
    lock_a.release()

# 回避策2: タイムアウト付きロック
def thread_with_timeout():
    acquired_a = lock_a.acquire(timeout=1.0)
    if not acquired_a:
        return  # タイムアウト → リトライ

    acquired_b = lock_b.acquire(timeout=1.0)
    if not acquired_b:
        lock_a.release()  # 保持しているロックも解放
        return  # タイムアウト → リトライ

    try:
        # 処理...
        pass
    finally:
        lock_b.release()
        lock_a.release()

# 回避策3: context manager で安全にロック管理
from contextlib import contextmanager

@contextmanager
def acquire_locks(*locks):
    """複数のロックをソート済みの順序で取得"""
    sorted_locks = sorted(locks, key=id)
    acquired = []
    try:
        for lock in sorted_locks:
            lock.acquire()
            acquired.append(lock)
        yield
    finally:
        for lock in reversed(acquired):
            lock.release()

# 使用例
def safe_thread():
    with acquire_locks(lock_a, lock_b):
        # 処理...（ロック順序は自動的に統一される）
        pass
```

### 6.3 ライブロックと飢餓

```
ライブロック（Livelock）:
  スレッドがデッドロックを回避しようとして、
  互いにロックを譲り合い続ける状態
  （廊下ですれ違おうとして同じ方向に動き続ける状況）

  Thread 1: lock(A) → try lock(B) → fail → unlock(A) → retry
  Thread 2: lock(B) → try lock(A) → fail → unlock(B) → retry
  → 永遠にリトライし続ける

  対策: ランダムなバックオフ（リトライ前にランダムな時間待つ）

飢餓（Starvation）:
  特定のスレッドがリソースを永遠に取得できない状態
  例: 優先度の低いスレッドが常に後回しにされる

  対策:
  - フェアネスを保証するロック（FIFO キュー）
  - 優先度の逆転を防止するプロトコル
  - タイムアウトと再試行
```

---

## 7. スレッドプールとタスクベース並行処理

### 7.1 スレッドプールの概念

```
スレッドプール:
  事前にスレッドを生成しておき、タスクを投入して再利用する

  ┌──────────────────────────────────────┐
  │ Thread Pool                          │
  │                                      │
  │  ┌─────────────┐  ┌──────────────┐  │
  │  │ Task Queue   │  │ Worker Pool  │  │
  │  │              │  │              │  │
  │  │ [Task 1] ─────→ │ Thread 1 ○  │  │
  │  │ [Task 2] ─────→ │ Thread 2 ○  │  │
  │  │ [Task 3]    │  │ Thread 3 ●  │  │
  │  │ [Task 4]    │  │ Thread 4 ●  │  │
  │  │ ...         │  │              │  │
  │  └─────────────┘  └──────────────┘  │
  │  ○ = アイドル  ● = 実行中           │
  └──────────────────────────────────────┘

利点:
  - スレッド作成コストの削減（再利用）
  - 同時実行数の制御
  - リソースの効率的利用
  - タスクの優先度付きスケジューリング
```

### 7.2 Rust の rayon（データ並列処理）

```rust
// rayon: Rust のデータ並列処理ライブラリ
use rayon::prelude::*;

fn main() {
    let numbers: Vec<i64> = (0..10_000_000).collect();

    // 並列 map + sum
    let sum: i64 = numbers.par_iter()
        .map(|&n| n * n)
        .sum();

    // 並列 filter + collect
    let evens: Vec<i64> = numbers.par_iter()
        .filter(|&&n| n % 2 == 0)
        .copied()
        .collect();

    // 並列ソート
    let mut data = vec![5, 3, 8, 1, 9, 2, 7, 4, 6];
    data.par_sort();

    // カスタムスレッドプール
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();

    pool.install(|| {
        let result: i64 = numbers.par_iter().sum();
        println!("Sum: {}", result);
    });

    // 並列 for_each
    (0..100).into_par_iter().for_each(|i| {
        // 各イテレーションが並列に実行される
        println!("Processing: {}", i);
    });
}
```

### 7.3 Work Stealing

```
Work Stealing アルゴリズム:
  各ワーカースレッドがローカルキューを持ち、
  自分のキューが空になったら他のスレッドのキューからタスクを「盗む」

  Thread 1: [T1, T2, T3] → T1 を実行
  Thread 2: [T4]         → T4 を実行
  Thread 3: []           → Thread 1 から T3 を盗む

  利点:
  - 負荷分散が自動的に行われる
  - ローカルキューへのアクセスはロック不要（deque の両端操作）
  - 盗む操作は稀にしか発生しないためオーバーヘッドが小さい

  使用している言語/ライブラリ:
  - Go のランタイムスケジューラ
  - Rust の rayon
  - Java の ForkJoinPool
  - .NET の Task Parallel Library (TPL)
  - Tokio (Rust の async ランタイム)
```

---

## 8. プロセス間通信（IPC）

### 8.1 IPC の種類

```
┌────────────────┬───────────────────────────────────────┐
│ IPC 手法       │ 特徴                                   │
├────────────────┼───────────────────────────────────────┤
│ パイプ         │ 一方向、親子プロセス間、バイトストリーム│
│ 名前付きパイプ │ 双方向可能、非親子プロセス間            │
│ ソケット       │ ネットワーク越しも可能、双方向          │
│ 共有メモリ     │ 最高速、同期が必要                      │
│ メッセージキュー│ 非同期、メッセージ単位                 │
│ シグナル       │ 非同期通知、ペイロードなし              │
│ ファイル       │ 最も単純、パフォーマンスは低い          │
│ mmap          │ ファイルをメモリにマップ                │
│ Unix ドメイン  │ 同一マシン内、TCP/IP より高速          │
│  ソケット      │                                        │
└────────────────┴───────────────────────────────────────┘
```

### 8.2 パイプとソケットの例

```python
import subprocess
import socket
import os

# パイプ: サブプロセスとの通信
result = subprocess.run(
    ["ls", "-la"],
    capture_output=True,
    text=True,
)
print(result.stdout)

# パイプチェーン: ls | grep ".py" | wc -l
p1 = subprocess.Popen(["ls"], stdout=subprocess.PIPE)
p2 = subprocess.Popen(["grep", ".py"], stdin=p1.stdout, stdout=subprocess.PIPE)
p3 = subprocess.Popen(["wc", "-l"], stdin=p2.stdout, stdout=subprocess.PIPE)
p1.stdout.close()
p2.stdout.close()
output = p3.communicate()[0]
print(f"Python files: {output.decode().strip()}")

# os.pipe(): 低レベルなパイプ
read_fd, write_fd = os.pipe()

pid = os.fork()
if pid == 0:
    # 子プロセス
    os.close(read_fd)
    os.write(write_fd, b"Hello from child!")
    os.close(write_fd)
    os._exit(0)
else:
    # 親プロセス
    os.close(write_fd)
    message = os.read(read_fd, 1024)
    os.close(read_fd)
    os.waitpid(pid, 0)
    print(f"Parent received: {message.decode()}")

# Unix ドメインソケット
SOCKET_PATH = "/tmp/example.sock"

# サーバー
def server():
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server_socket.bind(SOCKET_PATH)
    server_socket.listen(1)

    conn, _ = server_socket.accept()
    data = conn.recv(1024)
    print(f"Server received: {data.decode()}")
    conn.send(b"Response from server")
    conn.close()
    server_socket.close()

# クライアント
def client():
    client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client_socket.connect(SOCKET_PATH)
    client_socket.send(b"Hello from client")
    response = client_socket.recv(1024)
    print(f"Client received: {response.decode()}")
    client_socket.close()
```

---

## 9. コンテキストスイッチとスケジューリング

### 9.1 コンテキストスイッチ

```
コンテキストスイッチ:
  CPU が実行するタスクを切り替える際に、
  現在のタスクの状態を保存し、次のタスクの状態を復元する操作

  プロセスのコンテキストスイッチ:
    1. CPU レジスタの保存
    2. プログラムカウンタの保存
    3. ページテーブルの切り替え（TLB フラッシュ）
    4. キャッシュの無効化（一部）
    → コスト: 数マイクロ秒

  スレッドのコンテキストスイッチ:
    1. CPU レジスタの保存
    2. スタックポインタの切り替え
    → コスト: 数百ナノ秒（ページテーブル切替なし）

  goroutine のコンテキストスイッチ:
    1. 少数のレジスタの保存
    2. スタックポインタの切り替え
    → コスト: 数十ナノ秒（カーネル介入なし）

コスト比較:
  プロセス切替: ~5-10 μs
  スレッド切替: ~0.5-1 μs
  goroutine切替: ~0.01-0.1 μs
  async タスク切替: ~0.001-0.01 μs
```

### 9.2 スケジューリングアルゴリズム

```
OS のスケジューリング:
  1. FCFS（First-Come, First-Served）: 到着順
  2. SJF（Shortest Job First）: 最短ジョブ優先
  3. 優先度スケジューリング: 優先度に基づく
  4. ラウンドロビン: タイムスライスで公平に切り替え
  5. CFS（Completely Fair Scheduler）: Linux のデフォルト
     - 各タスクの仮想実行時間を追跡
     - 最も実行時間が少ないタスクを次に実行
     - 赤黒木で管理（O(log n)）

Go のスケジューラ:
  GMP モデル:
  - G (Goroutine): 実行単位
  - M (Machine): OS スレッド
  - P (Processor): 論理プロセッサ

  特徴:
  - Work Stealing で負荷分散
  - プリエンプティブ（Go 1.14+）
  - ネットワーク I/O でのブロッキングを検出して
    goroutine を別の OS スレッドに移動
```

---

## 10. 実践的な並行処理パターン

### 10.1 Worker Pool パターン

```go
package main

import (
    "fmt"
    "sync"
)

// Worker Pool パターン
func workerPool() {
    const numWorkers = 4
    jobs := make(chan int, 100)
    results := make(chan int, 100)

    // ワーカーの起動
    var wg sync.WaitGroup
    for w := 0; w < numWorkers; w++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            for job := range jobs {
                result := job * job // 処理
                results <- result
            }
        }(w)
    }

    // ジョブの投入
    go func() {
        for i := 0; i < 20; i++ {
            jobs <- i
        }
        close(jobs)
    }()

    // 結果の収集
    go func() {
        wg.Wait()
        close(results)
    }()

    for result := range results {
        fmt.Println(result)
    }
}
```

### 10.2 Fan-out / Fan-in パターン

```go
// Fan-out: 1つの入力を複数のワーカーに分配
// Fan-in: 複数のワーカーの出力を1つにまとめる

func fanOutFanIn() {
    // 入力チャネル
    input := make(chan int)
    go func() {
        for i := 0; i < 100; i++ {
            input <- i
        }
        close(input)
    }()

    // Fan-out: 3つのワーカーに分配
    numWorkers := 3
    workers := make([]<-chan int, numWorkers)
    for i := 0; i < numWorkers; i++ {
        workers[i] = worker(input)
    }

    // Fan-in: 全ワーカーの出力を統合
    merged := merge(workers...)

    for result := range merged {
        fmt.Println(result)
    }
}

func worker(input <-chan int) <-chan int {
    output := make(chan int)
    go func() {
        defer close(output)
        for n := range input {
            output <- n * n // 処理
        }
    }()
    return output
}

func merge(channels ...<-chan int) <-chan int {
    var wg sync.WaitGroup
    merged := make(chan int)

    for _, ch := range channels {
        wg.Add(1)
        go func(c <-chan int) {
            defer wg.Done()
            for v := range c {
                merged <- v
            }
        }(ch)
    }

    go func() {
        wg.Wait()
        close(merged)
    }()

    return merged
}
```

### 10.3 Pipeline パターン

```go
// Pipeline: データが複数のステージを順に通過する

func pipeline() {
    // Stage 1: 数値の生成
    gen := func(nums ...int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for _, n := range nums {
                out <- n
            }
        }()
        return out
    }

    // Stage 2: 二乗
    square := func(in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for n := range in {
                out <- n * n
            }
        }()
        return out
    }

    // Stage 3: フィルタ（偶数のみ）
    filterEven := func(in <-chan int) <-chan int {
        out := make(chan int)
        go func() {
            defer close(out)
            for n := range in {
                if n%2 == 0 {
                    out <- n
                }
            }
        }()
        return out
    }

    // パイプラインの接続
    result := filterEven(square(gen(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)))

    for v := range result {
        fmt.Println(v) // 4, 16, 36, 64, 100
    }
}
```

---

## まとめ

| 概念 | 説明 | 代表言語 |
|------|------|---------|
| OSスレッド | カーネル管理、重い | C, Java, Rust |
| 軽量スレッド | ランタイム管理、軽い | Go(goroutine), Erlang |
| Virtual Threads | JVM 上の軽量スレッド | Java 21+ |
| Mutex | 排他制御 | 全言語 |
| RWLock | 読み書きロック | 全言語 |
| Semaphore | 同時アクセス数の制限 | 全言語 |
| Condition Variable | 条件待ちと通知 | 全言語 |
| Arc/共有所有権 | スレッド間でデータ共有 | Rust |
| Atomic | ロックフリーな同期 | C++, Rust, Java |
| GIL | Python の制約 | CPython |
| Thread Pool | スレッドの再利用 | Java, Python, C++ |
| Work Stealing | 動的な負荷分散 | Go, Rust(rayon), Java |

並行処理を正しく実装するための原則:

1. **共有可変状態を最小化する**: 不変データやメッセージパッシングを優先
2. **ロックの粒度を適切に設定する**: 粗すぎるとスループットが低下、細かすぎるとデッドロックのリスク
3. **ロック順序を統一する**: デッドロック防止の最も基本的な対策
4. **RAII でロックを管理する**: ロックの解放忘れを防ぐ
5. **適切な並行モデルを選択する**: I/Oバウンドなら async、CPUバウンドならスレッド/プロセス
6. **テストとデバッグ**: ThreadSanitizer、競合検出ツールを活用

---

## 次に読むべきガイド
-> [[01-async-await.md]] -- async/await

---

## 参考文献
1. Herlihy, M. & Shavit, N. "The Art of Multiprocessor Programming." 2nd Ed, 2020.
2. Klabnik, S. & Nichols, C. "The Rust Programming Language." Ch.16, 2023.
3. Goetz, B. "Java Concurrency in Practice." Addison-Wesley, 2006.
4. Pike, R. "Concurrency is not Parallelism." Go Blog, 2012.
5. Silberschatz, A. et al. "Operating System Concepts." 10th Ed, Wiley, 2018.
6. Love, R. "Linux Kernel Development." 3rd Ed, Addison-Wesley, 2010.
7. Butcher, P. "Seven Concurrency Models in Seven Weeks." Pragmatic, 2014.
8. Donovan, A. & Kernighan, B. "The Go Programming Language." Addison-Wesley, 2015.
