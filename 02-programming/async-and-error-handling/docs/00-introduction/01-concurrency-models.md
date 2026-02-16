# 並行モデル概要

> プログラムが「複数のことを同時に行う」ための3つの主要モデル: マルチスレッド、イベントループ、アクターモデル。それぞれの仕組み、利点、欠点を比較する。

## この章で学ぶこと

- [ ] 並行（Concurrency）と並列（Parallelism）の違いを理解する
- [ ] 3つの主要な並行モデルの特徴を把握する
- [ ] 各モデルが得意なユースケースを学ぶ
- [ ] CSP（Communicating Sequential Processes）モデルを理解する
- [ ] 実務での並行モデル選択基準を身につける

---

## 1. 並行 vs 並列

### 1.1 基本概念

```
並行（Concurrency）:
  → 複数のタスクを「切り替えながら」進める
  → 1つのCPUコアでも可能
  → 「構造」の問題

  コア1: [タスクA] [タスクB] [タスクA] [タスクC] [タスクB]

並列（Parallelism）:
  → 複数のタスクを「同時に」実行する
  → 複数のCPUコアが必要
  → 「実行」の問題

  コア1: [タスクA] [タスクA] [タスクA]
  コア2: [タスクB] [タスクB] [タスクB]
  コア3: [タスクC] [タスクC] [タスクC]

Rob Pike（Go設計者）:
  「Concurrency is about dealing with lots of things at once.
   Parallelism is about doing lots of things at once.」
  （並行性は多くのことを一度に扱うこと。
   並列性は多くのことを一度にやること。）
```

### 1.2 並行と並列の関係

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   並行だが並列でない:                                │
│   → シングルコアでタスクを切り替え                    │
│   → 例: Node.js のイベントループ                     │
│   → 1コアだが複数のリクエストを処理                   │
│                                                     │
│   並列だが並行でない:                                │
│   → SIMD（同一命令を複数データに適用）                │
│   → 例: GPU の行列演算                               │
│   → 同じ処理を並列実行（別々のタスクではない）         │
│                                                     │
│   並行かつ並列:                                      │
│   → マルチコアで複数タスクを同時実行                  │
│   → 例: Go の goroutine + マルチコア                 │
│   → 複数のタスクが複数コアで同時に進行                │
│                                                     │
│   どちらでもない:                                    │
│   → 単一タスクを単一コアで逐次実行                   │
│   → 例: 普通の for ループ                            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 1.3 並行性のレベル

```
レベル1: プロセスレベルの並行性
  → OS が複数プロセスをスケジューリング
  → プロセス間はメモリ空間が分離
  → IPC（Inter-Process Communication）で通信
  → 例: fork(), マルチプロセスWebサーバー（Apache prefork）

レベル2: スレッドレベルの並行性
  → 1プロセス内で複数スレッド
  → メモリ空間を共有 → 同期が必要
  → 例: Java スレッド、C++ std::thread、Python threading

レベル3: コルーチン/ファイバーレベルの並行性
  → ユーザー空間で切り替え（カーネル不要）
  → 非常に軽量
  → 例: Go goroutine、Kotlin coroutines、Python asyncio

レベル4: 命令レベルの並列性（ILP）
  → CPU がパイプライン/スーパースカラで命令を並列実行
  → プログラマから透明
  → 例: CPU のアウトオブオーダー実行

レベル5: データレベルの並列性（DLP）
  → 同一命令を複数データに適用
  → 例: SIMD（SSE, AVX）、GPU のCUDA/OpenCL
```

### 1.4 コードで見る並行と並列

```go
package main

import (
    "fmt"
    "runtime"
    "sync"
    "time"
)

func main() {
    // 並行だが並列でない（1コアのみ使用）
    runtime.GOMAXPROCS(1) // 使用するOSスレッド数を1に制限

    var wg sync.WaitGroup
    start := time.Now()

    for i := 0; i < 4; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            time.Sleep(100 * time.Millisecond)
            fmt.Printf("goroutine %d done at %v\n", id, time.Since(start))
        }(i)
    }
    wg.Wait()
    fmt.Printf("1 core: total %v\n\n", time.Since(start))
    // → 約100ms（I/O待ちは並行処理されるので合計は100ms程度）

    // 並行かつ並列（全コア使用）
    runtime.GOMAXPROCS(runtime.NumCPU())
    start = time.Now()

    for i := 0; i < 4; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            // CPU集約型処理
            sum := 0
            for j := 0; j < 100_000_000; j++ {
                sum += j
            }
            fmt.Printf("goroutine %d done at %v\n", id, time.Since(start))
        }(i)
    }
    wg.Wait()
    fmt.Printf("all cores: total %v\n", time.Since(start))
    // → CPU集約型は並列実行により高速化
}
```

```python
import asyncio
import time
import multiprocessing

# Pythonでの並行（asyncio）と並列（multiprocessing）

# 並行: I/O待ちを効率的に処理
async def concurrent_io():
    """シングルスレッドだが、I/O待ちの間に他タスクを処理"""
    start = time.time()

    async def fetch(name: str, delay: float) -> str:
        await asyncio.sleep(delay)  # I/O待ちをシミュレート
        return f"{name} done"

    # 4つのI/Oタスクを並行実行
    results = await asyncio.gather(
        fetch("A", 0.1),
        fetch("B", 0.2),
        fetch("C", 0.15),
        fetch("D", 0.1),
    )
    print(f"Concurrent I/O: {time.time() - start:.3f}s")
    # → 約0.2秒（最も遅いタスクの時間）
    return results

# 並列: CPU集約型を複数コアで処理
def cpu_heavy(n: int) -> int:
    """CPU集約的な処理"""
    return sum(range(n))

def parallel_cpu():
    """複数プロセスで並列実行"""
    start = time.time()
    with multiprocessing.Pool(4) as pool:
        results = pool.map(cpu_heavy, [10_000_000] * 4)
    print(f"Parallel CPU: {time.time() - start:.3f}s")
    return results

if __name__ == "__main__":
    asyncio.run(concurrent_io())
    parallel_cpu()
```

---

## 2. マルチスレッドモデル

### 2.1 基本構造

```
仕組み:
  → OSスレッドを複数生成
  → 共有メモリでデータを交換
  → ロック（Mutex）で排他制御

  Thread 1 ─────────────────────────→
  Thread 2 ─────────────────────────→
  Thread 3 ─────────────────────────→
       ↕ 共有メモリ ↕
  ┌──────────────────┐
  │  Shared State    │ ← Mutex でロック
  └──────────────────┘

利点:
  ✓ 真の並列実行（マルチコア活用）
  ✓ CPU集約型に適する
  ✓ OSが自動的にスケジューリング

欠点:
  ✗ 共有状態のロック管理が複雑
  ✗ デッドロック、レースコンディション
  ✗ スレッド生成のオーバーヘッド（~1MB/スレッド）
  ✗ デバッグ困難

代表: Java, C++, Python(GILあり), Rust
```

### 2.2 Javaのマルチスレッド

```java
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

public class MultithreadingExamples {

    // 基本: Thread クラス
    public void basicThread() {
        Thread thread = new Thread(() -> {
            System.out.println("Running in thread: " + Thread.currentThread().getName());
        });
        thread.start();
    }

    // スレッドプール: ExecutorService
    public void threadPool() {
        // CPU数に合わせたプールサイズ
        int poolSize = Runtime.getRuntime().availableProcessors();
        ExecutorService executor = Executors.newFixedThreadPool(poolSize);

        List<Future<String>> futures = new ArrayList<>();

        for (int i = 0; i < 10; i++) {
            final int taskId = i;
            futures.add(executor.submit(() -> {
                Thread.sleep(100);
                return "Task " + taskId + " completed";
            }));
        }

        // 結果の収集
        for (Future<String> future : futures) {
            try {
                System.out.println(future.get(5, TimeUnit.SECONDS));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        executor.shutdown();
    }

    // 共有状態の管理: synchronized
    private int counter = 0;

    public synchronized void incrementSafe() {
        counter++;
    }

    // より細かいロック制御: ReentrantLock
    private final ReentrantLock lock = new ReentrantLock();
    private final Map<String, String> sharedMap = new HashMap<>();

    public void updateMap(String key, String value) {
        lock.lock();
        try {
            sharedMap.put(key, value);
        } finally {
            lock.unlock(); // 必ずfinallyでロック解放
        }
    }

    // ロック不要: Atomic変数
    private final AtomicLong atomicCounter = new AtomicLong(0);

    public void atomicIncrement() {
        atomicCounter.incrementAndGet(); // CAS操作でスレッドセーフ
    }

    // Producer-Consumer パターン
    public void producerConsumer() {
        BlockingQueue<String> queue = new LinkedBlockingQueue<>(100);

        // Producer
        new Thread(() -> {
            for (int i = 0; i < 1000; i++) {
                try {
                    queue.put("Item " + i); // キューが満杯ならブロック
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }
        }).start();

        // Consumer
        new Thread(() -> {
            while (true) {
                try {
                    String item = queue.take(); // キューが空ならブロック
                    process(item);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }).start();
    }
}
```

### 2.3 Rustのスレッドモデル

```rust
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

// Rust: 所有権システムでスレッド安全性をコンパイル時に保証

fn basic_threading() {
    let mut handles = vec![];

    for i in 0..4 {
        let handle = thread::spawn(move || {
            println!("Thread {} running", i);
            i * 2
        });
        handles.push(handle);
    }

    let results: Vec<i32> = handles.into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    println!("Results: {:?}", results);
}

// Arc<Mutex<T>> で共有状態
fn shared_state() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
            // ロックはスコープ終了時に自動解放
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Counter: {}", *counter.lock().unwrap());
}

// RwLock: 読み取りは複数同時、書き込みは排他
fn read_write_lock() {
    let data = Arc::new(RwLock::new(vec![1, 2, 3]));

    // 複数の読み取りスレッド
    let mut readers = vec![];
    for i in 0..5 {
        let data = Arc::clone(&data);
        readers.push(thread::spawn(move || {
            let values = data.read().unwrap(); // 読み取りロック
            println!("Reader {}: {:?}", i, *values);
        }));
    }

    // 書き込みスレッド
    {
        let data = Arc::clone(&data);
        thread::spawn(move || {
            let mut values = data.write().unwrap(); // 書き込みロック
            values.push(4);
        }).join().unwrap();
    }

    for reader in readers {
        reader.join().unwrap();
    }
}

// コンパイルエラーの例: Rust が防ぐデータ競合
// fn compile_error() {
//     let mut data = vec![1, 2, 3];
//
//     // ❌ コンパイルエラー: data を複数スレッドに move できない
//     thread::spawn(|| { data.push(4); });
//     thread::spawn(|| { data.push(5); });
//
//     // → Arc<Mutex<Vec<i32>>> を使う必要がある
// }
```

### 2.4 マルチスレッドの危険性

```
デッドロック:
  Thread A: lock(X) → lock(Y)
  Thread B: lock(Y) → lock(X)
  → 互いに相手のロック解放を待ち続ける

  防止策:
  1. ロックの順序を統一する
  2. タイムアウト付きロック（tryLock）
  3. ロック階層プロトコル

レースコンディション:
  Thread A: read(x) → x + 1 → write(x)
  Thread B: read(x) → x + 1 → write(x)
  → x を同時に読み、同じ値 +1 を書き込む
  → x が1しか増えない（本来2増えるべき）

  防止策:
  1. Mutex / synchronized
  2. Atomic操作（CAS）
  3. 不変データ構造

Priority Inversion:
  → 低優先度スレッドがロックを保持
  → 高優先度スレッドがロック待ち
  → 中優先度スレッドが低優先度を横取り
  → 高優先度が永遠に待つ

  防止策:
  1. Priority Inheritance Protocol
  2. ロック保持時間の最小化
```

```java
// デッドロックの例
public class DeadlockExample {
    private final Object lockA = new Object();
    private final Object lockB = new Object();

    // ❌ デッドロックの可能性
    public void method1() {
        synchronized (lockA) {
            System.out.println("Thread 1: locked A");
            // この間に Thread 2 が lockB を取得
            synchronized (lockB) {
                System.out.println("Thread 1: locked B");
            }
        }
    }

    public void method2() {
        synchronized (lockB) {
            System.out.println("Thread 2: locked B");
            // lockA は Thread 1 が保持中 → デッドロック
            synchronized (lockA) {
                System.out.println("Thread 2: locked A");
            }
        }
    }

    // ✅ 修正: ロック順序を統一
    public void method1Fixed() {
        synchronized (lockA) {  // 常にAが先
            synchronized (lockB) {
                System.out.println("Thread 1: locked A, B");
            }
        }
    }

    public void method2Fixed() {
        synchronized (lockA) {  // 常にAが先
            synchronized (lockB) {
                System.out.println("Thread 2: locked A, B");
            }
        }
    }
}
```

---

## 3. イベントループモデル

### 3.1 基本構造

```
仕組み:
  → シングルスレッドでイベントキューを処理
  → I/Oはノンブロッキング（OSに委任）
  → I/O完了時にコールバックをキューに追加

  ┌──────────────────────────────────────┐
  │           イベントループ              │
  │  ┌──────────────────────────────┐   │
  │  │ 1. コールスタック実行         │   │
  │  │ 2. マイクロタスク処理         │   │
  │  │ 3. マクロタスク1つ実行        │   │
  │  │ 4. → 1に戻る                 │   │
  │  └──────────────────────────────┘   │
  │         ↑ 完了通知                   │
  │  ┌──────────────────────────────┐   │
  │  │ OS / libuv（I/O管理）         │   │
  │  │ ネットワーク、ファイル、タイマー│   │
  │  └──────────────────────────────┘   │
  └──────────────────────────────────────┘

利点:
  ✓ 共有状態のロック不要（シングルスレッド）
  ✓ 大量の同時接続に強い（C10K問題の解決）
  ✓ メモリ効率が良い

欠点:
  ✗ CPU集約型に不向き（イベントループをブロック）
  ✗ マルチコアを直接活用できない
  ✗ コールバック地獄（async/awaitで改善）

代表: JavaScript(Node.js/ブラウザ), Python(asyncio)
```

### 3.2 Node.js イベントループの詳細

```
Node.js イベントループのフェーズ:

  ┌───────────────────────┐
  │        timers          │ ← setTimeout, setInterval
  ├───────────────────────┤
  │    pending callbacks   │ ← 遅延したI/Oコールバック
  ├───────────────────────┤
  │     idle, prepare      │ ← 内部使用
  ├───────────────────────┤
  │         poll           │ ← I/Oイベント取得、コールバック実行
  ├───────────────────────┤
  │        check           │ ← setImmediate
  ├───────────────────────┤
  │    close callbacks     │ ← close イベント
  └───────────────────────┘

マイクロタスク vs マクロタスク:
  マイクロタスク（高優先度）:
    - Promise.then/catch/finally
    - process.nextTick（Node.js）
    - queueMicrotask
    → 各フェーズの間に全て処理される

  マクロタスク（通常優先度）:
    - setTimeout/setInterval
    - setImmediate
    - I/Oコールバック
    → 1つずつ処理される
```

```typescript
// イベントループの実行順序を確認
console.log('1. 同期コード');

setTimeout(() => {
  console.log('5. setTimeout (マクロタスク)');
}, 0);

Promise.resolve().then(() => {
  console.log('3. Promise (マイクロタスク)');
}).then(() => {
  console.log('4. Promise chain (マイクロタスク)');
});

process.nextTick(() => {
  console.log('2. nextTick (マイクロタスク、最優先)');
});

console.log('1.5. 同期コード2');

// 出力順序:
// 1. 同期コード
// 1.5. 同期コード2
// 2. nextTick (マイクロタスク、最優先)
// 3. Promise (マイクロタスク)
// 4. Promise chain (マイクロタスク)
// 5. setTimeout (マクロタスク)
```

### 3.3 ブラウザのイベントループ

```typescript
// ブラウザのイベントループは Node.js と少し異なる
// requestAnimationFrame は独自のタイミング

console.log('1. 同期');

requestAnimationFrame(() => {
  console.log('4. requestAnimationFrame（描画前）');
});

setTimeout(() => {
  console.log('5. setTimeout');
}, 0);

Promise.resolve().then(() => {
  console.log('2. Promise (マイクロタスク)');
});

queueMicrotask(() => {
  console.log('3. queueMicrotask');
});

// 出力順序:
// 1. 同期
// 2. Promise (マイクロタスク)
// 3. queueMicrotask
// 4. requestAnimationFrame（次の描画フレーム前）
// 5. setTimeout
```

### 3.4 イベントループのブロッキング検出

```typescript
// イベントループのブロッキングを検出
function detectEventLoopBlocking(): void {
  let lastCheck = Date.now();

  setInterval(() => {
    const now = Date.now();
    const lag = now - lastCheck - 100; // 100ms間隔で設定

    if (lag > 50) { // 50ms以上の遅延
      console.warn(`Event loop blocked for ${lag}ms`);
      // ここでスタックトレースを取得するなどの処理
    }

    lastCheck = now;
  }, 100);
}

// Node.js: monitorEventLoopDelay API (v11.10+)
import { monitorEventLoopDelay } from 'perf_hooks';

const h = monitorEventLoopDelay({ resolution: 20 });
h.enable();

setInterval(() => {
  console.log({
    min: h.min / 1e6,       // ナノ秒 → ミリ秒
    max: h.max / 1e6,
    mean: h.mean / 1e6,
    p99: h.percentile(99) / 1e6,
  });
  h.reset();
}, 5000);
```

### 3.5 Python asyncio イベントループ

```python
import asyncio
import time

# Python の asyncio イベントループ

# 基本的なコルーチン
async def fetch_data(name: str, delay: float) -> str:
    print(f"[{time.time():.3f}] {name}: start")
    await asyncio.sleep(delay)  # I/O待ちをシミュレート
    print(f"[{time.time():.3f}] {name}: done")
    return f"{name} result"

# タスクの並行実行
async def main():
    # asyncio.create_task で並行実行
    task1 = asyncio.create_task(fetch_data("API-1", 0.2))
    task2 = asyncio.create_task(fetch_data("API-2", 0.3))
    task3 = asyncio.create_task(fetch_data("DB", 0.1))

    # 全て完了を待つ
    results = await asyncio.gather(task1, task2, task3)
    print(f"All results: {results}")

    # タイムアウト付き
    try:
        result = await asyncio.wait_for(
            fetch_data("Slow API", 5.0),
            timeout=1.0
        )
    except asyncio.TimeoutError:
        print("Timeout!")

asyncio.run(main())

# カスタムイベントループの利用
# uvloop: libuv ベースの高速イベントループ
import uvloop

async def high_performance_main():
    # uvloop は CPython のデフォルトイベントループより2-4倍高速
    pass

uvloop.install()
asyncio.run(high_performance_main())
```

---

## 4. アクターモデル

### 4.1 基本構造

```
仕組み:
  → 全てが「アクター」（独立したプロセス）
  → アクター間はメッセージパッシングで通信
  → 共有状態なし（各アクターが自分の状態を持つ）

  ┌─────────┐  メッセージ  ┌─────────┐
  │ Actor A │────────────→│ Actor B │
  │ state_a │             │ state_b │
  └─────────┘             └─────────┘
       │                       │
       │  メッセージ            │ メッセージ
       ↓                       ↓
  ┌─────────┐             ┌─────────┐
  │ Actor C │             │ Actor D │
  │ state_c │             │ state_d │
  └─────────┘             └─────────┘

利点:
  ✓ 共有状態なし（ロック不要）
  ✓ 分散システムに自然に拡張
  ✓ 耐障害性（アクターの再起動）
  ✓ スケーラビリティ

欠点:
  ✗ メッセージパッシングのオーバーヘッド
  ✗ デバッグが難しい（非同期メッセージ）
  ✗ 学習コストが高い

代表: Erlang/Elixir(BEAM), Akka(Scala/Java)
```

### 4.2 Erlang/Elixir のアクターモデル

```elixir
# Elixir: アクターモデルの典型的な実装

# GenServer: 汎用サーバープロセス
defmodule CounterServer do
  use GenServer

  # クライアントAPI
  def start_link(initial_value \\ 0) do
    GenServer.start_link(__MODULE__, initial_value, name: __MODULE__)
  end

  def increment() do
    GenServer.cast(__MODULE__, :increment)  # 非同期メッセージ
  end

  def get_value() do
    GenServer.call(__MODULE__, :get_value)  # 同期メッセージ（応答待ち）
  end

  # サーバーコールバック
  @impl true
  def init(initial_value) do
    {:ok, initial_value}
  end

  @impl true
  def handle_cast(:increment, state) do
    {:noreply, state + 1}
  end

  @impl true
  def handle_call(:get_value, _from, state) do
    {:reply, state, state}
  end
end

# Supervisor: 監視ツリーによる耐障害性
defmodule MyApp.Supervisor do
  use Supervisor

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, :ok, opts)
  end

  @impl true
  def init(:ok) do
    children = [
      {CounterServer, 0},
      {WebSocketHandler, []},
      {DatabasePool, pool_size: 10},
    ]

    # one_for_one: 1つのプロセスがクラッシュしたら、それだけ再起動
    # one_for_all: 1つがクラッシュしたら、全て再起動
    # rest_for_one: クラッシュ以降の全てを再起動
    Supervisor.init(children, strategy: :one_for_one)
  end
end

# 大量のプロセスを生成（軽量: ~2KB/プロセス）
defmodule MassiveSpawn do
  def run(count) do
    pids = for _ <- 1..count do
      spawn(fn ->
        receive do
          {:ping, sender} -> send(sender, :pong)
        end
      end)
    end

    # 全プロセスにメッセージ送信
    for pid <- pids do
      send(pid, {:ping, self()})
    end

    # 全プロセスからの応答を受信
    for _ <- pids do
      receive do
        :pong -> :ok
      end
    end

    IO.puts("#{count} processes completed")
  end
end

# MassiveSpawn.run(1_000_000)  # 100万プロセスも実用的に動作
```

### 4.3 Akka（Scala/Java）

```scala
import akka.actor.{Actor, ActorSystem, Props, ActorRef}

// Akka: JVM上のアクターモデル

// アクター定義
class CounterActor extends Actor {
  private var count = 0

  def receive: Receive = {
    case "increment" =>
      count += 1

    case "get" =>
      sender() ! count  // 応答を送信

    case "reset" =>
      count = 0
  }
}

// 使用例
object Main extends App {
  val system = ActorSystem("MySystem")
  val counter = system.actorOf(Props[CounterActor], "counter")

  // メッセージ送信（非同期、Fire-and-forget）
  counter ! "increment"
  counter ! "increment"
  counter ! "increment"

  // 応答を待つ（Ask パターン）
  import akka.pattern.ask
  import scala.concurrent.duration._
  implicit val timeout: akka.util.Timeout = 5.seconds

  val future = counter ? "get"  // Future[Any] を返す
  future.foreach(println)       // 3
}

// 監視（Supervision）
class ParentActor extends Actor {
  import akka.actor.SupervisorStrategy._
  import scala.concurrent.duration._

  override val supervisorStrategy = OneForOneStrategy(
    maxNrOfRetries = 3,
    withinTimeRange = 1.minute
  ) {
    case _: ArithmeticException => Resume    // 再開
    case _: NullPointerException => Restart  // 再起動
    case _: Exception => Escalate            // 上位に委譲
  }

  val child: ActorRef = context.actorOf(Props[ChildActor], "child")

  def receive: Receive = {
    case msg => child forward msg
  }
}
```

### 4.4 アクターモデルのパターン

```
パターン1: Request-Reply
  Client ──Request──→ Actor ──Reply──→ Client
  → 同期的なやり取り（タイムアウト付き）

パターン2: Fire-and-Forget
  Sender ──Message──→ Actor
  → 応答不要、非同期

パターン3: Publish-Subscribe
  Publisher ──Message──→ EventBus ──→ Subscriber1
                                  ──→ Subscriber2
                                  ──→ Subscriber3

パターン4: Scatter-Gather
  Coordinator ──Task──→ Worker1 ──Result──→ Aggregator
              ──Task──→ Worker2 ──Result──→
              ──Task──→ Worker3 ──Result──→

パターン5: Pipeline
  Stage1 ──→ Stage2 ──→ Stage3 ──→ Stage4
  → 各ステージがアクター
  → バックプレッシャー可能

パターン6: Circuit Breaker
  → 障害が一定回数連続したらアクターへのメッセージを遮断
  → 一定時間後に再試行
  → 障害の連鎖を防止
```

---

## 5. CSP（Communicating Sequential Processes）

### 5.1 基本構造

```
仕組み:
  → 軽量スレッド（goroutine）× 多数
  → チャネルでデータを送受信
  → 「メモリを共有して通信するな、通信してメモリを共有しろ」

  goroutine 1 ───→ [channel] ───→ goroutine 2
  goroutine 3 ───→ [channel] ───→ goroutine 4

利点:
  ✓ 軽量（goroutine: ~2KB、スレッド: ~1MB）
  ✓ チャネルによる安全な通信
  ✓ ランタイムが自動スケジューリング
  ✓ マルチコアを自動活用

代表: Go, Clojure(core.async)
```

### 5.2 Go のチャネルパターン

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// パターン1: Generator（ジェネレーター）
func fibonacci(ctx context.Context) <-chan int {
    ch := make(chan int)
    go func() {
        defer close(ch)
        a, b := 0, 1
        for {
            select {
            case <-ctx.Done():
                return
            case ch <- a:
                a, b = b, a+b
            }
        }
    }()
    return ch
}

// パターン2: Fan-Out / Fan-In
func fanOut(input <-chan int, workers int) []<-chan int {
    channels := make([]<-chan int, workers)
    for i := 0; i < workers; i++ {
        channels[i] = process(input)
    }
    return channels
}

func fanIn(channels ...<-chan int) <-chan int {
    merged := make(chan int)
    var wg sync.WaitGroup

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

func process(input <-chan int) <-chan int {
    output := make(chan int)
    go func() {
        defer close(output)
        for v := range input {
            output <- v * 2 // 処理
        }
    }()
    return output
}

// パターン3: Pipeline
func source(nums ...int) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for _, n := range nums {
            out <- n
        }
    }()
    return out
}

func square(in <-chan int) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for n := range in {
            out <- n * n
        }
    }()
    return out
}

func filter(in <-chan int, predicate func(int) bool) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for n := range in {
            if predicate(n) {
                out <- n
            }
        }
    }()
    return out
}

func main() {
    // Pipeline: source → square → filter
    numbers := source(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    squared := square(numbers)
    even := filter(squared, func(n int) bool { return n%2 == 0 })

    for n := range even {
        fmt.Println(n) // 4, 16, 36, 64, 100
    }
}

// パターン4: Worker Pool
func workerPool(jobs <-chan int, results chan<- int, workerCount int) {
    var wg sync.WaitGroup

    for i := 0; i < workerCount; i++ {
        wg.Add(1)
        go func(workerID int) {
            defer wg.Done()
            for job := range jobs {
                result := processJob(job) // 実際の処理
                results <- result
            }
        }(i)
    }

    go func() {
        wg.Wait()
        close(results)
    }()
}

func processJob(job int) int {
    time.Sleep(10 * time.Millisecond) // 処理をシミュレート
    return job * 2
}

// パターン5: Select による多重化
func multiplexer(ctx context.Context) {
    ticker := time.NewTicker(1 * time.Second)
    defer ticker.Stop()

    heartbeat := make(chan struct{})
    data := make(chan string)

    for {
        select {
        case <-ctx.Done():
            fmt.Println("Shutting down")
            return
        case <-ticker.C:
            fmt.Println("Tick")
        case <-heartbeat:
            fmt.Println("Heartbeat received")
        case msg := <-data:
            fmt.Println("Data:", msg)
        }
    }
}

// パターン6: Rate Limiter（レート制限）
func rateLimiter(requests <-chan string, ratePerSecond int) <-chan string {
    output := make(chan string)
    ticker := time.NewTicker(time.Second / time.Duration(ratePerSecond))

    go func() {
        defer close(output)
        defer ticker.Stop()
        for req := range requests {
            <-ticker.C // レート制限
            output <- req
        }
    }()

    return output
}
```

### 5.3 CSPとアクターモデルの比較

```
                    CSP（Go）              アクターモデル（Erlang）
通信方法           チャネル（匿名）         メッセージ（アドレス指定）
プロセス識別       匿名（チャネルで接続）    PID / 名前
ブロック           送受信でブロック可能       非ブロッキング（メールボックス）
同期               同期通信がデフォルト       非同期通信がデフォルト
分散               標準では非対応            ネイティブサポート
耐障害性           手動実装が必要            Supervisor ツリー
バッファ           バッファ付きチャネル可     メールボックスは無制限

CSP の長所:
  → 同期通信でデータフローが明確
  → select による複数チャネルの待ち受け
  → チャネルの方向性（送信専用/受信専用）を型で表現

アクターモデルの長所:
  → 分散環境に自然に拡張
  → 耐障害性のフレームワークが充実
  → アクター単位でのGC・メモリ管理
```

---

## 6. その他の並行モデル

### 6.1 Software Transactional Memory（STM）

```
STM: データベースのトランザクションに似たメモリ操作

  atomically $ do
    balance1 <- readTVar account1
    balance2 <- readTVar account2
    writeTVar account1 (balance1 - 100)
    writeTVar account2 (balance2 + 100)
  -- コンフリクトがあれば自動的にリトライ

利点:
  ✓ 楽観的並行制御（ロック不要）
  ✓ コンポーザブル（トランザクションの合成が可能）
  ✓ デッドロックなし

欠点:
  ✗ 副作用のある処理には使えない
  ✗ リトライのオーバーヘッド
  ✗ ライブロック（常にコンフリクト）の可能性

代表: Haskell(STM), Clojure(Ref/STM)
```

```haskell
-- Haskell STM: 銀行口座の送金
import Control.Concurrent.STM

type Account = TVar Int

transfer :: Account -> Account -> Int -> STM ()
transfer from to amount = do
    fromBalance <- readTVar from
    toBalance <- readTVar to
    if fromBalance >= amount
        then do
            writeTVar from (fromBalance - amount)
            writeTVar to (toBalance + amount)
        else retry  -- 残高不足なら再試行を待つ

main :: IO ()
main = do
    account1 <- newTVarIO 1000
    account2 <- newTVarIO 500

    -- アトミックに実行（コンフリクト時は自動リトライ）
    atomically $ transfer account1 account2 200

    balance1 <- readTVarIO account1
    balance2 <- readTVarIO account2
    putStrLn $ "Account 1: " ++ show balance1  -- 800
    putStrLn $ "Account 2: " ++ show balance2  -- 700
```

### 6.2 Structured Concurrency（構造化並行性）

```
構造化並行性:
  → 並行タスクにスコープ（境界）を設ける
  → 親タスク終了時に子タスクも必ず終了
  → リソースリークを防ぐ
  → Kotlin coroutines, Swift concurrency, Java 21 で採用

通常の並行:
  func parent() {
    spawn(child1)  // 生成して忘れる → リークの可能性
    spawn(child2)  // 生成して忘れる
    return         // 子タスクの完了を待たない
  }

構造化並行:
  func parent() {
    taskGroup {
      spawn(child1)  // スコープ内で生成
      spawn(child2)
    }  // ← ここで全ての子タスクの完了を待つ
    // child1, child2 が完了するまで先に進まない
  }
```

```swift
// Swift: Structured Concurrency
func fetchDashboard() async throws -> Dashboard {
    // TaskGroup: 構造化された並行実行
    try await withThrowingTaskGroup(of: DashboardComponent.self) { group in
        group.addTask { try await fetchUser() }
        group.addTask { try await fetchOrders() }
        group.addTask { try await fetchNotifications() }

        var components: [DashboardComponent] = []
        for try await component in group {
            components.append(component)
        }
        // グループを抜ける時、全タスクは完了済み
        return Dashboard(components: components)
    }
}

// TaskGroup のキャンセル
func fetchWithCancellation() async throws -> Data {
    try await withThrowingTaskGroup(of: Data.self) { group in
        group.addTask {
            try await fetchFromServer1() // 遅い
        }
        group.addTask {
            try await fetchFromServer2() // 速い
        }

        // 最初に完了したものを使い、残りはキャンセル
        guard let result = try await group.next() else {
            throw FetchError.noResult
        }
        group.cancelAll() // 残りのタスクをキャンセル
        return result
    }
}
```

```kotlin
// Kotlin: Coroutines による構造化並行性
import kotlinx.coroutines.*

suspend fun fetchDashboard(): Dashboard = coroutineScope {
    // coroutineScope 内の全タスクが完了するまで待つ
    val userDeferred = async { fetchUser() }
    val ordersDeferred = async { fetchOrders() }
    val notifsDeferred = async { fetchNotifications() }

    Dashboard(
        user = userDeferred.await(),
        orders = ordersDeferred.await(),
        notifications = notifsDeferred.await()
    )
    // いずれかが例外を投げたら、他も自動キャンセル
}

// SupervisorScope: 子の失敗が他の子に影響しない
suspend fun resilientFetch(): Dashboard = supervisorScope {
    val user = async { fetchUser() }
    val orders = async {
        try { fetchOrders() } catch (e: Exception) { emptyList() }
    }
    val notifs = async {
        try { fetchNotifications() } catch (e: Exception) { emptyList() }
    }

    Dashboard(
        user = user.await(),
        orders = orders.await(),
        notifications = notifs.await()
    )
}
```

### 6.3 データ並列モデル

```
データ並列（Data Parallelism）:
  → 同じ操作を複数のデータに同時適用
  → GPU計算、SIMD命令、MapReduce

  CPU SIMD:
    通常:     a[0]*b[0]  a[1]*b[1]  a[2]*b[2]  a[3]*b[3]  (4回の乗算)
    SIMD:    [a[0] a[1] a[2] a[3]] * [b[0] b[1] b[2] b[3]]  (1命令で4乗算)

  GPU (CUDA):
    → 数千のコアが同じカーネルを実行
    → 行列演算、機械学習に最適

  MapReduce:
    Map:    [data1, data2, data3, ...] → [result1, result2, result3, ...]
    Reduce: [result1, result2, result3, ...] → finalResult
    → 分散環境でのデータ処理（Hadoop, Spark）
```

```python
# Python: データ並列の例

# 1. multiprocessing.Pool でデータ並列
from multiprocessing import Pool

def square(x):
    return x ** 2

with Pool(4) as p:
    results = p.map(square, range(100))
    # 100個のデータを4プロセスで並列処理

# 2. NumPy: ベクトル化（暗黙的なデータ並列）
import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])
c = a * b  # 要素ごとの乗算（SIMD活用）
# [10, 40, 90, 160, 250]

# 3. concurrent.futures: 高レベルAPI
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# I/O並行: ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=10) as executor:
    urls = ["https://api.example.com/1", "https://api.example.com/2"]
    futures = [executor.submit(fetch_url, url) for url in urls]
    results = [f.result() for f in futures]

# CPU並列: ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=4) as executor:
    data_chunks = [data[i::4] for i in range(4)]
    futures = [executor.submit(process_chunk, chunk) for chunk in data_chunks]
    results = [f.result() for f in futures]
```

---

## 7. 比較と選択

### 7.1 総合比較表

```
┌──────────────┬────────────┬────────────┬────────────┬────────────┐
│              │ マルチ     │ イベント   │ アクター   │ CSP        │
│              │ スレッド   │ ループ     │ モデル     │            │
├──────────────┼────────────┼────────────┼────────────┼────────────┤
│ CPU集約      │ ◎         │ △         │ ○         │ ◎         │
├──────────────┼────────────┼────────────┼────────────┼────────────┤
│ I/O集約      │ ○         │ ◎         │ ◎         │ ◎         │
├──────────────┼────────────┼────────────┼────────────┼────────────┤
│ 同時接続数   │ ~千        │ ~十万      │ ~百万      │ ~百万      │
├──────────────┼────────────┼────────────┼────────────┼────────────┤
│ 安全性       │ 低(ロック) │ 中         │ 高         │ 高         │
├──────────────┼────────────┼────────────┼────────────┼────────────┤
│ デバッグ     │ 困難       │ 中程度     │ 困難       │ 中程度     │
├──────────────┼────────────┼────────────┼────────────┼────────────┤
│ メモリ効率   │ 低         │ 高         │ 高         │ 高         │
├──────────────┼────────────┼────────────┼────────────┼────────────┤
│ 分散対応     │ 手動       │ 手動       │ ネイティブ │ 手動       │
├──────────────┼────────────┼────────────┼────────────┼────────────┤
│ 学習コスト   │ 中         │ 低         │ 高         │ 中         │
└──────────────┴────────────┴────────────┴────────────┴────────────┘

選択指針:
  Web API（I/O集約）→ イベントループ or CSP
  リアルタイム通信 → アクターモデル
  画像/動画処理 → マルチスレッド
  マイクロサービス → アクターモデル or CSP
  分散システム → アクターモデル
  高パフォーマンスサーバー → CSP（Go）
  フロントエンド → イベントループ（JS）
```

### 7.2 ユースケース別推奨モデル

```typescript
// ユースケース1: REST API サーバー
// → イベントループ（Node.js）or CSP（Go）

// Node.js: 高い生産性、NPMエコシステム
import express from 'express';
const app = express();

app.get('/api/users/:id', async (req, res) => {
  const user = await db.getUser(req.params.id);
  res.json(user);
});

// Go: 高パフォーマンス、静的型付け
// func handleUser(w http.ResponseWriter, r *http.Request) {
//     user, err := db.GetUser(r.PathValue("id"))
//     json.NewEncoder(w).Encode(user)
// }
```

```elixir
# ユースケース2: リアルタイムチャット
# → アクターモデル（Elixir/Phoenix）

# 各チャットルームがアクター（プロセス）
defmodule ChatRoom do
  use GenServer

  def start_link(room_id) do
    GenServer.start_link(__MODULE__, room_id, name: via(room_id))
  end

  def join(room_id, user_id) do
    GenServer.call(via(room_id), {:join, user_id})
  end

  def send_message(room_id, user_id, message) do
    GenServer.cast(via(room_id), {:message, user_id, message})
  end

  # 状態: 参加者リスト
  def init(room_id) do
    {:ok, %{room_id: room_id, members: MapSet.new()}}
  end

  def handle_call({:join, user_id}, _from, state) do
    new_state = %{state | members: MapSet.put(state.members, user_id)}
    {:reply, :ok, new_state}
  end

  def handle_cast({:message, user_id, message}, state) do
    # 全メンバーにブロードキャスト
    Enum.each(state.members, fn member ->
      send_to_user(member, %{from: user_id, text: message})
    end)
    {:noreply, state}
  end

  defp via(room_id), do: {:via, Registry, {ChatRegistry, room_id}}
end
```

```go
// ユースケース3: データパイプライン
// → CSP（Go）

package main

import (
    "encoding/json"
    "fmt"
    "sync"
)

// ETL パイプライン: Extract → Transform → Load
type Record struct {
    ID   int
    Data string
}

func extract(source string) <-chan Record {
    out := make(chan Record, 100) // バッファ付きチャネル
    go func() {
        defer close(out)
        // データソースから読み取り
        for i := 0; i < 1000; i++ {
            out <- Record{ID: i, Data: fmt.Sprintf("raw-%d", i)}
        }
    }()
    return out
}

func transform(in <-chan Record, workers int) <-chan Record {
    out := make(chan Record, 100)
    var wg sync.WaitGroup

    for i := 0; i < workers; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for record := range in {
                // データ変換
                record.Data = fmt.Sprintf("transformed-%s", record.Data)
                out <- record
            }
        }()
    }

    go func() {
        wg.Wait()
        close(out)
    }()
    return out
}

func load(in <-chan Record) {
    for record := range in {
        // データベースに書き込み
        data, _ := json.Marshal(record)
        fmt.Println(string(data))
    }
}

func main() {
    // パイプライン構築
    records := extract("data-source")
    transformed := transform(records, 4) // 4ワーカーで並行変換
    load(transformed)
}
```

### 7.3 実務での選択フローチャート

```
要件を確認:

1. 何が支配的か？
   ├── I/O集約型（API、DB、ファイル）
   │   ├── 同時接続数 < 1,000 → 何でもOK
   │   ├── 同時接続数 1,000-10,000
   │   │   ├── チーム経験: JS → Node.js（イベントループ）
   │   │   ├── チーム経験: Python → asyncio
   │   │   └── パフォーマンス重視 → Go（CSP）
   │   └── 同時接続数 > 10,000
   │       ├── Go（CSP）
   │       └── Erlang/Elixir（アクターモデル）
   │
   └── CPU集約型（計算、画像処理、ML）
       ├── 単純並列 → マルチスレッド（Rust, C++, Java）
       ├── データ並列 → GPU / SIMD
       └── パイプライン → Go (CSP) or スレッド + キュー

2. 分散が必要か？
   ├── はい → アクターモデル（Erlang/Elixir, Akka）
   └── いいえ → 他のモデルで十分

3. 耐障害性が重要か？
   ├── はい → アクターモデル（Supervisor ツリー）
   └── いいえ → 他のモデルで十分

4. リアルタイム要件は？
   ├── WebSocket / SSE → イベントループ or アクターモデル
   └── REST API → イベントループ or CSP
```

---

## 8. ハイブリッドアプローチ

### 8.1 Node.js: イベントループ + Worker Threads

```typescript
// Node.js: I/Oはイベントループ、CPUはWorker Threads
import { Worker, isMainThread, parentPort, workerData } from 'worker_threads';
import { cpus } from 'os';

// メインスレッド: リクエスト処理（I/O）
if (isMainThread) {
  const numCPUs = cpus().length;
  const workerPool: Worker[] = [];
  const taskQueue: { data: any; resolve: Function; reject: Function }[] = [];

  // ワーカープールの初期化
  for (let i = 0; i < numCPUs; i++) {
    const worker = new Worker(__filename);
    worker.on('message', (result) => {
      // 次のタスクを処理
      const nextTask = taskQueue.shift();
      if (nextTask) {
        worker.postMessage(nextTask.data);
      }
    });
    workerPool.push(worker);
  }

  // CPU集約型タスクをワーカーに委譲
  function offloadToWorker(data: any): Promise<any> {
    return new Promise((resolve, reject) => {
      const freeWorker = workerPool.find(w => !w.isBusy);
      if (freeWorker) {
        freeWorker.postMessage(data);
      } else {
        taskQueue.push({ data, resolve, reject });
      }
    });
  }

  // Express サーバー
  import express from 'express';
  const app = express();

  app.post('/api/process-image', async (req, res) => {
    // I/O: リクエスト受信（イベントループ）
    const image = await receiveUpload(req);

    // CPU: 画像処理（ワーカースレッド）
    const processed = await offloadToWorker({ type: 'resize', image });

    // I/O: S3にアップロード（イベントループ）
    const url = await uploadToS3(processed);

    res.json({ url });
  });
}

// ワーカースレッド: CPU集約型処理
if (!isMainThread) {
  parentPort?.on('message', (data) => {
    const result = heavyComputation(data);
    parentPort?.postMessage(result);
  });
}
```

### 8.2 Python: asyncio + multiprocessing

```python
import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# CPU集約型処理
def cpu_bound_task(data: bytes) -> bytes:
    """画像処理などCPU集約型（別プロセスで実行）"""
    import hashlib
    result = hashlib.pbkdf2_hmac('sha256', data, b'salt', 100000)
    return result

# I/O集約型処理
async def io_bound_task(url: str) -> dict:
    """API呼び出しなどI/O集約型（asyncioで実行）"""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()

# ハイブリッド: asyncio + ProcessPoolExecutor
async def hybrid_handler(request_data: dict) -> dict:
    loop = asyncio.get_event_loop()

    # I/O: 非同期で外部データ取得
    external_data = await io_bound_task("https://api.example.com/data")

    # CPU: 別プロセスで重い計算
    with ProcessPoolExecutor(max_workers=4) as pool:
        computed = await loop.run_in_executor(
            pool,
            cpu_bound_task,
            external_data["payload"].encode()
        )

    # I/O: 結果を保存
    await save_result(computed)

    return {"status": "ok", "hash": computed.hex()}
```

---

## まとめ

| モデル | 核心 | 代表言語 | 適用場面 |
|--------|------|---------|----------|
| マルチスレッド | 共有メモリ + ロック | Java, C++, Rust | CPU集約型、レガシーシステム |
| イベントループ | シングルスレッド + 非同期I/O | JS, Python | Web API、フロントエンド |
| アクターモデル | メッセージパッシング | Erlang, Elixir | 分散、リアルタイム、高耐障害 |
| CSP | 軽量スレッド + チャネル | Go | 高パフォーマンスサーバー |
| STM | トランザクショナルメモリ | Haskell, Clojure | 複雑な共有状態管理 |
| 構造化並行性 | スコープ付き並行 | Kotlin, Swift, Java21 | モダンアプリケーション |

### 選択の原則

```
1. シンプルさを優先する
   → 必要以上に複雑なモデルを選ばない
   → イベントループで十分ならマルチスレッドは不要

2. チームのスキルセットに合わせる
   → 慣れた言語・モデルの方が生産性が高い
   → 新しいモデルの導入は十分な学習期間を設ける

3. ボトルネックに合わせる
   → I/O集約 → イベントループ / CSP
   → CPU集約 → マルチスレッド / データ並列
   → 混合 → ハイブリッドアプローチ

4. スケーラビリティ要件を考慮する
   → 垂直スケール → マルチスレッド
   → 水平スケール → アクターモデル / CSP
```

---

## 次に読むべきガイド
→ [[../01-async-patterns/00-callbacks.md]] — コールバック

---

## 参考文献
1. Hoare, C.A.R. "Communicating Sequential Processes." 1978.
2. Hewitt, C. "A Universal Modular Actor Formalism." 1973.
3. Pike, R. "Concurrency Is Not Parallelism." Waza Conference, 2012.
4. Armstrong, J. "Programming Erlang: Software for a Concurrent World." Pragmatic Bookshelf, 2013.
5. Goetz, B. "Java Concurrency in Practice." Addison-Wesley, 2006.
6. Klabnik, S. & Nichols, C. "The Rust Programming Language." No Starch Press, 2019.
7. Donovan, A. & Kernighan, B. "The Go Programming Language." Addison-Wesley, 2015.
8. Elizarov, R. "Structured Concurrency." Kotlin Blog, 2018.
9. Apple Developer Documentation. "Swift Concurrency."
10. Peierls, T. "STM in Haskell." Journal of Functional Programming, 2005.
