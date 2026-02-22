# 並行性 — スレッド、Mutex/RwLock、rayon

> Rust の所有権システムによるコンパイル時データ競合防止と、スレッド/ロック/並列イテレータによる実践的な並行処理を習得する

## この章で学ぶこと

1. **スレッドと所有権** — std::thread、Send/Sync トレイト、スコープドスレッド
2. **同期プリミティブ** — Mutex, RwLock, Condvar, Atomic, Barrier
3. **データ並列処理** — rayon による並列イテレータと作業分割
4. **チャネルによるメッセージパッシング** — mpsc, crossbeam-channel
5. **高度な並行パターン** — ロックフリーデータ構造、Producer-Consumer、ワークスティーリング

---

## 1. Rust の並行安全性モデル

```
┌──────────── Rust の並行安全性保証 ────────────┐
│                                                │
│  コンパイル時チェック                           │
│  ┌──────────────────────────────────────────┐  │
│  │  Send: 型 T を別スレッドに移動できる      │  │
│  │  Sync: &T を複数スレッドから共有できる    │  │
│  │                                          │  │
│  │  Send + Sync の例:                       │  │
│  │    i32, String, Vec<T>, Arc<T>           │  │
│  │                                          │  │
│  │  !Send の例:                             │  │
│  │    Rc<T>, *mut T                         │  │
│  │                                          │  │
│  │  !Sync の例:                             │  │
│  │    Cell<T>, RefCell<T>                   │  │
│  └──────────────────────────────────────────┘  │
│                                                │
│  → データ競合はコンパイルエラー               │
│  → デッドロックは防げない (論理エラー)         │
└────────────────────────────────────────────────┘
```

### Send / Sync の詳細ルール

```rust
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::Arc;

// Send: 型 T を別スレッドに「所有権ごと移動」できる
// Sync: &T を複数スレッドで「同時に参照」できる

// 基本ルール:
// T: Sync  ⟺  &T: Send
// つまり「共有参照を送れる」＝「共有できる」

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

fn check_traits() {
    // プリミティブ: Send + Sync
    assert_send::<i32>();
    assert_sync::<i32>();
    assert_send::<String>();
    assert_sync::<String>();

    // Arc<T>: T が Send + Sync なら Send + Sync
    assert_send::<Arc<i32>>();
    assert_sync::<Arc<i32>>();

    // Rc<T>: !Send, !Sync (参照カウントが非Atomic)
    // assert_send::<Rc<i32>>();  // コンパイルエラー!
    // assert_sync::<Rc<i32>>();  // コンパイルエラー!

    // Cell<T>: Send だが !Sync (内部可変性がスレッド安全でない)
    assert_send::<Cell<i32>>();
    // assert_sync::<Cell<i32>>();  // コンパイルエラー!

    // Mutex<T>: T が Send なら Send + Sync
    assert_send::<std::sync::Mutex<Vec<i32>>>();
    assert_sync::<std::sync::Mutex<Vec<i32>>>();
}

fn main() {
    check_traits();
    println!("全ての型チェックが通過");
}
```

### コード例: Send/Sync の実用的な意味

```rust
use std::rc::Rc;
use std::sync::Arc;

fn main() {
    // NG: Rc をスレッドに渡す → コンパイルエラー
    // let rc = Rc::new(42);
    // std::thread::spawn(move || {
    //     println!("{}", rc);
    // });
    // error: `Rc<i32>` cannot be sent between threads safely

    // OK: Arc を使えばスレッド間で共有可能
    let arc = Arc::new(42);
    let arc_clone = Arc::clone(&arc);
    let handle = std::thread::spawn(move || {
        println!("別スレッド: {}", arc_clone);
    });
    handle.join().unwrap();
    println!("メイン: {}", arc);

    // unsafe impl Send/Sync で手動マーク (危険!)
    // 自分で安全性を保証する場合のみ使用
    struct MyWrapper(*mut u8);
    unsafe impl Send for MyWrapper {}
    unsafe impl Sync for MyWrapper {}
}
```

---

## 2. スレッド基本操作

### コード例1: スレッドの生成と join

```rust
use std::thread;
use std::time::Duration;

fn main() {
    // スレッドの生成
    let handle = thread::spawn(|| {
        for i in 1..=5 {
            println!("[子スレッド] カウント: {}", i);
            thread::sleep(Duration::from_millis(100));
        }
        42 // 戻り値
    });

    // メインスレッドの処理
    for i in 1..=3 {
        println!("[メイン] カウント: {}", i);
        thread::sleep(Duration::from_millis(150));
    }

    // join で完了を待ち、戻り値を取得
    let result = handle.join().unwrap();
    println!("子スレッドの戻り値: {}", result);

    // スレッドビルダーで名前・スタックサイズを指定
    let builder = thread::Builder::new()
        .name("worker-1".into())
        .stack_size(4 * 1024 * 1024); // 4MB

    let handle = builder.spawn(|| {
        println!("スレッド名: {:?}", thread::current().name());
    }).unwrap();
    handle.join().unwrap();
}
```

### コード例2: スコープドスレッド (Rust 1.63+)

```rust
use std::thread;

fn main() {
    let data = vec![1, 2, 3, 4, 5];
    let mut results = vec![0; 5];

    // scope 内のスレッドはスコープ終了時に自動 join
    // → ローカル変数への参照を安全に渡せる
    thread::scope(|s| {
        // データの読み取りスレッド
        s.spawn(|| {
            let sum: i32 = data.iter().sum();
            println!("合計: {}", sum);
        });

        // データの変更スレッド (分割借用)
        for (i, slot) in results.iter_mut().enumerate() {
            s.spawn(move || {
                *slot = data[i] * data[i];
            });
        }
    }); // ← 全スレッドが完了するまでブロック

    println!("二乗: {:?}", results); // [1, 4, 9, 16, 25]
}
```

### コード例: スレッドプールの自作

```rust
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

type Job = Box<dyn FnOnce() + Send + 'static>;

struct ThreadPool {
    workers: Vec<Worker>,
    sender: Option<mpsc::Sender<Job>>,
}

struct Worker {
    id: usize,
    handle: Option<thread::JoinHandle<()>>,
}

impl ThreadPool {
    fn new(size: usize) -> Self {
        assert!(size > 0, "スレッド数は1以上が必要");

        let (sender, receiver) = mpsc::channel::<Job>();
        let receiver = Arc::new(Mutex::new(receiver));

        let mut workers = Vec::with_capacity(size);
        for id in 0..size {
            let receiver = Arc::clone(&receiver);
            let handle = thread::Builder::new()
                .name(format!("pool-worker-{}", id))
                .spawn(move || loop {
                    let job = receiver.lock().unwrap().recv();
                    match job {
                        Ok(job) => {
                            println!("[Worker {}] ジョブ実行開始", id);
                            job();
                        }
                        Err(_) => {
                            println!("[Worker {}] チャネル閉鎖、終了", id);
                            break;
                        }
                    }
                })
                .unwrap();

            workers.push(Worker {
                id,
                handle: Some(handle),
            });
        }

        ThreadPool {
            workers,
            sender: Some(sender),
        }
    }

    fn execute<F>(&self, f: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let job = Box::new(f);
        self.sender.as_ref().unwrap().send(job).unwrap();
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // sender を drop → チャネルを閉じる → worker がループを抜ける
        drop(self.sender.take());

        for worker in &mut self.workers {
            println!("Worker {} の終了を待機...", worker.id);
            if let Some(handle) = worker.handle.take() {
                handle.join().unwrap();
            }
        }
    }
}

fn main() {
    let pool = ThreadPool::new(4);

    for i in 0..8 {
        pool.execute(move || {
            println!("タスク {} 実行中 (スレッド: {:?})", i, thread::current().name());
            thread::sleep(std::time::Duration::from_millis(100));
            println!("タスク {} 完了", i);
        });
    }

    // pool が drop される → 全タスク完了を待機
    drop(pool);
    println!("全タスク完了");
}
```

---

## 3. 同期プリミティブ

### コード例3: Mutex と RwLock

```rust
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

fn main() {
    // Mutex — 排他ロック (読み書き両方を1スレッドに制限)
    let counter = Arc::new(Mutex::new(0u64));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        handles.push(thread::spawn(move || {
            for _ in 0..1000 {
                let mut num = counter.lock().unwrap();
                *num += 1;
                // num が drop されると自動的にロック解放
            }
        }));
    }

    for h in handles { h.join().unwrap(); }
    println!("カウンタ: {}", *counter.lock().unwrap()); // 10000

    // RwLock — 読み取りは並行、書き込みは排他
    let config = Arc::new(RwLock::new(String::from("初期設定")));

    let config_reader = Arc::clone(&config);
    let reader = thread::spawn(move || {
        let cfg = config_reader.read().unwrap(); // 読み取りロック
        println!("設定: {}", *cfg);
    });

    let config_writer = Arc::clone(&config);
    let writer = thread::spawn(move || {
        let mut cfg = config_writer.write().unwrap(); // 書き込みロック
        *cfg = "更新された設定".to_string();
    });

    reader.join().unwrap();
    writer.join().unwrap();
}
```

### ロックの動作

```
┌─────────── Mutex vs RwLock ─────────────┐
│                                          │
│  Mutex:                                  │
│    Thread A: [LOCK████████UNLOCK]       │
│    Thread B:      [wait][LOCK████UNLOCK]│
│    Thread C:           [wait....][LOCK] │
│    → 同時に1つだけ                       │
│                                          │
│  RwLock:                                 │
│    Reader A: [RLOCK██████████RUNLOCK]    │
│    Reader B: [RLOCK██████████RUNLOCK]    │
│    Reader C: [RLOCK██████████RUNLOCK]    │
│    → 読み取りは並行OK                    │
│                                          │
│    Writer:   [wait.........][WLOCK██UNL]│
│    → 書き込みは排他 (全reader完了待ち)   │
└──────────────────────────────────────────┘
```

### コード例: Mutex のポイズニング処理

```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let data = Arc::new(Mutex::new(vec![1, 2, 3]));

    // パニックするスレッドを生成
    let data_clone = Arc::clone(&data);
    let handle = thread::spawn(move || {
        let mut guard = data_clone.lock().unwrap();
        guard.push(4);
        panic!("意図的なパニック!"); // ← パニック時にロックがポイズニング
    });

    // パニックは join で Result::Err として返る
    let _ = handle.join();

    // ポイズニングされた Mutex のハンドリング
    match data.lock() {
        Ok(guard) => {
            println!("正常ロック: {:?}", *guard);
        }
        Err(poisoned) => {
            // ポイズニングされたが、中のデータは取得可能
            println!("ポイズニング検出! データ: {:?}", *poisoned.into_inner());
            // データの整合性は自分で判断する必要がある
        }
    }

    // parking_lot::Mutex はポイズニングしない (unwrap 不要)
    // use parking_lot::Mutex;
    // let m = Mutex::new(42);
    // let guard = m.lock(); // Result ではなく直接 MutexGuard
}
```

### コード例4: Atomic 操作

```rust
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

fn main() {
    let counter = Arc::new(AtomicU64::new(0));
    let running = Arc::new(AtomicBool::new(true));

    let mut handles = vec![];

    for _ in 0..4 {
        let counter = Arc::clone(&counter);
        let running = Arc::clone(&running);
        handles.push(thread::spawn(move || {
            while running.load(Ordering::Relaxed) {
                counter.fetch_add(1, Ordering::Relaxed);
                // Relaxed: 最も緩い順序保証 (カウンタに十分)
            }
        }));
    }

    thread::sleep(std::time::Duration::from_millis(100));
    running.store(false, Ordering::Relaxed); // 全スレッドに停止を通知

    for h in handles { h.join().unwrap(); }
    println!("カウンタ: {}", counter.load(Ordering::Relaxed));
}
```

### コード例: Ordering の違いを理解する

```rust
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;

fn main() {
    // Ordering の種類と用途
    // Relaxed:    最も緩い。他の操作との順序保証なし。カウンタに適する。
    // Acquire:    この load 以降の読み書きが、この load より前に並べ替えられない。
    // Release:    この store 以前の読み書きが、この store より後に並べ替えられない。
    // AcqRel:     Acquire + Release。RMW (Read-Modify-Write) 操作に使用。
    // SeqCst:     最も厳しい。全スレッドで全操作の順序が一致。

    // Release-Acquire ペアの例: フラグによるデータ公開
    let data = Arc::new(AtomicU32::new(0));
    let ready = Arc::new(AtomicBool::new(false));

    let data_clone = Arc::clone(&data);
    let ready_clone = Arc::clone(&ready);

    // プロデューサー
    let producer = thread::spawn(move || {
        data_clone.store(42, Ordering::Relaxed);      // データを書き込み
        ready_clone.store(true, Ordering::Release);    // Release: 上の書き込みが先に完了
    });

    // コンシューマー
    let data_clone2 = Arc::clone(&data);
    let ready_clone2 = Arc::clone(&ready);

    let consumer = thread::spawn(move || {
        // Acquire: ready=true を読んだ後、data の読み込みが確実に 42 になる
        while !ready_clone2.load(Ordering::Acquire) {
            std::hint::spin_loop(); // ビジーウェイト
        }
        let value = data_clone2.load(Ordering::Relaxed);
        assert_eq!(value, 42); // Release-Acquire により保証される
        println!("データ: {}", value);
    });

    producer.join().unwrap();
    consumer.join().unwrap();
}
```

### コード例5: Condvar (条件変数)

```rust
use std::sync::{Arc, Mutex, Condvar};
use std::thread;
use std::collections::VecDeque;

/// Condvar によるスレッドセーフなキュー
struct BlockingQueue<T> {
    queue: Mutex<VecDeque<T>>,
    not_empty: Condvar,
}

impl<T> BlockingQueue<T> {
    fn new() -> Self {
        BlockingQueue {
            queue: Mutex::new(VecDeque::new()),
            not_empty: Condvar::new(),
        }
    }

    fn push(&self, item: T) {
        let mut q = self.queue.lock().unwrap();
        q.push_back(item);
        self.not_empty.notify_one(); // 待機中のスレッドを1つ起こす
    }

    fn pop(&self) -> T {
        let mut q = self.queue.lock().unwrap();
        while q.is_empty() {
            q = self.not_empty.wait(q).unwrap(); // 通知まで待機
        }
        q.pop_front().unwrap()
    }
}

fn main() {
    let queue = Arc::new(BlockingQueue::new());

    let consumer_queue = Arc::clone(&queue);
    let consumer = thread::spawn(move || {
        for _ in 0..5 {
            let item = consumer_queue.pop();
            println!("消費: {}", item);
        }
    });

    for i in 0..5 {
        queue.push(i);
        thread::sleep(std::time::Duration::from_millis(50));
    }

    consumer.join().unwrap();
}
```

### コード例: Barrier による同期

```rust
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

fn main() {
    let num_threads = 4;
    let barrier = Arc::new(Barrier::new(num_threads));

    let mut handles = vec![];

    for id in 0..num_threads {
        let barrier = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            // Phase 1: 各スレッドが独立に初期化
            println!("[Thread {}] Phase 1: 初期化中...", id);
            thread::sleep(std::time::Duration::from_millis(100 * id as u64));
            println!("[Thread {}] Phase 1: 初期化完了", id);

            // 全スレッドが初期化完了するまで待機
            let result = barrier.wait();
            if result.is_leader() {
                println!("=== 全スレッド初期化完了、Phase 2 開始 ===");
            }

            // Phase 2: 全スレッドが同時に処理開始
            let start = Instant::now();
            println!("[Thread {}] Phase 2: 処理開始 at {:?}", id, start.elapsed());

            // 再度同期
            barrier.wait();
            println!("[Thread {}] Phase 2: 完了", id);
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}
```

### コード例: Once と OnceLock による一回限り初期化

```rust
use std::sync::{Once, OnceLock};

// Once: 初期化処理を一回だけ実行
static INIT: Once = Once::new();
static mut CONFIG: Option<String> = None;

fn get_config_legacy() -> &'static str {
    INIT.call_once(|| {
        // 初回呼び出し時のみ実行される
        unsafe {
            CONFIG = Some("production".to_string());
        }
    });
    unsafe { CONFIG.as_ref().unwrap().as_str() }
}

// OnceLock (Rust 1.70+): 型安全な一回限り初期化
static CONFIG_NEW: OnceLock<String> = OnceLock::new();

fn get_config() -> &'static str {
    CONFIG_NEW.get_or_init(|| {
        println!("設定を初期化中...");
        "production".to_string()
    })
}

fn main() {
    // 複数回呼んでも初期化は1回だけ
    println!("1回目: {}", get_config());
    println!("2回目: {}", get_config());
    println!("3回目: {}", get_config());
    // 出力:
    // 設定を初期化中...
    // 1回目: production
    // 2回目: production
    // 3回目: production
}
```

---

## 4. チャネルによるメッセージパッシング

### コード例: mpsc チャネル

```rust
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn main() {
    // mpsc: Multiple Producer, Single Consumer
    let (tx, rx) = mpsc::channel();

    // 複数のプロデューサー
    for id in 0..3 {
        let tx = tx.clone();
        thread::spawn(move || {
            for i in 0..5 {
                let msg = format!("Producer {} - Message {}", id, i);
                tx.send(msg).unwrap();
                thread::sleep(Duration::from_millis(50));
            }
        });
    }

    // 元の tx を drop (clone のみ残す)
    drop(tx);

    // コンシューマー: 全メッセージを受信
    let mut count = 0;
    while let Ok(msg) = rx.recv() {
        println!("受信: {}", msg);
        count += 1;
    }
    println!("合計 {} メッセージ受信", count); // 15

    // sync_channel: バッファ付きチャネル
    let (tx, rx) = mpsc::sync_channel::<i32>(3); // バッファサイズ 3

    thread::spawn(move || {
        for i in 0..10 {
            println!("送信: {} (バッファが満杯なら待機)", i);
            tx.send(i).unwrap(); // バッファが満杯ならブロック
        }
    });

    thread::sleep(Duration::from_millis(500));
    while let Ok(v) = rx.recv() {
        println!("受信: {}", v);
    }
}
```

### コード例: crossbeam-channel によるマルチコンシューマー

```rust
use std::thread;
use std::time::Duration;

// crossbeam-channel: MPMC (Multiple Producer, Multiple Consumer)
// Cargo.toml: crossbeam-channel = "0.5"

fn crossbeam_example() {
    use crossbeam_channel::{bounded, select, unbounded, Receiver, Sender};

    // bounded チャネル (バッファ付き)
    let (tx, rx): (Sender<String>, Receiver<String>) = bounded(10);

    // 複数のコンシューマー
    let mut consumers = vec![];
    for id in 0..3 {
        let rx = rx.clone();
        consumers.push(thread::spawn(move || {
            let mut processed = 0;
            while let Ok(msg) = rx.recv() {
                println!("[Consumer {}] 処理: {}", id, msg);
                processed += 1;
            }
            println!("[Consumer {}] 合計 {} 件処理", id, processed);
        }));
    }

    // プロデューサー
    for i in 0..30 {
        tx.send(format!("Job {}", i)).unwrap();
    }
    drop(tx); // チャネルを閉じる

    for c in consumers {
        c.join().unwrap();
    }
}

// select! マクロ: 複数チャネルの待ち受け
fn select_example() {
    use crossbeam_channel::{bounded, select, after, tick};

    let (tx1, rx1) = bounded(1);
    let (tx2, rx2) = bounded(1);

    // タイマー
    let timeout = after(Duration::from_secs(1));
    let ticker = tick(Duration::from_millis(200));

    thread::spawn(move || {
        thread::sleep(Duration::from_millis(300));
        tx1.send("チャネル1のデータ").unwrap();
    });

    thread::spawn(move || {
        thread::sleep(Duration::from_millis(500));
        tx2.send("チャネル2のデータ").unwrap();
    });

    loop {
        select! {
            recv(rx1) -> msg => {
                match msg {
                    Ok(m) => println!("rx1: {}", m),
                    Err(_) => println!("rx1 閉鎖"),
                }
            }
            recv(rx2) -> msg => {
                match msg {
                    Ok(m) => println!("rx2: {}", m),
                    Err(_) => println!("rx2 閉鎖"),
                }
            }
            recv(ticker) -> _ => {
                println!("tick");
            }
            recv(timeout) -> _ => {
                println!("タイムアウト!");
                break;
            }
        }
    }
}

fn main() {
    println!("=== crossbeam MPMC ===");
    crossbeam_example();
    println!("\n=== select! ===");
    select_example();
}
```

---

## 5. rayon — データ並列処理

### コード例6: 並列イテレータ

```rust
use rayon::prelude::*;

fn main() {
    let data: Vec<u64> = (0..10_000_000).collect();

    // 並列 map + filter + sum
    let sum: u64 = data.par_iter()
        .filter(|&&x| x % 2 == 0)
        .map(|&x| x * x)
        .sum();
    println!("偶数の二乗和: {}", sum);

    // 並列ソート
    let mut nums: Vec<i32> = (0..1_000_000).rev().collect();
    nums.par_sort_unstable();
    assert!(nums.windows(2).all(|w| w[0] <= w[1]));

    // 並列 for_each
    let results: Vec<String> = (0..100)
        .into_par_iter()
        .map(|i| format!("処理結果#{}", i))
        .collect();
    println!("結果数: {}", results.len());
}
```

### コード例7: カスタムスレッドプール

```rust
use rayon::ThreadPoolBuilder;

fn main() {
    // カスタムスレッドプール (スレッド数制御)
    let pool = ThreadPoolBuilder::new()
        .num_threads(4)
        .thread_name(|index| format!("worker-{}", index))
        .build()
        .unwrap();

    pool.install(|| {
        let data: Vec<i64> = (0..1_000_000).collect();
        let sum: i64 = data.par_iter().sum();
        println!("合計: {}", sum);
    });

    // join — 2つの処理を並列実行
    let (left, right) = rayon::join(
        || expensive_computation_a(),
        || expensive_computation_b(),
    );
    println!("A={}, B={}", left, right);
}

fn expensive_computation_a() -> u64 { (0..1_000_000u64).sum() }
fn expensive_computation_b() -> u64 { (0..500_000u64).map(|x| x * 2).sum() }
```

### コード例: rayon の高度な使い方

```rust
use rayon::prelude::*;
use std::collections::HashMap;

/// 並列 reduce による集計
fn parallel_word_count(texts: &[String]) -> HashMap<String, usize> {
    texts
        .par_iter()
        .fold(
            || HashMap::new(),
            |mut map, text| {
                for word in text.split_whitespace() {
                    *map.entry(word.to_lowercase()).or_insert(0) += 1;
                }
                map
            },
        )
        .reduce(
            || HashMap::new(),
            |mut a, b| {
                for (key, count) in b {
                    *a.entry(key).or_insert(0) += count;
                }
                a
            },
        )
}

/// 並列 find (最初に条件を満たす要素を探す)
fn parallel_find_prime(range: std::ops::Range<u64>) -> Option<u64> {
    range.into_par_iter().find_any(|&n| is_prime(n))
}

fn is_prime(n: u64) -> bool {
    if n < 2 { return false; }
    if n == 2 { return true; }
    if n % 2 == 0 { return false; }
    let limit = (n as f64).sqrt() as u64;
    (3..=limit).step_by(2).all(|i| n % i != 0)
}

/// 並列チャンクプロセッシング
fn parallel_chunk_processing(data: &[u8]) -> Vec<u32> {
    data.par_chunks(1024)
        .map(|chunk| {
            // 各チャンクの処理（例: チェックサム計算）
            chunk.iter().map(|&b| b as u32).sum()
        })
        .collect()
}

fn main() {
    // 並列ワードカウント
    let texts: Vec<String> = (0..1000)
        .map(|i| format!("hello world rust programming hello rust {}", i))
        .collect();
    let counts = parallel_word_count(&texts);
    println!("'hello' の出現回数: {}", counts.get("hello").unwrap_or(&0));
    println!("'rust' の出現回数: {}", counts.get("rust").unwrap_or(&0));

    // 並列検索
    let prime = parallel_find_prime(1_000_000..1_001_000);
    println!("見つかった素数: {:?}", prime);

    // チャンク処理
    let data: Vec<u8> = (0..10_000).map(|i| (i % 256) as u8).collect();
    let checksums = parallel_chunk_processing(&data);
    println!("チャンク数: {}, 最初のチェックサム: {}", checksums.len(), checksums[0]);
}
```

---

## 6. 高度な並行パターン

### コード例: Read-Copy-Update (RCU) パターン

```rust
use std::sync::{Arc, atomic::{AtomicPtr, Ordering}};
use std::thread;

/// Arc ベースの Read-Copy-Update パターン
/// 読み取りはロックフリー、書き込みは新しいデータを作成して置き換え
struct RcuConfig {
    data: std::sync::RwLock<Arc<ConfigData>>,
}

#[derive(Clone, Debug)]
struct ConfigData {
    database_url: String,
    max_connections: u32,
    timeout_ms: u64,
}

impl RcuConfig {
    fn new(data: ConfigData) -> Self {
        RcuConfig {
            data: std::sync::RwLock::new(Arc::new(data)),
        }
    }

    /// 読み取り: Arc のクローンを取得 (高速)
    fn read(&self) -> Arc<ConfigData> {
        Arc::clone(&self.data.read().unwrap())
    }

    /// 更新: 新しいデータで置き換え
    fn update<F>(&self, f: F)
    where
        F: FnOnce(&ConfigData) -> ConfigData,
    {
        let mut guard = self.data.write().unwrap();
        let new_data = f(&guard);
        *guard = Arc::new(new_data);
    }
}

fn main() {
    let config = Arc::new(RcuConfig::new(ConfigData {
        database_url: "postgres://localhost/mydb".to_string(),
        max_connections: 10,
        timeout_ms: 5000,
    }));

    // 読み取りスレッド群
    let mut readers = vec![];
    for id in 0..5 {
        let config = Arc::clone(&config);
        readers.push(thread::spawn(move || {
            for _ in 0..100 {
                let data = config.read();
                // 読み取りはロックフリー (Arc::clone のみ)
                let _url = &data.database_url;
                let _max = data.max_connections;
            }
            println!("[Reader {}] 完了", id);
        }));
    }

    // 書き込みスレッド
    let config_writer = Arc::clone(&config);
    let writer = thread::spawn(move || {
        for i in 0..10 {
            config_writer.update(|old| ConfigData {
                max_connections: old.max_connections + 1,
                ..old.clone()
            });
            thread::sleep(std::time::Duration::from_millis(10));
        }
        println!("[Writer] 完了");
    });

    for r in readers { r.join().unwrap(); }
    writer.join().unwrap();

    let final_config = config.read();
    println!("最終 max_connections: {}", final_config.max_connections);
}
```

### コード例: ダブルバッファリングパターン

```rust
use std::sync::{Arc, RwLock, atomic::{AtomicBool, Ordering}};
use std::thread;
use std::time::Duration;

/// ダブルバッファ: 読み取りと書き込みを分離
struct DoubleBuffer<T: Clone> {
    buffers: [RwLock<T>; 2],
    active: std::sync::atomic::AtomicUsize, // 現在のアクティブバッファ (0 or 1)
}

impl<T: Clone> DoubleBuffer<T> {
    fn new(initial: T) -> Self {
        DoubleBuffer {
            buffers: [
                RwLock::new(initial.clone()),
                RwLock::new(initial),
            ],
            active: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// 読み取り: アクティブバッファから (ロック競合最小)
    fn read(&self) -> std::sync::RwLockReadGuard<'_, T> {
        let idx = self.active.load(Ordering::Acquire);
        self.buffers[idx].read().unwrap()
    }

    /// 書き込み: 非アクティブバッファに書き込んでスワップ
    fn write<F>(&self, update_fn: F)
    where
        F: FnOnce(&T) -> T,
    {
        let active = self.active.load(Ordering::Acquire);
        let inactive = 1 - active;

        // 非アクティブバッファに新しいデータを書き込み
        {
            let current = self.buffers[active].read().unwrap();
            let new_data = update_fn(&current);
            let mut inactive_guard = self.buffers[inactive].write().unwrap();
            *inactive_guard = new_data;
        }

        // バッファをスワップ
        self.active.store(inactive, Ordering::Release);
    }
}

fn main() {
    let buffer = Arc::new(DoubleBuffer::new(vec![0u64; 100]));

    let running = Arc::new(AtomicBool::new(true));

    // 読み取りスレッド
    let buf_reader = Arc::clone(&buffer);
    let run_r = Arc::clone(&running);
    let reader = thread::spawn(move || {
        let mut reads = 0u64;
        while run_r.load(Ordering::Relaxed) {
            let data = buf_reader.read();
            let _sum: u64 = data.iter().sum();
            reads += 1;
        }
        println!("読み取り回数: {}", reads);
    });

    // 書き込みスレッド
    let buf_writer = Arc::clone(&buffer);
    for i in 0..100 {
        buf_writer.write(|old| {
            old.iter().map(|x| x + 1).collect()
        });
        thread::sleep(Duration::from_millis(1));
    }

    running.store(false, Ordering::Relaxed);
    reader.join().unwrap();

    let final_data = buffer.read();
    println!("最終値[0]: {}", final_data[0]); // 100
}
```

### コード例: Sharded Lock (分割ロック) パターン

```rust
use std::collections::HashMap;
use std::hash::{Hash, Hasher, DefaultHasher};
use std::sync::RwLock;

/// ハッシュベースのシャーディングで並行性を向上
struct ShardedMap<K, V> {
    shards: Vec<RwLock<HashMap<K, V>>>,
    num_shards: usize,
}

impl<K: Hash + Eq + Clone, V: Clone> ShardedMap<K, V> {
    fn new(num_shards: usize) -> Self {
        let mut shards = Vec::with_capacity(num_shards);
        for _ in 0..num_shards {
            shards.push(RwLock::new(HashMap::new()));
        }
        ShardedMap { shards, num_shards }
    }

    fn shard_index(&self, key: &K) -> usize {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish() as usize % self.num_shards
    }

    fn insert(&self, key: K, value: V) {
        let idx = self.shard_index(&key);
        let mut shard = self.shards[idx].write().unwrap();
        shard.insert(key, value);
    }

    fn get(&self, key: &K) -> Option<V> {
        let idx = self.shard_index(key);
        let shard = self.shards[idx].read().unwrap();
        shard.get(key).cloned()
    }

    fn remove(&self, key: &K) -> Option<V> {
        let idx = self.shard_index(key);
        let mut shard = self.shards[idx].write().unwrap();
        shard.remove(key)
    }

    fn len(&self) -> usize {
        self.shards.iter()
            .map(|s| s.read().unwrap().len())
            .sum()
    }
}

fn main() {
    use std::sync::Arc;
    use std::thread;

    let map = Arc::new(ShardedMap::<String, u64>::new(16));
    let mut handles = vec![];

    // 並行書き込み
    for t in 0..8 {
        let map = Arc::clone(&map);
        handles.push(thread::spawn(move || {
            for i in 0..10_000 {
                let key = format!("key_{}_{}", t, i);
                map.insert(key, i as u64);
            }
        }));
    }

    for h in handles { h.join().unwrap(); }
    println!("ShardedMap エントリ数: {}", map.len()); // 80000

    // 並行読み取り
    let value = map.get(&"key_3_500".to_string());
    println!("key_3_500 = {:?}", value);
}
```

---

## 7. 比較表

### 同期プリミティブ比較

| プリミティブ | 用途 | オーバーヘッド | 特徴 |
|---|---|---|---|
| `Mutex<T>` | 排他アクセス | 中 | 最も汎用的 |
| `RwLock<T>` | 読み取り並行 | 中〜高 | 読み取り多・書き込み少向け |
| `Atomic*` | 単一値の操作 | 低 | ロックフリー。カウンタ等 |
| `Condvar` | 待機/通知 | 中 | Mutex と組み合わせ |
| `Barrier` | 同期点 | 低 | 全スレッド到達まで待機 |
| `Once` / `OnceLock` | 一回限り初期化 | 最低 | lazy_static の代替 |
| `mpsc::channel` | メッセージ送受信 | 中 | MPSC のみ |
| `crossbeam::channel` | メッセージ送受信 | 低〜中 | MPMC、select 対応 |

### 並行パターン比較

| パターン | 適用場面 | Rustでの実現手段 |
|---|---|---|
| 共有メモリ | 状態の直接共有 | Arc<Mutex<T>>, Arc<RwLock<T>> |
| メッセージパッシング | 分離された状態 | mpsc, crossbeam-channel |
| データ並列 | 同じ操作を大量データに | rayon par_iter |
| アクターモデル | 独立したエンティティ | tokio::spawn + mpsc |
| ロックフリー | 超高頻度アクセス | Atomic*, crossbeam |
| RCU | 読み取り優位 | Arc + RwLock / AtomicPtr |
| シャーディング | 大量キーの並行アクセス | ShardedMap (DashMap) |
| ダブルバッファ | 読み書き分離 | Atomic index + buffer pair |

### Ordering の使い分け

| Ordering | 用途 | 性能 | 保証 |
|---|---|---|---|
| `Relaxed` | カウンタ、フラグ (単純) | 最高 | 順序保証なし |
| `Acquire` | ロック取得、データ読み取り | 高 | 後続の読み書きが前に来ない |
| `Release` | ロック解放、データ公開 | 高 | 先行の読み書きが後に来ない |
| `AcqRel` | CAS (compare-and-swap) | 中 | Acquire + Release |
| `SeqCst` | 全順序が必要な場合 | 低 | 全スレッドで順序一致 |

---

## 8. アンチパターン

### アンチパターン1: ロック粒度が大きすぎる

```rust
use std::sync::{Arc, Mutex};

// NG: 構造体全体を1つの Mutex で保護
struct BadCache {
    data: Mutex<(Vec<String>, std::collections::HashMap<String, String>)>,
}
// → data と index のどちらかだけ使いたい時も全体をロック

// OK: 細粒度ロック
struct GoodCache {
    data: Mutex<Vec<String>>,
    index: Mutex<std::collections::HashMap<String, String>>,
}
// → data と index を独立にロックできる
// ※ ただしデッドロックに注意: 常に同じ順序でロック
```

### アンチパターン2: Mutex のロック保持中に長時間処理

```rust
use std::sync::{Arc, Mutex};

// NG: ロック保持中にネットワーク呼び出し
fn bad_update(cache: &Mutex<Vec<String>>) {
    let mut data = cache.lock().unwrap();
    let new_item = fetch_from_network(); // 長時間ブロック!
    data.push(new_item);
}

// OK: ロック外で処理、最小限のロック
fn good_update(cache: &Mutex<Vec<String>>) {
    let new_item = fetch_from_network(); // ロック外で取得
    let mut data = cache.lock().unwrap();
    data.push(new_item);
    // ロック保持時間は最小限
}

fn fetch_from_network() -> String { "data".into() }
```

### アンチパターン3: デッドロック

```rust
use std::sync::{Arc, Mutex};
use std::thread;

// NG: ロック順序が逆 → デッドロックの危険
fn deadlock_example() {
    let a = Arc::new(Mutex::new(1));
    let b = Arc::new(Mutex::new(2));

    let a1 = Arc::clone(&a);
    let b1 = Arc::clone(&b);
    let t1 = thread::spawn(move || {
        let _a = a1.lock().unwrap(); // A → B の順
        thread::sleep(std::time::Duration::from_millis(10));
        let _b = b1.lock().unwrap();
    });

    let a2 = Arc::clone(&a);
    let b2 = Arc::clone(&b);
    let t2 = thread::spawn(move || {
        let _b = b2.lock().unwrap(); // B → A の順 (逆!) → デッドロック!
        thread::sleep(std::time::Duration::from_millis(10));
        let _a = a2.lock().unwrap();
    });

    // t1.join().unwrap();
    // t2.join().unwrap();
}

// OK: 常に同じ順序でロック
fn no_deadlock() {
    let a = Arc::new(Mutex::new(1));
    let b = Arc::new(Mutex::new(2));

    let a1 = Arc::clone(&a);
    let b1 = Arc::clone(&b);
    let t1 = thread::spawn(move || {
        let _a = a1.lock().unwrap(); // 常に A → B の順
        let _b = b1.lock().unwrap();
    });

    let a2 = Arc::clone(&a);
    let b2 = Arc::clone(&b);
    let t2 = thread::spawn(move || {
        let _a = a2.lock().unwrap(); // 常に A → B の順
        let _b = b2.lock().unwrap();
    });

    t1.join().unwrap();
    t2.join().unwrap();
}

fn main() {
    // deadlock_example(); // これはデッドロックする可能性がある
    no_deadlock(); // 安全
    println!("デッドロックなし");
}
```

### アンチパターン4: 不要なロック / Atomic の過剰使用

```rust
use std::sync::atomic::{AtomicU64, Ordering};

// NG: 単一スレッドなのに Atomic を使う
fn single_threaded_bad() {
    let counter = AtomicU64::new(0);
    for _ in 0..1_000_000 {
        counter.fetch_add(1, Ordering::SeqCst); // オーバーヘッドが無駄
    }
}

// OK: 単一スレッドなら通常の変数で十分
fn single_threaded_good() {
    let mut counter: u64 = 0;
    for _ in 0..1_000_000 {
        counter += 1;
    }
}

// NG: SeqCst を全てに使う (必要以上に厳しい順序保証)
fn overly_strict() {
    let counter = AtomicU64::new(0);
    counter.fetch_add(1, Ordering::SeqCst); // カウンタなら Relaxed で十分
}

// OK: 用途に合った Ordering を選択
fn appropriate_ordering() {
    let counter = AtomicU64::new(0);
    counter.fetch_add(1, Ordering::Relaxed); // カウンタ → Relaxed で十分
}

fn main() {
    single_threaded_good();
    appropriate_ordering();
}
```

---

## FAQ

### Q1: `Arc<Mutex<T>>` と `Arc<RwLock<T>>` はどちらを使うべき?

**A:** 書き込みが少なく読み取りが多い場合は `RwLock` が有利です。書き込みが頻繁、またはロック保持時間が短い場合は `Mutex` の方がオーバーヘッドが少ないです。迷ったら `Mutex` から始めましょう。

### Q2: `parking_lot` の Mutex と標準ライブラリの違いは?

**A:** `parking_lot::Mutex` は (1) ポイズニングなし(unwrap不要) (2) サイズが小さい(8バイト vs 40バイト) (3) 公平性オプションあり (4) `const fn` 対応。パフォーマンスクリティカルな用途で推奨されます。

### Q3: rayon はいつ使うべき?

**A:** CPU バウンドな処理で、データを分割して独立に処理できる場合に最適です。`.iter()` を `.par_iter()` に変えるだけで並列化できる手軽さが魅力です。I/Oバウンドな処理には tokio を使ってください。

### Q4: チャネルと共有メモリのどちらを使うべき?

**A:** 一般的にチャネル（メッセージパッシング）が推奨です。状態を各スレッド内に閉じ込め、メッセージでやり取りする方がバグが少なくなります。ただし、大量のデータを頻繁に共有する必要がある場合は共有メモリの方が効率的です。

```rust
// パターン: チャネルが適する場面
// - パイプライン処理 (A → B → C)
// - Producer-Consumer
// - イベント通知
// - 分散集計

// パターン: 共有メモリが適する場面
// - 頻繁な読み取り (キャッシュ、設定)
// - 大量データの共有 (データベース接続プール)
// - 高頻度のカウンタ/メトリクス
```

### Q5: `thread::scope` と `thread::spawn` の使い分けは?

**A:** ローカル変数を参照したい場合は `thread::scope` が安全で便利です。`'static` ライフタイムの制約がないため、Arc によるヒープ割当が不要になります。ただし、スコープ外でスレッドを管理したい場合（バックグラウンドスレッド等）は `thread::spawn` が必要です。

### Q6: DashMap とは何ですか?

**A:** `DashMap` は `dashmap` クレートが提供するスレッドセーフな HashMap です。内部的にシャーディング（前述の ShardedMap パターン）を使用しており、`Arc<RwLock<HashMap>>` より高い並行性能を発揮します。

```rust
// Cargo.toml: dashmap = "5"
use dashmap::DashMap;

fn main() {
    let map = DashMap::new();
    map.insert("key1", 42);
    map.insert("key2", 100);

    // 読み取り
    if let Some(value) = map.get("key1") {
        println!("key1 = {}", *value);
    }

    // 更新
    map.entry("key1").and_modify(|v| *v += 1).or_insert(0);

    // イテレート
    for entry in map.iter() {
        println!("{} = {}", entry.key(), entry.value());
    }
}
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| Send / Sync | コンパイル時にデータ競合を防止するマーカートレイト |
| thread::scope | ローカル変数の参照を安全に子スレッドに渡せる |
| Mutex | 排他ロック。最も基本的な同期手段 |
| RwLock | 読み取り並行ロック。読み取り優位な場面向け |
| Atomic | ロックフリーな単一値操作。カウンタ・フラグ |
| Condvar | 条件待ち。Producer-Consumer パターン |
| Barrier | 全スレッドの同期点。フェーズ処理向け |
| OnceLock | スレッドセーフな一回限り初期化 |
| mpsc / crossbeam | チャネルベースのメッセージパッシング |
| rayon | `.par_iter()` で簡単データ並列。CPU バウンド向け |
| ShardedMap / DashMap | 高並行性のスレッドセーフ HashMap |
| Ordering | Relaxed < Acquire/Release < SeqCst で使い分け |

## 次に読むべきガイド

- [FFI](./02-ffi-interop.md) — スレッド安全性とFFIの交差点
- [Tokioランタイム](../02-async/01-tokio-runtime.md) — 非同期タスクの並行管理
- [メモリレイアウト](./00-memory-layout.md) — false sharing とキャッシュライン

## 参考文献

1. **Rust Book — Fearless Concurrency**: https://doc.rust-lang.org/book/ch16-00-concurrency.html
2. **Rayon documentation**: https://docs.rs/rayon/latest/rayon/
3. **The Rustonomicon — Concurrency**: https://doc.rust-lang.org/nomicon/concurrency.html
4. **crossbeam documentation**: https://docs.rs/crossbeam/latest/crossbeam/
5. **DashMap documentation**: https://docs.rs/dashmap/latest/dashmap/
