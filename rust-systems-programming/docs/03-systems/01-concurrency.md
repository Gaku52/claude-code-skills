# 並行性 — スレッド、Mutex/RwLock、rayon

> Rust の所有権システムによるコンパイル時データ競合防止と、スレッド/ロック/並列イテレータによる実践的な並行処理を習得する

## この章で学ぶこと

1. **スレッドと所有権** — std::thread、Send/Sync トレイト、スコープドスレッド
2. **同期プリミティブ** — Mutex, RwLock, Condvar, Atomic, Barrier
3. **データ並列処理** — rayon による並列イテレータと作業分割

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

---

## 4. rayon — データ並列処理

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

---

## 5. 比較表

### 同期プリミティブ比較

| プリミティブ | 用途 | オーバーヘッド | 特徴 |
|---|---|---|---|
| `Mutex<T>` | 排他アクセス | 中 | 最も汎用的 |
| `RwLock<T>` | 読み取り並行 | 中〜高 | 読み取り多・書き込み少向け |
| `Atomic*` | 単一値の操作 | 低 | ロックフリー。カウンタ等 |
| `Condvar` | 待機/通知 | 中 | Mutex と組み合わせ |
| `Barrier` | 同期点 | 低 | 全スレッド到達まで待機 |
| `Once` | 一回限り初期化 | 最低 | lazy_static の内部 |

### 並行パターン比較

| パターン | 適用場面 | Rustでの実現手段 |
|---|---|---|
| 共有メモリ | 状態の直接共有 | Arc<Mutex<T>>, Arc<RwLock<T>> |
| メッセージパッシング | 分離された状態 | mpsc, crossbeam-channel |
| データ並列 | 同じ操作を大量データに | rayon par_iter |
| アクターモデル | 独立したエンティティ | tokio::spawn + mpsc |
| ロックフリー | 超高頻度アクセス | Atomic*, crossbeam |

---

## 6. アンチパターン

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

---

## FAQ

### Q1: `Arc<Mutex<T>>` と `Arc<RwLock<T>>` はどちらを使うべき?

**A:** 書き込みが少なく読み取りが多い場合は `RwLock` が有利です。書き込みが頻繁、またはロック保持時間が短い場合は `Mutex` の方がオーバーヘッドが少ないです。迷ったら `Mutex` から始めましょう。

### Q2: `parking_lot` の Mutex と標準ライブラリの違いは?

**A:** `parking_lot::Mutex` は (1) ポイズニングなし(unwrap不要) (2) サイズが小さい(8バイト vs 40バイト) (3) 公平性オプションあり (4) `const fn` 対応。パフォーマンスクリティカルな用途で推奨されます。

### Q3: rayon はいつ使うべき?

**A:** CPU バウンドな処理で、データを分割して独立に処理できる場合に最適です。`.iter()` を `.par_iter()` に変えるだけで並列化できる手軽さが魅力です。I/Oバウンドな処理には tokio を使ってください。

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
| rayon | `.par_iter()` で簡単データ並列。CPU バウンド向け |

## 次に読むべきガイド

- [FFI](./02-ffi-interop.md) — スレッド安全性とFFIの交差点
- [Tokioランタイム](../02-async/01-tokio-runtime.md) — 非同期タスクの並行管理
- [メモリレイアウト](./00-memory-layout.md) — false sharing とキャッシュライン

## 参考文献

1. **Rust Book — Fearless Concurrency**: https://doc.rust-lang.org/book/ch16-00-concurrency.html
2. **Rayon documentation**: https://docs.rs/rayon/latest/rayon/
3. **The Rustonomicon — Concurrency**: https://doc.rust-lang.org/nomicon/concurrency.html
