# Tokioランタイム — タスク管理とチャネル

> Tokio のマルチスレッドランタイムの内部構造、タスクスポーン、チャネルを使った非同期メッセージパッシングを習得する

## この章で学ぶこと

1. **ランタイム構成** — マルチスレッド/シングルスレッドランタイムの選択と設定
2. **タスク管理** — spawn, JoinSet, abort, タスクローカルストレージ
3. **非同期チャネル** — mpsc, oneshot, broadcast, watch の使い分け

---

## 1. Tokio ランタイムのアーキテクチャ

```
┌────────────────────── Tokio Runtime ──────────────────────┐
│                                                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ Worker      │  │ Worker      │  │ Worker      │      │
│  │ Thread #1   │  │ Thread #2   │  │ Thread #N   │      │
│  │             │  │             │  │             │      │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │      │
│  │ │ Local Q │ │  │ │ Local Q │ │  │ │ Local Q │ │      │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │      │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      │
│         │                │                │              │
│  ┌──────┴────────────────┴────────────────┴──────┐      │
│  │              Global Task Queue                  │      │
│  │          (Work-Stealing Scheduler)             │      │
│  └─────────────────────┬─────────────────────────┘      │
│                        │                                  │
│  ┌─────────────────────┴─────────────────────────┐      │
│  │           I/O Driver (mio/epoll/kqueue)        │      │
│  │           Timer Driver                          │      │
│  └────────────────────────────────────────────────┘      │
│                                                            │
│  ┌────────────────────────────────────────────────┐      │
│  │        Blocking Thread Pool (spawn_blocking)    │      │
│  └────────────────────────────────────────────────┘      │
└────────────────────────────────────────────────────────────┘
```

---

## 2. ランタイムの構築と設定

### コード例1: ランタイム構成の選択

```rust
// パターン1: マクロによる簡易設定 (マルチスレッド)
#[tokio::main]
async fn main() {
    println!("マルチスレッドランタイム");
}

// パターン2: シングルスレッド (テスト・軽量用途)
#[tokio::main(flavor = "current_thread")]
async fn main() {
    println!("シングルスレッドランタイム");
}

// パターン3: 手動ビルド (詳細制御)
fn main() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)                       // ワーカースレッド数
        .max_blocking_threads(64)                // ブロッキングスレッド上限
        .thread_name("my-worker")                // スレッド名
        .thread_stack_size(3 * 1024 * 1024)     // スタックサイズ 3MB
        .enable_all()                             // I/O + Timer ドライバ有効
        .build()
        .unwrap();

    runtime.block_on(async {
        println!("手動構築ランタイム");
    });
}
```

---

## 3. タスク管理

### コード例2: spawn と JoinHandle

```rust
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    // spawn — 新しいタスクを非同期に実行
    let handle = tokio::spawn(async {
        sleep(Duration::from_millis(100)).await;
        42
    });

    // JoinHandle で結果を取得
    let result = handle.await.unwrap();
    println!("結果: {}", result); // 結果: 42

    // spawn で生成したタスクは即座に実行開始
    // .await は結果の「取得」であり「開始」ではない
    let h1 = tokio::spawn(async { sleep(Duration::from_secs(1)).await; "A" });
    let h2 = tokio::spawn(async { sleep(Duration::from_secs(1)).await; "B" });

    // h1 と h2 は並行実行される → 合計約1秒
    let (a, b) = (h1.await.unwrap(), h2.await.unwrap());
    println!("{}, {}", a, b);
}
```

### コード例3: JoinSet による動的タスク管理

```rust
use tokio::task::JoinSet;
use tokio::time::{sleep, Duration};

async fn process_item(id: u32) -> String {
    sleep(Duration::from_millis(50 * id as u64)).await;
    format!("Item#{} 処理完了", id)
}

#[tokio::main]
async fn main() {
    let mut set = JoinSet::new();

    // 動的にタスクを追加
    for id in 1..=10 {
        set.spawn(process_item(id));
    }

    // 完了順に結果を取得
    while let Some(result) = set.join_next().await {
        match result {
            Ok(msg) => println!("{}", msg),
            Err(e) => eprintln!("タスクエラー: {}", e),
        }
    }

    // abort_all で全タスクをキャンセル
    let mut set2 = JoinSet::new();
    for i in 0..5 {
        set2.spawn(async move {
            sleep(Duration::from_secs(10)).await;
            i
        });
    }
    set2.abort_all(); // 全タスクを即座にキャンセル
}
```

### タスクキャンセルの仕組み

```
┌────────────── タスクのライフサイクル ──────────────┐
│                                                     │
│  spawn()                                            │
│    │                                                │
│    ▼                                                │
│  Running ──── .await on poll ────→ Suspended        │
│    │              │                    │            │
│    │              │ I/O Ready          │            │
│    │              ◄────────────────────┘            │
│    │                                                │
│    ├── abort() ──→ Cancelled (JoinError)           │
│    │                                                │
│    └── 正常完了 ──→ Completed (Ok(T))               │
│                                                     │
│  drop(JoinHandle):                                  │
│    タスクは継続 (デタッチ)。結果は取得不可           │
└─────────────────────────────────────────────────────┘
```

---

## 4. チャネル

### コード例4: mpsc (多対一) チャネル

```rust
use tokio::sync::mpsc;

#[derive(Debug)]
enum Command {
    Get { key: String },
    Set { key: String, value: String },
    Shutdown,
}

#[tokio::main]
async fn main() {
    // バッファ付き mpsc チャネル
    let (tx, mut rx) = mpsc::channel::<Command>(32);

    // ワーカータスク (受信側)
    let worker = tokio::spawn(async move {
        let mut store = std::collections::HashMap::new();

        while let Some(cmd) = rx.recv().await {
            match cmd {
                Command::Set { key, value } => {
                    println!("[worker] SET {} = {}", key, value);
                    store.insert(key, value);
                }
                Command::Get { key } => {
                    let val = store.get(&key).cloned().unwrap_or_default();
                    println!("[worker] GET {} => {}", key, val);
                }
                Command::Shutdown => {
                    println!("[worker] シャットダウン");
                    break;
                }
            }
        }
    });

    // 複数の送信者
    let tx2 = tx.clone();
    tokio::spawn(async move {
        tx2.send(Command::Set {
            key: "name".into(),
            value: "Alice".into(),
        }).await.unwrap();
    });

    tx.send(Command::Set {
        key: "age".into(),
        value: "30".into(),
    }).await.unwrap();

    tx.send(Command::Get { key: "name".into() }).await.unwrap();
    tx.send(Command::Shutdown).await.unwrap();

    worker.await.unwrap();
}
```

### コード例5: oneshot (一対一・一回限り)

```rust
use tokio::sync::oneshot;

async fn compute_answer(reply: oneshot::Sender<u64>) {
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    let answer = 42;
    let _ = reply.send(answer); // 一回だけ送信
}

#[tokio::main]
async fn main() {
    let (tx, rx) = oneshot::channel();

    tokio::spawn(compute_answer(tx));

    // タイムアウト付き受信
    tokio::select! {
        result = rx => {
            println!("回答: {}", result.unwrap());
        }
        _ = tokio::time::sleep(tokio::time::Duration::from_secs(5)) => {
            println!("タイムアウト");
        }
    }
}
```

### コード例6: broadcast と watch

```rust
use tokio::sync::{broadcast, watch};

#[tokio::main]
async fn main() {
    // broadcast — 多対多。全受信者にクローン送信
    let (btx, _) = broadcast::channel::<String>(16);
    let mut brx1 = btx.subscribe();
    let mut brx2 = btx.subscribe();

    btx.send("ブロードキャスト!".into()).unwrap();

    println!("rx1: {}", brx1.recv().await.unwrap());
    println!("rx2: {}", brx2.recv().await.unwrap());

    // watch — 最新値のみ保持。設定変更通知に最適
    let (wtx, mut wrx) = watch::channel("初期値".to_string());

    tokio::spawn(async move {
        loop {
            wrx.changed().await.unwrap();
            println!("設定変更: {}", *wrx.borrow());
        }
    });

    wtx.send("更新値1".into()).unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    wtx.send("更新値2".into()).unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
}
```

---

## 5. 比較表

### チャネル種別比較

| チャネル | 送信者 | 受信者 | バッファ | ユースケース |
|---|---|---|---|---|
| `mpsc` | 複数 | 1つ | 有限サイズ | コマンドキュー、ワーカープール |
| `mpsc::unbounded` | 複数 | 1つ | 無制限 | メモリ許容時の簡易キュー |
| `oneshot` | 1つ | 1つ | 1メッセージ | リクエスト-レスポンス |
| `broadcast` | 複数 | 複数 | リングバッファ | イベント通知、Pub/Sub |
| `watch` | 1つ | 複数 | 最新値のみ | 設定変更、状態監視 |

### spawn 種別比較

| API | スレッド | 用途 | 制約 |
|---|---|---|---|
| `tokio::spawn` | ワーカー | 非同期タスク | `Send + 'static` |
| `spawn_blocking` | ブロッキングプール | 同期 I/O、CPU処理 | `Send + 'static` |
| `spawn_local` | カレントスレッド | `!Send` な Future | `LocalSet` 内のみ |
| `block_on` | 呼び出しスレッド | ランタイム外→内 | ランタイム内では使用不可 |

---

## 6. アンチパターン

### アンチパターン1: チャネルバッファの不適切なサイズ

```rust
// NG: バッファサイズ1は送信者をほぼ常にブロック
let (tx, rx) = mpsc::channel(1);

// NG: unbounded で無制限にメモリ消費
let (tx, rx) = mpsc::unbounded_channel();
// → 受信側が遅いと送信側がメモリを無限に使う

// OK: 想定負荷に基づいたバッファサイズ
let (tx, rx) = mpsc::channel(256);  // 想定ピーク負荷の2-4倍

// OK: バックプレッシャーを意識した設計
let (tx, rx) = mpsc::channel(64);
// send().await は バッファ満杯時に待機する → 自然な流量制御
```

### アンチパターン2: タスクリークの放置

```rust
// NG: spawn したタスクの JoinHandle を捨てる
fn start_background() {
    tokio::spawn(async {
        loop {
            // 永遠に動き続けるタスク — 誰も止められない
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    });
    // JoinHandle が drop → タスクはデタッチされ制御不能
}

// OK: JoinHandle を保持してグレースフルシャットダウン
struct Service {
    handle: tokio::task::JoinHandle<()>,
    shutdown_tx: tokio::sync::watch::Sender<bool>,
}

impl Service {
    fn new() -> Self {
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::watch::channel(false);
        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() { break; }
                    }
                    _ = tokio::time::sleep(tokio::time::Duration::from_secs(1)) => {
                        println!("処理中...");
                    }
                }
            }
            println!("グレースフルに停止");
        });
        Service { handle, shutdown_tx }
    }

    async fn stop(self) {
        let _ = self.shutdown_tx.send(true);
        let _ = self.handle.await;
    }
}
```

---

## FAQ

### Q1: `spawn_blocking` と `block_on` の違いは?

**A:** `spawn_blocking` は非同期ランタイム内からブロッキング処理を専用スレッドに逃がす手段です。`block_on` は同期コードから非同期ランタイムを起動する手段です。

```rust
// spawn_blocking: async内 → 同期処理を別スレッドへ
async fn example() {
    let result = tokio::task::spawn_blocking(|| {
        std::fs::read_to_string("large.txt") // ブロッキング I/O
    }).await.unwrap();
}

// block_on: 同期コード → async を実行
fn sync_context() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(async { fetch_data().await });
}
```

### Q2: `select!` でキャンセルされたブランチはどうなる?

**A:** 選ばれなかったブランチの Future は drop されます。リソースリークを避けるため、途中状態のクリーンアップが必要な処理は `CancellationToken` で明示的に管理しましょう。

### Q3: tokio と OS スレッドのパフォーマンス差は?

**A:** tokio タスクは約 256 バイト、OS スレッドは約 8MB(スタック)のメモリを消費します。10万同時接続では tokio が圧倒的に有利です。ただし CPU バウンドの処理ではスレッドプールの方が適切です。

---

## まとめ

| 項目 | 要点 |
|---|---|
| ランタイム選択 | `multi_thread` がデフォルト。テストは `current_thread` |
| タスク spawn | `Send + 'static` 制約。JoinHandle で結果取得 |
| JoinSet | 動的タスク集合。完了順に結果取得。一括キャンセル可 |
| mpsc | 最も汎用的。バックプレッシャー付きキュー |
| oneshot | リクエスト-レスポンスパターン |
| broadcast | Pub/Sub パターン |
| watch | 状態監視・設定変更通知 |
| シャットダウン | watch + select! でグレースフル停止 |

## 次に読むべきガイド

- [非同期パターン](./02-async-patterns.md) — Stream、並行制限、リトライパターン
- [ネットワーク](./03-networking.md) — 非同期HTTP/WebSocket/gRPC
- [並行性](../03-systems/01-concurrency.md) — スレッドとロックの基礎

## 参考文献

1. **Tokio Documentation**: https://docs.rs/tokio/latest/tokio/
2. **Tokio Tutorial**: https://tokio.rs/tokio/tutorial
3. **Alice Ryhl - Actors with Tokio**: https://ryhl.io/blog/actors-with-tokio/
