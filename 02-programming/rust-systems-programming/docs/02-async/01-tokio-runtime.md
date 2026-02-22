# Tokioランタイム — タスク管理とチャネル

> Tokio のマルチスレッドランタイムの内部構造、タスクスポーン、チャネルを使った非同期メッセージパッシングを習得する

## この章で学ぶこと

1. **ランタイム構成** — マルチスレッド/シングルスレッドランタイムの選択と設定
2. **タスク管理** — spawn, JoinSet, abort, タスクローカルストレージ
3. **非同期チャネル** — mpsc, oneshot, broadcast, watch の使い分け
4. **同期プリミティブ** — Mutex, RwLock, Semaphore, Notify, Barrier
5. **タスクローカルストレージ** — task_local! によるタスクごとのコンテキスト伝搬

---

## 1. Tokio ランタイムのアーキテクチャ

### 1.1 全体構造

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

### 1.2 ワークスティーリングスケジューラの動作

```
┌──────────── Work-Stealing Scheduler ──────────────┐
│                                                      │
│  Worker #1          Worker #2          Worker #3    │
│  ┌──────┐          ┌──────┐          ┌──────┐    │
│  │ T1   │          │ T4   │          │      │    │
│  │ T2   │          │ T5   │          │ (空) │    │
│  │ T3   │          │      │          │      │    │
│  └──────┘          └──────┘          └──────┘    │
│      │                                    ▲        │
│      │         ワークスティーリング         │        │
│      └────────────────────────────────────┘        │
│      Worker #3 が Worker #1 からタスクを盗む       │
│                                                      │
│  メリット:                                           │
│  - 負荷の自動分散                                    │
│  - アイドルスレッドの最小化                           │
│  - キャッシュ局所性の維持 (ローカルキュー優先)       │
└──────────────────────────────────────────────────────┘
```

### 1.3 I/O Driver の役割

I/O Driver は OS のイベント通知機構（Linux: epoll、macOS: kqueue、Windows: IOCP）を抽象化し、非同期 I/O イベントを Tokio ランタイムに橋渡しします。

```
┌─────────── I/O Driver の動作フロー ───────────┐
│                                                  │
│  ① タスクがソケット読み取りを要求               │
│     → I/O Driver に関心を登録 (epoll_ctl)       │
│     → タスクは Pending を返して中断              │
│                                                  │
│  ② I/O Driver がイベントを監視                  │
│     → epoll_wait でブロック (他タスクは実行中)   │
│                                                  │
│  ③ データ到着                                   │
│     → epoll_wait が返る                          │
│     → 対応する Waker を呼ぶ                      │
│                                                  │
│  ④ タスクが再スケジュール                       │
│     → ワーカースレッドが poll() を再実行         │
│     → Ready(data) を返す                         │
└──────────────────────────────────────────────────┘
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

// パターン3: ワーカースレッド数を指定
#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    println!("4スレッドランタイム");
}

// パターン4: 手動ビルド (詳細制御)
fn main() {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)                       // ワーカースレッド数
        .max_blocking_threads(64)                // ブロッキングスレッド上限
        .thread_name("my-worker")                // スレッド名
        .thread_stack_size(3 * 1024 * 1024)     // スタックサイズ 3MB
        .enable_all()                             // I/O + Timer ドライバ有効
        .on_thread_start(|| {
            println!("スレッド開始: {:?}", std::thread::current().id());
        })
        .on_thread_stop(|| {
            println!("スレッド停止: {:?}", std::thread::current().id());
        })
        .build()
        .unwrap();

    runtime.block_on(async {
        println!("手動構築ランタイム");
    });
}

// パターン5: シングルスレッドランタイムの手動構築
fn main_single() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime.block_on(async {
        println!("シングルスレッドランタイム (手動)");
    });
}
```

### コード例2: 複数ランタイムの使い分け

```rust
use tokio::runtime::Runtime;

/// CPU集中処理用とI/O処理用のランタイムを分離
struct AppRuntime {
    io_runtime: Runtime,
    cpu_runtime: Runtime,
}

impl AppRuntime {
    fn new() -> Self {
        let io_runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .thread_name("io-worker")
            .enable_all()
            .build()
            .unwrap();

        let cpu_runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(num_cpus::get())
            .thread_name("cpu-worker")
            .enable_all()
            .build()
            .unwrap();

        AppRuntime { io_runtime, cpu_runtime }
    }

    /// I/Oバウンドタスクを実行
    fn spawn_io<F>(&self, future: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        self.io_runtime.spawn(future)
    }

    /// CPUバウンドタスクをブロッキングプールで実行
    fn spawn_cpu<F, R>(&self, f: F) -> tokio::task::JoinHandle<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.cpu_runtime.spawn_blocking(f)
    }
}
```

### コード例3: ランタイムメトリクスの監視

```rust
use tokio::runtime::Handle;

#[tokio::main]
async fn main() {
    let handle = Handle::current();

    // ランタイムメトリクスの取得 (tokio_unstable が必要)
    // RUSTFLAGS="--cfg tokio_unstable" cargo run
    #[cfg(tokio_unstable)]
    {
        let metrics = handle.metrics();
        println!("ワーカースレッド数: {}", metrics.num_workers());
        println!("アクティブタスク数: {}", metrics.active_tasks_count());
        println!("ブロッキングスレッド数: {}", metrics.num_blocking_threads());

        // 定期的にメトリクスを出力
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5));
            loop {
                interval.tick().await;
                let m = Handle::current().metrics();
                println!(
                    "[metrics] active_tasks={}, blocking_threads={}",
                    m.active_tasks_count(),
                    m.num_blocking_threads(),
                );
            }
        });
    }

    // メインアプリケーション処理
    tokio::time::sleep(tokio::time::Duration::from_secs(30)).await;
}
```

---

## 3. タスク管理

### コード例4: spawn と JoinHandle

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

    // JoinHandle の is_finished() で完了チェック (ノンブロッキング)
    let h3 = tokio::spawn(async {
        sleep(Duration::from_secs(2)).await;
        "done"
    });

    // ポーリング的な完了チェック
    for _ in 0..5 {
        if h3.is_finished() {
            println!("タスク完了!");
            break;
        }
        println!("まだ実行中...");
        sleep(Duration::from_millis(500)).await;
    }
    let result = h3.await.unwrap();
    println!("結果: {}", result);
}
```

### コード例5: JoinSet による動的タスク管理

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

    // JoinSet の len() でアクティブタスク数を確認
    println!("残りタスク: {}", set2.len());
}
```

### コード例6: JoinSet を使った並行度制限付きタスク実行

```rust
use tokio::task::JoinSet;
use tokio::time::{sleep, Duration};

/// 最大 max_concurrent 個のタスクを同時実行する
async fn process_with_limit(items: Vec<u32>, max_concurrent: usize) -> Vec<String> {
    let mut set = JoinSet::new();
    let mut results = Vec::new();
    let mut iter = items.into_iter();

    // 最初に max_concurrent 個のタスクを投入
    for _ in 0..max_concurrent {
        if let Some(item) = iter.next() {
            set.spawn(async move {
                sleep(Duration::from_millis(100)).await;
                format!("Item#{} 処理完了", item)
            });
        }
    }

    // 1つ完了するたびに次のタスクを投入
    while let Some(result) = set.join_next().await {
        match result {
            Ok(msg) => results.push(msg),
            Err(e) => eprintln!("エラー: {}", e),
        }

        // 次のアイテムがあればタスクを追加
        if let Some(item) = iter.next() {
            set.spawn(async move {
                sleep(Duration::from_millis(100)).await;
                format!("Item#{} 処理完了", item)
            });
        }
    }

    results
}

#[tokio::main]
async fn main() {
    let items: Vec<u32> = (1..=20).collect();
    let results = process_with_limit(items, 5).await; // 最大5並行
    println!("処理完了: {} 件", results.len());
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
│                                                     │
│  abort() 呼び出し時:                                │
│    - 次の .await ポイントでキャンセルされる          │
│    - 実行中の同期コードは完了まで実行される          │
│    - Drop が正しく実行される (RAII安全)             │
└─────────────────────────────────────────────────────┘
```

### コード例7: タスクのキャンセルとクリーンアップ

```rust
use tokio::time::{sleep, Duration};

struct Resource {
    name: String,
}

impl Resource {
    fn new(name: &str) -> Self {
        println!("  [{}] リソース作成", name);
        Resource { name: name.to_string() }
    }
}

impl Drop for Resource {
    fn drop(&mut self) {
        println!("  [{}] リソース解放 (Drop)", self.name);
    }
}

#[tokio::main]
async fn main() {
    // abort によるキャンセル
    let handle = tokio::spawn(async {
        let _resource = Resource::new("task-resource");
        println!("  タスク開始。長い処理...");
        sleep(Duration::from_secs(60)).await;
        println!("  タスク完了 (この行は実行されない)");
    });

    // 500ms 後にキャンセル
    sleep(Duration::from_millis(500)).await;
    handle.abort();

    match handle.await {
        Ok(_) => println!("タスク正常完了"),
        Err(e) if e.is_cancelled() => println!("タスクはキャンセルされた"),
        Err(e) => println!("タスクがパニック: {}", e),
    }
    // 出力:
    //   [task-resource] リソース作成
    //   タスク開始。長い処理...
    //   [task-resource] リソース解放 (Drop)
    //   タスクはキャンセルされた
}
```

### コード例8: タスクローカルストレージ

```rust
use tokio::task_local;

task_local! {
    static REQUEST_ID: String;
    static USER_ID: u64;
}

async fn handle_request() {
    let req_id = REQUEST_ID.with(|id| id.clone());
    let user_id = USER_ID.with(|id| *id);
    println!("リクエスト {} (ユーザー {}): 処理中...", req_id, user_id);

    // サブ関数でもアクセス可能
    process_data().await;
}

async fn process_data() {
    let req_id = REQUEST_ID.with(|id| id.clone());
    println!("  [{}] データ処理中...", req_id);
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    println!("  [{}] データ処理完了", req_id);
}

#[tokio::main]
async fn main() {
    // task_local! 変数にスコープを設定して実行
    let handle1 = tokio::spawn(
        REQUEST_ID.scope("req-001".to_string(),
            USER_ID.scope(42,
                handle_request()
            )
        )
    );

    let handle2 = tokio::spawn(
        REQUEST_ID.scope("req-002".to_string(),
            USER_ID.scope(99,
                handle_request()
            )
        )
    );

    let _ = tokio::join!(handle1, handle2);
}
```

### コード例9: spawn_local と LocalSet

```rust
use tokio::task::LocalSet;
use std::rc::Rc;

#[tokio::main]
async fn main() {
    // LocalSet: !Send な Future を実行するための環境
    let local = LocalSet::new();

    local.run_until(async {
        // Rc は Send ではないが、spawn_local なら使える
        let data = Rc::new(vec![1, 2, 3]);

        let data_clone = data.clone();
        tokio::task::spawn_local(async move {
            println!("ローカルタスク: {:?}", data_clone);
        }).await.unwrap();

        // 複数のローカルタスクを生成
        let mut handles = Vec::new();
        for i in 0..5 {
            let d = data.clone();
            let handle = tokio::task::spawn_local(async move {
                tokio::time::sleep(tokio::time::Duration::from_millis(i * 10)).await;
                format!("Task {} completed with data len {}", i, d.len())
            });
            handles.push(handle);
        }

        for handle in handles {
            println!("{}", handle.await.unwrap());
        }
    }).await;
}
```

---

## 4. チャネル

### コード例10: mpsc (多対一) チャネル

```rust
use tokio::sync::mpsc;

#[derive(Debug)]
enum Command {
    Get { key: String },
    Set { key: String, value: String },
    Delete { key: String },
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
                Command::Delete { key } => {
                    if store.remove(&key).is_some() {
                        println!("[worker] DELETE {} (成功)", key);
                    } else {
                        println!("[worker] DELETE {} (キー未存在)", key);
                    }
                }
                Command::Shutdown => {
                    println!("[worker] シャットダウン");
                    break;
                }
            }
        }
        println!("[worker] ストア最終状態: {:?}", store);
    });

    // 複数の送信者
    let tx2 = tx.clone();
    tokio::spawn(async move {
        tx2.send(Command::Set {
            key: "name".into(),
            value: "Alice".into(),
        }).await.unwrap();
    });

    let tx3 = tx.clone();
    tokio::spawn(async move {
        tx3.send(Command::Set {
            key: "role".into(),
            value: "admin".into(),
        }).await.unwrap();
    });

    tx.send(Command::Set {
        key: "age".into(),
        value: "30".into(),
    }).await.unwrap();

    tx.send(Command::Get { key: "name".into() }).await.unwrap();
    tx.send(Command::Delete { key: "role".into() }).await.unwrap();
    tx.send(Command::Shutdown).await.unwrap();

    worker.await.unwrap();
}
```

### コード例11: mpsc を使ったリクエスト-レスポンスパターン

```rust
use tokio::sync::{mpsc, oneshot};

#[derive(Debug)]
enum DbCommand {
    Get {
        key: String,
        reply: oneshot::Sender<Option<String>>,
    },
    Set {
        key: String,
        value: String,
        reply: oneshot::Sender<bool>,
    },
}

/// データベースアクタータスク
async fn db_actor(mut rx: mpsc::Receiver<DbCommand>) {
    let mut store = std::collections::HashMap::new();

    while let Some(cmd) = rx.recv().await {
        match cmd {
            DbCommand::Get { key, reply } => {
                let value = store.get(&key).cloned();
                let _ = reply.send(value);
            }
            DbCommand::Set { key, value, reply } => {
                store.insert(key, value);
                let _ = reply.send(true);
            }
        }
    }
}

/// データベースクライアント
#[derive(Clone)]
struct DbClient {
    tx: mpsc::Sender<DbCommand>,
}

impl DbClient {
    async fn get(&self, key: &str) -> Option<String> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx.send(DbCommand::Get {
            key: key.to_string(),
            reply: reply_tx,
        }).await.ok()?;
        reply_rx.await.ok()?
    }

    async fn set(&self, key: &str, value: &str) -> bool {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx.send(DbCommand::Set {
            key: key.to_string(),
            value: value.to_string(),
            reply: reply_tx,
        }).await.ok();
        reply_rx.await.unwrap_or(false)
    }
}

#[tokio::main]
async fn main() {
    let (tx, rx) = mpsc::channel(32);
    tokio::spawn(db_actor(rx));

    let client = DbClient { tx };

    // 複数クライアントが並行アクセス
    let c1 = client.clone();
    let c2 = client.clone();

    let h1 = tokio::spawn(async move {
        c1.set("name", "Alice").await;
        c1.set("email", "alice@example.com").await;
    });

    let h2 = tokio::spawn(async move {
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        let name = c2.get("name").await;
        println!("name = {:?}", name);
        let email = c2.get("email").await;
        println!("email = {:?}", email);
    });

    let _ = tokio::join!(h1, h2);
}
```

### コード例12: oneshot (一対一・一回限り)

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

    // oneshot の drop 検出
    let (tx2, rx2) = oneshot::channel::<String>();
    drop(tx2); // 送信側を drop

    match rx2.await {
        Ok(value) => println!("受信: {}", value),
        Err(_) => println!("送信側が閉じた (値が送られなかった)"),
    }
}
```

### コード例13: broadcast と watch

```rust
use tokio::sync::{broadcast, watch};

#[tokio::main]
async fn main() {
    // ── broadcast — 多対多。全受信者にクローン送信 ──
    let (btx, _) = broadcast::channel::<String>(16);
    let mut brx1 = btx.subscribe();
    let mut brx2 = btx.subscribe();

    btx.send("ブロードキャスト!".into()).unwrap();

    println!("rx1: {}", brx1.recv().await.unwrap());
    println!("rx2: {}", brx2.recv().await.unwrap());

    // 遅延サブスクライバー: subscribe した後に送信されたメッセージのみ受信
    let mut brx3 = btx.subscribe();
    btx.send("新メッセージ".into()).unwrap();
    println!("rx3: {}", brx3.recv().await.unwrap());

    // バッファオーバーフロー時のエラーハンドリング
    let (btx2, _) = broadcast::channel::<u32>(2); // バッファサイズ2
    let mut brx = btx2.subscribe();
    btx2.send(1).unwrap();
    btx2.send(2).unwrap();
    btx2.send(3).unwrap(); // バッファオーバーフロー → 最も古いメッセージが失われる

    match brx.recv().await {
        Ok(val) => println!("受信: {}", val),
        Err(broadcast::error::RecvError::Lagged(n)) => {
            println!("{} メッセージが失われた", n);
        }
        Err(broadcast::error::RecvError::Closed) => {
            println!("チャネル閉鎖");
        }
    }

    // ── watch — 最新値のみ保持。設定変更通知に最適 ──
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

### コード例14: watch を使った設定ホットリロード

```rust
use tokio::sync::watch;
use serde::Deserialize;
use std::sync::Arc;

#[derive(Debug, Clone, Deserialize)]
struct AppConfig {
    log_level: String,
    max_connections: u32,
    feature_flags: Vec<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        AppConfig {
            log_level: "info".to_string(),
            max_connections: 100,
            feature_flags: vec![],
        }
    }
}

async fn config_watcher(config_tx: watch::Sender<Arc<AppConfig>>) {
    // 設定ファイルの変更を監視 (簡易版)
    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(5));

    loop {
        interval.tick().await;

        // 設定ファイルを読み込み (実際にはファイルシステムウォッチャーを使う)
        match tokio::fs::read_to_string("config.toml").await {
            Ok(content) => {
                match toml::from_str::<AppConfig>(&content) {
                    Ok(new_config) => {
                        println!("設定リロード: {:?}", new_config);
                        let _ = config_tx.send(Arc::new(new_config));
                    }
                    Err(e) => eprintln!("設定パースエラー: {}", e),
                }
            }
            Err(_) => {} // ファイルが存在しない場合はスキップ
        }
    }
}

async fn worker(id: u32, mut config_rx: watch::Receiver<Arc<AppConfig>>) {
    loop {
        // 設定変更の通知を待つ
        tokio::select! {
            Ok(()) = config_rx.changed() => {
                let config = config_rx.borrow().clone();
                println!("ワーカー {} が新設定を受信: log_level={}", id, config.log_level);
            }
            _ = tokio::time::sleep(tokio::time::Duration::from_secs(1)) => {
                let config = config_rx.borrow().clone();
                println!(
                    "ワーカー {} 処理中 (max_conn={})",
                    id, config.max_connections
                );
            }
        }
    }
}
```

---

## 5. 同期プリミティブ

### コード例15: tokio::sync::Mutex と RwLock

```rust
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

#[derive(Debug)]
struct SharedCache {
    data: std::collections::HashMap<String, String>,
    hits: u64,
    misses: u64,
}

#[tokio::main]
async fn main() {
    // Mutex: await 中も安全にロックを保持できる
    let cache = Arc::new(Mutex::new(SharedCache {
        data: std::collections::HashMap::new(),
        hits: 0,
        misses: 0,
    }));

    // 複数タスクからのアクセス
    let mut handles = Vec::new();
    for i in 0..10 {
        let cache = cache.clone();
        handles.push(tokio::spawn(async move {
            let mut c = cache.lock().await; // 非同期ロック取得
            c.data.insert(format!("key_{}", i), format!("value_{}", i));
            // ロック中に await しても安全 (tokio::sync::Mutex の場合)
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }));
    }
    for h in handles { h.await.unwrap(); }

    // RwLock: 読み取りは並行、書き込みは排他
    let config = Arc::new(RwLock::new(vec!["setting1".to_string()]));

    // 読み取りタスク (並行実行可能)
    let c1 = config.clone();
    let c2 = config.clone();
    let r1 = tokio::spawn(async move {
        let guard = c1.read().await;
        println!("Reader 1: {:?}", *guard);
    });
    let r2 = tokio::spawn(async move {
        let guard = c2.read().await;
        println!("Reader 2: {:?}", *guard);
    });

    // 書き込みタスク (排他)
    let c3 = config.clone();
    let w1 = tokio::spawn(async move {
        let mut guard = c3.write().await;
        guard.push("setting2".to_string());
        println!("Writer: 設定追加");
    });

    let _ = tokio::join!(r1, r2, w1);
}
```

### コード例16: Notify — イベント通知

```rust
use std::sync::Arc;
use tokio::sync::Notify;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    let notify = Arc::new(Notify::new());

    // 待機側
    let n1 = notify.clone();
    let waiter = tokio::spawn(async move {
        println!("通知を待機中...");
        n1.notified().await;
        println!("通知を受信!");
    });

    // 通知側
    sleep(Duration::from_millis(500)).await;
    println!("通知を送信");
    notify.notify_one(); // 1つの待機タスクを起床

    waiter.await.unwrap();

    // notify_waiters(): 全ての待機タスクを起床
    let notify = Arc::new(Notify::new());
    let mut handles = Vec::new();

    for i in 0..5 {
        let n = notify.clone();
        handles.push(tokio::spawn(async move {
            n.notified().await;
            println!("ワーカー {} が起床", i);
        }));
    }

    sleep(Duration::from_millis(100)).await;
    notify.notify_waiters(); // 全員起床

    for h in handles { h.await.unwrap(); }
}
```

### コード例17: Barrier — 同期ポイント

```rust
use std::sync::Arc;
use tokio::sync::Barrier;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    let barrier = Arc::new(Barrier::new(4)); // 4タスクが揃うまで待機

    let mut handles = Vec::new();
    for i in 0..4 {
        let b = barrier.clone();
        handles.push(tokio::spawn(async move {
            // フェーズ1: 初期化
            println!("ワーカー {}: 初期化中...", i);
            sleep(Duration::from_millis(i as u64 * 100)).await;
            println!("ワーカー {}: 初期化完了。バリアで待機", i);

            // 全ワーカーが到達するまで待機
            let result = b.wait().await;
            if result.is_leader() {
                println!("ワーカー {} はリーダー (最後に到達)", i);
            }

            // フェーズ2: 全員揃ったら実行
            println!("ワーカー {}: メイン処理開始!", i);
        }));
    }

    for h in handles { h.await.unwrap(); }
}
```

---

## 6. 比較表

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

### 同期プリミティブ比較

| プリミティブ | 用途 | std 版との違い |
|---|---|---|
| `tokio::sync::Mutex` | 排他ロック | `.await` 中も安全にロック保持可能 |
| `tokio::sync::RwLock` | 読み書きロック | 非同期コンテキスト対応 |
| `tokio::sync::Semaphore` | リソース制限 | 非同期 `acquire().await` |
| `tokio::sync::Notify` | イベント通知 | `std` に相当なし。条件変数的用途 |
| `tokio::sync::Barrier` | 同期ポイント | 非同期対応。全タスク到達まで待機 |
| `tokio::sync::OnceCell` | 遅延初期化 | 非同期初期化関数に対応 |

### std::sync vs tokio::sync の使い分け

| 状況 | 推奨 | 理由 |
|---|---|---|
| ロック範囲に `.await` を含まない | `std::sync::Mutex` | 軽量、オーバーヘッド小 |
| ロック範囲に `.await` を含む | `tokio::sync::Mutex` | デッドロック防止 |
| 短時間のアトミック操作 | `std::sync::atomic` | 最軽量 |
| 複数タスク間の通知 | `tokio::sync::Notify` | 非同期対応必須 |
| 設定共有 (読み取り多) | `Arc<std::sync::RwLock>` | 読み取りロックが軽量 |
| 設定変更通知 | `tokio::sync::watch` | 変更通知が組み込み |

---

## 7. アンチパターン

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

### アンチパターン3: tokio::sync::Mutex の不必要な使用

```rust
// NG: await を含まないのに tokio::sync::Mutex を使う
use tokio::sync::Mutex;
async fn bad_counter(counter: &Mutex<u64>) {
    let mut c = counter.lock().await;
    *c += 1;
    // await ポイントなし → std::sync::Mutex で十分
}

// OK: await がないなら std::sync::Mutex が軽量
use std::sync::Mutex;
async fn good_counter(counter: &Mutex<u64>) {
    let mut c = counter.lock().unwrap();
    *c += 1;
    // drop(c); — スコープ終了で自動解放
}

// ベスト: 単純なカウンタならアトミック
use std::sync::atomic::{AtomicU64, Ordering};
async fn best_counter(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}
```

### アンチパターン4: ランタイム内での block_on 呼び出し

```rust
// NG: 既に非同期ランタイム内で block_on を呼ぶとパニック
#[tokio::main]
async fn main() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    // rt.block_on(some_future()); // パニック: "Cannot start a runtime from within a runtime"
}

// OK: 非同期ランタイム内では .await を使う
#[tokio::main]
async fn main() {
    let result = some_future().await;
    println!("{}", result);
}

// OK: 同期コンテキストから非同期コードを呼ぶ場合
fn sync_function() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(async { some_future().await });
    println!("{}", result);
}

async fn some_future() -> String {
    "result".to_string()
}
```

---

## 8. 実践パターン集

### 8.1 アクターパターン

```rust
use tokio::sync::{mpsc, oneshot};

/// アクターが処理するメッセージ
enum ActorMessage {
    Increment,
    Decrement,
    GetValue { reply: oneshot::Sender<i64> },
}

/// カウンターアクター
struct CounterActor {
    value: i64,
    rx: mpsc::Receiver<ActorMessage>,
}

impl CounterActor {
    fn new(rx: mpsc::Receiver<ActorMessage>) -> Self {
        CounterActor { value: 0, rx }
    }

    async fn run(mut self) {
        while let Some(msg) = self.rx.recv().await {
            match msg {
                ActorMessage::Increment => self.value += 1,
                ActorMessage::Decrement => self.value -= 1,
                ActorMessage::GetValue { reply } => {
                    let _ = reply.send(self.value);
                }
            }
        }
    }
}

/// アクターへのハンドル (クローン可能)
#[derive(Clone)]
struct CounterHandle {
    tx: mpsc::Sender<ActorMessage>,
}

impl CounterHandle {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel(32);
        let actor = CounterActor::new(rx);
        tokio::spawn(actor.run());
        CounterHandle { tx }
    }

    async fn increment(&self) {
        let _ = self.tx.send(ActorMessage::Increment).await;
    }

    async fn decrement(&self) {
        let _ = self.tx.send(ActorMessage::Decrement).await;
    }

    async fn get_value(&self) -> i64 {
        let (reply_tx, reply_rx) = oneshot::channel();
        let _ = self.tx.send(ActorMessage::GetValue { reply: reply_tx }).await;
        reply_rx.await.unwrap_or(0)
    }
}

#[tokio::main]
async fn main() {
    let counter = CounterHandle::new();

    // 複数タスクから並行操作
    let mut handles = Vec::new();
    for _ in 0..100 {
        let c = counter.clone();
        handles.push(tokio::spawn(async move {
            c.increment().await;
        }));
    }
    for h in handles { h.await.unwrap(); }

    let value = counter.get_value().await;
    println!("最終値: {}", value); // 100
}
```

### 8.2 ワーカープールパターン

```rust
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};

struct Job {
    id: u32,
    data: String,
}

struct WorkerPool {
    job_tx: mpsc::Sender<Job>,
    handles: Vec<tokio::task::JoinHandle<()>>,
}

impl WorkerPool {
    fn new(num_workers: usize) -> Self {
        let (job_tx, job_rx) = mpsc::channel::<Job>(100);
        let job_rx = std::sync::Arc::new(tokio::sync::Mutex::new(job_rx));
        let mut handles = Vec::new();

        for worker_id in 0..num_workers {
            let rx = job_rx.clone();
            let handle = tokio::spawn(async move {
                loop {
                    let job = {
                        let mut rx = rx.lock().await;
                        rx.recv().await
                    };

                    match job {
                        Some(job) => {
                            println!("ワーカー {}: ジョブ {} 処理中 ({})",
                                worker_id, job.id, job.data);
                            sleep(Duration::from_millis(100)).await;
                            println!("ワーカー {}: ジョブ {} 完了", worker_id, job.id);
                        }
                        None => {
                            println!("ワーカー {}: シャットダウン", worker_id);
                            break;
                        }
                    }
                }
            });
            handles.push(handle);
        }

        WorkerPool { job_tx, handles }
    }

    async fn submit(&self, job: Job) -> Result<(), mpsc::error::SendError<Job>> {
        self.job_tx.send(job).await
    }

    async fn shutdown(self) {
        drop(self.job_tx); // チャネルを閉じる
        for handle in self.handles {
            let _ = handle.await;
        }
    }
}

#[tokio::main]
async fn main() {
    let pool = WorkerPool::new(4);

    // ジョブ投入
    for i in 0..20 {
        pool.submit(Job {
            id: i,
            data: format!("データ_{}", i),
        }).await.unwrap();
    }

    pool.shutdown().await;
    println!("全ジョブ完了");
}
```

### 8.3 定期タスクスケジューラー

```rust
use tokio::time::{interval, Duration, Instant};

struct Scheduler {
    tasks: Vec<ScheduledTask>,
}

struct ScheduledTask {
    name: String,
    interval: Duration,
    task: Box<dyn Fn() -> tokio::task::JoinHandle<()> + Send + Sync>,
}

impl Scheduler {
    fn new() -> Self {
        Scheduler { tasks: Vec::new() }
    }

    fn add_task<F, Fut>(&mut self, name: &str, interval_ms: u64, task_fn: F)
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        self.tasks.push(ScheduledTask {
            name: name.to_string(),
            interval: Duration::from_millis(interval_ms),
            task: Box::new(move || tokio::spawn(task_fn())),
        });
    }

    async fn run(self, shutdown: tokio::sync::watch::Receiver<bool>) {
        let mut handles = Vec::new();

        for scheduled in &self.tasks {
            let name = scheduled.name.clone();
            let interval_duration = scheduled.interval;
            let mut shutdown_rx = shutdown.clone();

            // 各タスクのスケジューラーループ
            let handle = tokio::spawn(async move {
                let mut ticker = interval(interval_duration);

                loop {
                    tokio::select! {
                        _ = ticker.tick() => {
                            let start = Instant::now();
                            println!("[{}] 実行開始", name);
                            // ここで実際のタスクを実行
                            tokio::time::sleep(Duration::from_millis(50)).await;
                            println!("[{}] 実行完了 ({:?})", name, start.elapsed());
                        }
                        Ok(()) = shutdown_rx.changed() => {
                            if *shutdown_rx.borrow() {
                                println!("[{}] シャットダウン", name);
                                break;
                            }
                        }
                    }
                }
            });
            handles.push(handle);
        }

        for h in handles {
            let _ = h.await;
        }
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

```rust
use tokio_util::sync::CancellationToken;

async fn careful_select() {
    let token = CancellationToken::new();

    let token_clone = token.clone();
    let task = tokio::spawn(async move {
        tokio::select! {
            _ = token_clone.cancelled() => {
                println!("キャンセルされた。クリーンアップ...");
                // リソースの明示的解放
            }
            result = long_running_task() => {
                println!("タスク完了: {:?}", result);
            }
        }
    });

    // 条件に応じてキャンセル
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    token.cancel();
    let _ = task.await;
}

async fn long_running_task() -> String {
    tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
    "done".to_string()
}
```

### Q3: tokio と OS スレッドのパフォーマンス差は?

**A:** tokio タスクは約 256 バイト、OS スレッドは約 8MB(スタック)のメモリを消費します。10万同時接続では tokio が圧倒的に有利です。ただし CPU バウンドの処理ではスレッドプールの方が適切です。

| 指標 | tokio タスク | OS スレッド |
|---|---|---|
| メモリ (初期) | ~256 バイト | ~8 MB (スタック) |
| 生成コスト | ~数マイクロ秒 | ~数ミリ秒 |
| コンテキストスイッチ | ユーザー空間 (~ns) | カーネル空間 (~μs) |
| 10万タスク時のメモリ | ~25 MB | ~800 GB (非現実的) |
| 最適な用途 | I/Oバウンド | CPUバウンド |

### Q4: mpsc チャネルのバッファサイズの目安は?

**A:** 一般的な指針として、ピーク時の秒間メッセージ数の 2〜4 倍を設定します。小さすぎると送信側がブロックされ、大きすぎるとメモリを無駄に消費します。バックプレッシャーが効くサイズにするのがポイントです。

```rust
// ウェブサーバー (100 req/s 想定): 256 程度
let (tx, rx) = mpsc::channel(256);

// 高スループットパイプライン (10K msg/s): 1024〜4096
let (tx, rx) = mpsc::channel(4096);

// 低頻度コマンド (設定変更等): 8〜32
let (tx, rx) = mpsc::channel(16);
```

### Q5: タスクがパニックした場合の挙動は?

**A:** `tokio::spawn` で生成したタスクがパニックした場合、他のタスクには影響しません。JoinHandle の `.await` が `JoinError` を返します。

```rust
#[tokio::main]
async fn main() {
    let handle = tokio::spawn(async {
        panic!("タスク内でパニック!");
    });

    match handle.await {
        Ok(_) => println!("正常完了"),
        Err(e) if e.is_panic() => {
            // パニックの値を取得
            let panic_value = e.into_panic();
            if let Some(msg) = panic_value.downcast_ref::<&str>() {
                eprintln!("パニック: {}", msg);
            }
        }
        Err(e) if e.is_cancelled() => println!("キャンセル"),
        Err(e) => eprintln!("その他のエラー: {}", e),
    }
    // 他のタスクは正常に動作を継続
}
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| ランタイム選択 | `multi_thread` がデフォルト。テストは `current_thread` |
| ワークスティーリング | 負荷の自動分散。ローカルキュー → グローバルキュー → 他スレッドから盗む |
| タスク spawn | `Send + 'static` 制約。JoinHandle で結果取得 |
| JoinSet | 動的タスク集合。完了順に結果取得。一括キャンセル可 |
| spawn_local | `!Send` な Future 用。LocalSet 内でのみ使用 |
| task_local! | タスクごとのコンテキスト情報 (リクエストID等) |
| mpsc | 最も汎用的。バックプレッシャー付きキュー |
| oneshot | リクエスト-レスポンスパターン |
| broadcast | Pub/Sub パターン |
| watch | 状態監視・設定変更通知 |
| Mutex / RwLock | await を含むクリティカルセクションには tokio 版を使用 |
| Notify | イベント通知。条件変数的な用途 |
| Barrier | 全タスクの同期ポイント |
| シャットダウン | watch + select! でグレースフル停止 |
| アクターパターン | mpsc + oneshot で安全なメッセージパッシング |

## 次に読むべきガイド

- [非同期パターン](./02-async-patterns.md) — Stream、並行制限、リトライパターン
- [ネットワーク](./03-networking.md) — 非同期HTTP/WebSocket/gRPC
- [並行性](../03-systems/01-concurrency.md) — スレッドとロックの基礎

## 参考文献

1. **Tokio Documentation**: https://docs.rs/tokio/latest/tokio/
2. **Tokio Tutorial**: https://tokio.rs/tokio/tutorial
3. **Alice Ryhl - Actors with Tokio**: https://ryhl.io/blog/actors-with-tokio/
4. **Tokio Mini-Redis (学習用実装)**: https://github.com/tokio-rs/mini-redis
5. **Tokio Metrics**: https://docs.rs/tokio-metrics/latest/tokio_metrics/
6. **Jon Gjengset - Decrusting tokio**: https://www.youtube.com/watch?v=o2ob8zkeq2s
