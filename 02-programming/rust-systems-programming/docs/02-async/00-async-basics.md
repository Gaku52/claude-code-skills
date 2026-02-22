# async/await基礎 — Rustの非同期プログラミングモデル

> Future trait を中心とした Rust 非同期ランタイムの仕組みと async/await 構文の基礎を理解する

## この章で学ぶこと

1. **Future trait の仕組み** — Poll ベースの遅延評価モデルとゼロコスト抽象化
2. **async/await 構文** — 非同期関数の定義・呼び出し・合成パターン
3. **ランタイムの役割** — Executor、Reactor、Waker の協調動作
4. **Pin と Unpin** — 自己参照型の安全性を保証するメモリ固定の仕組み
5. **ライフタイムと非同期** — async 関数における借用と所有権の扱い

---

## 1. 同期 vs 非同期の全体像

### 1.1 基本概念図

```
┌──────────────────── 同期 (Sync) ────────────────────┐
│                                                      │
│  Thread 1: [タスクA ██████████████████████████████]  │
│  Thread 2: [タスクB ██████████████████████████████]  │
│  Thread 3: [タスクC ██████████████████████████████]  │
│                                                      │
│  → スレッド数 = 同時タスク数 (10K接続 = 10Kスレッド)   │
└──────────────────────────────────────────────────────┘

┌──────────────────── 非同期 (Async) ──────────────────┐
│                                                      │
│  Thread 1: [A██][B██][A██][C████][B██][A██]         │
│  Thread 2: [C██][A██][B████][C██][B██]              │
│                                                      │
│  → 少数スレッドで多数タスクを処理                      │
│  → I/O待ち中に他のタスクを実行                        │
└──────────────────────────────────────────────────────┘
```

### 1.2 なぜ非同期が必要なのか

同期モデルでは、I/O 操作（ネットワーク通信、ディスクアクセスなど）の完了を待つ間、スレッド全体がブロックされます。1万の同時接続を処理するには1万のスレッドが必要になり、メモリ消費（1スレッドあたり約8MBのスタック = 80GB）とコンテキストスイッチのオーバーヘッドが爆発的に増大します。

非同期モデルでは、I/O 待ち中にスレッドを他のタスクに活用できるため、少数のスレッド（通常はCPUコア数）で数万～数十万の同時接続を効率的に処理できます。

```rust
// 同期モデル: スレッドプールのサイズが同時処理数の上限
fn sync_handler(stream: TcpStream) {
    // この関数がブロックしている間、スレッドは他の仕事ができない
    let data = read_from_db();       // ~10ms ブロック
    let enriched = call_api(data);   // ~50ms ブロック
    stream.write_all(&enriched);     // ~1ms ブロック
}

// 非同期モデル: I/O 待ち中にスレッドが他のタスクを処理
async fn async_handler(stream: TcpStream) {
    // .await で中断し、完了したら再開される
    let data = read_from_db().await;       // 中断→再開
    let enriched = call_api(data).await;   // 中断→再開
    stream.write_all(&enriched).await;     // 中断→再開
}
```

### 1.3 Rust 非同期モデルの特徴

Rust の非同期モデルは他の言語と異なる重要な特徴を持ちます。

| 特徴 | Rust | Go | JavaScript | Python |
|---|---|---|---|---|
| 実行モデル | ゼロコスト Future | Goroutine (GC付き) | イベントループ | コルーチン |
| ランタイム | ユーザー選択 (tokio等) | 言語組み込み | V8エンジン組み込み | asyncio 標準 |
| スケジューリング | 協調的 | プリエンプティブ | 協調的 | 協調的 |
| メモリ割当 | スタック上 (ゼロアロケーション可) | ヒープ (Goroutineスタック) | ヒープ (Promise) | ヒープ |
| スレッドモデル | マルチスレッド対応 | M:Nスケジューリング | シングルスレッド | シングルスレッド (GIL) |
| 型安全性 | コンパイル時検証 | 実行時パニック可 | なし | 型ヒントのみ |

Rust の非同期は「ゼロコスト抽象化」を目指しており、async/await 構文はコンパイル時に状態マシンに変換されます。ランタイムが言語に組み込まれていないため、用途に応じて tokio、async-std、smol などのランタイムを選択できます。

---

## 2. Future trait の核心

### 2.1 Future trait の定義

```rust
use std::pin::Pin;
use std::task::{Context, Poll};

// 標準ライブラリの Future trait (簡略化)
pub trait Future {
    type Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
}

// Poll 列挙体
pub enum Poll<T> {
    Ready(T),   // 完了。値 T を返す
    Pending,     // 未完了。Waker で再通知される
}
```

`Future` trait は非同期計算の最も基本的な抽象化です。`poll` メソッドが呼ばれるたびに、計算が進行し、完了していれば `Ready(value)` を返し、まだ完了していなければ `Pending` を返します。`Pending` を返す際には、`Context` 内の `Waker` を登録し、準備が整った時点でランタイムに通知します。

### 2.2 手動 Future 実装

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

/// 指定時間後に完了する自作 Future
struct Delay {
    when: Instant,
}

impl Delay {
    fn new(duration: Duration) -> Self {
        Delay {
            when: Instant::now() + duration,
        }
    }
}

impl Future for Delay {
    type Output = String;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if Instant::now() >= self.when {
            Poll::Ready("時間経過!".to_string())
        } else {
            // 実際のランタイムではタイマーを登録して Waker を呼ぶ
            // ここでは即座に再ポーリングを要求 (ビジーウェイト)
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

#[tokio::main]
async fn main() {
    let msg = Delay::new(Duration::from_secs(1)).await;
    println!("{}", msg); // "時間経過!"
}
```

### 2.3 状態マシンとしての Future

async ブロックは、コンパイラによって状態マシンに変換されます。各 `.await` ポイントが状態遷移点となります。

```rust
// この async 関数:
async fn example() -> u64 {
    let a = step_one().await;    // 中断点1
    let b = step_two(a).await;   // 中断点2
    a + b
}

// コンパイラは概念的に以下のような状態マシンを生成:
enum ExampleFuture {
    // 初期状態: step_one の完了を待っている
    State0 {
        step_one_future: StepOneFuture,
    },
    // step_one 完了後: step_two の完了を待っている
    State1 {
        a: u64,
        step_two_future: StepTwoFuture,
    },
    // 完了状態
    Done,
}

impl Future for ExampleFuture {
    type Output = u64;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<u64> {
        loop {
            match self.as_mut().get_mut() {
                ExampleFuture::State0 { step_one_future } => {
                    // step_one をポーリング
                    match Pin::new(step_one_future).poll(cx) {
                        Poll::Ready(a) => {
                            // 次の状態に遷移
                            *self.as_mut().get_mut() = ExampleFuture::State1 {
                                a,
                                step_two_future: step_two(a),
                            };
                        }
                        Poll::Pending => return Poll::Pending,
                    }
                }
                ExampleFuture::State1 { a, step_two_future } => {
                    match Pin::new(step_two_future).poll(cx) {
                        Poll::Ready(b) => {
                            let result = *a + b;
                            *self.as_mut().get_mut() = ExampleFuture::Done;
                            return Poll::Ready(result);
                        }
                        Poll::Pending => return Poll::Pending,
                    }
                }
                ExampleFuture::Done => panic!("polled after completion"),
            }
        }
    }
}
```

この変換により、ヒープアロケーションなしで非同期処理が実現されます。各状態は enum のバリアントとして表現され、必要な変数はバリアントのフィールドとして保持されます。

### 2.4 Waker の仕組み

`Waker` は、Future が再びポーリングされるべきタイミングをランタイムに通知するためのメカニズムです。

```
┌──────────── Waker のライフサイクル ────────────┐
│                                                  │
│  ① Executor が Future::poll() を呼ぶ           │
│     Context に Waker を含めて渡す                │
│                                                  │
│  ② Future が Pending を返す                     │
│     → I/O ドライバに Waker を登録               │
│     → Executor は別のタスクを実行               │
│                                                  │
│  ③ I/O イベント発生 (データ到着等)              │
│     → Reactor が Waker.wake() を呼ぶ            │
│                                                  │
│  ④ Executor がタスクを再キューイング            │
│     → 次のポーリングサイクルで poll() 再実行     │
│                                                  │
│  ⑤ Future が Ready(value) を返す               │
│     → タスク完了                                │
└──────────────────────────────────────────────────┘
```

```rust
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// 値が設定されるまで待機する Future
struct SharedState {
    completed: bool,
    value: Option<String>,
    waker: Option<Waker>,
}

struct WaitForValue {
    shared: Arc<Mutex<SharedState>>,
}

impl Future for WaitForValue {
    type Output = String;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<String> {
        let mut state = self.shared.lock().unwrap();

        if state.completed {
            // 値が設定されていれば完了
            Poll::Ready(state.value.take().unwrap())
        } else {
            // Waker を保存して、値が設定された時に通知してもらう
            state.waker = Some(cx.waker().clone());
            Poll::Pending
        }
    }
}

/// 外部から値を設定して Waker を起動する
fn set_value(shared: &Arc<Mutex<SharedState>>, value: String) {
    let mut state = shared.lock().unwrap();
    state.value = Some(value);
    state.completed = true;
    // Waker が登録されていれば通知
    if let Some(waker) = state.waker.take() {
        waker.wake(); // Executor にタスクの再ポーリングを要求
    }
}
```

### 2.5 カスタム Future: タイマー付きリトライ

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use pin_project::pin_project;

/// 内部の Future にタイムアウトを付与する Future
#[pin_project]
struct WithTimeout<F> {
    #[pin]
    future: F,
    #[pin]
    delay: tokio::time::Sleep,
    timed_out: bool,
}

impl<F> WithTimeout<F> {
    fn new(future: F, timeout: std::time::Duration) -> Self {
        WithTimeout {
            future,
            delay: tokio::time::sleep(timeout),
            timed_out: false,
        }
    }
}

impl<F: Future> Future for WithTimeout<F> {
    type Output = Result<F::Output, &'static str>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();

        // まずタイムアウトを確認
        if this.delay.poll(cx).is_ready() {
            return Poll::Ready(Err("タイムアウト"));
        }

        // 本体の Future をポーリング
        match this.future.poll(cx) {
            Poll::Ready(value) => Poll::Ready(Ok(value)),
            Poll::Pending => Poll::Pending,
        }
    }
}

// 使用例
#[tokio::main]
async fn main() {
    let slow_operation = async {
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        "完了"
    };

    let result = WithTimeout::new(
        slow_operation,
        std::time::Duration::from_secs(2),
    ).await;

    match result {
        Ok(value) => println!("成功: {}", value),
        Err(e) => println!("エラー: {}", e), // "エラー: タイムアウト"
    }
}
```

---

## 3. async/await 構文

### 3.1 基本的な async 関数

```rust
use tokio::time::{sleep, Duration};

/// async fn は Future を返す関数のシンタックスシュガー
async fn fetch_data(url: &str) -> Result<String, reqwest::Error> {
    // .await で Future の完了を待つ
    let response = reqwest::get(url).await?;
    let body = response.text().await?;
    Ok(body)
}

/// 上記は以下と等価 (脱糖後)
fn fetch_data_desugared(url: &str) -> impl Future<Output = Result<String, reqwest::Error>> + '_ {
    async move {
        let response = reqwest::get(url).await?;
        let body = response.text().await?;
        Ok(body)
    }
}

#[tokio::main]
async fn main() {
    match fetch_data("https://httpbin.org/get").await {
        Ok(body) => println!("取得成功: {}バイト", body.len()),
        Err(e) => eprintln!("エラー: {}", e),
    }
}
```

### 3.2 非同期処理の実行フロー

```
┌─────────────────────────────────────────────────┐
│            async fn fetch_data() の内部           │
│                                                   │
│  ① async fn 呼び出し                              │
│     → Future (状態マシン) を生成 (まだ実行しない)   │
│                                                   │
│  ② .await                                        │
│     → Executor が poll() を呼ぶ                   │
│                                                   │
│  ③ Poll::Pending (I/O未完了)                      │
│     → Waker を登録してタスクを中断                  │
│     → Executor は他のタスクを実行                   │
│                                                   │
│  ④ I/O 完了通知 (Reactor → Waker)                 │
│     → Executor がタスクを再スケジュール             │
│                                                   │
│  ⑤ 再 poll() → Poll::Ready(value)                │
│     → .await が value を返す                      │
└─────────────────────────────────────────────────┘
```

### 3.3 async ブロックとクロージャ

```rust
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    // async ブロック: 無名の async 関数のようなもの
    let greeting = async {
        sleep(Duration::from_millis(100)).await;
        "Hello from async block!"
    };
    println!("{}", greeting.await);

    // async move ブロック: キャプチャした変数の所有権を移動
    let name = String::from("Rust");
    let greet = async move {
        // name の所有権がこのブロックに移動
        format!("Hello, {}!", name)
    };
    println!("{}", greet.await);
    // println!("{}", name); // コンパイルエラー: name は移動済み

    // 非同期クロージャ (nightly feature または async ブロックで代替)
    let urls = vec!["https://a.com", "https://b.com", "https://c.com"];
    let fetch_all = urls.iter().map(|url| {
        let url = url.to_string(); // クロージャの外でクローン
        async move {
            // reqwest::get(&url).await
            format!("fetched: {}", url)
        }
    });

    let results: Vec<String> = futures::future::join_all(fetch_all).await;
    for r in &results {
        println!("{}", r);
    }
}
```

### 3.4 async 関数のライフタイム

async 関数の引数にライフタイムが含まれる場合、返される Future のライフタイムに影響します。

```rust
// 参照を受け取る async 関数
// 戻り値の Future は引数のライフタイムに束縛される
async fn process_data(data: &[u8]) -> usize {
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    data.len()
}

// 上記の脱糖形式:
// fn process_data<'a>(data: &'a [u8]) -> impl Future<Output = usize> + 'a

#[tokio::main]
async fn main() {
    let data = vec![1, 2, 3, 4, 5];

    // OK: data は .await の完了まで生存する
    let len = process_data(&data).await;
    println!("長さ: {}", len);

    // NG: spawn するには 'static が必要
    // tokio::spawn(process_data(&data)); // コンパイルエラー!

    // OK: 所有権を移動して 'static にする
    let data_clone = data.clone();
    tokio::spawn(async move {
        let len = process_data(&data_clone).await;
        println!("スポーンタスク内: 長さ = {}", len);
    }).await.unwrap();
}
```

### 3.5 再帰的な async 関数

async 関数は通常、再帰的に呼び出すことができません。コンパイラが生成する状態マシンのサイズが無限になるためです。`Box::pin` を使って回避します。

```rust
use std::pin::Pin;
use std::future::Future;

/// ディレクトリツリーを非同期で走査する例
async fn traverse_directory(path: std::path::PathBuf) -> Vec<String> {
    let mut results = Vec::new();

    if let Ok(mut entries) = tokio::fs::read_dir(&path).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            let entry_path = entry.path();
            if entry_path.is_dir() {
                // 再帰呼び出し: Box::pin で Future をヒープに配置
                let sub_results: Pin<Box<dyn Future<Output = Vec<String>> + Send>>
                    = Box::pin(traverse_directory(entry_path));
                results.extend(sub_results.await);
            } else {
                results.push(entry_path.display().to_string());
            }
        }
    }

    results
}

// より簡潔な書き方 (async-recursion クレート使用)
// #[async_recursion::async_recursion]
// async fn traverse_directory(path: PathBuf) -> Vec<String> {
//     // 通常の再帰呼び出しが可能
//     traverse_directory(sub_path).await;
// }
```

---

## 4. 複数 Future の合成

### 4.1 join! と select!

```rust
use tokio::time::{sleep, Duration};

async fn task_a() -> String {
    sleep(Duration::from_millis(100)).await;
    "A完了".to_string()
}

async fn task_b() -> String {
    sleep(Duration::from_millis(200)).await;
    "B完了".to_string()
}

async fn task_c() -> String {
    sleep(Duration::from_millis(50)).await;
    "C完了".to_string()
}

#[tokio::main]
async fn main() {
    // join! — 全ての Future を並行実行し、全完了を待つ
    let (a, b, c) = tokio::join!(task_a(), task_b(), task_c());
    println!("{}, {}, {}", a, b, c);
    // 200ms で全完了 (最も遅い task_b に合わせる)

    // select! — 最初に完了した Future の結果を取得
    tokio::select! {
        val = task_a() => println!("Aが先: {}", val),
        val = task_b() => println!("Bが先: {}", val),
        val = task_c() => println!("Cが先: {}", val),
    }
    // "Cが先: C完了" (50ms で最速)
}
```

### 4.2 エラーハンドリング付き並行処理

```rust
use anyhow::Result;

async fn fetch_user(id: u64) -> Result<String> {
    // API呼び出しをシミュレート
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
    Ok(format!("User#{}", id))
}

async fn fetch_profile(id: u64) -> Result<String> {
    tokio::time::sleep(tokio::time::Duration::from_millis(80)).await;
    Ok(format!("Profile#{}", id))
}

#[tokio::main]
async fn main() -> Result<()> {
    // try_join! — いずれかがエラーなら即座に返す
    let (user, profile) = tokio::try_join!(
        fetch_user(1),
        fetch_profile(1),
    )?;

    println!("{}, {}", user, profile);

    // JoinSet — 動的にタスクを追加して全完了を待つ
    let mut set = tokio::task::JoinSet::new();
    for id in 1..=5 {
        set.spawn(fetch_user(id));
    }

    while let Some(result) = set.join_next().await {
        let user = result??;
        println!("取得: {}", user);
    }

    Ok(())
}
```

### 4.3 FutureExt による拡張メソッド

```rust
use futures::FutureExt;
use tokio::time::{sleep, Duration};

#[tokio::main]
async fn main() {
    // fuse() — select! で安全に使うための変換
    let mut future_a = Box::pin(sleep(Duration::from_millis(100)).fuse());
    let mut future_b = Box::pin(sleep(Duration::from_millis(200)).fuse());

    // 両方が完了するまでループ
    let mut a_done = false;
    let mut b_done = false;

    while !a_done || !b_done {
        tokio::select! {
            _ = &mut future_a, if !a_done => {
                println!("A 完了");
                a_done = true;
            }
            _ = &mut future_b, if !b_done => {
                println!("B 完了");
                b_done = true;
            }
        }
    }

    // map() — Future の結果を変換
    let result = async { 42 }
        .map(|x| x * 2)
        .await;
    println!("結果: {}", result); // 84

    // then() — Future の結果から新しい Future を生成
    let result = async { 10 }
        .then(|x| async move {
            sleep(Duration::from_millis(10)).await;
            x + 5
        })
        .await;
    println!("結果: {}", result); // 15
}
```

### 4.4 join_all と try_join_all

```rust
use futures::future::{join_all, try_join_all};
use anyhow::Result;

async fn fetch_item(id: u32) -> Result<String> {
    tokio::time::sleep(tokio::time::Duration::from_millis(id as u64 * 10)).await;
    if id == 5 {
        anyhow::bail!("アイテム5の取得に失敗");
    }
    Ok(format!("Item#{}", id))
}

#[tokio::main]
async fn main() -> Result<()> {
    // join_all — 全 Future の結果を Vec で取得 (エラーも含む)
    let futures: Vec<_> = (1..=10).map(|id| fetch_item(id)).collect();
    let results: Vec<Result<String>> = join_all(futures).await;

    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(item) => println!("[{}] {}", i, item),
            Err(e) => eprintln!("[{}] エラー: {}", i, e),
        }
    }

    // try_join_all — いずれかがエラーなら即座にエラーを返す
    let futures: Vec<_> = (1..=3).map(|id| fetch_item(id)).collect();
    let results: Vec<String> = try_join_all(futures).await?;
    println!("全成功: {:?}", results);

    Ok(())
}
```

### 4.5 select! の高度な使い方

```rust
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration, interval};

#[tokio::main]
async fn main() {
    let (tx, mut rx) = mpsc::channel::<String>(32);
    let mut heartbeat = interval(Duration::from_secs(5));
    let shutdown = tokio::signal::ctrl_c();
    tokio::pin!(shutdown);

    // 送信側
    tokio::spawn(async move {
        for i in 0..10 {
            sleep(Duration::from_millis(500)).await;
            let _ = tx.send(format!("メッセージ #{}", i)).await;
        }
    });

    loop {
        tokio::select! {
            // biased; を指定すると、上から順に優先的にチェック
            biased;

            // シャットダウンシグナル (最優先)
            _ = &mut shutdown => {
                println!("Ctrl+C 受信。シャットダウン...");
                break;
            }

            // メッセージ受信
            Some(msg) = rx.recv() => {
                println!("受信: {}", msg);
            }

            // ハートビート
            _ = heartbeat.tick() => {
                println!("ハートビート送信");
            }

            // 全チャネルが閉じた場合
            else => {
                println!("全チャネル閉鎖。終了。");
                break;
            }
        }
    }
}
```

---

## 5. Pin と Unpin

### 5.1 Pin の必要性

```
┌─────────────────────────────────────────────┐
│              Pin の必要性                      │
│                                               │
│  async fn foo() {                            │
│      let data = vec![1, 2, 3];               │
│      let ref_to_data = &data;  ← 自己参照    │
│      some_async_op().await;    ← 中断点      │
│      println!("{:?}", ref_to_data);           │
│  }                                            │
│                                               │
│  中断時に Future がメモリ上で移動すると       │
│  ref_to_data が無効になる → Pin で防ぐ       │
│                                               │
│  Pin<&mut T>:                                │
│    T が Unpin なら → 移動OK (大半の型)       │
│    T が !Unpin なら → 移動禁止 (async 生成物) │
└─────────────────────────────────────────────┘
```

### 5.2 Pin の実践的な使い方

```rust
use std::pin::Pin;
use std::future::Future;
use tokio::time::{sleep, Duration};

// pin! マクロ (tokio::pin! または std::pin::pin!) でスタックピン
#[tokio::main]
async fn main() {
    // スタック上にピン留め
    let future = sleep(Duration::from_millis(100));
    tokio::pin!(future);

    // Pin<&mut Sleep> として使える
    // select! 内で &mut 参照として使う場合に必要
    tokio::select! {
        _ = &mut future => {
            println!("スリープ完了");
        }
    }

    // Box::pin でヒープ上にピン留め
    let boxed_future: Pin<Box<dyn Future<Output = ()>>> =
        Box::pin(async {
            sleep(Duration::from_millis(100)).await;
            println!("Boxed Future 完了");
        });
    boxed_future.await;
}

// 動的ディスパッチが必要な場合 (trait object)
async fn execute_any(future: Pin<Box<dyn Future<Output = String> + Send>>) -> String {
    future.await
}
```

### 5.3 Unpin の理解

```rust
use std::marker::Unpin;
use std::pin::Pin;

// 通常の型は Unpin を自動実装する
// → Pin<&mut T> があっても自由に移動できる
struct MyStruct {
    value: i32,
}
// MyStruct: Unpin (自動実装)

// async ブロック/関数が生成する Future は !Unpin
// → Pin で固定されている間は移動できない
fn takes_unpin<T: Unpin>(_: &T) {}

fn example() {
    let x = MyStruct { value: 42 };
    takes_unpin(&x); // OK: MyStruct は Unpin

    // let future = async { 42 };
    // takes_unpin(&future); // コンパイルエラー: async は !Unpin

    // Unpin を手動で実装する場合 (通常は不要):
    // 自己参照を含まないカスタム Future に対して
    struct SimpleFuture;
    impl Unpin for SimpleFuture {}
}
```

---

## 6. Executor / Reactor / Waker アーキテクチャ

### 6.1 3つのコンポーネントの協調

```
┌──────────────── 非同期ランタイムの構造 ─────────────────┐
│                                                           │
│  ┌─────────────┐     ┌─────────────┐                    │
│  │  Executor    │     │  Reactor    │                    │
│  │  (タスク実行)│     │  (I/O監視)  │                    │
│  │             │     │             │                    │
│  │  タスクキュー│     │  epoll /    │                    │
│  │  ┌───┐┌───┐│     │  kqueue /   │                    │
│  │  │T1 ││T2 ││     │  IOCP       │                    │
│  │  └───┘└───┘│     │             │                    │
│  │  ┌───┐┌───┐│     │  ┌────────┐ │                    │
│  │  │T3 ││T4 ││     │  │ソケット │ │                    │
│  │  └───┘└───┘│     │  │ タイマー│ │                    │
│  └──────┬──────┘     │  │ ファイル│ │                    │
│         │            │  └────────┘ │                    │
│         │ poll()     └──────┬──────┘                    │
│         │                   │                            │
│         │    ┌──────────────┘                            │
│         │    │ wake()                                    │
│         ▼    ▼                                           │
│  ┌─────────────────┐                                    │
│  │     Waker       │                                    │
│  │  (通知メカニズム) │                                    │
│  │                 │                                    │
│  │  Reactor → Waker.wake()                              │
│  │        → Executor がタスクを再キューイング            │
│  └─────────────────┘                                    │
└───────────────────────────────────────────────────────────┘
```

### 6.2 ミニ Executor の実装

```rust
use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};

/// 最小限の Executor 実装
struct MiniExecutor {
    queue: VecDeque<Pin<Box<dyn Future<Output = ()>>>>,
}

/// タスクが再ポーリングを要求するための Waker
struct MiniWaker;

impl Wake for MiniWaker {
    fn wake(self: Arc<Self>) {
        // 実際のランタイムではここでタスクを再キューイング
        // この簡易版では何もしない (ビジーポーリング)
    }
}

impl MiniExecutor {
    fn new() -> Self {
        MiniExecutor {
            queue: VecDeque::new(),
        }
    }

    fn spawn(&mut self, future: impl Future<Output = ()> + 'static) {
        self.queue.push_back(Box::pin(future));
    }

    fn run(&mut self) {
        let waker = Waker::from(Arc::new(MiniWaker));
        let mut cx = Context::from_waker(&waker);

        while let Some(mut future) = self.queue.pop_front() {
            match future.as_mut().poll(&mut cx) {
                Poll::Ready(()) => {
                    // タスク完了
                }
                Poll::Pending => {
                    // 未完了: キューの末尾に戻す (ビジーポーリング)
                    self.queue.push_back(future);
                }
            }
        }
    }
}

// 使用例 (tokio なしで動く)
fn main() {
    let mut executor = MiniExecutor::new();

    executor.spawn(async {
        println!("タスク1 開始");
        // 注: 実際の非同期I/Oはランタイムのサポートが必要
        println!("タスク1 完了");
    });

    executor.spawn(async {
        println!("タスク2 開始");
        println!("タスク2 完了");
    });

    executor.run();
}
```

---

## 7. 非同期エラーハンドリングのパターン

### 7.1 Result と ? 演算子

```rust
use anyhow::{Context, Result};
use tokio::time::{sleep, Duration};

#[derive(Debug)]
struct ApiResponse {
    status: u16,
    body: String,
}

async fn fetch_api(url: &str) -> Result<ApiResponse> {
    let response = reqwest::get(url)
        .await
        .context(format!("HTTP リクエスト失敗: {}", url))?;

    let status = response.status().as_u16();
    let body = response.text()
        .await
        .context("レスポンスボディの読み取り失敗")?;

    Ok(ApiResponse { status, body })
}

async fn fetch_with_fallback(primary: &str, fallback: &str) -> Result<String> {
    // プライマリを試行し、失敗したらフォールバック
    match fetch_api(primary).await {
        Ok(resp) if resp.status == 200 => Ok(resp.body),
        Ok(resp) => {
            eprintln!("プライマリがステータス {} を返却。フォールバックに切替", resp.status);
            let resp = fetch_api(fallback).await?;
            Ok(resp.body)
        }
        Err(e) => {
            eprintln!("プライマリがエラー: {}。フォールバックに切替", e);
            let resp = fetch_api(fallback).await?;
            Ok(resp.body)
        }
    }
}
```

### 7.2 カスタムエラー型

```rust
use thiserror::Error;

#[derive(Error, Debug)]
enum ServiceError {
    #[error("ネットワークエラー: {0}")]
    Network(#[from] reqwest::Error),

    #[error("タイムアウト: {0}秒超過")]
    Timeout(u64),

    #[error("レートリミット: {retry_after}秒後にリトライ")]
    RateLimit { retry_after: u64 },

    #[error("認証失敗: {0}")]
    Auth(String),

    #[error("内部エラー: {0}")]
    Internal(#[from] anyhow::Error),
}

async fn call_service(url: &str, token: &str) -> Result<String, ServiceError> {
    let client = reqwest::Client::new();
    let response = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        client.get(url).bearer_auth(token).send(),
    )
    .await
    .map_err(|_| ServiceError::Timeout(30))?
    .map_err(ServiceError::Network)?;

    match response.status().as_u16() {
        200 => Ok(response.text().await.map_err(ServiceError::Network)?),
        401 => Err(ServiceError::Auth("無効なトークン".into())),
        429 => {
            let retry = response
                .headers()
                .get("Retry-After")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse().ok())
                .unwrap_or(60);
            Err(ServiceError::RateLimit { retry_after: retry })
        }
        _ => Err(ServiceError::Internal(
            anyhow::anyhow!("予期しないステータス: {}", response.status()),
        )),
    }
}
```

---

## 8. 比較表

### 8.1 ランタイム比較

| 項目 | tokio | async-std | smol |
|---|---|---|---|
| エコシステム | 最大規模 | 中規模 | 軽量 |
| マルチスレッド | デフォルト対応 | デフォルト対応 | 対応 |
| I/O | 独自 (mio ベース) | 独自 | polling ベース |
| タイマー | `tokio::time` | `async_std::task` | `async-io` |
| 採用実績 | Axum, tonic 等 | 一部 | 組み込み向け |
| 依存サイズ | 中 | 中 | 小 |
| ワークスティーリング | あり | あり | あり |
| カスタムランタイム構築 | Builder API | 限定的 | 容易 |

### 8.2 同期 vs 非同期の選択基準

| 基準 | 同期処理が適切 | 非同期処理が適切 |
|---|---|---|
| I/O パターン | CPU集中的処理 | I/O集中的処理 |
| 同時接続数 | 少数 (〜100) | 多数 (1K〜100K+) |
| レイテンシ要件 | 予測可能性重視 | スループット重視 |
| コード複雑性 | シンプルさ重視 | 多少の複雑さ許容 |
| ライブラリ | 同期APIのみの場合 | 非同期エコシステム活用 |
| デバッグ | スタックトレース明快 | 非同期対応ツール必要 |
| メモリ使用量 | スレッドスタック (8MB/スレッド) | Future (数百バイト/タスク) |
| 起動コスト | スレッド生成 (~数ms) | タスクスポーン (~数μs) |

### 8.3 Future 合成メソッドの比較

| メソッド | 用途 | 動作 | エラー時 |
|---|---|---|---|
| `join!` | 全並行・全完了 | 全 Future が完了するまで待つ | 全完了後に個別チェック |
| `try_join!` | 全並行・最初のエラーで中断 | いずれかがエラーで即座に返る | 他は drop (キャンセル) |
| `select!` | 全並行・最初の完了 | 最初に完了した結果を取得 | 他は drop (キャンセル) |
| `join_all` | 動的数の並行・全完了 | Vec<Future> を渡す | Vec<Result> で返る |
| `try_join_all` | 動的数の並行・最初のエラー | Vec<Future> を渡す | 最初のエラーで中断 |
| `FuturesUnordered` | 動的数の並行・完了順取得 | Stream として結果を返す | 個別に処理 |

---

## 9. アンチパターン

### 9.1 async 内でのブロッキング呼び出し

```rust
// NG: async 内で std::thread::sleep (ランタイム全体をブロック!)
async fn bad_delay() {
    std::thread::sleep(std::time::Duration::from_secs(5));
    // 他の全タスクが5秒間停止する
}

// OK: tokio の非同期スリープを使う
async fn good_delay() {
    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
    // 他のタスクは中断中も実行を続ける
}

// OK: CPU集中処理は spawn_blocking で逃がす
async fn cpu_heavy() {
    let result = tokio::task::spawn_blocking(|| {
        // 重い同期計算
        (0..10_000_000u64).sum::<u64>()
    }).await.unwrap();
    println!("結果: {}", result);
}

// OK: ブロッキングライブラリは spawn_blocking でラップ
async fn read_file_blocking(path: String) -> std::io::Result<String> {
    tokio::task::spawn_blocking(move || {
        std::fs::read_to_string(path)
    }).await.unwrap()
}
```

### 9.2 Future を .await せずに放置

```rust
// NG: async fn の戻り値を無視 (何も実行されない!)
async fn send_notification() {
    println!("通知送信");
}

async fn bad_example() {
    send_notification(); // ← .await なし! 実行されない!
    // コンパイラ警告: unused future
}

// OK: 必ず .await するか spawn する
async fn good_example() {
    send_notification().await;           // パターン1: 同期的に待つ
    tokio::spawn(send_notification());   // パターン2: バックグラウンド実行
}
```

### 9.3 不要な Arc/Mutex の使用

```rust
// NG: 非同期コンテキストで std::sync::Mutex を使う
use std::sync::Mutex;

async fn bad_shared_state() {
    let data = std::sync::Arc::new(Mutex::new(Vec::new()));
    let d = data.clone();
    tokio::spawn(async move {
        let mut lock = d.lock().unwrap(); // .await 中にロックを保持する危険性
        // 長い非同期処理...
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await; // ← デッドロックの原因!
        lock.push(42);
    });
}

// OK: tokio::sync::Mutex を使う (await 中もロックを安全に保持)
async fn good_shared_state() {
    let data = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new()));
    let d = data.clone();
    tokio::spawn(async move {
        let mut lock = d.lock().await; // 非同期ロック取得
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        lock.push(42);
    });
}

// ベスト: ロックの粒度を小さくする
async fn best_shared_state() {
    let data = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let d = data.clone();
    tokio::spawn(async move {
        // 非同期処理
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        // ロックは最小限の範囲で
        {
            let mut lock = d.lock().unwrap();
            lock.push(42);
        } // ← ここで即座にロック解放
    });
}
```

### 9.4 async 関数内での大量のメモリ確保

```rust
// NG: async 関数のスタックフレーム（状態マシン）が巨大になる
async fn bad_large_stack() {
    let buffer = [0u8; 1_000_000]; // 1MB の配列が Future の状態に含まれる
    some_async_op().await;
    println!("{}", buffer.len());
}

// OK: Box で ヒープに配置
async fn good_large_heap() {
    let buffer = vec![0u8; 1_000_000]; // ヒープに配置
    some_async_op().await;
    println!("{}", buffer.len());
}

async fn some_async_op() {
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
}
```

---

## 10. 実践パターン集

### 10.1 グレースフルシャットダウン

```rust
use tokio::sync::watch;
use tokio::time::{sleep, Duration};

async fn graceful_shutdown_example() {
    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);

    // ワーカータスク群
    let mut handles = Vec::new();
    for i in 0..4 {
        let mut rx = shutdown_tx.subscribe();
        let handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = rx.changed() => {
                        if *rx.borrow() {
                            println!("ワーカー {} がシャットダウン中...", i);
                            // クリーンアップ処理
                            sleep(Duration::from_millis(100)).await;
                            println!("ワーカー {} 停止完了", i);
                            return;
                        }
                    }
                    _ = sleep(Duration::from_secs(1)) => {
                        println!("ワーカー {} 処理中...", i);
                    }
                }
            }
        });
        handles.push(handle);
    }

    // 3秒後にシャットダウン
    sleep(Duration::from_secs(3)).await;
    println!("シャットダウンシグナル送信");
    let _ = shutdown_tx.send(true);

    // 全ワーカーの完了を待つ
    for handle in handles {
        let _ = handle.await;
    }
    println!("全ワーカー停止。プログラム終了。");
}
```

### 10.2 非同期イテレーション (for await 的パターン)

```rust
use tokio::sync::mpsc;
use tokio_stream::{wrappers::ReceiverStream, StreamExt};

async fn async_iteration_example() {
    let (tx, rx) = mpsc::channel::<i32>(32);

    // プロデューサー
    tokio::spawn(async move {
        for i in 0..20 {
            let _ = tx.send(i).await;
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }
    });

    // コンシューマー: Stream として処理
    let stream = ReceiverStream::new(rx);
    let results: Vec<i32> = stream
        .filter(|x| *x % 2 == 0)       // 偶数のみ
        .map(|x| x * x)                 // 二乗
        .take(5)                         // 最初の5つ
        .collect()
        .await;

    println!("結果: {:?}", results); // [0, 4, 16, 36, 64]
}
```

### 10.3 CancellationToken パターン

```rust
use tokio_util::sync::CancellationToken;
use tokio::time::{sleep, Duration};

async fn cancellation_token_example() {
    let token = CancellationToken::new();

    // 子トークン: 親がキャンセルされると自動的にキャンセル
    let child_token = token.child_token();

    let task = tokio::spawn({
        let token = child_token.clone();
        async move {
            loop {
                tokio::select! {
                    _ = token.cancelled() => {
                        println!("キャンセルされました。クリーンアップ中...");
                        // リソース解放など
                        break;
                    }
                    _ = sleep(Duration::from_secs(1)) => {
                        println!("作業中...");
                    }
                }
            }
        }
    });

    // 3秒後にキャンセル
    sleep(Duration::from_secs(3)).await;
    token.cancel();
    let _ = task.await;
    println!("タスクが正常にキャンセルされました");
}
```

### 10.4 非同期リソース管理 (Drop と非同期)

```rust
/// 非同期クリーンアップが必要なリソースの管理パターン
struct AsyncResource {
    name: String,
    // 非同期クリーンアップ用のシグナル
    cleanup_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl AsyncResource {
    async fn new(name: &str) -> Self {
        println!("リソース '{}' を作成", name);
        let (tx, rx) = tokio::sync::oneshot::channel();

        // バックグラウンドでクリーンアップ待機
        let resource_name = name.to_string();
        tokio::spawn(async move {
            let _ = rx.await;
            // 非同期クリーンアップ処理
            println!("リソース '{}' の非同期クリーンアップ完了", resource_name);
        });

        AsyncResource {
            name: name.to_string(),
            cleanup_tx: Some(tx),
        }
    }

    async fn use_resource(&self) {
        println!("リソース '{}' を使用中", self.name);
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
}

impl Drop for AsyncResource {
    fn drop(&mut self) {
        // Drop は同期なので、シグナルを送ってバックグラウンドタスクに委譲
        if let Some(tx) = self.cleanup_tx.take() {
            let _ = tx.send(());
        }
        println!("リソース '{}' をドロップ (同期部分)", self.name);
    }
}
```

---

## FAQ

### Q1: `#[tokio::main]` は何をしているの?

**A:** 非同期ランタイム(Executor)を起動し、`main` 関数の `async` ブロックを実行するマクロです。

```rust
// これは:
#[tokio::main]
async fn main() { /* ... */ }

// 以下と等価:
fn main() {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async { /* ... */ })
}
```

### Q2: `Send` + `'static` 制約が必要な理由は?

**A:** `tokio::spawn` はタスクを別スレッドで実行する可能性があるため、`Send`(スレッド間移動可能)と `'static`(借用を含まない)が必要です。

```rust
// エラー: ローカル参照を含む Future は spawn できない
async fn bad() {
    let data = String::from("hello");
    let r = &data;
    tokio::spawn(async move {
        // println!("{}", r); // ← コンパイルエラー: &String は 'static でない
    });
}

// OK: 所有権を move する
async fn good() {
    let data = String::from("hello");
    tokio::spawn(async move {
        println!("{}", data); // OK: data の所有権を移動
    });
}
```

### Q3: async trait はどう書く?

**A:** Rust 1.75+ では `async fn` を trait 内で直接使えます。それ以前は `async-trait` クレートを使います。

```rust
// Rust 1.75+ (ネイティブ対応)
trait Service {
    async fn call(&self, req: Request) -> Response;
}

// 注意: trait の async fn はデフォルトでは Send ではない
// Send 制約が必要な場合は明示する:
trait SendService: Send + Sync {
    fn call(&self, req: Request) -> impl Future<Output = Response> + Send;
}

// Rust 1.74以前 (async-trait クレート)
use async_trait::async_trait;

#[async_trait]
trait Service {
    async fn call(&self, req: Request) -> Response;
}
```

### Q4: `tokio::spawn` と `tokio::join!` の使い分けは?

**A:** `join!` は現在のタスク内で複数の Future を並行に実行します。`spawn` は新しいタスクを生成してデタッチ実行します。

```rust
// join!: 並行だが、同一タスク内。キャンセルが容易
let (a, b) = tokio::join!(future_a, future_b);

// spawn: 独立タスク。Send + 'static が必要
let handle = tokio::spawn(future_a);
// JoinHandle で結果を取得するか、デタッチ
```

**使い分けの基準:**
- 結果を待つ必要がある短い並行処理 → `join!`
- 独立したバックグラウンドタスク → `spawn`
- ライフタイムの制約がある (参照を含む) → `join!`
- 動的な数のタスク → `JoinSet` (spawn ベース)

### Q5: 非同期コードのデバッグ方法は?

**A:** 以下のツールと手法を活用します。

```rust
// 1. tokio-console (トレーシングベースのデバッグツール)
// Cargo.toml:
// [dependencies]
// console-subscriber = "0.4"

#[tokio::main]
async fn main() {
    console_subscriber::init(); // tokio-console 有効化
    // ... アプリケーションコード
}

// 2. tracing によるログ出力
use tracing::{info, instrument};

#[instrument] // 関数の呼び出しを自動トレース
async fn process_request(id: u64) -> String {
    info!("リクエスト処理開始");
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    info!("リクエスト処理完了");
    format!("結果: {}", id)
}

// 3. タスクダンプ (tokio の unstable feature)
// RUSTFLAGS="--cfg tokio_unstable" cargo run
// で tokio::runtime::Handle::dump() が使える
```

### Q6: async fn の中で同期コードを呼ぶのはいつ問題になる?

**A:** 同期コードが「長時間ブロック」する場合に問題です。1マイクロ秒程度の短い同期処理は問題ありません。目安として 10〜100μs 以上かかる処理は `spawn_blocking` を検討してください。

```rust
// 問題なし: 短い同期処理
async fn ok_example() {
    let hash = sha256(&data); // マイクロ秒オーダー
    // ...
}

// 問題あり: 長い同期処理
async fn bad_example() {
    let compressed = zstd::compress(&large_data, 19); // ミリ秒～秒オーダー
    // → spawn_blocking に逃がすべき
}
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| Future trait | `poll()` が `Ready` か `Pending` を返す遅延評価モデル |
| async/await | Future を生成・待機するシンタックスシュガー |
| 状態マシン | async ブロックはコンパイル時に状態マシンに変換 (ゼロコスト) |
| ランタイム | Executor (タスク実行) + Reactor (I/O監視) + Waker (通知) |
| tokio | 最も広く使われる非同期ランタイム |
| join! | 複数 Future の並行実行・全完了待ち |
| try_join! | 複数 Future の並行実行・最初のエラーで中断 |
| select! | 複数 Future のうち最初の完了を取得 |
| Pin | 自己参照を含む Future のメモリ安全性を保証 |
| Unpin | 移動可能な型のマーカートレイト (通常の型は自動実装) |
| spawn_blocking | 同期処理をブロッキングスレッドプールに逃がす |
| CancellationToken | 協調的なタスクキャンセルの推奨パターン |
| Send + 'static | spawn に必要な制約。参照を含む場合は join! を使用 |
| エラーハンドリング | thiserror + anyhow の組み合わせが実践的 |

## 次に読むべきガイド

- [Tokioランタイム](./01-tokio-runtime.md) — タスク管理とチャネルの詳細
- [非同期パターン](./02-async-patterns.md) — Stream、並行制限、リトライ
- [ネットワーク](./03-networking.md) — HTTP/WebSocket/gRPC

## 参考文献

1. **Asynchronous Programming in Rust**: https://rust-lang.github.io/async-book/
2. **Tokio Tutorial**: https://tokio.rs/tokio/tutorial
3. **Pin and suffering (Fasterthanlime)**: https://fasterthanli.me/articles/pin-and-suffering
4. **Jon Gjengset - Decrusting the tokio crate**: https://www.youtube.com/watch?v=o2ob8zkeq2s
5. **Without Boats - Zero-cost async IO**: https://without.boats/blog/zero-cost-async-io/
6. **Tokio Console**: https://github.com/tokio-rs/console
