# メッセージパッシング

> 「共有メモリではなく、メッセージの送受信で通信する」。データ競合を根本的に回避する並行処理の設計原則。

## この章で学ぶこと

- [ ] メッセージパッシングの概念・歴史・理論的基盤を理解する
- [ ] チャネルベースの通信（Go, Rust）を実装できる
- [ ] アクターモデル（Erlang/Elixir, Akka）の設計思想と実装を理解する
- [ ] CSP（Communicating Sequential Processes）の形式的背景を知る
- [ ] 共有メモリとメッセージパッシングの使い分けを正確に判断できる
- [ ] パイプライン・ファンアウト/ファンインなど主要パターンを実装できる
- [ ] デッドロック・ライブロック等の障害パターンを回避できる
- [ ] 分散システムにおけるメッセージパッシングの応用を理解する


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [async/await（非同期プログラミング）](./01-async-await.md) の内容を理解していること

---

## 1. メッセージパッシングの基本概念

### 1.1 定義と歴史的背景

メッセージパッシング（Message Passing）は、並行に動作する計算主体（プロセス、スレッド、アクター等）が、共有メモリを介さずメッセージの送受信のみで通信する方式である。この概念は 1970 年代に Carl Hewitt のアクターモデル（1973）と Tony Hoare の CSP（Communicating Sequential Processes, 1978）という二つの独立した研究から発展した。

共有メモリモデルでは複数のスレッドが同一メモリ空間を読み書きするため、ミューテックスやセマフォなどの同期機構が必要になる。一方メッセージパッシングでは、各計算主体が独自のメモリ空間を持ち、データのやり取りはメッセージのコピーまたは所有権の移動で行われる。これにより、データ競合を構造的に排除できる。

```
共有メモリモデル:
  ┌─────────┐     ┌──────────────┐     ┌─────────┐
  │ Thread A │────>│  共有データ   │<────│ Thread B │
  └─────────┘  ^  └──────────────┘  ^  └─────────┘
               |                    |
         Mutex/Lock            Mutex/Lock

  問題点:
  - デッドロック: 複数ロックの順序不整合
  - データ競合: ロック忘れ、不十分な同期
  - 優先度逆転: 低優先度スレッドがロック保持
  - コンボイ効果: 遅いスレッドが全体を律速

メッセージパッシングモデル:
  ┌─────────┐  message   ┌─────────┐  message   ┌─────────┐
  │ Process A│───────────>│ Process B│───────────>│ Process C│
  │ [状態 a] │  channel   │ [状態 b] │  channel   │ [状態 c] │
  └─────────┘            └─────────┘            └─────────┘

  各プロセスが独自の状態を所有。通信はメッセージのみ。
  データ競合は構造的に発生しない。
```

### 1.2 Go の格言と設計思想

Go 言語の公式ドキュメントには次の格言がある:

> "Do not communicate by sharing memory; share memory by communicating."
> （共有メモリで通信するな、通信でメモリを共有せよ）

これは「データを複数のゴルーチンから直接操作する代わりに、チャネルを通じてデータの所有権を移動させよ」という意味である。あるゴルーチンがチャネルにデータを送信した後、そのデータには送信側は触れないという規約を守ることで、同時アクセスの問題を回避する。

### 1.3 同期型と非同期型の分類

メッセージパッシングは、送受信のタイミングにより大きく二つに分類される。

```
同期型（Synchronous / Rendezvous）:
  ┌──────┐              ┌──────┐
  │送信側│─── send ────>│受信側│
  │      │   (block)    │      │
  │ wait │<── ack  ─────│ recv │
  └──────┘              └──────┘
  送信側は受信側が受け取るまでブロックする。
  Go のバッファなしチャネル、Ada のランデブーが該当。

非同期型（Asynchronous / Buffered）:
  ┌──────┐              ┌────────────┐     ┌──────┐
  │送信側│─── send ────>│ メッセージ  │────>│受信側│
  │      │  (non-block) │  キュー     │     │      │
  └──────┘              └────────────┘     └──────┘
  送信側は即座に戻る。メッセージはキューに蓄積。
  Erlang のメールボックス、Go のバッファ付きチャネルが該当。
```

| 特性 | 同期型 | 非同期型 |
|------|--------|----------|
| 送信時のブロック | 受信まで待機 | 即座に返る（バッファ満杯時を除く） |
| レイテンシ | 高い（待機コスト） | 低い（送信側は即座に続行） |
| デッドロックリスク | 高い（相互待機の可能性） | 低い（だがバッファ枯渇に注意） |
| メモリ使用 | 少ない | バッファ分のメモリが必要 |
| デバッグ容易性 | 高い（因果関係が明確） | 低い（タイミング依存のバグ） |
| 代表例 | Go unbuffered chan, Ada rendezvous | Erlang mailbox, Go buffered chan |
| バックプレッシャー | 自然に発生 | 明示的な設計が必要 |

---

## 2. 理論的基盤: CSP とアクターモデル

### 2.1 CSP（Communicating Sequential Processes）

CSP は Tony Hoare が 1978 年に発表した形式的プロセス代数である。CSP では、プロセスは逐次的な計算を行う独立した実体であり、名前付きチャネルを通じてのみ通信する。

CSP の主要な特徴は以下の通りである:

1. **チャネルは一等市民**: チャネルは名前を持ち、型付けされ、変数として渡せる
2. **同期通信が基本**: 送信と受信は同時に発生する（ランデブー）
3. **選択的通信**: `select`/`alt` 構文で複数のチャネルを同時に待機
4. **逐次合成**: プロセス内部は逐次的に実行される

```
CSP のプロセス合成演算子:

  P ; Q        逐次合成: P の後に Q を実行
  P || Q       並行合成: P と Q を並行に実行
  P [] Q       外部選択: 環境が P か Q を選ぶ
  P |~| Q      内部選択: プロセスが P か Q を非決定的に選ぶ

  チャネル通信:
  c!v          チャネル c に値 v を送信
  c?x          チャネル c から値を受信して x に束縛

  例: バッファ（サイズ1）の CSP 記述
  BUFFER = left?x -> right!x -> BUFFER
```

Go 言語のゴルーチンとチャネルは CSP に強く影響を受けている。Go の `select` 文は CSP の外部選択に対応する。

### 2.2 アクターモデル

アクターモデルは Carl Hewitt が 1973 年に提唱し、Gul Agha が 1986 年に体系化した並行計算モデルである。アクターモデルでは、アクター（Actor）が計算の基本単位であり、各アクターは以下の能力を持つ:

1. **メッセージの受信**: 非同期メールボックスからメッセージを取り出す
2. **内部状態の変更**: 自身の状態を更新する（外部からはアクセス不可）
3. **メッセージの送信**: 他のアクターにメッセージを送る
4. **新しいアクターの生成**: 子アクターを作成する

```
アクターモデルの構造:

  ┌─────────────────────────────────────┐
  │            Actor System             │
  │                                     │
  │  ┌──────────┐     ┌──────────┐     │
  │  │ Actor A  │ msg │ Actor B  │     │
  │  │┌────────┐│────>│┌────────┐│     │
  │  ││Mailbox ││     ││Mailbox ││     │
  │  │├────────┤│     │├────────┤│     │
  │  ││Behavior││     ││Behavior││     │
  │  │├────────┤│     │├────────┤│     │
  │  ││ State  ││     ││ State  ││     │
  │  │└────────┘│     │└────────┘│     │
  │  └──────────┘     └────┬─────┘     │
  │        ^               │ spawn     │
  │        │ msg           v           │
  │  ┌─────┴────┐     ┌──────────┐     │
  │  │ Actor D  │     │ Actor C  │     │
  │  │┌────────┐│     │┌────────┐│     │
  │  ││Mailbox ││<────││Mailbox ││     │
  │  │├────────┤│ msg │├────────┤│     │
  │  ││Behavior││     ││Behavior││     │
  │  │├────────┤│     │├────────┤│     │
  │  ││ State  ││     ││ State  ││     │
  │  │└────────┘│     │└────────┘│     │
  │  └──────────┘     └──────────┘     │
  │                                     │
  └─────────────────────────────────────┘

  各アクターは:
  (1) メールボックス(受信キュー)
  (2) ビヘイビア(メッセージ処理ロジック)
  (3) 内部状態(外部非公開)
  を持つ独立した計算単位
```

### 2.3 CSP とアクターモデルの比較

| 特性 | CSP | アクターモデル |
|------|-----|---------------|
| 通信方式 | 名前付きチャネル（同期） | 直接アドレッシング（非同期） |
| 識別対象 | チャネル | アクター（アドレス） |
| バッファリング | なし（基本は同期） | メールボックス（無制限キュー） |
| メッセージ順序 | チャネルごとに FIFO | 保証なし（同一送信者間は FIFO） |
| 合成性 | 代数的合成が可能 | 動的なトポロジ変更が容易 |
| 障害処理 | モデル外（別途必要） | 監視ツリーで体系化（Erlang） |
| 代表的実装 | Go, Clojure core.async | Erlang/OTP, Akka, Orleans |
| 理論的強み | 形式検証（FDR等） | 位置透過性、分散に自然 |
| 適用領域 | 構造化された並行処理 | 大規模分散・耐障害システム |

---

## 3. Go のチャネル: CSP の実践

### 3.1 チャネルの基本操作

Go のチャネルは型安全なメッセージパッシングの仕組みである。チャネルの方向を制限することで、送信専用・受信専用を型レベルで強制できる。

```go
package main

import (
    "fmt"
    "time"
)

func main() {
    // === バッファなしチャネル（同期型） ===
    unbuffered := make(chan string)

    go func() {
        // 受信側が準備できるまでブロックされる
        unbuffered <- "hello"
        fmt.Println("送信完了") // 受信後に出力される
    }()

    time.Sleep(100 * time.Millisecond) // 送信側がブロックされていることを確認
    msg := <-unbuffered
    fmt.Println("受信:", msg)

    // === バッファ付きチャネル（非同期型） ===
    buffered := make(chan int, 3) // 3要素までバッファリング

    // バッファに空きがある限りブロックされない
    buffered <- 10
    buffered <- 20
    buffered <- 30
    // buffered <- 40 // バッファ満杯でブロック（デッドロック）

    fmt.Println(<-buffered) // 10（FIFO）
    fmt.Println(<-buffered) // 20
    fmt.Println(<-buffered) // 30

    // === チャネルの方向制約 ===
    // chan<- int : 送信専用
    // <-chan int : 受信専用
    producer := func(out chan<- int) {
        for i := 0; i < 5; i++ {
            out <- i
        }
        close(out) // 送信完了を通知
    }

    consumer := func(in <-chan int) {
        for v := range in { // close されるまで繰り返す
            fmt.Printf("受信: %d\n", v)
        }
    }

    ch := make(chan int, 5)
    go producer(ch)
    consumer(ch) // メインゴルーチンで受信
}
```

### 3.2 select 文による多重化

Go の `select` 文は複数のチャネル操作を同時に待機し、準備できたものを非決定的に選択する。CSP の外部選択演算子に対応する。

```go
package main

import (
    "context"
    "fmt"
    "math/rand"
    "time"
)

// タイムアウト付きの処理
func fetchWithTimeout(url string, timeout time.Duration) (string, error) {
    result := make(chan string, 1)
    errCh := make(chan error, 1)

    go func() {
        // シミュレーション: ランダムな遅延
        delay := time.Duration(rand.Intn(500)) * time.Millisecond
        time.Sleep(delay)
        result <- fmt.Sprintf("Response from %s (took %v)", url, delay)
    }()

    select {
    case res := <-result:
        return res, nil
    case err := <-errCh:
        return "", err
    case <-time.After(timeout):
        return "", fmt.Errorf("timeout after %v", timeout)
    }
}

// 最速レスポンスの採用（First Response Wins）
func queryMultipleServers(servers []string) string {
    result := make(chan string, len(servers))

    for _, server := range servers {
        go func(s string) {
            // 各サーバーに問い合わせ
            delay := time.Duration(rand.Intn(300)) * time.Millisecond
            time.Sleep(delay)
            result <- fmt.Sprintf("Response from %s", s)
        }(server)
    }

    return <-result // 最初のレスポンスを返す
}

// Context によるキャンセル伝播
func longRunningTask(ctx context.Context) error {
    for i := 0; ; i++ {
        select {
        case <-ctx.Done():
            fmt.Printf("タスク中断: %v\n", ctx.Err())
            return ctx.Err()
        default:
            fmt.Printf("処理中: ステップ %d\n", i)
            time.Sleep(100 * time.Millisecond)
        }
    }
}

func main() {
    // タイムアウトの例
    res, err := fetchWithTimeout("https://example.com", 200*time.Millisecond)
    if err != nil {
        fmt.Println("エラー:", err)
    } else {
        fmt.Println(res)
    }

    // 最速レスポンスの例
    servers := []string{"server-a", "server-b", "server-c"}
    fmt.Println(queryMultipleServers(servers))

    // Context キャンセルの例
    ctx, cancel := context.WithTimeout(context.Background(), 350*time.Millisecond)
    defer cancel()
    longRunningTask(ctx)
}
```

### 3.3 ワーカープールパターン

並行処理の代表的パターンであるワーカープールは、固定数のワーカーゴルーチンがジョブチャネルからタスクを取得して処理する構成である。

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

// Job はワーカーが処理するタスクを表す
type Job struct {
    ID      int
    Payload string
}

// Result はジョブの処理結果を表す
type Result struct {
    JobID    int
    Output   string
    Duration time.Duration
}

// worker はジョブチャネルから仕事を受け取り、結果チャネルに書き込む
func worker(id int, jobs <-chan Job, results chan<- Result, wg *sync.WaitGroup) {
    defer wg.Done()
    for job := range jobs {
        start := time.Now()
        // 処理のシミュレーション
        time.Sleep(50 * time.Millisecond)
        output := fmt.Sprintf("Worker %d processed job %d: %s",
            id, job.ID, job.Payload)
        results <- Result{
            JobID:    job.ID,
            Output:   output,
            Duration: time.Since(start),
        }
    }
}

func main() {
    const numWorkers = 4
    const numJobs = 20

    jobs := make(chan Job, numJobs)
    results := make(chan Result, numJobs)

    // ワーカーの起動
    var wg sync.WaitGroup
    for w := 1; w <= numWorkers; w++ {
        wg.Add(1)
        go worker(w, jobs, results, &wg)
    }

    // ジョブの投入
    for j := 1; j <= numJobs; j++ {
        jobs <- Job{ID: j, Payload: fmt.Sprintf("task-%d", j)}
    }
    close(jobs) // これ以上ジョブがないことを通知

    // 全ワーカーの完了を待機し、結果チャネルを閉じる
    go func() {
        wg.Wait()
        close(results)
    }()

    // 結果の収集
    for result := range results {
        fmt.Printf("Job %2d: %s (%v)\n",
            result.JobID, result.Output, result.Duration)
    }
}
```

```
ワーカープールのデータフロー:

  ┌─────────┐     ┌──────────────────────────┐     ┌──────────┐
  │         │     │      jobs チャネル         │     │          │
  │ Producer│────>│ [job1][job2][job3]...     │     │ Collector│
  │         │     └──────┬───┬───┬───────────┘     │          │
  └─────────┘            │   │   │                  └────^─────┘
                         v   v   v                       │
                    ┌────┐┌────┐┌────┐                   │
                    │ W1 ││ W2 ││ W3 │   Workers         │
                    └──┬─┘└──┬─┘└──┬─┘                   │
                       │     │     │                      │
                       v     v     v                      │
                    ┌──────────────────────────┐          │
                    │    results チャネル       │──────────┘
                    │ [res1][res2][res3]...    │
                    └──────────────────────────┘

  特徴:
  - ワーカー数を固定してリソース消費を制御
  - ジョブチャネルの close で全ワーカーが終了
  - WaitGroup で全ワーカーの完了を検知
  - バックプレッシャーはチャネルバッファで自然に発生
```

### 3.4 パイプラインパターン

パイプラインは、処理をステージに分解し、各ステージがチャネルで接続される構成である。Unix のパイプ（`cmd1 | cmd2 | cmd3`）に似た構造を持つ。

```go
package main

import (
    "fmt"
    "math"
    "sync"
)

// generate はスライスの要素をチャネルに送出する（ソースステージ）
func generate(nums ...int) <-chan int {
    out := make(chan int)
    go func() {
        for _, n := range nums {
            out <- n
        }
        close(out)
    }()
    return out
}

// square は入力の各値を二乗する（変換ステージ）
func square(in <-chan int) <-chan int {
    out := make(chan int)
    go func() {
        for n := range in {
            out <- n * n
        }
        close(out)
    }()
    return out
}

// filter は条件を満たす値のみ通過させる（フィルタステージ）
func filter(in <-chan int, pred func(int) bool) <-chan int {
    out := make(chan int)
    go func() {
        for n := range in {
            if pred(n) {
                out <- n
            }
        }
        close(out)
    }()
    return out
}

// fanOut は1つの入力チャネルを n 個のワーカーに分配する
func fanOut(in <-chan int, n int, process func(int) int) []<-chan int {
    outs := make([]<-chan int, n)
    for i := 0; i < n; i++ {
        out := make(chan int)
        outs[i] = out
        go func() {
            for v := range in {
                out <- process(v)
            }
            close(out)
        }()
    }
    return outs
}

// fanIn は複数のチャネルを1つに統合する
func fanIn(channels ...<-chan int) <-chan int {
    out := make(chan int)
    var wg sync.WaitGroup
    for _, ch := range channels {
        wg.Add(1)
        go func(c <-chan int) {
            defer wg.Done()
            for v := range c {
                out <- v
            }
        }(ch)
    }
    go func() {
        wg.Wait()
        close(out)
    }()
    return out
}

func main() {
    // 基本パイプライン: generate -> square -> filter -> 出力
    isPrime := func(n int) bool {
        if n < 2 {
            return false
        }
        for i := 2; i <= int(math.Sqrt(float64(n))); i++ {
            if n%i == 0 {
                return false
            }
        }
        return true
    }

    pipeline := filter(
        square(generate(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)),
        isPrime,
    )

    fmt.Println("素数である平方数:")
    for v := range pipeline {
        fmt.Println(v) // 4, 9, 25, 49 が出力される
    }
}
```

---

## 4. Rust のチャネル: 所有権による安全性

### 4.1 標準ライブラリの mpsc チャネル

Rust の `std::sync::mpsc`（Multi-Producer, Single-Consumer）チャネルは、所有権システムと連携してコンパイル時にデータ競合を防ぐ。

```rust
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn main() {
    // === 基本的な使用 ===
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        let data = String::from("Hello from thread!");
        tx.send(data).unwrap();
        // data は move されたため、ここでは使用不可
        // println!("{}", data); // コンパイルエラー!
    });

    let received = rx.recv().unwrap();
    println!("受信: {}", received);

    // === 複数プロデューサー ===
    let (tx, rx) = mpsc::channel();

    for i in 0..5 {
        let tx_clone = tx.clone();
        thread::spawn(move || {
            let msg = format!("スレッド {} からのメッセージ", i);
            tx_clone.send(msg).unwrap();
            thread::sleep(Duration::from_millis(100));
        });
    }
    drop(tx); // 元の tx を破棄（全送信者が drop されると rx が終了）

    // イテレータとして受信（全送信者が drop されるまで）
    for msg in rx {
        println!("{}", msg);
    }

    // === 同期チャネル（バッファ付き） ===
    let (tx, rx) = mpsc::sync_channel(2); // バッファサイズ 2

    thread::spawn(move || {
        tx.send(1).unwrap(); // 即座に返る
        tx.send(2).unwrap(); // 即座に返る
        println!("バッファ満杯前");
        tx.send(3).unwrap(); // バッファ満杯、受信されるまでブロック
        println!("3つ目送信完了");
    });

    thread::sleep(Duration::from_millis(500));
    println!("受信開始");
    for val in rx {
        println!("値: {}", val);
    }
}
```

### 4.2 crossbeam チャネル: 高機能な代替

`crossbeam-channel` クレートは、MPMC（Multi-Producer, Multi-Consumer）チャネルと `select!` マクロを提供する。

```rust
use crossbeam_channel::{bounded, unbounded, select, Receiver, Sender};
use std::thread;
use std::time::Duration;

// タイムアウト付きのリクエスト処理
fn request_with_timeout(timeout: Duration) -> Result<String, String> {
    let (tx, rx) = bounded(1);

    thread::spawn(move || {
        // 処理のシミュレーション
        thread::sleep(Duration::from_millis(200));
        let _ = tx.send("処理完了".to_string());
    });

    select! {
        recv(rx) -> msg => msg.map_err(|e| e.to_string()),
        default(timeout) => Err("タイムアウト".to_string()),
    }
}

// ファンイン: 複数のソースを1つに統合
fn fan_in(receivers: Vec<Receiver<String>>) -> Receiver<String> {
    let (tx, rx) = unbounded();

    for r in receivers {
        let tx = tx.clone();
        thread::spawn(move || {
            for msg in r {
                if tx.send(msg).is_err() {
                    break;
                }
            }
        });
    }

    rx
}

fn main() {
    // タイムアウトの例
    match request_with_timeout(Duration::from_millis(300)) {
        Ok(msg) => println!("成功: {}", msg),
        Err(e) => println!("失敗: {}", e),
    }

    // 複数チャネルの select
    let (s1, r1) = bounded::<String>(0);
    let (s2, r2) = bounded::<String>(0);

    thread::spawn(move || {
        thread::sleep(Duration::from_millis(100));
        s1.send("チャネル1".to_string()).unwrap();
    });

    thread::spawn(move || {
        thread::sleep(Duration::from_millis(200));
        s2.send("チャネル2".to_string()).unwrap();
    });

    // 両方のチャネルを待機
    for _ in 0..2 {
        select! {
            recv(r1) -> msg => println!("r1: {:?}", msg),
            recv(r2) -> msg => println!("r2: {:?}", msg),
        }
    }
}
```

---

## 5. アクターモデルの実装

### 5.1 Erlang/Elixir: OTP によるアクター

Erlang/OTP はアクターモデルの最も成熟した実装である。Erlang のプロセスは OS スレッドではなく、VM が管理する軽量プロセス（約 300 バイト）であり、数百万プロセスの同時実行が可能である。

```elixir
# === GenServer を使ったアクター（Elixir） ===
defmodule BankAccount do
  use GenServer

  # --- クライアント API ---
  def start_link(initial_balance) do
    GenServer.start_link(__MODULE__, initial_balance)
  end

  def deposit(account, amount) when amount > 0 do
    GenServer.call(account, {:deposit, amount})
  end

  def withdraw(account, amount) when amount > 0 do
    GenServer.call(account, {:withdraw, amount})
  end

  def balance(account) do
    GenServer.call(account, :balance)
  end

  # 非同期通知（返答不要）
  def notify(account, message) do
    GenServer.cast(account, {:notify, message})
  end

  # --- サーバーコールバック ---
  @impl true
  def init(initial_balance) do
    {:ok, %{balance: initial_balance, history: []}}
  end

  @impl true
  def handle_call({:deposit, amount}, _from, state) do
    new_balance = state.balance + amount
    new_state = %{
      balance: new_balance,
      history: [{:deposit, amount, new_balance} | state.history]
    }
    {:reply, {:ok, new_balance}, new_state}
  end

  @impl true
  def handle_call({:withdraw, amount}, _from, state) do
    if state.balance >= amount do
      new_balance = state.balance - amount
      new_state = %{
        balance: new_balance,
        history: [{:withdraw, amount, new_balance} | state.history]
      }
      {:reply, {:ok, new_balance}, new_state}
    else
      {:reply, {:error, :insufficient_funds}, state}
    end
  end

  @impl true
  def handle_call(:balance, _from, state) do
    {:reply, state.balance, state}
  end

  @impl true
  def handle_cast({:notify, message}, state) do
    IO.puts("通知: #{message}")
    {:noreply, state}
  end
end

# 使用例
{:ok, account} = BankAccount.start_link(1000)
{:ok, balance} = BankAccount.deposit(account, 500)
IO.puts("残高: #{balance}")  # => 残高: 1500

{:ok, balance} = BankAccount.withdraw(account, 200)
IO.puts("残高: #{balance}")  # => 残高: 1300

{:error, :insufficient_funds} = BankAccount.withdraw(account, 5000)
IO.puts("残高: #{BankAccount.balance(account)}")  # => 残高: 1300
```

### 5.2 Erlang/OTP の監視ツリー

アクターモデルの最大の強みの一つは、障害を隔離し自動回復できる監視ツリー（Supervision Tree）である。

```
Erlang/OTP 監視ツリーの構造:

        ┌──────────────┐
        │  Application │
        │  Supervisor  │
        └──────┬───────┘
               │
       ┌───────┼───────┐
       v       v       v
  ┌────────┐┌──────┐┌──────────┐
  │ Worker ││ Sub  ││ Worker   │
  │ A      ││ Sup  ││ C        │
  └────────┘└──┬───┘└──────────┘
               │
          ┌────┼────┐
          v    v    v
      ┌────┐┌────┐┌────┐
      │ W1 ││ W2 ││ W3 │
      └────┘└────┘└────┘

  再起動戦略:
  - :one_for_one   : 落ちた子だけ再起動
  - :one_for_all   : 1つ落ちたら全子を再起動
  - :rest_for_one  : 落ちた子以降を再起動

  "Let it crash" 哲学:
  エラー処理をアクター内部に書くのではなく、
  アクターをクラッシュさせて監視者が再起動する。
  防御的プログラミングよりシンプルで堅牢。
```

```elixir
# 監視ツリーの定義
defmodule MyApp.Supervisor do
  use Supervisor

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, :ok, opts)
  end

  @impl true
  def init(:ok) do
    children = [
      # 子プロセスの仕様
      {BankAccount, 0},
      {MyApp.Cache, []},
      {MyApp.EventBus, []}
    ]

    # one_for_one: 個別に再起動
    # max_restarts: 5秒間に3回まで再起動を許容
    Supervisor.init(children,
      strategy: :one_for_one,
      max_restarts: 3,
      max_seconds: 5
    )
  end
end
```

---

## 6. 主要なメッセージパッシングパターン

### 6.1 パターン一覧

メッセージパッシングには、繰り返し現れる設計パターンがある。以下に主要なパターンとその用途をまとめる。

| パターン | 概要 | 用途 | 言語例 |
|---------|------|------|--------|
| Request-Reply | メッセージ送信後、返答を待つ | RPC 的な同期処理 | Go (chan pair), Erlang (call) |
| Fire-and-Forget | 送信のみで返答を待たない | ログ送信、通知 | Erlang (cast), Akka (tell) |
| Pipeline | ステージを直列接続 | データ変換フロー | Go (chan chain), Unix pipe |
| Fan-Out/Fan-In | 分散処理と結果統合 | 並列バッチ処理 | Go (multiple goroutines) |
| Publish-Subscribe | 購読者全員にブロードキャスト | イベント配信 | Erlang (pg), Redis Pub/Sub |
| Scatter-Gather | 複数に問い合わせ、全結果を集約 | 分散検索 | Go (select + WaitGroup) |
| Router | メッセージを条件に応じて振り分け | ロードバランシング | Akka (Router), Go (select) |
| Dead Letter | 配送不能メッセージの処理 | エラーハンドリング | Akka (DeadLetter), RabbitMQ |

### 6.2 Fan-Out / Fan-In パターンの詳細

Fan-Out は一つのソースから複数のワーカーに作業を分配し、Fan-In は複数のワーカーの結果を一つのチャネルに統合するパターンである。

```
Fan-Out / Fan-In パターン:

                Fan-Out                    Fan-In

  ┌──────┐    ┌─────────┐              ┌─────────┐    ┌──────┐
  │      │───>│ Worker1 │──────────────│         │    │      │
  │      │    └─────────┘              │         │    │      │
  │ Source│    ┌─────────┐              │ Merger  │───>│ Sink │
  │      │───>│ Worker2 │──────────────│         │    │      │
  │      │    └─────────┘              │         │    │      │
  │      │    ┌─────────┐              │         │    │      │
  │      │───>│ Worker3 │──────────────│         │    │      │
  └──────┘    └─────────┘              └─────────┘    └──────┘

  データの流れ:
  1. Source が jobs チャネルにタスクを投入
  2. 複数の Worker が jobs チャネルから取得（Fan-Out）
  3. 各 Worker が結果を results チャネルに書き込み
  4. Merger が全 results を集約（Fan-In）
  5. Sink が最終結果を利用

  利点:
  - CPU バウンドな処理を並列化
  - ワーカー数の調整でスループットを制御
  - 各ワーカーは独立しており障害が隔離される
```

### 6.3 Publish-Subscribe パターン

Publish-Subscribe（Pub/Sub）パターンでは、発行者（Publisher）がトピックにメッセージを送信し、そのトピックを購読している全ての購読者（Subscriber）がメッセージを受信する。

```go
package main

import (
    "fmt"
    "sync"
    "time"
)

// EventBus はトピックベースの Pub/Sub を実装する
type EventBus struct {
    mu          sync.RWMutex
    subscribers map[string][]chan interface{}
}

func NewEventBus() *EventBus {
    return &EventBus{
        subscribers: make(map[string][]chan interface{}),
    }
}

// Subscribe はトピックを購読し、メッセージを受信するチャネルを返す
func (eb *EventBus) Subscribe(topic string, bufferSize int) <-chan interface{} {
    eb.mu.Lock()
    defer eb.mu.Unlock()

    ch := make(chan interface{}, bufferSize)
    eb.subscribers[topic] = append(eb.subscribers[topic], ch)
    return ch
}

// Publish はトピックの全購読者にメッセージを送信する
func (eb *EventBus) Publish(topic string, msg interface{}) {
    eb.mu.RLock()
    defer eb.mu.RUnlock()

    for _, ch := range eb.subscribers[topic] {
        // 非ブロッキング送信（バッファ満杯なら破棄）
        select {
        case ch <- msg:
        default:
            // 購読者が追いつかない場合は破棄
            fmt.Printf("警告: トピック %s の購読者がメッセージを処理できません\n", topic)
        }
    }
}

// Close はトピックの全チャネルを閉じる
func (eb *EventBus) Close(topic string) {
    eb.mu.Lock()
    defer eb.mu.Unlock()

    for _, ch := range eb.subscribers[topic] {
        close(ch)
    }
    delete(eb.subscribers, topic)
}

func main() {
    bus := NewEventBus()

    // 購読者 A: "orders" トピック
    ordersA := bus.Subscribe("orders", 10)
    go func() {
        for msg := range ordersA {
            fmt.Printf("[購読者A] 注文受信: %v\n", msg)
        }
    }()

    // 購読者 B: "orders" トピック
    ordersB := bus.Subscribe("orders", 10)
    go func() {
        for msg := range ordersB {
            fmt.Printf("[購読者B] 注文受信: %v\n", msg)
        }
    }()

    // 購読者 C: "logs" トピック
    logs := bus.Subscribe("logs", 100)
    go func() {
        for msg := range logs {
            fmt.Printf("[ログ] %v\n", msg)
        }
    }()

    // 発行
    bus.Publish("orders", map[string]interface{}{
        "id": 1, "item": "Book", "price": 2980,
    })
    bus.Publish("orders", map[string]interface{}{
        "id": 2, "item": "Pen", "price": 150,
    })
    bus.Publish("logs", "Order processing started")

    time.Sleep(100 * time.Millisecond)
    bus.Close("orders")
    bus.Close("logs")
}
```

### 6.4 Request-Reply パターン（Go 実装）

Request-Reply は、リクエストに応答チャネルを埋め込み、呼び出し側が返答を待つパターンである。

```go
package main

import (
    "fmt"
    "time"
)

// Request は応答チャネルを含むリクエスト
type Request struct {
    Query    string
    ReplyCh  chan Response  // 応答を受け取るチャネル
}

// Response はリクエストへの応答
type Response struct {
    Answer string
    Err    error
}

// server はリクエストを受け取って処理するゴルーチン
func server(requests <-chan Request) {
    for req := range requests {
        // リクエストの処理
        time.Sleep(50 * time.Millisecond) // 処理時間
        req.ReplyCh <- Response{
            Answer: fmt.Sprintf("Answer to: %s", req.Query),
            Err:    nil,
        }
    }
}

// ask はサーバーにリクエストを送り、応答を待つ
func ask(requests chan<- Request, query string, timeout time.Duration) (Response, error) {
    replyCh := make(chan Response, 1)
    requests <- Request{Query: query, ReplyCh: replyCh}

    select {
    case resp := <-replyCh:
        return resp, resp.Err
    case <-time.After(timeout):
        return Response{}, fmt.Errorf("request timed out")
    }
}

func main() {
    requests := make(chan Request, 10)
    go server(requests)

    resp, err := ask(requests, "What is message passing?", time.Second)
    if err != nil {
        fmt.Println("エラー:", err)
    } else {
        fmt.Println("応答:", resp.Answer)
    }

    close(requests)
}
```

---

## 7. 共有メモリとメッセージパッシングの使い分け

### 7.1 判断基準の詳細

どちらのモデルを使うべきかは、問題の性質によって決まる。以下の判断フローチャートを参考にされたい。

```
使い分け判断フローチャート:

  処理単位は独立しているか？
  │
  ├── Yes ──> データのコピーコストは許容できるか？
  │           │
  │           ├── Yes ──> メッセージパッシングを推奨
  │           │           (チャネル or アクター)
  │           │
  │           └── No ───> 大きなデータ構造の共有が必要
  │                       ├── 読み取り専用 ──> 不変データ共有
  │                       └── 読み書き ──────> 共有メモリ + RWMutex
  │
  └── No ───> 密結合な状態を共有する必要があるか？
              │
              ├── Yes ──> 共有メモリ(Mutex/Atomic)を推奨
              │           ※ ただし競合条件に細心の注意
              │
              └── No ───> タスク分解を再検討
                          独立した単位に分割できないか再設計
```

### 7.2 詳細比較表

| 評価軸 | メッセージパッシング | 共有メモリ |
|--------|---------------------|-----------|
| データ競合の安全性 | 構造的に安全 | プログラマの責任 |
| パフォーマンス（低レイテンシ） | メッセージコピーのオーバーヘッド | 直接アクセスで高速 |
| パフォーマンス（スループット） | パイプラインで高スループット | ロック競合で低下する場合あり |
| スケーラビリティ | 水平スケールが容易 | 単一マシン内に限定 |
| デバッグ容易性 | メッセージトレースで追跡可能 | 非決定的な再現困難バグ |
| コード複雑性 | パターンに従えば低い | ロック設計が複雑化しやすい |
| メモリ効率 | メッセージコピーで消費増 | 直接共有で効率的 |
| 障害隔離 | プロセス境界で隔離 | 1スレッドの異常が全体に波及 |
| 分散対応 | ネットワーク越しに自然に拡張 | 分散共有メモリは極めて困難 |
| 型安全性 | チャネル型で静的検査（Go, Rust） | ロック対象の型安全性は限定的 |
| テスタビリティ | メッセージの注入・検査が容易 | モック困難、タイミング依存 |

### 7.3 各モデルが適する具体的シナリオ

**メッセージパッシングが適する場面:**

- ETL パイプライン（Extract-Transform-Load）のステージ間通信
- マイクロサービス間の通信（gRPC, メッセージキュー）
- ゲームサーバーのプレイヤーセッション管理（各プレイヤーをアクターに）
- IoT デバイスからのデータ収集と処理
- Web サーバーのリクエストハンドリング（リクエストごとにゴルーチン）
- 障害に強いテレコムシステム（Erlang/OTP の設計ターゲット）

**共有メモリが適する場面:**

- インメモリキャッシュ（`sync.Map` や `RwLock<HashMap>`）
- アトミックカウンター（メトリクス収集、レート制限）
- 読み取り中心のコンフィグ共有（`RwLock` で読み取り並行性を確保）
- 高頻度取引システムのオーダーブック（ナノ秒レベルのレイテンシ要件）
- 科学計算の行列演算（大規模データのコピーは非現実的）

---

## 8. 分散メッセージパッシング

### 8.1 ネットワーク越しのメッセージパッシング

メッセージパッシングの大きな利点は、プロセス間通信からネットワーク通信への移行が自然であることだ。アクターモデルでは「位置透過性（Location Transparency）」により、同一マシンか異なるマシンかを意識せずにメッセージを送受信できる。

```
分散メッセージパッシングのアーキテクチャ:

  Node A (Tokyo)                    Node B (Osaka)
  ┌─────────────────┐              ┌─────────────────┐
  │  ┌─────┐        │    TCP/UDP   │        ┌─────┐  │
  │  │Act-1│──────────────────────────────>│Act-3│  │
  │  └─────┘        │   serialize  │        └─────┘  │
  │  ┌─────┐        │   + send     │        ┌─────┐  │
  │  │Act-2│<──────────────────────────────│Act-4│  │
  │  └─────┘        │  deserialize │        └─────┘  │
  │                 │   + deliver  │                 │
  └─────────────────┘              └─────────────────┘

  ネットワーク越しの考慮事項:
  1. シリアライゼーション（Protocol Buffers, JSON, MessagePack）
  2. メッセージ配送保証（at-most-once, at-least-once, exactly-once）
  3. 順序保証（因果順序, 全順序）
  4. 障害検出（ハートビート, タイムアウト）
  5. ネットワーク分断（CAP定理との関連）
```

### 8.2 メッセージ配送保証

分散システムでのメッセージ配送には三つの保証レベルがある:

| 保証レベル | 説明 | トレードオフ | 実装例 |
|-----------|------|-------------|--------|
| At-most-once | 最大1回配送。重複なし | メッセージ喪失の可能性 | UDP, Erlang デフォルト |
| At-least-once | 最低1回配送。喪失なし | 重複の可能性。冪等性が必要 | TCP + ACK, Kafka |
| Exactly-once | 正確に1回配送 | 高コスト。完全な実現は困難 | Kafka トランザクション |

### 8.3 メッセージキューとの関係

プロセス間のチャネルやアクターのメールボックスを、ネットワーク越しに拡張したものがメッセージキュー（MQ）である。

```
メッセージキューの位置づけ:

  プロセス内          プロセス間            マシン間
  ┌─────────┐       ┌─────────────┐      ┌──────────────┐
  │ Channel │       │ Unix Domain │      │ Message Queue│
  │ (Go)    │       │ Socket      │      │ (RabbitMQ,   │
  │         │       │ Named Pipe  │      │  Kafka, NATS)│
  └─────────┘       └─────────────┘      └──────────────┘
  ← 低レイテンシ                      高信頼性・永続化 →
  ← シンプル                          豊富な機能 →

  メッセージキューの付加価値:
  - 永続化: メッセージをディスクに保存
  - 再試行: 配送失敗時の自動再送
  - ルーティング: トピック、パターンマッチ
  - 監視: メッセージ量、遅延のメトリクス
  - デッドレター: 処理不能メッセージの退避
```

---

## 9. アンチパターンと落とし穴

### 9.1 アンチパターン 1: チャネルの過剰使用

Go においてよくあるアンチパターンは、単純なミューテックスで十分な場面でもチャネルを使おうとすることである。

```go
// === アンチパターン: チャネルによるカウンター ===
// 不必要に複雑で、パフォーマンスも悪い
type ChannelCounter struct {
    incrementCh chan struct{}
    getCh       chan chan int
}

func NewChannelCounter() *ChannelCounter {
    c := &ChannelCounter{
        incrementCh: make(chan struct{}),
        getCh:       make(chan chan int),
    }
    go func() {
        count := 0
        for {
            select {
            case <-c.incrementCh:
                count++
            case replyCh := <-c.getCh:
                replyCh <- count
            }
        }
    }()
    return c
}

func (c *ChannelCounter) Increment() { c.incrementCh <- struct{}{} }
func (c *ChannelCounter) Get() int {
    replyCh := make(chan int, 1)
    c.getCh <- replyCh
    return <-replyCh
}

// === 推奨: sync/atomic による単純なカウンター ===
// シンプルで高速
import "sync/atomic"

type AtomicCounter struct {
    count int64
}

func (c *AtomicCounter) Increment() { atomic.AddInt64(&c.count, 1) }
func (c *AtomicCounter) Get() int64 { return atomic.LoadInt64(&c.count) }

// 判断基準:
// - 単純な状態の保護 → sync.Mutex / sync/atomic
// - データの所有権移動 → チャネル
// - 複数の非同期イベントの待機 → チャネル + select
// - パイプライン処理 → チャネル
```

### 9.2 アンチパターン 2: ゴルーチンリーク

チャネルの受信側がいなくなった場合、送信側のゴルーチンが永遠にブロックされ、メモリリーク（ゴルーチンリーク）が発生する。

```go
// === アンチパターン: ゴルーチンリーク ===
func leakySearch(query string) string {
    results := make(chan string)

    // 3つのバックエンドに問い合わせ
    go func() { results <- searchBackendA(query) }()
    go func() { results <- searchBackendB(query) }()
    go func() { results <- searchBackendC(query) }()

    // 最初の結果だけ返す
    return <-results
    // 問題: 残り2つのゴルーチンは永遠にブロックされる!
    // results に送信しようとするが、受信者がいない
}

// === 修正版: Context によるキャンセルとバッファ ===
func safeSearch(ctx context.Context, query string) string {
    // バッファサイズを送信者数と同じにする
    results := make(chan string, 3)

    ctx, cancel := context.WithCancel(ctx)
    defer cancel() // 最初の結果を得たら他をキャンセル

    search := func(backend func(context.Context, string) string) {
        select {
        case results <- backend(ctx, query):
        case <-ctx.Done():
            return // キャンセルされたら終了
        }
    }

    go search(searchBackendA)
    go search(searchBackendB)
    go search(searchBackendC)

    return <-results
}

// キャンセルを受け取るバックエンドの例
func searchBackendA(ctx context.Context, query string) string {
    select {
    case <-time.After(200 * time.Millisecond):
        return "Result from A"
    case <-ctx.Done():
        return "" // キャンセル時は即座に戻る
    }
}
```

### 9.3 アンチパターン 3: デッドロック（循環待ち）

メッセージパッシングでも、二つのプロセスが互いのメッセージを待つことでデッドロックが発生する。

```go
// === アンチパターン: チャネル間のデッドロック ===
func deadlock() {
    chA := make(chan int)
    chB := make(chan int)

    // ゴルーチン 1: chA に送信してから chB を受信
    go func() {
        chA <- 1      // chB が受信するまでブロック
        val := <-chB  // 永遠に到達しない
        fmt.Println(val)
    }()

    // ゴルーチン 2: chB に送信してから chA を受信
    go func() {
        chB <- 2      // chA が受信するまでブロック
        val := <-chA  // 永遠に到達しない
        fmt.Println(val)
    }()

    // 両方のゴルーチンが互いを待ってデッドロック!
}

// === 修正版: select で非決定的に選択 ===
func noDeadlock() {
    chA := make(chan int, 1)
    chB := make(chan int, 1)

    go func() {
        for i := 0; i < 5; i++ {
            select {
            case chA <- i:
            case val := <-chB:
                fmt.Println("G1 received:", val)
            }
        }
    }()

    go func() {
        for i := 0; i < 5; i++ {
            select {
            case chB <- i + 100:
            case val := <-chA:
                fmt.Println("G2 received:", val)
            }
        }
    }()

    time.Sleep(time.Second)
}
```

### 9.4 アンチパターン 4: 無制限バッファによるメモリ枯渇

非同期メッセージパッシングにおいて、バッファサイズを無制限にすると、消費者が追いつかない場合にメモリが枯渇する。

```go
// === アンチパターン: 無制限バッファ（メモリ枯渇の危険）===
func unboundedBuffer() {
    // Go のチャネルは固定バッファだが、
    // スライスで無制限キューを実装するとメモリ枯渇のリスク
    type UnboundedChan struct {
        in   chan int
        out  chan int
        buf  []int
    }
    // 生産速度 > 消費速度の場合、buf が無限に成長する

    // === 推奨: バックプレッシャーの設計 ===
    // バッファ付きチャネルで自然なバックプレッシャー
    ch := make(chan int, 100) // 100 要素で生産者がブロック

    // または明示的なレート制限
    // rate.NewLimiter(rate.Every(time.Millisecond), 100)
}
```

---

## 10. パフォーマンス特性と設計指針

### 10.1 チャネル / メールボックスのコスト

メッセージパッシングのコストは、主に以下の要素で構成される:

```
メッセージパッシングのコスト構造:

  ┌──────────────────────────────────────────┐
  │          メッセージ送信のコスト            │
  │                                          │
  │  1. メモリアロケーション                   │
  │     - メッセージオブジェクトの生成          │
  │     - エンベロープ（メタデータ）の付加      │
  │                                          │
  │  2. データのコピーまたは所有権移動          │
  │     - 値渡し: データのディープコピー        │
  │     - 所有権移動: ポインタ移動(低コスト)    │
  │                                          │
  │  3. 同期オーバーヘッド                     │
  │     - チャネル内部のロック取得              │
  │     - ゴルーチンの起床（コンテキストスイッチ）│
  │                                          │
  │  4. スケジューリング                       │
  │     - 受信側ゴルーチンのランキュー追加      │
  │     - M:N スケジューラのコスト              │
  └──────────────────────────────────────────┘

  Go チャネルの内部:
  ┌─────────────────────────────────┐
  │          hchan 構造体            │
  │  ┌───────────────────────────┐  │
  │  │ buf: リングバッファ        │  │
  │  │ qcount: 現在の要素数       │  │
  │  │ dataqsiz: バッファサイズ   │  │
  │  │ elemsize: 要素サイズ       │  │
  │  │ sendx: 送信インデックス     │  │
  │  │ recvx: 受信インデックス     │  │
  │  │ recvq: 受信待ちキュー       │  │
  │  │ sendq: 送信待ちキュー       │  │
  │  │ lock: mutex                │  │
  │  └───────────────────────────┘  │
  └─────────────────────────────────┘
```

### 10.2 設計指針のまとめ

| 指針 | 説明 | 具体例 |
|------|------|--------|
| 所有権の明確化 | メッセージ送信後は送信側がデータに触れない | Go: 送信後のスライス操作を避ける |
| バッファサイズの適正化 | 大きすぎず小さすぎないバッファを選ぶ | ワーカー数の 2-3 倍が目安 |
| タイムアウトの設定 | 全てのチャネル操作にタイムアウトを設ける | `select` + `time.After` / Context |
| グレースフルシャットダウン | チャネルの close で終了を伝播する | 送信側が close、受信側が `range` で検知 |
| エラー伝播の設計 | エラーもメッセージとして送信する | `Result[T]` 型（値 + エラー）を送信 |
| 監視と可観測性 | メッセージ数、遅延を計測する | Prometheus メトリクスの埋め込み |

---

## 11. 言語別メッセージパッシング機能の総合比較

| 言語 / ランタイム | モデル | チャネル型 | select 相当 | バッファ | 所有権 | 分散対応 |
|-------------------|--------|-----------|-------------|---------|--------|----------|
| Go | CSP | `chan T` | `select` | 任意サイズ | 規約ベース | 標準なし（gRPC等） |
| Rust (std) | CSP | `mpsc::Sender/Receiver` | なし（recv のみ） | Unbounded/Sync | 型システムで強制 | 標準なし |
| Rust (crossbeam) | CSP | `Sender/Receiver` | `select!` | bounded/unbounded | 型システムで強制 | 標準なし |
| Rust (tokio) | CSP | `mpsc`, `broadcast`, `watch` | `tokio::select!` | 任意サイズ | 型システムで強制 | 標準なし |
| Erlang/OTP | Actor | メールボックス | `receive` (パターンマッチ) | 無制限 | コピー（不変値） | ネイティブ（Distributed Erlang） |
| Elixir | Actor | メールボックス | `receive` | 無制限 | コピー（不変値） | ネイティブ（OTP） |
| Scala (Akka) | Actor | `ActorRef` | `receive` (パターンマッチ) | メールボックス設定 | JVM GC | Akka Cluster |
| Kotlin | CSP | `Channel<T>` | `select` | Rendezvous/Buffered | コルーチンスコープ | 標準なし |
| Clojure | CSP | `core.async/chan` | `alts!` / `alts!!` | 任意サイズ | 不変データ構造 | 標準なし |
| Swift | Actor | `actor` 型 | `async let` + `TaskGroup` | なし（直接呼出し） | 値型 + Sendable | 標準なし |

---

## 12. 演習問題

### 12.1 基礎演習: 温度変換パイプライン

**課題**: 華氏温度のスライスをチャネルベースのパイプラインで摂氏に変換し、氷点下のもののみ出力するプログラムを Go で実装せよ。

```
要件:
1. generate ステージ: 華氏温度のスライスをチャネルに送出
2. convert ステージ: 華氏から摂氏に変換（C = (F - 32) * 5/9）
3. filter ステージ: 0度未満のみ通過
4. 各ステージは独立したゴルーチンで動作すること
```

<details>
<summary>解答例（クリックで展開）</summary>

```go
package main

import "fmt"

func generateF(temps ...float64) <-chan float64 {
    out := make(chan float64)
    go func() {
        for _, t := range temps {
            out <- t
        }
        close(out)
    }()
    return out
}

func toCelsius(in <-chan float64) <-chan float64 {
    out := make(chan float64)
    go func() {
        for f := range in {
            out <- (f - 32.0) * 5.0 / 9.0
        }
        close(out)
    }()
    return out
}

func belowFreezing(in <-chan float64) <-chan float64 {
    out := make(chan float64)
    go func() {
        for c := range in {
            if c < 0 {
                out <- c
            }
        }
        close(out)
    }()
    return out
}

func main() {
    fahrenheits := []float64{32, 212, 0, -40, 50, 14, 20, 100}
    for c := range belowFreezing(toCelsius(generateF(fahrenheits...))) {
        fmt.Printf("%.1f°C (氷点下)\n", c)
    }
}
```
</details>

### 12.2 中級演習: タイムアウト付き並行 Web クローラー

**課題**: 複数の URL を並行にフェッチし、タイムアウト制御付きで結果を集約するプログラムを設計せよ。

```
要件:
1. URL のリストを受け取り、各 URL を別々のゴルーチンでフェッチ
2. 各フェッチには個別のタイムアウト（3秒）を設定
3. 全体のタイムアウト（10秒）も設定
4. 結果は成功/失敗を含む構造体のスライスで返す
5. Context によるキャンセル伝播を実装
6. 最大同時接続数を制限（セマフォパターン）
```

<details>
<summary>解答の骨格（クリックで展開）</summary>

```go
package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type FetchResult struct {
    URL      string
    Body     string
    Duration time.Duration
    Err      error
}

func crawl(ctx context.Context, urls []string, maxConcurrency int,
    perURLTimeout, totalTimeout time.Duration) []FetchResult {

    ctx, cancel := context.WithTimeout(ctx, totalTimeout)
    defer cancel()

    results := make([]FetchResult, 0, len(urls))
    var mu sync.Mutex
    var wg sync.WaitGroup

    // セマフォ: 同時実行数の制限
    semaphore := make(chan struct{}, maxConcurrency)

    for _, url := range urls {
        wg.Add(1)
        go func(u string) {
            defer wg.Done()

            semaphore <- struct{}{} // セマフォ取得
            defer func() { <-semaphore }()

            start := time.Now()
            urlCtx, urlCancel := context.WithTimeout(ctx, perURLTimeout)
            defer urlCancel()

            body, err := fetch(urlCtx, u) // fetch は仮想関数
            result := FetchResult{
                URL:      u,
                Body:     body,
                Duration: time.Since(start),
                Err:      err,
            }

            mu.Lock()
            results = append(results, result)
            mu.Unlock()
        }(url)
    }

    wg.Wait()
    return results
}

func fetch(ctx context.Context, url string) (string, error) {
    // 仮実装: HTTP クライアントで取得する想定
    select {
    case <-time.After(100 * time.Millisecond):
        return fmt.Sprintf("Content of %s", url), nil
    case <-ctx.Done():
        return "", ctx.Err()
    }
}

func main() {
    urls := []string{
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
        "https://example.com/d",
    }
    results := crawl(context.Background(), urls, 2,
        3*time.Second, 10*time.Second)
    for _, r := range results {
        if r.Err != nil {
            fmt.Printf("[FAIL] %s: %v\n", r.URL, r.Err)
        } else {
            fmt.Printf("[OK]   %s (%v)\n", r.URL, r.Duration)
        }
    }
}
```
</details>

### 12.3 上級演習: Erlang/Elixir 風の監視ツリー（Go 実装）

**課題**: Go で簡易的な Supervisor パターンを実装せよ。

```
要件:
1. Worker インターフェース（Start, Stop メソッド）を定義
2. Supervisor が複数の Worker を管理
3. Worker がパニックした場合、Supervisor が自動的に再起動
4. 再起動戦略: one-for-one（落ちた Worker のみ再起動）
5. 再起動回数の上限を設定（5秒間に3回まで）
6. 上限超過時は Supervisor 自体がエラーを報告
```

<details>
<summary>設計のヒント（クリックで展開）</summary>

```go
// Worker インターフェース
type Worker interface {
    Name() string
    Run(ctx context.Context) error
}

// Supervisor の構造
type Supervisor struct {
    workers      []Worker
    maxRestarts  int
    window       time.Duration
    restartLog   []time.Time // 再起動タイムスタンプの履歴
}

// 実装のポイント:
// 1. 各 Worker を別ゴルーチンで起動
// 2. errgroup または独自のエラーチャネルで障害を検知
// 3. recover() でパニックを捕捉
// 4. 再起動判定: window 内の再起動回数が maxRestarts 以下か確認
// 5. Context のキャンセルでグレースフルシャットダウン
```
</details>

---

## 13. よくある質問（FAQ）

### Q1: チャネルとミューテックス、どちらを使うべきですか？

**A**: Go の公式 Wiki（"Share Memory By Communicating"）に基づく判断基準は以下の通りである:

- **チャネルを使う場面**: データの所有権を移動させる場合、複数の非同期イベントを調整する場合、パイプライン処理
- **ミューテックスを使う場面**: 単純なカウンター、キャッシュ、設定値の保護など、データの保護のみが目的の場合

一般的に、「データが流れる」場合はチャネル、「データを守る」場合はミューテックスが適する。Go の標準ライブラリ自体も、`sync.Map` や `sync.Pool` など共有メモリモデルを多用しており、チャネル一辺倒ではない。

### Q2: Erlang のアクターモデルは本当に数百万プロセスを扱えますか？

**A**: Erlang VM（BEAM）のプロセスは OS スレッドではなく、VM が管理する軽量エンティティである。各プロセスは約 300 バイトの初期メモリで開始し、ヒープは独立しているためガベージコレクションもプロセス単位で行われる。WhatsApp は 2012 年時点で 1 台のサーバーに 200 万の同時接続を Erlang で処理していた事例がある。ただし、プロセス数が増えるとスケジューラのオーバーヘッドが増加するため、実際のスケーラビリティはワークロードのパターンに依存する。

### Q3: メッセージパッシングはマイクロサービスのサービス間通信とどう関係しますか？

**A**: メッセージパッシングの概念は、プロセス内（チャネル）、プロセス間（IPC）、マシン間（ネットワーク）のすべてのスケールに適用される。マイクロサービスにおける非同期メッセージング（Kafka, RabbitMQ, NATS 等）は、アクターモデルの考え方をシステムアーキテクチャレベルに拡張したものと言える。各サービスがアクターに相当し、メッセージキューがメールボックスに相当する。同期的な gRPC/REST 呼び出しは Request-Reply パターンに、イベント駆動アーキテクチャは Pub/Sub パターンに対応する。

### Q4: Go のチャネルは内部的にどう実装されていますか？

**A**: Go のチャネルは `runtime.hchan` 構造体として実装されている。内部にはリングバッファ（バッファ付きの場合）、送信待ちキュー（sudog のリスト）、受信待ちキュー、そしてミューテックスを持つ。つまり、チャネル自体は内部的にロックを使用している。しかし、プログラマがロックを直接扱う必要がなく、チャネルの send/receive という高レベル API を通じて安全に通信できる点が重要である。バッファなしチャネルの場合、送信側は受信側のスタックに直接値をコピーする最適化（direct send）が行われる。

### Q5: メッセージパッシングではデッドロックは発生しませんか？

**A**: メッセージパッシングでもデッドロックは発生しうる。例えば、二つのプロセスが互いの同期チャネルに送信しようとする場合（循環待ち）にデッドロックが起きる。Go ランタイムは全ゴルーチンがブロックされた状態を検出して `fatal error: all goroutines are asleep - deadlock!` を報告するが、一部のゴルーチンだけがデッドロックした場合は検出されない。防止策として、(1) 同期チャネルの代わりにバッファ付きチャネルを使う、(2) `select` + `default` でノンブロッキング送信を行う、(3) タイムアウトを設定する、などがある。

### Q6: アクターモデルと CSP の実用上の違いは何ですか？

**A**: 最大の違いは通信のアドレッシング方法である。CSP ではチャネルに名前をつけて通信するため、送信者は受信者を知らなくてもよい（チャネルを知っていればよい）。一方アクターモデルでは、相手のアクターアドレス（PID）を知っている必要がある。実用上の影響として、CSP は静的なデータフロー（パイプライン等）の構築に適し、アクターモデルは動的にトポロジが変化するシステム（チャットルーム、ゲームのプレイヤー管理等）に適する。また、Erlang のアクターモデルは監視ツリーによる障害回復が組み込まれている点が大きな利点である。

---

## 14. 発展的トピック

### 14.1 構造化並行性（Structured Concurrency）

近年注目されているのが「構造化並行性」の概念である。従来のゴルーチンやスレッドは「発射して忘れる（fire-and-forget）」方式であり、親の制御から離れてしまう問題がある。構造化並行性では、並行タスクが必ず親のスコープ内で完了することを保証する。

```
非構造化並行性（従来）:
  main() {
      go task1()    // どこかで動いている（いつ終わるか不明）
      go task2()    // リークの可能性
      // main が先に終了するとタスクが孤児化
  }

構造化並行性:
  main() {
      scope {
          task1()   // スコープ内で完了が保証される
          task2()   // 例外はスコープに伝播
      }             // 全タスクの完了を待つ
      // ここに到達した時点で全タスクが完了済み
  }
```

Go では `errgroup` パッケージが構造化並行性の一形態を提供する。Kotlin の `coroutineScope`、Swift の `TaskGroup`、Java の Project Loom の `StructuredTaskScope` が同様の概念を実装している。

### 14.2 セッション型（Session Types）

セッション型は、チャネル上の通信プロトコルを型レベルで表現する理論的手法である。例えば「整数を送信 → 文字列を受信 → 終了」というプロトコルを型として表現し、プロトコル違反をコンパイル時に検出する。研究段階の技術だが、Rust の `session-types` クレートなどで実験的に利用可能である。

### 14.3 リアクティブストリーム

リアクティブストリーム（Reactive Streams）は、非同期バックプレッシャーを伴うストリーム処理の仕様である。Java の `Flow API`、Reactor（Spring WebFlux）、RxJava、Akka Streams がこの仕様を実装している。メッセージパッシングの考え方をストリーム処理に適用し、消費者のペースに合わせて生産者が送信速度を調整する仕組みを提供する。

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## まとめ

| モデル | 通信方式 | 安全性 | スケーラビリティ | 障害隔離 | 代表言語 |
|-------|---------|--------|----------------|---------|---------|
| チャネル（CSP） | 型付き同期/非同期メッセージ | 高い | 単一プロセス内 | 限定的 | Go, Rust, Kotlin |
| アクター | 非同期メールボックス | 最も高い | 分散対応 | 監視ツリー | Erlang, Elixir, Akka |
| 共有メモリ | Mutex + 共有変数 | プログラマ依存 | 単一マシン内 | なし | C, C++, Java |
| リアクティブストリーム | バックプレッシャー付きストリーム | 高い | ストリーム処理に特化 | 限定的 | Java, Scala, Kotlin |

メッセージパッシングの核心は「データの所有権を明確にし、通信を通じて協調する」ことである。Go のチャネル、Rust の所有権付き mpsc、Erlang のアクターモデルは、それぞれ異なるアプローチでこの原則を実現している。どのモデルを選択するかは、アプリケーションの特性（レイテンシ要件、障害耐性、スケーラビリティ、チームの習熟度）に応じて判断すべきである。

---

## 次に読むべきガイド


---

## 参考文献

1. Hoare, C. A. R. "Communicating Sequential Processes." Prentice Hall, 1985. (CSP の原典。PDF が著者のWebサイトで無料公開されている)
2. Hewitt, C., Bishop, P., & Steiger, R. "A Universal Modular ACTOR Formalism for Artificial Intelligence." IJCAI, 1973. (アクターモデルの原論文)
3. Armstrong, J. "Making Reliable Distributed Systems in the Presence of Software Errors." PhD Thesis, Royal Institute of Technology, Stockholm, 2003. (Erlang/OTP の設計哲学を体系的に解説した Joe Armstrong の博士論文)
4. Donovan, A. & Kernighan, B. "The Go Programming Language." Addison-Wesley, Ch.8-9, 2015. (Go のゴルーチンとチャネルの公式参考書)
5. Klabnik, S. & Nichols, C. "The Rust Programming Language." No Starch Press, Ch.16, 2019. (Rust の並行性と所有権の公式ガイド)
6. Agha, G. "Actors: A Model of Concurrent Computation in Distributed Systems." MIT Press, 1986. (アクターモデルの理論的体系化)
7. Go Blog. "Share Memory By Communicating." https://go.dev/blog/codelab-share (Go 公式ブログの並行処理ガイド)
8. Erlang Documentation. "OTP Design Principles." https://www.erlang.org/doc/design_principles/ (Erlang/OTP の設計原則公式ドキュメント)
