# Goroutine と Channel -- Go並行プログラミングの基盤

> goroutineは軽量スレッド、channelは型安全な通信路であり、CSPモデルに基づくGoの並行処理の中核を成す。

---

## この章で学ぶこと

1. **goroutine** -- go文による軽量並行実行の仕組み
2. **channel** -- データの安全な受け渡しとselect文
3. **sync.WaitGroup** -- goroutineの完了待ち合わせ
4. **パターンと実践** -- 実際のアプリケーションにおけるgoroutine・channel活用法

---

## 1. goroutine の基本

### コード例 1: goroutine の起動

```go
func main() {
    go func() {
        fmt.Println("goroutine で実行")
    }()

    go sayHello("World")

    time.Sleep(100 * time.Millisecond) // 待機（実際はWaitGroupを使う）
}

func sayHello(name string) {
    fmt.Printf("Hello, %s!\n", name)
}
```

### goroutine の内部動作

goroutineはOSスレッドの上に多重化される軽量な実行単位である。Goランタイムは M:N スケジューリングモデルを採用し、M個のgoroutine を N個のOSスレッド上で実行する。

```go
// goroutine の特性を理解するデモ
package main

import (
    "fmt"
    "runtime"
    "sync"
)

func main() {
    // GOMAXPROCS はOSスレッド数を制御
    fmt.Printf("論理CPU数: %d\n", runtime.NumCPU())
    fmt.Printf("GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))

    var wg sync.WaitGroup
    const N = 100000

    // 10万のgoroutineを起動しても問題なく動作する
    for i := 0; i < N; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            // 各goroutineは約2KBのスタックから開始
            // 必要に応じて最大1GBまで拡張される
            _ = id
        }(i)
    }

    wg.Wait()
    fmt.Printf("全 %d goroutine 完了\n", N)
    fmt.Printf("現在のgoroutine数: %d\n", runtime.NumGoroutine())
}
```

### goroutine スケジューラの GMP モデル

```
G = Goroutine (実行単位)
M = Machine  (OSスレッド)
P = Processor (論理プロセッサ、GOMAXPROCS で設定)

                 Global Run Queue
                 [G10][G11][G12]...
                        │
        ┌───────────────┼───────────────┐
        │               │               │
   ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
   │   P0    │    │   P1    │    │   P2    │
   │ Local Q │    │ Local Q │    │ Local Q │
   │[G1][G2] │    │[G4][G5] │    │[G7][G8] │
   │         │    │         │    │         │
   │ current │    │ current │    │ current │
   │   G3    │    │   G6    │    │   G9    │
   └────┬────┘    └────┬────┘    └────┬────┘
        │              │              │
   ┌────▼────┐    ┌────▼────┐    ┌────▼────┐
   │   M0    │    │   M1    │    │   M2    │
   │ (Thread)│    │ (Thread)│    │ (Thread)│
   └─────────┘    └─────────┘    └─────────┘
        │              │              │
   ─────┴──────────────┴──────────────┴──── OS

ワークスティーリング:
  P0のLocal Queueが空 → P1のLocal Queueから半分を盗む
  → Global Queueからも取得する
```

### コード例: goroutineのスケジューリング観察

```go
package main

import (
    "fmt"
    "runtime"
    "time"
)

func main() {
    // GOMAXPROCS を 1 に設定して協調スケジューリングを観察
    runtime.GOMAXPROCS(1)

    go func() {
        for i := 0; i < 5; i++ {
            fmt.Printf("goroutine A: %d\n", i)
            // runtime.Gosched() を呼ばないと、
            // この goroutine が CPU を占有し続ける可能性がある
            runtime.Gosched() // 明示的にスケジューラに制御を渡す
        }
    }()

    go func() {
        for i := 0; i < 5; i++ {
            fmt.Printf("goroutine B: %d\n", i)
            runtime.Gosched()
        }
    }()

    time.Sleep(time.Second)
}
```

---

## 2. channel の基本

### コード例 2: バッファなしチャネル（同期チャネル）

```go
func main() {
    ch := make(chan int)    // バッファなしチャネル

    go func() {
        ch <- 42            // 送信（受信されるまでブロック）
    }()

    value := <-ch           // 受信（送信されるまでブロック）
    fmt.Println(value)      // 42
}
```

バッファなしチャネルの核心は「ランデブー（rendezvous）」セマンティクスにある。送信と受信が同時に行われることが保証される。これにより、2つのgoroutine間で確実に同期ポイントを作ることができる。

```go
// ランデブーの応用: goroutine間の確実な通知
func main() {
    done := make(chan struct{}) // struct{} はゼロバイトでメモリ効率が良い

    go func() {
        fmt.Println("処理実行中...")
        time.Sleep(time.Second)
        fmt.Println("処理完了")
        done <- struct{}{} // 完了通知
    }()

    <-done // 完了を待つ
    fmt.Println("メインgoroutineも終了")
}
```

### コード例 3: バッファ付きチャネル

```go
func main() {
    ch := make(chan string, 3) // バッファサイズ3

    ch <- "a" // ブロックしない（バッファに余裕あり）
    ch <- "b"
    ch <- "c"
    // ch <- "d" // ここでブロック（バッファ満杯）

    fmt.Println(<-ch) // "a" (FIFO)
    fmt.Println(<-ch) // "b"
}
```

### バッファサイズの設計指針

```go
// バッファサイズ 0: 同期通信（ランデブー）
// 送信者は受信者が準備できるまでブロック
done := make(chan struct{})

// バッファサイズ 1: シグナリング
// 通知を1つバッファリングできる
signal := make(chan os.Signal, 1)

// バッファサイズ N: パイプライン/ワークキュー
// 生産者と消費者の速度差を吸収
jobs := make(chan Job, 100)

// バッファサイズの決定基準:
//   0: 同期が必要な場合
//   1: 通知用途（signal.Notify等）
//   N: 生産者/消費者のスループット差を吸収
//   大きすぎるバッファはメモリを浪費し、
//   問題の発見を遅らせる可能性がある
```

### コード例: チャネルの方向制約

```go
// 送信専用チャネルと受信専用チャネルで型安全を確保
func producer(out chan<- int) {
    // chan<- int は送信専用。受信しようとするとコンパイルエラー
    for i := 0; i < 10; i++ {
        out <- i * i
    }
    close(out)
}

func consumer(in <-chan int) {
    // <-chan int は受信専用。送信しようとするとコンパイルエラー
    for v := range in {
        fmt.Println(v)
    }
}

func main() {
    ch := make(chan int, 5)

    go producer(ch)  // chan int は chan<- int に暗黙変換
    consumer(ch)     // chan int は <-chan int に暗黙変換
}
```

---

## 3. select文

### コード例 4: select文の基本

```go
func main() {
    ch1 := make(chan string)
    ch2 := make(chan string)

    go func() { time.Sleep(100 * time.Millisecond); ch1 <- "one" }()
    go func() { time.Sleep(200 * time.Millisecond); ch2 <- "two" }()

    for i := 0; i < 2; i++ {
        select {
        case msg := <-ch1:
            fmt.Println("ch1:", msg)
        case msg := <-ch2:
            fmt.Println("ch2:", msg)
        }
    }
}
```

### コード例: select の応用パターン

```go
// タイムアウト付きの操作
func fetchWithTimeout(url string, timeout time.Duration) ([]byte, error) {
    result := make(chan []byte, 1)
    errCh := make(chan error, 1)

    go func() {
        resp, err := http.Get(url)
        if err != nil {
            errCh <- err
            return
        }
        defer resp.Body.Close()
        body, err := io.ReadAll(resp.Body)
        if err != nil {
            errCh <- err
            return
        }
        result <- body
    }()

    select {
    case body := <-result:
        return body, nil
    case err := <-errCh:
        return nil, err
    case <-time.After(timeout):
        return nil, fmt.Errorf("fetch %s: timeout after %v", url, timeout)
    }
}

// ノンブロッキング操作（default付きselect）
func tryReceive(ch <-chan int) (int, bool) {
    select {
    case v := <-ch:
        return v, true
    default:
        return 0, false // チャネルにデータがなければ即座にリターン
    }
}

// 定期的なポーリングとシグナル受信の組み合わせ
func monitor(ctx context.Context, events <-chan Event) {
    ticker := time.NewTicker(5 * time.Second)
    defer ticker.Stop()

    for {
        select {
        case event := <-events:
            processEvent(event)
        case <-ticker.C:
            checkHealth()
        case <-ctx.Done():
            fmt.Println("モニタリング停止:", ctx.Err())
            return
        }
    }
}
```

### select のランダム選択と公平性

```go
// 複数のチャネルが同時に準備完了している場合、
// select はランダムに1つを選択する（公平性の保証）
func demonstrateFairness() {
    ch1 := make(chan string, 100)
    ch2 := make(chan string, 100)

    // 両方のチャネルにデータを投入
    for i := 0; i < 100; i++ {
        ch1 <- "A"
        ch2 <- "B"
    }

    countA, countB := 0, 0
    for i := 0; i < 200; i++ {
        select {
        case <-ch1:
            countA++
        case <-ch2:
            countB++
        }
    }

    // 結果は約100:100（ランダムなので厳密ではない）
    fmt.Printf("ch1: %d, ch2: %d\n", countA, countB)
}
```

---

## 4. WaitGroup と完了待ち

### コード例 5: WaitGroup の基本

```go
func main() {
    var wg sync.WaitGroup
    urls := []string{
        "https://example.com",
        "https://go.dev",
        "https://github.com",
    }

    for _, url := range urls {
        wg.Add(1)
        go func(u string) {
            defer wg.Done()
            resp, err := http.Get(u)
            if err != nil {
                log.Printf("error: %s: %v", u, err)
                return
            }
            defer resp.Body.Close()
            fmt.Printf("%s: %d\n", u, resp.StatusCode)
        }(url)
    }

    wg.Wait() // 全goroutineの完了を待つ
}
```

### WaitGroup の注意点

```go
// NG: goroutine 内で wg.Add を呼ぶ
// wg.Wait() が Add より先に実行される可能性がある
func badPattern() {
    var wg sync.WaitGroup
    for i := 0; i < 10; i++ {
        go func(id int) {
            wg.Add(1)   // ← NG: goroutine起動後にAddしている
            defer wg.Done()
            process(id)
        }(i)
    }
    wg.Wait() // Add前にWaitが実行される可能性
}

// OK: goroutine 起動前に wg.Add を呼ぶ
func goodPattern() {
    var wg sync.WaitGroup
    for i := 0; i < 10; i++ {
        wg.Add(1) // ← OK: goroutine起動前にAdd
        go func(id int) {
            defer wg.Done()
            process(id)
        }(i)
    }
    wg.Wait()
}
```

### コード例: WaitGroup + エラー収集

```go
// WaitGroup で並行処理しつつ、エラーも収集する
func fetchAllURLs(ctx context.Context, urls []string) ([]Result, []error) {
    var (
        wg      sync.WaitGroup
        mu      sync.Mutex
        results []Result
        errs    []error
    )

    for _, url := range urls {
        wg.Add(1)
        go func(u string) {
            defer wg.Done()

            result, err := fetchURL(ctx, u)

            mu.Lock()
            defer mu.Unlock()
            if err != nil {
                errs = append(errs, fmt.Errorf("fetch %s: %w", u, err))
            } else {
                results = append(results, result)
            }
        }(url)
    }

    wg.Wait()
    return results, errs
}
```

---

## 5. チャネルのクローズとrange

### コード例 6: チャネルのクローズとrange

```go
func generate(n int) <-chan int {
    ch := make(chan int)
    go func() {
        defer close(ch) // 送信完了後にクローズ
        for i := 0; i < n; i++ {
            ch <- i * i
        }
    }()
    return ch
}

func main() {
    for v := range generate(5) { // クローズされるまでループ
        fmt.Println(v) // 0, 1, 4, 9, 16
    }
}
```

### クローズの安全な扱い方

```go
// クローズ済みチャネルからの受信
func main() {
    ch := make(chan int, 3)
    ch <- 1
    ch <- 2
    close(ch)

    // 方法1: ok パターン
    v, ok := <-ch
    fmt.Println(v, ok)  // 1 true
    v, ok = <-ch
    fmt.Println(v, ok)  // 2 true
    v, ok = <-ch
    fmt.Println(v, ok)  // 0 false（クローズ済み、ゼロ値）

    // 方法2: range（推奨）
    ch2 := make(chan int, 3)
    ch2 <- 10
    ch2 <- 20
    close(ch2)
    for v := range ch2 {
        fmt.Println(v) // 10, 20
    }
}

// 複数の送信者がいる場合のクローズ
type FanIn struct {
    ch   chan int
    once sync.Once
}

func (f *FanIn) Close() {
    f.once.Do(func() {
        close(f.ch) // sync.Once で二重クローズを防止
    })
}
```

---

## 6. 実践的なgoroutine・channelパターン

### パターン1: Fan-out / Fan-in

```go
// Fan-out: 1つのチャネルから複数のgoroutineに分配
// Fan-in:  複数のチャネルを1つのチャネルにマージ

func fanOut(input <-chan int, workers int) []<-chan int {
    outputs := make([]<-chan int, workers)
    for i := 0; i < workers; i++ {
        outputs[i] = worker(input)
    }
    return outputs
}

func worker(input <-chan int) <-chan int {
    output := make(chan int)
    go func() {
        defer close(output)
        for v := range input {
            output <- heavyComputation(v)
        }
    }()
    return output
}

func fanIn(channels ...<-chan int) <-chan int {
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

func main() {
    // 入力チャネルの作成
    input := make(chan int, 100)
    go func() {
        defer close(input)
        for i := 0; i < 1000; i++ {
            input <- i
        }
    }()

    // Fan-out: 5つのワーカーに分配
    outputs := fanOut(input, 5)

    // Fan-in: 結果をマージ
    results := fanIn(outputs...)

    // 結果の消費
    for result := range results {
        fmt.Println(result)
    }
}
```

### パターン2: Pipeline

```go
// ステージを連結するパイプラインパターン
func generate(ctx context.Context, nums ...int) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for _, n := range nums {
            select {
            case out <- n:
            case <-ctx.Done():
                return
            }
        }
    }()
    return out
}

func square(ctx context.Context, in <-chan int) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for n := range in {
            select {
            case out <- n * n:
            case <-ctx.Done():
                return
            }
        }
    }()
    return out
}

func filter(ctx context.Context, in <-chan int, pred func(int) bool) <-chan int {
    out := make(chan int)
    go func() {
        defer close(out)
        for n := range in {
            if pred(n) {
                select {
                case out <- n:
                case <-ctx.Done():
                    return
                }
            }
        }
    }()
    return out
}

func main() {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    // パイプライン構築: 生成 → 二乗 → フィルタ
    nums := generate(ctx, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    squared := square(ctx, nums)
    even := filter(ctx, squared, func(n int) bool {
        return n%2 == 0
    })

    for result := range even {
        fmt.Println(result) // 4, 16, 36, 64, 100
    }
}
```

### パターン3: セマフォ（同時実行数制限）

```go
// バッファ付きチャネルをセマフォとして使用
type Semaphore struct {
    ch chan struct{}
}

func NewSemaphore(maxConcurrency int) *Semaphore {
    return &Semaphore{
        ch: make(chan struct{}, maxConcurrency),
    }
}

func (s *Semaphore) Acquire(ctx context.Context) error {
    select {
    case s.ch <- struct{}{}:
        return nil
    case <-ctx.Done():
        return ctx.Err()
    }
}

func (s *Semaphore) Release() {
    <-s.ch
}

// 使用例: 最大10並列でHTTPリクエスト
func fetchConcurrently(ctx context.Context, urls []string) []Result {
    sem := NewSemaphore(10) // 最大10同時実行
    var wg sync.WaitGroup
    results := make([]Result, len(urls))

    for i, url := range urls {
        wg.Add(1)
        go func(idx int, u string) {
            defer wg.Done()

            if err := sem.Acquire(ctx); err != nil {
                results[idx] = Result{Error: err}
                return
            }
            defer sem.Release()

            resp, err := http.Get(u)
            if err != nil {
                results[idx] = Result{Error: err}
                return
            }
            defer resp.Body.Close()
            body, _ := io.ReadAll(resp.Body)
            results[idx] = Result{Body: body}
        }(i, url)
    }

    wg.Wait()
    return results
}
```

### パターン4: errgroup による並行処理

```go
import "golang.org/x/sync/errgroup"

// errgroup は WaitGroup + エラー集約 + context連携
func processOrders(ctx context.Context, orders []Order) error {
    g, ctx := errgroup.WithContext(ctx)

    // 同時実行数制限 (Go 1.20+)
    g.SetLimit(5)

    for _, order := range orders {
        order := order // Go 1.21以前はローカルコピーが必要
        g.Go(func() error {
            // ctx が自動的に連携される
            // いずれかの goroutine がエラーを返すと、
            // ctx がキャンセルされ、他の goroutine も停止する
            return processOrder(ctx, order)
        })
    }

    return g.Wait() // 最初のエラーを返す
}

// errgroup で複数種類の処理を並行実行
func aggregateData(ctx context.Context, userID string) (*Dashboard, error) {
    g, ctx := errgroup.WithContext(ctx)

    var (
        profile  *Profile
        orders   []Order
        activity []Activity
    )

    g.Go(func() error {
        var err error
        profile, err = fetchProfile(ctx, userID)
        return err
    })

    g.Go(func() error {
        var err error
        orders, err = fetchOrders(ctx, userID)
        return err
    })

    g.Go(func() error {
        var err error
        activity, err = fetchActivity(ctx, userID)
        return err
    })

    if err := g.Wait(); err != nil {
        return nil, fmt.Errorf("aggregateData(%s): %w", userID, err)
    }

    return &Dashboard{
        Profile:  profile,
        Orders:   orders,
        Activity: activity,
    }, nil
}
```

### パターン5: Or-Done チャネル

```go
// 複数の処理のうち最初に完了したものの結果を使う
func firstResult(ctx context.Context, fns ...func(context.Context) (string, error)) (string, error) {
    ctx, cancel := context.WithCancel(ctx)
    defer cancel()

    type result struct {
        val string
        err error
    }

    ch := make(chan result, len(fns))

    for _, fn := range fns {
        fn := fn
        go func() {
            val, err := fn(ctx)
            ch <- result{val, err}
        }()
    }

    // 最初の成功結果を返す
    var lastErr error
    for i := 0; i < len(fns); i++ {
        r := <-ch
        if r.err == nil {
            cancel() // 他のgoroutineをキャンセル
            return r.val, nil
        }
        lastErr = r.err
    }

    return "", fmt.Errorf("all attempts failed, last error: %w", lastErr)
}

// 使用例: 最速のDNSサーバーから結果を取得
result, err := firstResult(ctx,
    func(ctx context.Context) (string, error) {
        return queryDNS(ctx, "8.8.8.8", domain)
    },
    func(ctx context.Context) (string, error) {
        return queryDNS(ctx, "1.1.1.1", domain)
    },
    func(ctx context.Context) (string, error) {
        return queryDNS(ctx, "9.9.9.9", domain)
    },
)
```

### パターン6: Ticker と定期処理

```go
// 定期的なバックグラウンド処理
func startPeriodicTask(ctx context.Context, interval time.Duration, task func(context.Context) error) {
    ticker := time.NewTicker(interval)
    defer ticker.Stop()

    // 起動直後にも1回実行する
    if err := task(ctx); err != nil {
        log.Printf("初回実行エラー: %v", err)
    }

    for {
        select {
        case <-ticker.C:
            if err := task(ctx); err != nil {
                log.Printf("定期処理エラー: %v", err)
            }
        case <-ctx.Done():
            log.Println("定期処理停止")
            return
        }
    }
}

// 使用例
func main() {
    ctx, cancel := context.WithCancel(context.Background())

    go startPeriodicTask(ctx, 30*time.Second, func(ctx context.Context) error {
        return cleanupExpiredSessions(ctx)
    })

    go startPeriodicTask(ctx, 5*time.Minute, func(ctx context.Context) error {
        return reportMetrics(ctx)
    })

    // SIGTERM で停止
    sigCh := make(chan os.Signal, 1)
    signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)
    <-sigCh
    cancel()
    time.Sleep(time.Second) // クリーンアップ待ち
}
```

---

## 7. goroutine リーク検出と防止

### goroutine リークの典型的なケース

```go
// ケース1: 受信されないチャネルへの送信
func leakySearch(query string) string {
    ch := make(chan string)
    go func() { ch <- searchAPI1(query) }()
    go func() { ch <- searchAPI2(query) }()
    return <-ch // 1つだけ受信。もう1つのgoroutineはリーク
}

// 修正: バッファ付きチャネル
func safeSearch(query string) string {
    ch := make(chan string, 2) // 全goroutineが送信可能
    go func() { ch <- searchAPI1(query) }()
    go func() { ch <- searchAPI2(query) }()
    return <-ch
}

// ケース2: 消費されないチャネルの生成
func leakyProducer() <-chan int {
    ch := make(chan int)
    go func() {
        i := 0
        for {
            ch <- i // 消費者がいなくなると永遠にブロック
            i++
        }
    }()
    return ch
}

// 修正: context でキャンセル
func safeProducer(ctx context.Context) <-chan int {
    ch := make(chan int)
    go func() {
        defer close(ch)
        i := 0
        for {
            select {
            case ch <- i:
                i++
            case <-ctx.Done():
                return // キャンセルで確実に終了
            }
        }
    }()
    return ch
}

// ケース3: ロックの待ち状態で永遠にブロック
func leakyLock() {
    var mu sync.Mutex
    mu.Lock()
    go func() {
        mu.Lock()   // デッドロック: 親goroutineがUnlockしない
        defer mu.Unlock()
        doWork()
    }()
    // mu.Unlock() を忘れている
}
```

### goroutine リークの検出方法

```go
// 方法1: runtime.NumGoroutine() でモニタリング
func monitorGoroutines(ctx context.Context) {
    ticker := time.NewTicker(10 * time.Second)
    defer ticker.Stop()

    baseline := runtime.NumGoroutine()

    for {
        select {
        case <-ticker.C:
            current := runtime.NumGoroutine()
            if current > baseline*2 {
                log.Printf("⚠️ goroutine リーク疑い: baseline=%d, current=%d",
                    baseline, current)
                // スタックトレースを出力
                buf := make([]byte, 1<<20)
                n := runtime.Stack(buf, true)
                log.Printf("Stack trace:\n%s", buf[:n])
            }
        case <-ctx.Done():
            return
        }
    }
}

// 方法2: テストでのリーク検出 (go.uber.org/goleak)
import "go.uber.org/goleak"

func TestMain(m *testing.M) {
    goleak.VerifyTestMain(m)
}

func TestNoLeak(t *testing.T) {
    defer goleak.VerifyNone(t)

    ctx, cancel := context.WithCancel(context.Background())
    ch := safeProducer(ctx)

    // いくつか消費
    for i := 0; i < 10; i++ {
        <-ch
    }

    cancel() // goroutineの停止を保証
}
```

---

## 8. ASCII図解

### 図1: goroutine と OS スレッドの関係 (M:N スケジューリング)

```
┌─────────────── Go ランタイム ───────────────┐
│                                              │
│  G = goroutine    M = OS Thread   P = Proc   │
│                                              │
│  ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐             │
│  │G1 │ │G2 │ │G3 │ │G4 │ │G5 │  Run Queue  │
│  └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘ └─┬─┘             │
│    │      │      │      │      │              │
│  ┌─▼──────▼─┐  ┌─▼──────▼─┐                  │
│  │   P1     │  │   P2     │   Processor      │
│  └────┬─────┘  └────┬─────┘                  │
│       │              │                        │
│  ┌────▼─────┐  ┌────▼─────┐                  │
│  │   M1     │  │   M2     │   OS Thread      │
│  └──────────┘  └──────────┘                  │
└──────────────────────────────────────────────┘
```

### 図2: バッファなし vs バッファ付きチャネル

```
バッファなし (make(chan int)):
  送信側 ──[同期]──> 受信側
  送信は受信側が準備できるまでブロック

  Sender    Channel    Receiver
    │                    │
    ├──── send ────>     │ (ブロック)
    │              ├──── recv
    │              │
    │    (同期完了)  │

バッファ付き (make(chan int, 3)):
  送信側 ──[buf]──[buf]──[buf]──> 受信側
  バッファが満杯のときだけブロック

  Sender    Buffer[3]    Receiver
    │      [_][_][_]       │
    ├─send─>[A][_][_]      │
    ├─send─>[A][B][_]      │
    │       [A][B][_]──recv─┤  → A
    │       [_][B][_]       │
```

### 図3: select文の動作

```
select {
case msg := <-ch1:     ┐
case msg := <-ch2:     ├── 準備できたケースを実行
case ch3 <- value:     │   複数準備完了ならランダム選択
default:               ┘   defaultはブロック回避用
}

       ┌─ ch1 ready? ──YES──> case 1 実行
       │
select─┼─ ch2 ready? ──YES──> case 2 実行
       │
       ├─ ch3 ready? ──YES──> case 3 実行
       │
       └─ none ready ──────> default実行
                              (defaultなければブロック)
```

### 図4: Fan-out / Fan-in パターン

```
Fan-out (1つのソースから複数のワーカーに分配):

            ┌──> Worker1 ──┐
  Source ───┼──> Worker2 ──┼──> Merged Output
            ├──> Worker3 ──┤
            └──> Worker4 ──┘

Pipeline (ステージの連鎖):

  Generate ──> Transform ──> Filter ──> Consume
    (chan)       (chan)        (chan)

Or-Done (最初の結果を採用):

  Query DNS A ──┐
  Query DNS B ──┼──> First Result
  Query DNS C ──┘
```

### 図5: goroutine ライフサイクル

```
┌─────────────────────────────────────────────┐
│              goroutine ライフサイクル          │
│                                             │
│  ┌──────────┐    ┌──────────┐               │
│  │ Runnable │───>│ Running  │               │
│  │ (待機中) │    │ (実行中) │               │
│  └──────────┘    └────┬─────┘               │
│       ▲               │                     │
│       │               ├──> I/O待ち           │
│       │               │    → Waiting状態     │
│       │               │    → I/O完了でRunnable│
│       │               │                     │
│       │               ├──> channel待ち       │
│       │               │    → Waiting状態     │
│       │               │    → 送受信でRunnable │
│       │               │                     │
│       │               ├──> preemption        │
│       │               │    → Runnable に戻る  │
│       │               │                     │
│       └───────────────┘                     │
│                       │                     │
│                       ▼                     │
│               ┌──────────┐                  │
│               │   Dead   │                  │
│               │ (終了)   │                  │
│               └──────────┘                  │
└─────────────────────────────────────────────┘
```

---

## 9. 比較表

### 表1: チャネルの種類

| 種類 | 構文 | 送信ブロック条件 | 受信ブロック条件 | 用途 |
|------|------|---------------|---------------|------|
| バッファなし | `make(chan T)` | 受信側がいない | 送信側がいない | 同期的通信 |
| バッファ付き | `make(chan T, n)` | バッファ満杯 | バッファ空 | 非同期的通信 |
| 送信専用 | `chan<- T` | 同上 | コンパイルエラー | APIの制約 |
| 受信専用 | `<-chan T` | コンパイルエラー | 同上 | APIの制約 |

### 表2: goroutine vs OS Thread vs async/await

| 項目 | goroutine | OS Thread | async/await (JS) |
|------|-----------|-----------|-----------------|
| 初期スタックサイズ | ~2KB | ~1MB | N/A |
| 切替コスト | 低い (ユーザー空間) | 高い (カーネル) | 低い (イベントループ) |
| 同時数目安 | 数十万〜数百万 | 数千 | 数万 |
| スケジューリング | Go ランタイム (M:N) | OS | イベントループ |
| メモリモデル | happens-before | OS依存 | シングルスレッド |
| ブロッキングI/O | 自動的にスレッド追加 | スレッド占有 | 不可（別の仕組み必要） |

### 表3: 並行パターンの選択指針

| パターン | 用途 | 複雑度 | goroutine数 |
|---------|------|--------|------------|
| WaitGroup | 全完了を待つ | 低 | 既知 |
| errgroup | エラー付き並行処理 | 低 | 既知 |
| Fan-out/Fan-in | 並列処理と結果集約 | 中 | ワーカー数 |
| Pipeline | ステージ処理 | 中 | ステージ数 |
| Semaphore | 同時実行数制限 | 低 | 制限付き |
| Or-Done | 最速結果の採用 | 中 | 候補数 |
| Worker Pool | 継続的なジョブ処理 | 中 | 固定 |

### 表4: チャネル操作の結果一覧

| 操作 | nilチャネル | クローズ済み | バッファ空/満杯 | 正常 |
|------|-----------|------------|--------------|------|
| 送信 `ch <-` | 永久ブロック | **panic** | ブロック(満杯) | 送信 |
| 受信 `<-ch` | 永久ブロック | ゼロ値, false | ブロック(空) | 受信 |
| close | **panic** | **panic** | 残りデータ受信可 | クローズ |
| len | 0 | バッファ内の数 | バッファ内の数 | バッファ内の数 |
| cap | 0 | バッファサイズ | バッファサイズ | バッファサイズ |

---

## 10. アンチパターン

### アンチパターン 1: goroutineリーク

```go
// BAD: チャネルから受信されず goroutine が永遠にブロック
func leakySearch(query string) string {
    ch := make(chan string)
    go func() { ch <- searchAPI1(query) }()
    go func() { ch <- searchAPI2(query) }()
    return <-ch // 1つだけ受信。もう1つの goroutine はリーク
}

// GOOD: context でキャンセル、またはバッファ付きチャネル
func safeSearch(ctx context.Context, query string) string {
    ch := make(chan string, 2) // バッファで全goroutineが送信可能
    go func() { ch <- searchAPI1(query) }()
    go func() { ch <- searchAPI2(query) }()
    return <-ch
}
```

### アンチパターン 2: ループ変数のキャプチャ (Go 1.21以前)

```go
// BAD (Go 1.21以前): ループ変数が共有される
for _, url := range urls {
    go func() {
        fetch(url) // 全goroutineが最後のurlを参照
    }()
}

// GOOD: 引数で渡す（Go 1.22以降は不要）
for _, url := range urls {
    url := url // ローカルコピー
    go func() {
        fetch(url)
    }()
}
```

### アンチパターン 3: time.Sleep での同期

```go
// BAD: time.Sleep で goroutine の完了を待つ
func badSync() {
    go processData()
    time.Sleep(5 * time.Second) // 5秒で終わるとは限らない
}

// GOOD: WaitGroup または channel で同期
func goodSync() {
    done := make(chan struct{})
    go func() {
        defer close(done)
        processData()
    }()
    <-done // 確実に完了を待つ
}
```

### アンチパターン 4: チャネルの過剰使用

```go
// BAD: 単純なカウンタにチャネルを使う
type Counter struct {
    ch chan int
}
func (c *Counter) Inc() {
    c.ch <- 1
}
func (c *Counter) Value() int {
    return <-c.ch
}

// GOOD: 単純なカウンタには atomic を使う
type Counter struct {
    count atomic.Int64
}
func (c *Counter) Inc() {
    c.count.Add(1)
}
func (c *Counter) Value() int64 {
    return c.count.Load()
}
```

---

## 11. FAQ

### Q1: goroutineはいくつまで起動できるか？

理論上は数百万。各goroutineの初期スタックは約2KBで、必要に応じて動的に拡張される。ただし、CPUバウンドの処理では`runtime.GOMAXPROCS`（デフォルト=CPU数）以上の並列度は出ない。I/Oバウンドなら大量に起動する意味がある。100万goroutineでも約2GBのメモリで起動可能だが、スケジューリングオーバーヘッドは増加する。

### Q2: チャネルとMutexのどちらを使うべきか？

原則: 「データの所有権を移転するならチャネル、共有状態を保護するならMutex」。Go Proverbsでは「Don't communicate by sharing memory; share memory by communicating」とされるが、単純なカウンタやキャッシュにはMutexが適切。パイプラインやイベント通知にはチャネルが自然。パフォーマンスが重要な場合、チャネルはMutexより遅い（内部でもMutexを使っている）ことに注意。

### Q3: クローズされたチャネルに送信するとどうなるか？

panicが発生する。チャネルのクローズは「送信側」が行い、「受信側」はrangeやok判定で検知する。複数の送信者がいる場合はsync.Onceでクローズを保護する。受信側からチャネルをクローズすべきではない。

### Q4: GOMAXPROCS を変更すべきか？

通常は不要。デフォルトでCPUコア数に設定され、ほとんどのワークロードで最適。ただし、コンテナ環境（Docker/Kubernetes）ではホストのCPU数が見えてしまうことがある。`uber-go/automaxprocs` パッケージを使うと、コンテナに割り当てられたCPU数に自動調整される。

### Q5: goroutine を外部から停止する方法は？

Go には goroutine を外部から強制的に停止する方法はない。代わりに `context.Context` のキャンセルシグナルを協調的にチェックする設計パターンを使う。goroutine 内部で定期的に `ctx.Done()` チャネルを確認し、キャンセルされたらリターンする。これは意図的な設計決定であり、リソースの安全なクリーンアップを保証する。

### Q6: nilチャネルは何に使うか？

nilチャネルへの送受信は永久にブロックする。これは select文で特定のcaseを動的に無効化するのに使える。例えば、あるチャネルからの受信を一時的に停止したい場合、そのチャネル変数をnilに設定すると、selectでそのcaseは選択されなくなる。

```go
func dynamicSelect(ch1, ch2 <-chan int) {
    for ch1 != nil || ch2 != nil {
        select {
        case v, ok := <-ch1:
            if !ok {
                ch1 = nil // ch1 を無効化
                continue
            }
            fmt.Println("ch1:", v)
        case v, ok := <-ch2:
            if !ok {
                ch2 = nil // ch2 を無効化
                continue
            }
            fmt.Println("ch2:", v)
        }
    }
}
```

### Q7: for-range over channel と for-select のどちらを使うべきか？

`for v := range ch` は1つのチャネルからの受信に最適で、チャネルがクローズされると自動的にループが終了する。一方、`for-select` は複数のチャネルの待ち合わせや、context のキャンセル検知を同時に行う場合に使う。単一チャネルでも context キャンセルが必要なら for-select を選ぶ。

---

## まとめ

| 概念 | 要点 |
|------|------|
| goroutine | `go f()` で起動。軽量（~2KB）、M:Nスケジューリング |
| GMP モデル | Goroutine-Machine-Processor の3層構造 |
| channel | 型安全な通信路。バッファ有/無の2種類 |
| 方向制約 | `chan<-` 送信専用、`<-chan` 受信専用 |
| select | 複数チャネルの待ち合わせ。非決定的選択 |
| WaitGroup | goroutine群の完了待ち |
| errgroup | WaitGroup + エラー集約 + context連携 |
| close | チャネルの閉鎖。送信側が行う |
| range over channel | クローズまでループ受信 |
| Fan-out/Fan-in | 並列分散と結果集約 |
| Pipeline | ステージ連鎖の処理フロー |
| Semaphore | チャネルで同時実行数制限 |

---

## 次に読むべきガイド

- [01-sync-primitives.md](./01-sync-primitives.md) -- Mutex/atomic等の同期プリミティブ
- [02-concurrency-patterns.md](./02-concurrency-patterns.md) -- Fan-out/Fan-in等の並行パターン
- [03-context.md](./03-context.md) -- Context によるキャンセル制御

---

## 参考文献

1. **Go Blog, "Share Memory By Communicating"** -- https://go.dev/blog/codelab-share
2. **Go Blog, "Go Concurrency Patterns"** -- https://go.dev/blog/concurrency-patterns
3. **Go Blog, "Advanced Go Concurrency Patterns"** -- https://go.dev/blog/io2013-talk-concurrency
4. **Go Blog, "Go Concurrency Patterns: Pipelines and cancellation"** -- https://go.dev/blog/pipelines
5. **Hoare, C.A.R. (1978) "Communicating Sequential Processes"** -- https://www.cs.cmu.edu/~crary/819-f09/Hoare78.pdf
6. **golang.org/x/sync/errgroup** -- https://pkg.go.dev/golang.org/x/sync/errgroup
7. **uber-go/goleak** -- https://github.com/uber-go/goleak
