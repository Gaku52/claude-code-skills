# 並行パターン -- Fan-out/Fan-in, Pipeline, Worker Pool

> Goの並行パターンはgoroutineとchannelを組み合わせ、Fan-out/Fan-in・Pipeline・Worker Pool・Contextにより実用的な並行処理を構築する。

---

## この章で学ぶこと

1. **Pipeline パターン** -- ステージをチャネルで連結する処理フロー
2. **Fan-out / Fan-in** -- 並列分散と結果集約
3. **Worker Pool** -- goroutine数を制限した並行処理

---

## 1. Pipeline パターン

### コード例 1: 基本的なPipeline

```go
func generate(nums ...int) <-chan int {
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

func main() {
    ch := generate(2, 3, 4)
    out := square(ch)
    for v := range out {
        fmt.Println(v) // 4, 9, 16
    }
}
```

### コード例 2: Fan-out / Fan-in

```go
func fanOut(in <-chan int, workers int) []<-chan int {
    channels := make([]<-chan int, workers)
    for i := 0; i < workers; i++ {
        channels[i] = square(in) // 同じ入力を複数workerで処理
    }
    return channels
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
```

### コード例 3: Worker Pool

```go
func workerPool(ctx context.Context, jobs <-chan Job, results chan<- Result, numWorkers int) {
    var wg sync.WaitGroup
    for i := 0; i < numWorkers; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            for job := range jobs {
                select {
                case <-ctx.Done():
                    return
                default:
                    result := process(job)
                    results <- result
                }
            }
        }(i)
    }
    go func() {
        wg.Wait()
        close(results)
    }()
}
```

### コード例 4: errgroup による並行処理

```go
import "golang.org/x/sync/errgroup"

func fetchAll(ctx context.Context, urls []string) ([]string, error) {
    g, ctx := errgroup.WithContext(ctx)
    results := make([]string, len(urls))

    for i, url := range urls {
        i, url := i, url
        g.Go(func() error {
            body, err := fetch(ctx, url)
            if err != nil {
                return err // 1つでもエラーなら全体キャンセル
            }
            results[i] = body
            return nil
        })
    }

    if err := g.Wait(); err != nil {
        return nil, err
    }
    return results, nil
}
```

### コード例 5: セマフォパターン

```go
func processWithLimit(items []Item, maxConcurrency int) {
    sem := make(chan struct{}, maxConcurrency)
    var wg sync.WaitGroup

    for _, item := range items {
        wg.Add(1)
        sem <- struct{}{} // セマフォ取得（満杯ならブロック）
        go func(it Item) {
            defer wg.Done()
            defer func() { <-sem }() // セマフォ解放
            process(it)
        }(item)
    }
    wg.Wait()
}
```

---

## 2. ASCII図解

### 図1: Pipeline パターン

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ generate │───>│ filter   │───>│ square   │───>│ consumer │
│  ch out  │    │ ch in/out│    │ ch in/out│    │  ch in   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘

データフロー:
  [1,2,3,4,5] → [2,4] → [4,16] → print
```

### 図2: Fan-out / Fan-in

```
                    Fan-out              Fan-in
               ┌──> Worker1 ──┐
               │              │
Input ─────────┼──> Worker2 ──┼──────> Output
               │              │
               └──> Worker3 ──┘

  Input ch ──┬──> [Worker 1] ──┐
             ├──> [Worker 2] ──┼──> Merged ch
             └──> [Worker 3] ──┘
```

### 図3: Worker Pool

```
┌─────────────────────────────────────────┐
│              Worker Pool                 │
│                                          │
│  Jobs Queue    Workers        Results    │
│  ┌─────────┐  ┌──────────┐  ┌────────┐ │
│  │ Job 1   │  │ Worker 1 │  │ Res 1  │ │
│  │ Job 2   │──>│ Worker 2 │──>│ Res 2 │ │
│  │ Job 3   │  │ Worker 3 │  │ Res 3  │ │
│  │ ...     │  └──────────┘  │ ...    │ │
│  └─────────┘   (固定数)      └────────┘ │
│                                          │
│  同時実行数 = Worker数 (制御可能)         │
└─────────────────────────────────────────┘
```

---

## 3. 比較表

### 表1: 並行パターンの比較

| パターン | 用途 | 複雑度 | goroutine数 |
|---------|------|-------|------------|
| Pipeline | 直列データ処理 | 低 | ステージ数分 |
| Fan-out/Fan-in | 並列分散処理 | 中 | ワーカー数分 |
| Worker Pool | 制限付き並列処理 | 中 | 固定(設定可能) |
| errgroup | エラー付き並列 | 低 | タスク数分 |
| セマフォ | 同時実行数制限 | 低 | タスク数分(制限付) |
| Pipeline + Cancel | キャンセル可能処理 | 高 | ステージ数分 |

### 表2: errgroup vs WaitGroup

| 項目 | errgroup | sync.WaitGroup |
|------|----------|---------------|
| エラー伝搬 | 最初のエラーを返す | なし |
| キャンセル | context連携で自動 | 手動 |
| 同時実行制限 | `SetLimit(n)` | 手動(セマフォ) |
| パッケージ | `golang.org/x/sync` | 標準`sync` |
| 戻り値 | `error` | なし |

---

## 4. アンチパターン

### アンチパターン 1: 無制限goroutine起動

```go
// BAD: リクエスト毎にgoroutineを無制限に起動
func handler(w http.ResponseWriter, r *http.Request) {
    for _, item := range getItems() { // 10万件
        go process(item) // goroutine爆発、OOM
    }
}

// GOOD: Worker Poolで制限
func handler(w http.ResponseWriter, r *http.Request) {
    items := getItems()
    sem := make(chan struct{}, 100) // 最大100並行
    var wg sync.WaitGroup
    for _, item := range items {
        wg.Add(1)
        sem <- struct{}{}
        go func(it Item) {
            defer wg.Done()
            defer func() { <-sem }()
            process(it)
        }(item)
    }
    wg.Wait()
}
```

### アンチパターン 2: channelの閉じ忘れ

```go
// BAD: チャネルを閉じない → 受信側がrange で永遠にブロック
func produce(ch chan<- int) {
    for i := 0; i < 10; i++ {
        ch <- i
    }
    // close(ch) が無い
}

// GOOD: deferで確実にcloseする
func produce(ch chan<- int) {
    defer close(ch)
    for i := 0; i < 10; i++ {
        ch <- i
    }
}
```

---

## 5. FAQ

### Q1: Worker Poolのワーカー数はどう決めるか？

CPUバウンド処理: `runtime.NumCPU()` を基準にする。I/Oバウンド処理: 外部リソースの許容並行数に合わせる（DB接続プール数、API rate limit等）。ベンチマークで最適値を探る。

### Q2: errgroupとWaitGroupのどちらを使うべきか？

エラーハンドリングが必要ならerrgroup、単純な完了待ちならWaitGroup。errgroupはContextとの連携も容易で、現代のGoコードでは多くの場合errgroupが推奨される。

### Q3: Pipelineパターンで遅いステージがあるとどうなるか？

最も遅いステージがボトルネックになる。対策は (1) 遅いステージをFan-outで並列化、(2) バッファ付きチャネルで一時的な速度差を吸収、(3) バッチ処理でスループット向上。

---

## まとめ

| 概念 | 要点 |
|------|------|
| Pipeline | チャネルでステージを連結。データの流れが明確 |
| Fan-out/Fan-in | 処理を分散し結果を集約 |
| Worker Pool | goroutine数を制限して安定運用 |
| errgroup | エラー付き並行処理の標準パターン |
| セマフォ | バッファ付きチャネルで同時実行数制限 |

---

## 次に読むべきガイド

- [03-context.md](./03-context.md) -- Contextによるキャンセル制御
- [../02-web/00-net-http.md](../02-web/00-net-http.md) -- HTTPサーバーでの並行処理
- [../03-tools/02-profiling.md](../03-tools/02-profiling.md) -- 並行処理のプロファイリング

---

## 参考文献

1. **Go Blog, "Go Concurrency Patterns: Pipelines and cancellation"** -- https://go.dev/blog/pipelines
2. **Go Blog, "Advanced Go Concurrency Patterns"** -- https://go.dev/blog/io2013-talk-concurrency
3. **golang.org/x/sync/errgroup** -- https://pkg.go.dev/golang.org/x/sync/errgroup
