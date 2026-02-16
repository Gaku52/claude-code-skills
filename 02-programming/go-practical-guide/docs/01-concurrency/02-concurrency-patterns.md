# 並行パターン -- Fan-out/Fan-in, Pipeline, Worker Pool

> Goの並行パターンはgoroutineとchannelを組み合わせ、Fan-out/Fan-in・Pipeline・Worker Pool・Contextにより実用的な並行処理を構築する。

---

## この章で学ぶこと

1. **Pipeline パターン** -- ステージをチャネルで連結する処理フロー
2. **Fan-out / Fan-in** -- 並列分散と結果集約
3. **Worker Pool** -- goroutine数を制限した並行処理
4. **errgroup / セマフォ** -- エラーハンドリング付き並行制御
5. **Or-Done / Tee / Bridge** -- 高度なチャネル合成パターン
6. **Rate Limiter** -- スループット制御パターン
7. **実践的なユースケース** -- 本番環境での適用事例

---

## 1. Pipeline パターン

Pipelineパターンは、データ処理を複数のステージに分割し、各ステージをチャネルで連結する設計パターンである。各ステージは独立したgoroutineで動作し、入力チャネルからデータを受け取り、処理結果を出力チャネルに送信する。

### 1.1 Pipelineの設計原則

Pipelineを設計する際の基本原則は以下の通りである。

- **単一責任**: 各ステージは1つの処理だけを担当する
- **チャネル所有権**: チャネルを作成したgoroutineがcloseの責任を持つ
- **バッファリング**: ステージ間の速度差はバッファ付きチャネルで吸収する
- **キャンセル対応**: すべてのステージがcontextによるキャンセルに対応する

### コード例 1: 基本的なPipeline

```go
package main

import (
	"context"
	"fmt"
)

// generate はスライスの要素を順番にチャネルに送信するジェネレータステージ
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

// square は入力値を二乗して出力するステージ
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

// filter は条件に合う値だけを通過させるステージ
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
	// Pipeline: generate → filter(偶数) → square → 出力
	ch := generate(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	even := filter(ch, func(n int) bool { return n%2 == 0 })
	out := square(even)

	for v := range out {
		fmt.Println(v) // 4, 16, 36, 64, 100
	}
}
```

### コード例 2: Context対応Pipeline

本番環境のPipelineでは、キャンセル対応が必須である。すべてのステージがcontextを受け取り、キャンセルシグナルに応答する設計にする。

```go
package main

import (
	"context"
	"fmt"
	"time"
)

// generateWithCtx はcontext対応のジェネレータ
func generateWithCtx(ctx context.Context, nums ...int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for _, n := range nums {
			select {
			case <-ctx.Done():
				return // キャンセル時は即座に終了
			case out <- n:
			}
		}
	}()
	return out
}

// squareWithCtx はcontext対応の二乗ステージ
func squareWithCtx(ctx context.Context, in <-chan int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for {
			select {
			case <-ctx.Done():
				return
			case n, ok := <-in:
				if !ok {
					return
				}
				select {
				case <-ctx.Done():
					return
				case out <- n * n:
				}
			}
		}
	}()
	return out
}

// accumulate は入力値を累積加算するステージ
func accumulate(ctx context.Context, in <-chan int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		sum := 0
		for {
			select {
			case <-ctx.Done():
				return
			case n, ok := <-in:
				if !ok {
					// 最終結果を送信
					select {
					case <-ctx.Done():
					case out <- sum:
					}
					return
				}
				sum += n
			}
		}
	}()
	return out
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	ch := generateWithCtx(ctx, 1, 2, 3, 4, 5)
	squared := squareWithCtx(ctx, ch)
	result := accumulate(ctx, squared)

	for v := range result {
		fmt.Println("Sum of squares:", v) // 55
	}
}
```

### コード例 3: バッチ処理Pipeline

大量データを処理する場合、個別処理ではなくバッチ処理でスループットを向上させる。

```go
package main

import (
	"context"
	"fmt"
)

// batch は入力チャネルの値をバッチサイズごとにまとめるステージ
func batch(ctx context.Context, in <-chan int, size int) <-chan []int {
	out := make(chan []int)
	go func() {
		defer close(out)
		buf := make([]int, 0, size)
		for {
			select {
			case <-ctx.Done():
				return
			case v, ok := <-in:
				if !ok {
					// 残りのバッファを送信
					if len(buf) > 0 {
						select {
						case <-ctx.Done():
						case out <- buf:
						}
					}
					return
				}
				buf = append(buf, v)
				if len(buf) >= size {
					// バッファのコピーを送信（元のスライスは再利用）
					batch := make([]int, len(buf))
					copy(batch, buf)
					select {
					case <-ctx.Done():
						return
					case out <- batch:
					}
					buf = buf[:0]
				}
			}
		}
	}()
	return out
}

// processBatch はバッチ単位で処理を行うステージ
func processBatch(ctx context.Context, in <-chan []int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for {
			select {
			case <-ctx.Done():
				return
			case items, ok := <-in:
				if !ok {
					return
				}
				// バッチ内の合計を計算
				sum := 0
				for _, v := range items {
					sum += v
				}
				select {
				case <-ctx.Done():
					return
				case out <- sum:
				}
			}
		}
	}()
	return out
}

func main() {
	ctx := context.Background()

	// 1〜100の数値を生成
	gen := func() <-chan int {
		ch := make(chan int)
		go func() {
			defer close(ch)
			for i := 1; i <= 100; i++ {
				ch <- i
			}
		}()
		return ch
	}

	// Pipeline: 生成 → バッチ化(10個ずつ) → バッチ処理
	numbers := gen()
	batched := batch(ctx, numbers, 10)
	results := processBatch(ctx, batched)

	for sum := range results {
		fmt.Println("Batch sum:", sum)
	}
}
```

---

## 2. Fan-out / Fan-in

Fan-outは1つの入力チャネルから複数のワーカーに処理を分散させるパターンである。Fan-inは複数のチャネルからの出力を1つのチャネルに統合するパターンである。これらを組み合わせることで、CPUバウンドやI/Oバウンドな処理を効率的に並列化できる。

### 2.1 Fan-out / Fan-inの適用基準

Fan-out/Fan-inが有効なケースは以下の通りである。

- 各処理が独立しており、順序が重要でない場合
- I/Oバウンドな処理（API呼び出し、DB問い合わせ等）を並列化したい場合
- CPUバウンドな処理をマルチコアで分散したい場合

### コード例 4: Fan-out / Fan-in 基本実装

```go
package main

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// fanOut は入力チャネルから複数のワーカーに処理を分散する
func fanOut(ctx context.Context, in <-chan int, workers int) []<-chan int {
	channels := make([]<-chan int, workers)
	for i := 0; i < workers; i++ {
		channels[i] = worker(ctx, in, i)
	}
	return channels
}

// worker は個別のワーカーgoroutine
func worker(ctx context.Context, in <-chan int, id int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for {
			select {
			case <-ctx.Done():
				return
			case n, ok := <-in:
				if !ok {
					return
				}
				// 重い処理をシミュレート
				time.Sleep(time.Duration(rand.Intn(100)) * time.Millisecond)
				result := n * n
				select {
				case <-ctx.Done():
					return
				case out <- result:
				}
			}
		}
	}()
	return out
}

// fanIn は複数のチャネルを1つに統合する
func fanIn(ctx context.Context, channels ...<-chan int) <-chan int {
	var wg sync.WaitGroup
	merged := make(chan int)

	// 各入力チャネルからmergedチャネルに転送するgoroutineを起動
	for _, ch := range channels {
		wg.Add(1)
		go func(c <-chan int) {
			defer wg.Done()
			for {
				select {
				case <-ctx.Done():
					return
				case v, ok := <-c:
					if !ok {
						return
					}
					select {
					case <-ctx.Done():
						return
					case merged <- v:
					}
				}
			}
		}(ch)
	}

	// 全ワーカー完了後にmergedチャネルを閉じる
	go func() {
		wg.Wait()
		close(merged)
	}()

	return merged
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// 入力データ生成
	input := make(chan int)
	go func() {
		defer close(input)
		for i := 1; i <= 20; i++ {
			select {
			case <-ctx.Done():
				return
			case input <- i:
			}
		}
	}()

	// Fan-out: 4つのワーカーに分散
	workers := fanOut(ctx, input, 4)

	// Fan-in: 結果を統合
	results := fanIn(ctx, workers...)

	// 結果を収集
	for result := range results {
		fmt.Println("Result:", result)
	}
}
```

### コード例 5: 順序を保持するFan-out / Fan-in

通常のFan-out/Fan-inでは結果の順序が保証されない。順序を保持する必要がある場合は、インデックスを付与する。

```go
package main

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"
)

// IndexedItem はインデックス付きデータ
type IndexedItem struct {
	Index  int
	Value  int
	Result int
}

// orderedFanOut は順序情報を保持するFan-out
func orderedFanOut(ctx context.Context, items []int, workers int) <-chan IndexedItem {
	input := make(chan IndexedItem, len(items))
	go func() {
		defer close(input)
		for i, v := range items {
			select {
			case <-ctx.Done():
				return
			case input <- IndexedItem{Index: i, Value: v}:
			}
		}
	}()

	// 各ワーカーの出力チャネル
	outputs := make([]<-chan IndexedItem, workers)
	for w := 0; w < workers; w++ {
		out := make(chan IndexedItem)
		outputs[w] = out
		go func() {
			defer close(out)
			for item := range input {
				select {
				case <-ctx.Done():
					return
				default:
					// 重い処理をシミュレート
					time.Sleep(10 * time.Millisecond)
					item.Result = item.Value * item.Value
					select {
					case <-ctx.Done():
						return
					case out <- item:
					}
				}
			}
		}()
	}

	// Fan-in
	merged := make(chan IndexedItem)
	var wg sync.WaitGroup
	for _, ch := range outputs {
		wg.Add(1)
		go func(c <-chan IndexedItem) {
			defer wg.Done()
			for item := range c {
				select {
				case <-ctx.Done():
					return
				case merged <- item:
				}
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
	ctx := context.Background()
	items := []int{10, 20, 30, 40, 50, 60, 70, 80, 90, 100}

	results := make([]IndexedItem, 0, len(items))
	for item := range orderedFanOut(ctx, items, 4) {
		results = append(results, item)
	}

	// インデックスでソートして元の順序を復元
	sort.Slice(results, func(i, j int) bool {
		return results[i].Index < results[j].Index
	})

	for _, r := range results {
		fmt.Printf("Index=%d, Value=%d, Result=%d\n", r.Index, r.Value, r.Result)
	}
}
```

---

## 3. Worker Pool

Worker Poolパターンは、固定数のgoroutine(ワーカー)を事前に起動し、ジョブキューからタスクを取得して処理するパターンである。goroutineの無制限起動を防ぎ、リソースの使用量を制御できる。

### 3.1 Worker Poolの設計指針

- **ワーカー数**: CPUバウンドは`runtime.NumCPU()`、I/Oバウンドは外部リソースの制約に合わせる
- **ジョブキュー**: バッファ付きチャネルでバックプレッシャーを制御
- **結果チャネル**: ワーカーの処理結果を集約
- **エラーハンドリング**: ジョブ単位でエラーを扱う

### コード例 6: 汎用Worker Pool

```go
package main

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

// Job は処理すべきタスクを表す
type Job struct {
	ID      int
	Payload string
}

// Result は処理結果を表す
type Result struct {
	JobID    int
	Output   string
	Duration time.Duration
	Err      error
}

// WorkerPool は固定数のワーカーでジョブを処理するプール
type WorkerPool struct {
	numWorkers int
	jobs       chan Job
	results    chan Result
	wg         sync.WaitGroup
}

// NewWorkerPool は新しいWorkerPoolを生成する
func NewWorkerPool(numWorkers, jobBufferSize int) *WorkerPool {
	return &WorkerPool{
		numWorkers: numWorkers,
		jobs:       make(chan Job, jobBufferSize),
		results:    make(chan Result, jobBufferSize),
	}
}

// Start はワーカーを起動する
func (wp *WorkerPool) Start(ctx context.Context) {
	for i := 0; i < wp.numWorkers; i++ {
		wp.wg.Add(1)
		go wp.runWorker(ctx, i)
	}

	// 全ワーカー終了後にresultsチャネルを閉じる
	go func() {
		wp.wg.Wait()
		close(wp.results)
	}()
}

// runWorker は個別ワーカーの処理ループ
func (wp *WorkerPool) runWorker(ctx context.Context, id int) {
	defer wp.wg.Done()
	for {
		select {
		case <-ctx.Done():
			log.Printf("Worker %d: shutting down (context cancelled)", id)
			return
		case job, ok := <-wp.jobs:
			if !ok {
				log.Printf("Worker %d: no more jobs", id)
				return
			}
			start := time.Now()
			output, err := processJob(ctx, job)
			wp.results <- Result{
				JobID:    job.ID,
				Output:   output,
				Duration: time.Since(start),
				Err:      err,
			}
		}
	}
}

// Submit はジョブをキューに追加する
func (wp *WorkerPool) Submit(job Job) {
	wp.jobs <- job
}

// Close はジョブキューを閉じる（新しいジョブの受け付けを停止）
func (wp *WorkerPool) Close() {
	close(wp.jobs)
}

// Results は結果チャネルを返す
func (wp *WorkerPool) Results() <-chan Result {
	return wp.results
}

// processJob は個別ジョブの処理ロジック
func processJob(ctx context.Context, job Job) (string, error) {
	// 処理シミュレート
	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case <-time.After(50 * time.Millisecond):
		return fmt.Sprintf("Processed: %s", job.Payload), nil
	}
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	pool := NewWorkerPool(5, 100)
	pool.Start(ctx)

	// ジョブを投入
	go func() {
		for i := 0; i < 50; i++ {
			pool.Submit(Job{ID: i, Payload: fmt.Sprintf("task-%d", i)})
		}
		pool.Close()
	}()

	// 結果を収集
	successCount := 0
	errorCount := 0
	for result := range pool.Results() {
		if result.Err != nil {
			errorCount++
			log.Printf("Job %d failed: %v", result.JobID, result.Err)
		} else {
			successCount++
		}
	}

	fmt.Printf("Completed: %d success, %d errors\n", successCount, errorCount)
}
```

### コード例 7: 動的スケーリングWorker Pool

負荷に応じてワーカー数を動的に調整するWorker Pool。

```go
package main

import (
	"context"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"
)

// DynamicPool は動的にスケーリングするWorkerPool
type DynamicPool struct {
	minWorkers  int
	maxWorkers  int
	activeCount int64
	jobs        chan func()
	wg          sync.WaitGroup
	mu          sync.Mutex
	workerCount int
}

// NewDynamicPool は動的プールを生成する
func NewDynamicPool(minWorkers, maxWorkers, queueSize int) *DynamicPool {
	dp := &DynamicPool{
		minWorkers: minWorkers,
		maxWorkers: maxWorkers,
		jobs:       make(chan func(), queueSize),
	}
	return dp
}

// Start は最小ワーカー数でプールを起動する
func (dp *DynamicPool) Start(ctx context.Context) {
	for i := 0; i < dp.minWorkers; i++ {
		dp.addWorker(ctx)
	}

	// 負荷監視goroutine
	go dp.monitor(ctx)
}

// addWorker はワーカーを1つ追加する
func (dp *DynamicPool) addWorker(ctx context.Context) {
	dp.mu.Lock()
	dp.workerCount++
	dp.mu.Unlock()

	dp.wg.Add(1)
	go func() {
		defer dp.wg.Done()
		defer func() {
			dp.mu.Lock()
			dp.workerCount--
			dp.mu.Unlock()
		}()

		for {
			select {
			case <-ctx.Done():
				return
			case job, ok := <-dp.jobs:
				if !ok {
					return
				}
				atomic.AddInt64(&dp.activeCount, 1)
				job()
				atomic.AddInt64(&dp.activeCount, -1)
			}
		}
	}()
}

// monitor は負荷を監視してワーカー数を調整する
func (dp *DynamicPool) monitor(ctx context.Context) {
	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			queueLen := len(dp.jobs)
			active := atomic.LoadInt64(&dp.activeCount)
			dp.mu.Lock()
			current := dp.workerCount
			dp.mu.Unlock()

			// キューが溢れそうならワーカーを追加
			if queueLen > current && current < dp.maxWorkers {
				toAdd := min(dp.maxWorkers-current, queueLen-current)
				for i := 0; i < toAdd; i++ {
					dp.addWorker(ctx)
				}
				log.Printf("Scaled up: %d -> %d workers (queue=%d, active=%d)",
					current, current+toAdd, queueLen, active)
			}
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Submit はジョブを投入する
func (dp *DynamicPool) Submit(job func()) {
	dp.jobs <- job
}

// Shutdown はプールをシャットダウンする
func (dp *DynamicPool) Shutdown() {
	close(dp.jobs)
	dp.wg.Wait()
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	pool := NewDynamicPool(2, 10, 100)
	pool.Start(ctx)

	var completed int64

	// 大量のジョブを一気に投入
	for i := 0; i < 200; i++ {
		i := i
		pool.Submit(func() {
			time.Sleep(10 * time.Millisecond)
			atomic.AddInt64(&completed, 1)
			if i%50 == 0 {
				log.Printf("Progress: job %d completed", i)
			}
		})
	}

	pool.Shutdown()
	fmt.Printf("All %d jobs completed\n", atomic.LoadInt64(&completed))
}
```

---

## 4. errgroup による並行処理

`golang.org/x/sync/errgroup`は、複数のgoroutineの完了待ちとエラーハンドリングを統合するパッケージである。`sync.WaitGroup`を置き換え、エラー伝搬とContextキャンセルを自動化する。

### コード例 8: errgroup 基本パターン

```go
package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"time"

	"golang.org/x/sync/errgroup"
)

// fetchURL はURLからコンテンツを取得する
func fetchURL(ctx context.Context, url string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("fetch %s: %w", url, err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read body: %w", err)
	}
	return string(body), nil
}

// fetchAll は複数URLを並行取得する
func fetchAll(ctx context.Context, urls []string) ([]string, error) {
	g, ctx := errgroup.WithContext(ctx)
	results := make([]string, len(urls))

	for i, url := range urls {
		i, url := i, url // ループ変数キャプチャ
		g.Go(func() error {
			body, err := fetchURL(ctx, url)
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

func main() {
	ctx := context.Background()
	urls := []string{
		"https://httpbin.org/get",
		"https://httpbin.org/headers",
		"https://httpbin.org/ip",
	}

	results, err := fetchAll(ctx, urls)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	for i, r := range results {
		fmt.Printf("URL %d: %d bytes\n", i, len(r))
	}
}
```

### コード例 9: errgroup.SetLimit による同時実行数制限

```go
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"golang.org/x/sync/errgroup"
)

// Task は処理タスクを表す
type Task struct {
	ID   int
	Name string
}

// processTask は個別タスクの処理
func processTask(ctx context.Context, task Task) error {
	log.Printf("Start task %d: %s", task.ID, task.Name)

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-time.After(100 * time.Millisecond):
		log.Printf("Done task %d: %s", task.ID, task.Name)
		return nil
	}
}

func main() {
	ctx := context.Background()

	tasks := make([]Task, 50)
	for i := range tasks {
		tasks[i] = Task{ID: i, Name: fmt.Sprintf("task-%d", i)}
	}

	g, ctx := errgroup.WithContext(ctx)
	g.SetLimit(5) // 同時実行数を5に制限

	for _, task := range tasks {
		task := task
		g.Go(func() error {
			return processTask(ctx, task)
		})
	}

	if err := g.Wait(); err != nil {
		log.Fatalf("Error: %v", err)
	}
	fmt.Println("All tasks completed")
}
```

### コード例 10: errgroup.TryGo によるノンブロッキング投入

```go
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"golang.org/x/sync/errgroup"
)

func main() {
	ctx := context.Background()
	g, ctx := errgroup.WithContext(ctx)
	g.SetLimit(3) // 最大3つまで同時実行

	submitted := 0
	dropped := 0

	for i := 0; i < 20; i++ {
		i := i
		// TryGoはリミットに達している場合はfalseを返す（ブロックしない）
		if g.TryGo(func() error {
			log.Printf("Processing task %d", i)
			time.Sleep(100 * time.Millisecond)
			return nil
		}) {
			submitted++
		} else {
			dropped++
			log.Printf("Task %d dropped (pool full)", i)
		}
	}

	if err := g.Wait(); err != nil {
		log.Fatalf("Error: %v", err)
	}

	fmt.Printf("Submitted: %d, Dropped: %d\n", submitted, dropped)
}
```

---

## 5. セマフォパターン

セマフォパターンは、バッファ付きチャネルを使って同時実行数を制限するシンプルなパターンである。errgroupのSetLimitより軽量で、細かい制御が可能。

### コード例 11: 基本セマフォ

```go
package main

import (
	"fmt"
	"sync"
	"time"
)

// Item は処理対象の項目
type Item struct {
	ID   int
	Data string
}

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

func process(item Item) {
	time.Sleep(50 * time.Millisecond) // 処理シミュレート
	fmt.Printf("Processed: %d\n", item.ID)
}

func main() {
	items := make([]Item, 100)
	for i := range items {
		items[i] = Item{ID: i, Data: fmt.Sprintf("data-%d", i)}
	}

	start := time.Now()
	processWithLimit(items, 10) // 最大10並行
	fmt.Printf("Elapsed: %v\n", time.Since(start))
}
```

### コード例 12: golang.org/x/sync/semaphore の活用

標準拡張ライブラリの`semaphore`パッケージは、重み付きセマフォを提供する。

```go
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"golang.org/x/sync/semaphore"
)

func main() {
	ctx := context.Background()

	// 重み付きセマフォ（総重み10）
	sem := semaphore.NewWeighted(10)

	type Task struct {
		Name   string
		Weight int64 // リソース消費量
	}

	tasks := []Task{
		{"light-1", 1},
		{"light-2", 1},
		{"medium-1", 3},
		{"heavy-1", 5},
		{"medium-2", 3},
		{"light-3", 1},
		{"heavy-2", 5},
		{"light-4", 1},
	}

	for _, task := range tasks {
		task := task

		// タスクの重みに応じてセマフォを取得
		if err := sem.Acquire(ctx, task.Weight); err != nil {
			log.Printf("Failed to acquire semaphore for %s: %v", task.Name, err)
			continue
		}

		go func() {
			defer sem.Release(task.Weight)
			log.Printf("Start: %s (weight=%d)", task.Name, task.Weight)
			time.Sleep(100 * time.Millisecond)
			log.Printf("Done: %s", task.Name)
		}()
	}

	// 全タスクの完了を待つ（セマフォを全重み分取得）
	if err := sem.Acquire(ctx, 10); err != nil {
		log.Fatal(err)
	}
	fmt.Println("All tasks completed")
}
```

---

## 6. Or-Done パターン

Or-Doneパターンは、チャネルの読み取りとキャンセルを同時に扱うヘルパーである。select文の冗長さを解消する。

### コード例 13: Or-Done チャネル

```go
package main

import (
	"context"
	"fmt"
)

// orDone はcontextのキャンセルを考慮してチャネルから読み取る
func orDone(ctx context.Context, in <-chan int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for {
			select {
			case <-ctx.Done():
				return
			case v, ok := <-in:
				if !ok {
					return
				}
				select {
				case <-ctx.Done():
					return
				case out <- v:
				}
			}
		}
	}()
	return out
}

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// 無限に値を生成するチャネル
	infinite := make(chan int)
	go func() {
		defer close(infinite)
		for i := 0; ; i++ {
			select {
			case <-ctx.Done():
				return
			case infinite <- i:
			}
		}
	}()

	// 最初の10個だけ取得
	count := 0
	for v := range orDone(ctx, infinite) {
		fmt.Println(v)
		count++
		if count >= 10 {
			cancel()
			break
		}
	}
}
```

---

## 7. Tee パターン

Teeパターンは、1つのチャネルの値を2つのチャネルに分配するパターンである。Unixのteeコマンドと同じ概念で、同じデータを2つの異なる処理パイプラインに送信する場合に使う。

### コード例 14: Tee チャネル

```go
package main

import (
	"context"
	"fmt"
)

// tee は入力チャネルの値を2つの出力チャネルに複製する
func tee(ctx context.Context, in <-chan int) (<-chan int, <-chan int) {
	out1 := make(chan int)
	out2 := make(chan int)

	go func() {
		defer close(out1)
		defer close(out2)
		for {
			select {
			case <-ctx.Done():
				return
			case v, ok := <-in:
				if !ok {
					return
				}
				// 両方の出力チャネルに送信（ローカル変数で保護）
				ch1, ch2 := out1, out2
				for i := 0; i < 2; i++ {
					select {
					case <-ctx.Done():
						return
					case ch1 <- v:
						ch1 = nil // 送信済みならnil化してブロック
					case ch2 <- v:
						ch2 = nil
					}
				}
			}
		}
	}()

	return out1, out2
}

func main() {
	ctx := context.Background()

	// データ生成
	input := make(chan int)
	go func() {
		defer close(input)
		for i := 1; i <= 5; i++ {
			input <- i
		}
	}()

	// Teeで分岐
	ch1, ch2 := tee(ctx, input)

	// 2つの処理パイプライン
	done := make(chan struct{})
	go func() {
		defer close(done)
		for v := range ch1 {
			fmt.Printf("Pipeline A: %d * 2 = %d\n", v, v*2)
		}
	}()

	for v := range ch2 {
		fmt.Printf("Pipeline B: %d ^ 2 = %d\n", v, v*v)
	}

	<-done
}
```

---

## 8. Bridge パターン

Bridgeパターンは、「チャネルのチャネル」（`<-chan <-chan T`）を単一のチャネルにフラット化するパターンである。複数のデータソースを動的に追加・切り替える場合に有用。

### コード例 15: Bridge チャネル

```go
package main

import (
	"context"
	"fmt"
)

// bridge はチャネルのチャネルを単一チャネルにフラット化する
func bridge(ctx context.Context, chanStream <-chan <-chan int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for {
			var stream <-chan int
			select {
			case <-ctx.Done():
				return
			case maybeStream, ok := <-chanStream:
				if !ok {
					return
				}
				stream = maybeStream
			}

			for {
				select {
				case <-ctx.Done():
					return
				case v, ok := <-stream:
					if !ok {
						break // 次のストリームへ
					}
					select {
					case <-ctx.Done():
						return
					case out <- v:
					}
					continue
				}
				break
			}
		}
	}()
	return out
}

// genRange はstart〜endの範囲のチャネルを生成する
func genRange(start, end int) <-chan int {
	ch := make(chan int)
	go func() {
		defer close(ch)
		for i := start; i <= end; i++ {
			ch <- i
		}
	}()
	return ch
}

func main() {
	ctx := context.Background()

	// チャネルのチャネルを生成
	chanStream := make(chan (<-chan int))
	go func() {
		defer close(chanStream)
		chanStream <- genRange(1, 3)
		chanStream <- genRange(10, 13)
		chanStream <- genRange(100, 102)
	}()

	// Bridgeでフラット化
	for v := range bridge(ctx, chanStream) {
		fmt.Println(v) // 1, 2, 3, 10, 11, 12, 13, 100, 101, 102
	}
}
```

---

## 9. Rate Limiter パターン

Rate Limiterパターンは、処理のスループットを一定レートに制限するパターンである。API呼び出しの制限やリソースの過負荷防止に使用する。

### コード例 16: time.Ticker ベースのRate Limiter

```go
package main

import (
	"context"
	"fmt"
	"log"
	"time"
)

// RateLimiter はトークンバケット方式のレートリミッター
type RateLimiter struct {
	ticker *time.Ticker
	tokens chan struct{}
}

// NewRateLimiter は指定レートのリミッターを生成する
func NewRateLimiter(rate int, burst int) *RateLimiter {
	rl := &RateLimiter{
		ticker: time.NewTicker(time.Second / time.Duration(rate)),
		tokens: make(chan struct{}, burst),
	}

	// 初期トークンをバーストサイズ分投入
	for i := 0; i < burst; i++ {
		rl.tokens <- struct{}{}
	}

	// 定期的にトークンを補充
	go func() {
		for range rl.ticker.C {
			select {
			case rl.tokens <- struct{}{}:
			default: // バケットが満杯なら破棄
			}
		}
	}()

	return rl
}

// Wait はトークンが利用可能になるまで待機する
func (rl *RateLimiter) Wait(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-rl.tokens:
		return nil
	}
}

// Stop はリミッターを停止する
func (rl *RateLimiter) Stop() {
	rl.ticker.Stop()
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// 毎秒5リクエスト、バースト10
	limiter := NewRateLimiter(5, 10)
	defer limiter.Stop()

	for i := 0; i < 30; i++ {
		if err := limiter.Wait(ctx); err != nil {
			log.Printf("Rate limiter error: %v", err)
			break
		}
		fmt.Printf("[%s] Request %d\n", time.Now().Format("15:04:05.000"), i)
	}
}
```

### コード例 17: golang.org/x/time/rate の活用

```go
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"golang.org/x/time/rate"
)

func main() {
	ctx := context.Background()

	// 毎秒10イベント、バースト3
	limiter := rate.NewLimiter(rate.Limit(10), 3)

	// Wait: トークンが利用可能になるまでブロック
	for i := 0; i < 20; i++ {
		if err := limiter.Wait(ctx); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("[%s] Event %d\n", time.Now().Format("15:04:05.000"), i)
	}

	// Allow: ノンブロッキング（即座にtrue/falseを返す）
	fmt.Println("\n--- Allow (non-blocking) ---")
	for i := 0; i < 10; i++ {
		if limiter.Allow() {
			fmt.Printf("Event %d: allowed\n", i)
		} else {
			fmt.Printf("Event %d: rate limited\n", i)
		}
	}

	// Reserve: トークン予約（待機時間を取得）
	fmt.Println("\n--- Reserve ---")
	r := limiter.Reserve()
	if r.OK() {
		fmt.Printf("Delay: %v\n", r.Delay())
		time.Sleep(r.Delay())
		fmt.Println("Executed after delay")
	}
}
```

---

## 10. ASCII図解

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

### 図4: Or-Done パターン

```
通常のchannelリード:            Or-Done パターン:

for v := range ch {             for v := range orDone(ctx, ch) {
    // ctxキャンセル非対応         // ctxキャンセル自動対応
    process(v)                      process(v)
}                               }

⚠️ ctx.Done()を見ていない       ✅ ctx.Done()を自動チェック
```

### 図5: Tee パターン

```
                    ┌──> Pipeline A (ch1)
                    │
  Input ── Tee ─────┤
                    │
                    └──> Pipeline B (ch2)

  同じデータを2つの独立した処理に分配
```

### 図6: Bridge パターン

```
  Chan of Chans:
  ┌────────────────────────────────────────┐
  │ chanStream: <-chan (<-chan int)         │
  │                                        │
  │  ┌──── [1, 2, 3] ────┐                │
  │  ├──── [10, 11, 12] ──┤  → Bridge →  [1, 2, 3, 10, 11, 12, 100, 101]
  │  └──── [100, 101] ────┘                │
  │                                        │
  └────────────────────────────────────────┘
```

### 図7: Rate Limiter（トークンバケット）

```
  ┌─────────────────────────────────┐
  │       Token Bucket              │
  │                                 │
  │  バケット容量 = Burst Size      │
  │  ┌────────────────────────┐    │
  │  │ ○ ○ ○ ○ ○ _ _ _ _ _  │    │  ○ = トークン
  │  │ 5/10 トークン残り       │    │  _ = 空き
  │  └────────────────────────┘    │
  │       ↑              ↓         │
  │   Rate(r/s)で     リクエスト時  │
  │   トークン補充    にトークン消費 │
  └─────────────────────────────────┘
```

---

## 11. 比較表

### 表1: 並行パターンの比較

| パターン | 用途 | 複雑度 | goroutine数 | キャンセル対応 |
|---------|------|-------|------------|-------------|
| Pipeline | 直列データ処理 | 低 | ステージ数分 | ステージ毎にselect |
| Fan-out/Fan-in | 並列分散処理 | 中 | ワーカー数分 | context伝搬 |
| Worker Pool | 制限付き並列処理 | 中 | 固定(設定可能) | context + jobs close |
| errgroup | エラー付き並列 | 低 | タスク数分 | 自動(WithContext) |
| セマフォ | 同時実行数制限 | 低 | タスク数分(制限付) | context渡し |
| Or-Done | キャンセル対応読み取り | 低 | 1 | 組み込み |
| Tee | チャネル分配 | 低 | 1 | context対応 |
| Bridge | チャネルフラット化 | 中 | 1 | context対応 |
| Rate Limiter | スループット制御 | 中 | 1-2 | context/Timer |
| Pipeline + Cancel | キャンセル可能処理 | 高 | ステージ数分 | 全ステージ対応 |

### 表2: errgroup vs WaitGroup vs セマフォ

| 項目 | errgroup | sync.WaitGroup | セマフォ |
|------|----------|---------------|---------|
| エラー伝搬 | 最初のエラーを返す | なし | なし |
| キャンセル | context連携で自動 | 手動 | 手動 |
| 同時実行制限 | `SetLimit(n)` | 不可（別途実装） | バッファサイズ |
| ノンブロッキング投入 | `TryGo()` | 不可 | `TryAcquire` |
| パッケージ | `golang.org/x/sync` | 標準`sync` | チャネル or `x/sync` |
| 戻り値 | `error` | なし | なし |
| 重み付き | 不可 | 不可 | `semaphore.Weighted` |
| 用途 | エラー付き並行処理 | 単純な完了待ち | リソース制限 |

### 表3: Rate Limiterの実装比較

| 実装 | パッケージ | アルゴリズム | バースト | 分散対応 |
|------|----------|------------|---------|---------|
| time.Ticker | 標準 | 固定レート | なし | なし |
| バッファ付きチャネル | 標準 | トークンバケット | あり | なし |
| rate.Limiter | `x/time/rate` | トークンバケット | あり | なし |
| Redis + Lua | redis-go | スライディングウィンドウ | あり | あり |
| leaky bucket | カスタム | Leaky Bucket | なし | なし |

### 表4: パターン選択フローチャート

| 要件 | 推奨パターン |
|------|------------|
| データを順番に変換したい | Pipeline |
| 同じ処理を並列化したい | Fan-out/Fan-in |
| goroutine数を制限したい | Worker Pool / セマフォ |
| エラーで全体を止めたい | errgroup |
| API呼び出し頻度を制限したい | Rate Limiter |
| 1つの入力を複数に分配したい | Tee |
| 複数ソースを逐次読みたい | Bridge |
| キャンセル対応の読み取り | Or-Done |

---

## 12. アンチパターン

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

### アンチパターン 3: 送信と受信の非対称なclose

```go
// BAD: 受信側でチャネルを閉じる
func consumer(ch <-chan int) {
	for v := range ch {
		process(v)
	}
	close(ch) // コンパイルエラー（receive-onlyチャネルはcloseできない）
}

// BAD: 複数のgoroutineで同じチャネルをcloseする
func multipleProducers(ch chan<- int) {
	for i := 0; i < 3; i++ {
		go func(id int) {
			ch <- id
			close(ch) // panic: close of closed channel
		}(i)
	}
}

// GOOD: 1つの所有者がcloseの責任を持つ
func multipleProducers(ch chan<- int) {
	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			ch <- id
		}(i)
	}
	go func() {
		wg.Wait()
		close(ch) // WaitGroup完了後に1回だけclose
	}()
}
```

### アンチパターン 4: select文でのビジーウェイト

```go
// BAD: default節でビジーウェイト
func waitForResult(ch <-chan int) int {
	for {
		select {
		case v := <-ch:
			return v
		default:
			// CPU 100% 消費するビジーループ
		}
	}
}

// GOOD: default節を使わずにブロック（またはtime.Afterでタイムアウト）
func waitForResult(ch <-chan int, timeout time.Duration) (int, error) {
	select {
	case v := <-ch:
		return v, nil
	case <-time.After(timeout):
		return 0, fmt.Errorf("timeout waiting for result")
	}
}
```

### アンチパターン 5: goroutineリーク

```go
// BAD: 誰も受信しないチャネルに送信するgoroutineがリーク
func leakyFunction() <-chan int {
	ch := make(chan int)
	go func() {
		result := heavyComputation()
		ch <- result // 受信者がいなければ永遠にブロック → goroutineリーク
	}()
	return ch
}

// 呼び出し側で結果を使わない場合
func caller() {
	_ = leakyFunction() // チャネルを捨てる → goroutineが永遠にリーク
}

// GOOD: contextでキャンセル可能にする
func safeFunction(ctx context.Context) <-chan int {
	ch := make(chan int, 1) // バッファ1で送信側がブロックしない
	go func() {
		result := heavyComputation()
		select {
		case <-ctx.Done():
			return // キャンセルされたら終了
		case ch <- result:
		}
	}()
	return ch
}
```

---

## 13. 実践ユースケース

### ユースケース 1: 画像処理パイプライン

```go
package main

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

// Image は処理対象の画像
type Image struct {
	ID       int
	Filename string
	Data     []byte
	Width    int
	Height   int
}

// ProcessedImage は処理済み画像
type ProcessedImage struct {
	Original  Image
	Thumbnail []byte
	Optimized []byte
	Duration  time.Duration
}

// download は画像をダウンロードするステージ
func download(ctx context.Context, urls <-chan string) <-chan Image {
	out := make(chan Image)
	go func() {
		defer close(out)
		id := 0
		for url := range urls {
			select {
			case <-ctx.Done():
				return
			default:
				// ダウンロード処理（シミュレート）
				time.Sleep(20 * time.Millisecond)
				img := Image{
					ID:       id,
					Filename: url,
					Data:     make([]byte, 1024),
					Width:    1920,
					Height:   1080,
				}
				id++
				select {
				case <-ctx.Done():
					return
				case out <- img:
				}
			}
		}
	}()
	return out
}

// resize はサムネイルを生成するステージ（Fan-outで並列化）
func resize(ctx context.Context, in <-chan Image, workers int) <-chan ProcessedImage {
	out := make(chan ProcessedImage)
	var wg sync.WaitGroup

	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for img := range in {
				select {
				case <-ctx.Done():
					return
				default:
					start := time.Now()
					// リサイズ処理（シミュレート）
					time.Sleep(30 * time.Millisecond)
					processed := ProcessedImage{
						Original:  img,
						Thumbnail: make([]byte, 256),
						Optimized: make([]byte, 512),
						Duration:  time.Since(start),
					}
					select {
					case <-ctx.Done():
						return
					case out <- processed:
					}
				}
			}
		}(i)
	}

	go func() {
		wg.Wait()
		close(out)
	}()

	return out
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// 画像URLを生成
	urls := make(chan string)
	go func() {
		defer close(urls)
		for i := 0; i < 20; i++ {
			urls <- fmt.Sprintf("https://example.com/image-%d.jpg", i)
		}
	}()

	// Pipeline: ダウンロード → リサイズ（4ワーカー）
	images := download(ctx, urls)
	results := resize(ctx, images, 4)

	// 結果を収集
	count := 0
	for result := range results {
		count++
		log.Printf("Processed %s in %v", result.Original.Filename, result.Duration)
	}
	fmt.Printf("Total: %d images processed\n", count)
}
```

### ユースケース 2: マイクロサービスのデータ集約

```go
package main

import (
	"context"
	"fmt"
	"time"

	"golang.org/x/sync/errgroup"
)

// UserProfile はユーザープロフィール
type UserProfile struct {
	UserID   int
	Name     string
	Email    string
	Orders   []Order
	Reviews  []Review
	Points   int
	IsPremium bool
}

type Order struct {
	ID     int
	Amount float64
}

type Review struct {
	ID      int
	Content string
}

// getUserProfile は複数サービスからデータを集約する
func getUserProfile(ctx context.Context, userID int) (*UserProfile, error) {
	profile := &UserProfile{UserID: userID}

	g, ctx := errgroup.WithContext(ctx)

	// ユーザー基本情報を取得
	g.Go(func() error {
		name, email, err := fetchUserInfo(ctx, userID)
		if err != nil {
			return fmt.Errorf("user info: %w", err)
		}
		profile.Name = name
		profile.Email = email
		return nil
	})

	// 注文履歴を取得
	g.Go(func() error {
		orders, err := fetchOrders(ctx, userID)
		if err != nil {
			return fmt.Errorf("orders: %w", err)
		}
		profile.Orders = orders
		return nil
	})

	// レビューを取得
	g.Go(func() error {
		reviews, err := fetchReviews(ctx, userID)
		if err != nil {
			return fmt.Errorf("reviews: %w", err)
		}
		profile.Reviews = reviews
		return nil
	})

	// ポイント残高を取得
	g.Go(func() error {
		points, err := fetchPoints(ctx, userID)
		if err != nil {
			return fmt.Errorf("points: %w", err)
		}
		profile.Points = points
		return nil
	})

	// プレミアム判定
	g.Go(func() error {
		premium, err := checkPremium(ctx, userID)
		if err != nil {
			return fmt.Errorf("premium: %w", err)
		}
		profile.IsPremium = premium
		return nil
	})

	if err := g.Wait(); err != nil {
		return nil, err
	}
	return profile, nil
}

// 以下はモックサービス呼び出し
func fetchUserInfo(ctx context.Context, id int) (string, string, error) {
	time.Sleep(50 * time.Millisecond)
	return "Tanaka", "tanaka@example.com", nil
}

func fetchOrders(ctx context.Context, id int) ([]Order, error) {
	time.Sleep(80 * time.Millisecond)
	return []Order{{ID: 1, Amount: 1000}, {ID: 2, Amount: 2000}}, nil
}

func fetchReviews(ctx context.Context, id int) ([]Review, error) {
	time.Sleep(60 * time.Millisecond)
	return []Review{{ID: 1, Content: "Great product!"}}, nil
}

func fetchPoints(ctx context.Context, id int) (int, error) {
	time.Sleep(30 * time.Millisecond)
	return 5000, nil
}

func checkPremium(ctx context.Context, id int) (bool, error) {
	time.Sleep(20 * time.Millisecond)
	return true, nil
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	start := time.Now()
	profile, err := getUserProfile(ctx, 1)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Profile: %+v\n", profile)
	fmt.Printf("Elapsed: %v (vs sequential ~240ms)\n", time.Since(start))
}
```

---

## 14. FAQ

### Q1: Worker Poolのワーカー数はどう決めるか？

CPUバウンド処理: `runtime.NumCPU()` を基準にする。I/Oバウンド処理: 外部リソースの許容並行数に合わせる（DB接続プール数、API rate limit等）。ベンチマークで最適値を探る。一般的なガイドラインとして:

- **CPUバウンド**: `runtime.NumCPU()` またはそれ以下
- **I/Oバウンド（ローカルディスク）**: `runtime.NumCPU() * 2`
- **I/Oバウンド（ネットワーク）**: 50〜200（外部サービスの制限に依存）
- **混合ワークロード**: ベンチマークで最適値を特定

### Q2: errgroupとWaitGroupのどちらを使うべきか？

エラーハンドリングが必要ならerrgroup、単純な完了待ちならWaitGroup。errgroupはContextとの連携も容易で、現代のGoコードでは多くの場合errgroupが推奨される。errgroup固有の利点として `SetLimit` による同時実行制限と `TryGo` によるノンブロッキング投入がある。

### Q3: Pipelineパターンで遅いステージがあるとどうなるか？

最も遅いステージがボトルネックになる。対策は (1) 遅いステージをFan-outで並列化、(2) バッファ付きチャネルで一時的な速度差を吸収、(3) バッチ処理でスループット向上。ボトルネックの特定にはpprof/traceを活用する。

### Q4: goroutineリークをどう検出するか？

(1) `runtime.NumGoroutine()` を定期的に監視する。(2) テストでは `goleak` パッケージ（`go.uber.org/goleak`）を使う。(3) pprofの `/debug/pprof/goroutine` エンドポイントで確認する。(4) `context.WithCancel` を適切に使い、キャンセル時に全goroutineが終了することを保証する。

### Q5: Fan-outの結果の順序を保証するには？

インデックス付きの結果構造体を使い、全結果収集後にソートする。または、結果をインデックスでマッピングする配列を事前に確保し、各ワーカーが自分のインデックスに直接書き込む（`results[i] = ...` パターン）。

### Q6: select文でのチャネル優先順位は？

Go言語のselect文には優先順位がない（複数のcaseが同時に準備できた場合はランダムに選択される）。優先順位が必要な場合はネストしたselect文を使う:

```go
// ctx.Done()を優先的にチェック
select {
case <-ctx.Done():
    return ctx.Err()
default:
    select {
    case <-ctx.Done():
        return ctx.Err()
    case v := <-dataCh:
        process(v)
    }
}
```

### Q7: チャネルのバッファサイズはどう決めるか？

- **バッファなし（0）**: 送受信が同期。厳密な手順制御が必要な場合
- **小バッファ（1〜10）**: ステージ間の小さな速度差を吸収
- **中バッファ（10〜100）**: I/O待ちの多いPipelineで有効
- **大バッファ（100+）**: バースト的な入力を受け付ける場合。メモリ消費に注意

---

## まとめ

| 概念 | 要点 |
|------|------|
| Pipeline | チャネルでステージを連結。データの流れが明確 |
| Fan-out/Fan-in | 処理を分散し結果を集約 |
| Worker Pool | goroutine数を制限して安定運用 |
| errgroup | エラー付き並行処理の標準パターン |
| セマフォ | バッファ付きチャネルで同時実行数制限 |
| Or-Done | キャンセル対応のチャネル読み取りヘルパー |
| Tee | 1入力を2出力に分配 |
| Bridge | チャネルのチャネルをフラット化 |
| Rate Limiter | スループットを制御して外部リソースを保護 |

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
4. **golang.org/x/time/rate** -- https://pkg.go.dev/golang.org/x/time/rate
5. **golang.org/x/sync/semaphore** -- https://pkg.go.dev/golang.org/x/sync/semaphore
6. **Katherine Cox-Buday, "Concurrency in Go"** -- O'Reilly Media
