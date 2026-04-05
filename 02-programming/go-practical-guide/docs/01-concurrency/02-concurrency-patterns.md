# Concurrency Patterns -- Fan-out/Fan-in, Pipeline, Worker Pool

> Go's concurrency patterns combine goroutines and channels to build practical concurrent processing through Fan-out/Fan-in, Pipeline, Worker Pool, and Context.

---

## What You Will Learn in This Chapter

1. **Pipeline pattern** -- A processing flow that connects stages via channels
2. **Fan-out / Fan-in** -- Parallel distribution and result aggregation
3. **Worker Pool** -- Concurrent processing with a limited number of goroutines
4. **errgroup / semaphore** -- Concurrency control with error handling
5. **Or-Done / Tee / Bridge** -- Advanced channel composition patterns
6. **Rate Limiter** -- Throughput control patterns
7. **Practical use cases** -- Application examples in production environments


## Prerequisites

Before reading this guide, the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding of [Synchronization Primitives -- Mutex, RWMutex, Once, Pool, atomic](./01-sync-primitives.md)

---

## 1. Pipeline Pattern

The Pipeline pattern is a design pattern that divides data processing into multiple stages and connects each stage with channels. Each stage runs in its own goroutine, receives data from an input channel, and sends the processing results to an output channel.

### 1.1 Design Principles of a Pipeline

The basic principles when designing a Pipeline are as follows:

- **Single responsibility**: Each stage is responsible for only one operation
- **Channel ownership**: The goroutine that creates a channel is responsible for closing it
- **Buffering**: Speed differences between stages are absorbed with buffered channels
- **Cancellation support**: Every stage should support cancellation via context

### Code Example 1: Basic Pipeline

```go
package main

import (
	"context"
	"fmt"
)

// generate is a generator stage that sends slice elements to a channel in order
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

// square is a stage that squares the input values and outputs them
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

// filter is a stage that passes through only values matching the condition
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
	// Pipeline: generate -> filter(even) -> square -> output
	ch := generate(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
	even := filter(ch, func(n int) bool { return n%2 == 0 })
	out := square(even)

	for v := range out {
		fmt.Println(v) // 4, 16, 36, 64, 100
	}
}
```

### Code Example 2: Context-Aware Pipeline

In production Pipelines, cancellation support is essential. Design every stage to accept a context and respond to cancellation signals.

```go
package main

import (
	"context"
	"fmt"
	"time"
)

// generateWithCtx is a context-aware generator
func generateWithCtx(ctx context.Context, nums ...int) <-chan int {
	out := make(chan int)
	go func() {
		defer close(out)
		for _, n := range nums {
			select {
			case <-ctx.Done():
				return // Exit immediately on cancellation
			case out <- n:
			}
		}
	}()
	return out
}

// squareWithCtx is a context-aware squaring stage
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

// accumulate is a stage that accumulates input values
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
					// Send the final result
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

### Code Example 3: Batch Processing Pipeline

When processing large amounts of data, improve throughput by using batch processing instead of per-item processing.

```go
package main

import (
	"context"
	"fmt"
)

// batch is a stage that groups input channel values into batches of a given size
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
					// Send the remaining buffer
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
					// Send a copy of the buffer (the original slice will be reused)
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

// processBatch is a stage that processes data in batch units
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
				// Calculate the sum within the batch
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

	// Generate numbers from 1 to 100
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

	// Pipeline: generate -> batch (10 at a time) -> batch processing
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

Fan-out is a pattern that distributes work from a single input channel across multiple workers. Fan-in is a pattern that merges outputs from multiple channels into a single channel. By combining these, you can efficiently parallelize CPU-bound or I/O-bound processing.

### 2.1 Criteria for Applying Fan-out / Fan-in

Cases where Fan-out/Fan-in is effective include:

- When each operation is independent and order does not matter
- When you want to parallelize I/O-bound operations (API calls, DB queries, etc.)
- When you want to distribute CPU-bound operations across multiple cores

### Code Example 4: Basic Fan-out / Fan-in Implementation

```go
package main

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// fanOut distributes work from the input channel to multiple workers
func fanOut(ctx context.Context, in <-chan int, workers int) []<-chan int {
	channels := make([]<-chan int, workers)
	for i := 0; i < workers; i++ {
		channels[i] = worker(ctx, in, i)
	}
	return channels
}

// worker is an individual worker goroutine
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
				// Simulate heavy processing
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

// fanIn merges multiple channels into one
func fanIn(ctx context.Context, channels ...<-chan int) <-chan int {
	var wg sync.WaitGroup
	merged := make(chan int)

	// Start a goroutine that forwards values from each input channel to the merged channel
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

	// Close the merged channel after all workers complete
	go func() {
		wg.Wait()
		close(merged)
	}()

	return merged
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Generate input data
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

	// Fan-out: distribute across 4 workers
	workers := fanOut(ctx, input, 4)

	// Fan-in: merge the results
	results := fanIn(ctx, workers...)

	// Collect the results
	for result := range results {
		fmt.Println("Result:", result)
	}
}
```

### Code Example 5: Order-Preserving Fan-out / Fan-in

Normal Fan-out/Fan-in does not guarantee the order of results. When order must be preserved, attach an index.

```go
package main

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"
)

// IndexedItem is an indexed data item
type IndexedItem struct {
	Index  int
	Value  int
	Result int
}

// orderedFanOut is a Fan-out that preserves ordering information
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

	// Output channel for each worker
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
					// Simulate heavy processing
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

	// Sort by index to restore the original order
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

The Worker Pool pattern pre-starts a fixed number of goroutines (workers) and has them retrieve tasks from a job queue for processing. It prevents unbounded goroutine creation and allows you to control resource usage.

### 3.1 Worker Pool Design Guidelines

- **Number of workers**: For CPU-bound work, use `runtime.NumCPU()`; for I/O-bound work, match the constraints of external resources
- **Job queue**: Control backpressure with a buffered channel
- **Result channel**: Aggregate processing results from workers
- **Error handling**: Handle errors on a per-job basis

### Code Example 6: Generic Worker Pool

```go
package main

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

// Job represents a task to be processed
type Job struct {
	ID      int
	Payload string
}

// Result represents a processing result
type Result struct {
	JobID    int
	Output   string
	Duration time.Duration
	Err      error
}

// WorkerPool is a pool that processes jobs with a fixed number of workers
type WorkerPool struct {
	numWorkers int
	jobs       chan Job
	results    chan Result
	wg         sync.WaitGroup
}

// NewWorkerPool creates a new WorkerPool
func NewWorkerPool(numWorkers, jobBufferSize int) *WorkerPool {
	return &WorkerPool{
		numWorkers: numWorkers,
		jobs:       make(chan Job, jobBufferSize),
		results:    make(chan Result, jobBufferSize),
	}
}

// Start launches the workers
func (wp *WorkerPool) Start(ctx context.Context) {
	for i := 0; i < wp.numWorkers; i++ {
		wp.wg.Add(1)
		go wp.runWorker(ctx, i)
	}

	// Close the results channel after all workers finish
	go func() {
		wp.wg.Wait()
		close(wp.results)
	}()
}

// runWorker is the processing loop for an individual worker
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

// Submit adds a job to the queue
func (wp *WorkerPool) Submit(job Job) {
	wp.jobs <- job
}

// Close closes the job queue (stops accepting new jobs)
func (wp *WorkerPool) Close() {
	close(wp.jobs)
}

// Results returns the results channel
func (wp *WorkerPool) Results() <-chan Result {
	return wp.results
}

// processJob is the processing logic for an individual job
func processJob(ctx context.Context, job Job) (string, error) {
	// Simulate processing
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

	// Submit jobs
	go func() {
		for i := 0; i < 50; i++ {
			pool.Submit(Job{ID: i, Payload: fmt.Sprintf("task-%d", i)})
		}
		pool.Close()
	}()

	// Collect results
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

### Code Example 7: Dynamically Scaling Worker Pool

A Worker Pool that dynamically adjusts the number of workers based on load.

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

// DynamicPool is a WorkerPool that scales dynamically
type DynamicPool struct {
	minWorkers  int
	maxWorkers  int
	activeCount int64
	jobs        chan func()
	wg          sync.WaitGroup
	mu          sync.Mutex
	workerCount int
}

// NewDynamicPool creates a dynamic pool
func NewDynamicPool(minWorkers, maxWorkers, queueSize int) *DynamicPool {
	dp := &DynamicPool{
		minWorkers: minWorkers,
		maxWorkers: maxWorkers,
		jobs:       make(chan func(), queueSize),
	}
	return dp
}

// Start launches the pool with the minimum number of workers
func (dp *DynamicPool) Start(ctx context.Context) {
	for i := 0; i < dp.minWorkers; i++ {
		dp.addWorker(ctx)
	}

	// Load monitoring goroutine
	go dp.monitor(ctx)
}

// addWorker adds one worker
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

// monitor watches the load and adjusts the number of workers
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

			// Add workers if the queue is about to overflow
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

// Submit submits a job
func (dp *DynamicPool) Submit(job func()) {
	dp.jobs <- job
}

// Shutdown shuts down the pool
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

	// Submit a large number of jobs all at once
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

## 4. Concurrent Processing with errgroup

`golang.org/x/sync/errgroup` is a package that integrates waiting for multiple goroutines to complete with error handling. It replaces `sync.WaitGroup` and automates error propagation and context cancellation.

### Code Example 8: Basic errgroup Pattern

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

// fetchURL retrieves content from a URL
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

// fetchAll fetches multiple URLs concurrently
func fetchAll(ctx context.Context, urls []string) ([]string, error) {
	g, ctx := errgroup.WithContext(ctx)
	results := make([]string, len(urls))

	for i, url := range urls {
		i, url := i, url // Loop variable capture
		g.Go(func() error {
			body, err := fetchURL(ctx, url)
			if err != nil {
				return err // A single error cancels the whole group
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

### Code Example 9: Limiting Concurrency with errgroup.SetLimit

```go
package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"golang.org/x/sync/errgroup"
)

// Task represents a processing task
type Task struct {
	ID   int
	Name string
}

// processTask processes an individual task
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
	g.SetLimit(5) // Limit concurrency to 5

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

### Code Example 10: Non-Blocking Submission with errgroup.TryGo

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
	g.SetLimit(3) // Up to 3 concurrent executions

	submitted := 0
	dropped := 0

	for i := 0; i < 20; i++ {
		i := i
		// TryGo returns false if the limit has been reached (does not block)
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

## 5. Semaphore Pattern

The semaphore pattern is a simple pattern that uses a buffered channel to limit concurrency. It is lighter-weight than errgroup's SetLimit and allows for fine-grained control.

### Code Example 11: Basic Semaphore

```go
package main

import (
	"fmt"
	"sync"
	"time"
)

// Item is an item to be processed
type Item struct {
	ID   int
	Data string
}

func processWithLimit(items []Item, maxConcurrency int) {
	sem := make(chan struct{}, maxConcurrency)
	var wg sync.WaitGroup

	for _, item := range items {
		wg.Add(1)
		sem <- struct{}{} // Acquire the semaphore (blocks if full)
		go func(it Item) {
			defer wg.Done()
			defer func() { <-sem }() // Release the semaphore
			process(it)
		}(item)
	}
	wg.Wait()
}

func process(item Item) {
	time.Sleep(50 * time.Millisecond) // Simulate processing
	fmt.Printf("Processed: %d\n", item.ID)
}

func main() {
	items := make([]Item, 100)
	for i := range items {
		items[i] = Item{ID: i, Data: fmt.Sprintf("data-%d", i)}
	}

	start := time.Now()
	processWithLimit(items, 10) // Max 10 concurrent
	fmt.Printf("Elapsed: %v\n", time.Since(start))
}
```

### Code Example 12: Using golang.org/x/sync/semaphore

The standard extension library's `semaphore` package provides a weighted semaphore.

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

	// Weighted semaphore (total weight of 10)
	sem := semaphore.NewWeighted(10)

	type Task struct {
		Name   string
		Weight int64 // Resource consumption
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

		// Acquire the semaphore according to the task's weight
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

	// Wait for all tasks to complete (acquire the full weight of the semaphore)
	if err := sem.Acquire(ctx, 10); err != nil {
		log.Fatal(err)
	}
	fmt.Println("All tasks completed")
}
```

---

## 6. Or-Done Pattern

The Or-Done pattern is a helper that handles channel reads and cancellation at the same time. It eliminates the verbosity of select statements.

### Code Example 13: Or-Done Channel

```go
package main

import (
	"context"
	"fmt"
)

// orDone reads from a channel while respecting context cancellation
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

	// Channel that generates values infinitely
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

	// Take only the first 10 values
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

## 7. Tee Pattern

The Tee pattern is a pattern that distributes values from a single channel to two channels. It uses the same concept as the Unix tee command and is used when you want to send the same data to two different processing pipelines.

### Code Example 14: Tee Channel

```go
package main

import (
	"context"
	"fmt"
)

// tee duplicates values from the input channel to two output channels
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
				// Send to both output channels (protected by local variables)
				ch1, ch2 := out1, out2
				for i := 0; i < 2; i++ {
					select {
					case <-ctx.Done():
						return
					case ch1 <- v:
						ch1 = nil // Nil out after sending to block further sends
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

	// Generate data
	input := make(chan int)
	go func() {
		defer close(input)
		for i := 1; i <= 5; i++ {
			input <- i
		}
	}()

	// Branch with Tee
	ch1, ch2 := tee(ctx, input)

	// Two processing pipelines
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

## 8. Bridge Pattern

The Bridge pattern is a pattern that flattens a "channel of channels" (`<-chan <-chan T`) into a single channel. It is useful when dynamically adding or switching between multiple data sources.

### Code Example 15: Bridge Channel

```go
package main

import (
	"context"
	"fmt"
)

// bridge flattens a channel of channels into a single channel
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
						break // Move to the next stream
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

// genRange generates a channel for the range from start to end
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

	// Generate a channel of channels
	chanStream := make(chan (<-chan int))
	go func() {
		defer close(chanStream)
		chanStream <- genRange(1, 3)
		chanStream <- genRange(10, 13)
		chanStream <- genRange(100, 102)
	}()

	// Flatten with Bridge
	for v := range bridge(ctx, chanStream) {
		fmt.Println(v) // 1, 2, 3, 10, 11, 12, 13, 100, 101, 102
	}
}
```

---

## 9. Rate Limiter Pattern

The Rate Limiter pattern is a pattern that limits processing throughput to a constant rate. It is used for limiting API calls and preventing resource overload.

### Code Example 16: time.Ticker-Based Rate Limiter

```go
package main

import (
	"context"
	"fmt"
	"log"
	"time"
)

// RateLimiter is a token bucket rate limiter
type RateLimiter struct {
	ticker *time.Ticker
	tokens chan struct{}
}

// NewRateLimiter creates a limiter with the specified rate
func NewRateLimiter(rate int, burst int) *RateLimiter {
	rl := &RateLimiter{
		ticker: time.NewTicker(time.Second / time.Duration(rate)),
		tokens: make(chan struct{}, burst),
	}

	// Inject initial tokens equal to the burst size
	for i := 0; i < burst; i++ {
		rl.tokens <- struct{}{}
	}

	// Replenish tokens periodically
	go func() {
		for range rl.ticker.C {
			select {
			case rl.tokens <- struct{}{}:
			default: // Discard if the bucket is full
			}
		}
	}()

	return rl
}

// Wait waits until a token becomes available
func (rl *RateLimiter) Wait(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-rl.tokens:
		return nil
	}
}

// Stop stops the limiter
func (rl *RateLimiter) Stop() {
	rl.ticker.Stop()
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// 5 requests per second, burst of 10
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

### Code Example 17: Using golang.org/x/time/rate

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

	// 10 events per second, burst of 3
	limiter := rate.NewLimiter(rate.Limit(10), 3)

	// Wait: blocks until a token becomes available
	for i := 0; i < 20; i++ {
		if err := limiter.Wait(ctx); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("[%s] Event %d\n", time.Now().Format("15:04:05.000"), i)
	}

	// Allow: non-blocking (returns true/false immediately)
	fmt.Println("\n--- Allow (non-blocking) ---")
	for i := 0; i < 10; i++ {
		if limiter.Allow() {
			fmt.Printf("Event %d: allowed\n", i)
		} else {
			fmt.Printf("Event %d: rate limited\n", i)
		}
	}

	// Reserve: token reservation (retrieves the wait time)
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

## 10. ASCII Diagrams

### Figure 1: Pipeline Pattern

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ generate │───>│ filter   │───>│ square   │───>│ consumer │
│  ch out  │    │ ch in/out│    │ ch in/out│    │  ch in   │
└──────────┘    └──────────┘    └──────────┘    └──────────┘

Data flow:
  [1,2,3,4,5] → [2,4] → [4,16] → print
```

### Figure 2: Fan-out / Fan-in

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

### Figure 3: Worker Pool

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
│  └─────────┘  (fixed count) └────────┘ │
│                                          │
│  Concurrency = Number of Workers (controllable) │
└─────────────────────────────────────────┘
```

### Figure 4: Or-Done Pattern

```
Normal channel read:            Or-Done pattern:

for v := range ch {             for v := range orDone(ctx, ch) {
    // No ctx cancel support       // Automatic ctx cancel support
    process(v)                      process(v)
}                               }

[!] Does not watch ctx.Done()   [OK] Automatically checks ctx.Done()
```

### Figure 5: Tee Pattern

```
                    ┌──> Pipeline A (ch1)
                    │
  Input ── Tee ─────┤
                    │
                    └──> Pipeline B (ch2)

  Distributes the same data to two independent processes
```

### Figure 6: Bridge Pattern

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

### Figure 7: Rate Limiter (Token Bucket)

```
  ┌─────────────────────────────────┐
  │       Token Bucket              │
  │                                 │
  │  Bucket capacity = Burst Size   │
  │  ┌────────────────────────┐    │
  │  │ O O O O O _ _ _ _ _    │    │  O = token
  │  │ 5/10 tokens remaining  │    │  _ = empty slot
  │  └────────────────────────┘    │
  │       ↑              ↓         │
  │   Refill tokens  Consume tokens │
  │   at Rate(r/s)   on request     │
  └─────────────────────────────────┘
```

---

## 11. Comparison Tables

### Table 1: Comparison of Concurrency Patterns

| Pattern | Use Case | Complexity | Goroutine Count | Cancellation Support |
|---------|----------|------------|-----------------|----------------------|
| Pipeline | Serial data processing | Low | One per stage | select in each stage |
| Fan-out/Fan-in | Parallel distributed processing | Medium | One per worker | context propagation |
| Worker Pool | Limited parallel processing | Medium | Fixed (configurable) | context + jobs close |
| errgroup | Parallel with error handling | Low | One per task | Automatic (WithContext) |
| Semaphore | Concurrency limit | Low | Per task (limited) | Pass context |
| Or-Done | Cancellable reads | Low | 1 | Built-in |
| Tee | Channel distribution | Low | 1 | Context-aware |
| Bridge | Channel flattening | Medium | 1 | Context-aware |
| Rate Limiter | Throughput control | Medium | 1-2 | context/Timer |
| Pipeline + Cancel | Cancellable processing | High | One per stage | All stages |

### Table 2: errgroup vs WaitGroup vs Semaphore

| Item | errgroup | sync.WaitGroup | Semaphore |
|------|----------|----------------|-----------|
| Error propagation | Returns the first error | None | None |
| Cancellation | Automatic via context integration | Manual | Manual |
| Concurrency limit | `SetLimit(n)` | Not available (implement separately) | Buffer size |
| Non-blocking submission | `TryGo()` | Not available | `TryAcquire` |
| Package | `golang.org/x/sync` | Standard `sync` | channel or `x/sync` |
| Return value | `error` | None | None |
| Weighted | Not available | Not available | `semaphore.Weighted` |
| Use case | Concurrent processing with errors | Simple completion wait | Resource limiting |

### Table 3: Comparison of Rate Limiter Implementations

| Implementation | Package | Algorithm | Burst | Distributed Support |
|----------------|---------|-----------|-------|---------------------|
| time.Ticker | Standard | Fixed rate | No | No |
| Buffered channel | Standard | Token bucket | Yes | No |
| rate.Limiter | `x/time/rate` | Token bucket | Yes | No |
| Redis + Lua | redis-go | Sliding window | Yes | Yes |
| leaky bucket | Custom | Leaky bucket | No | No |

### Table 4: Pattern Selection Flowchart

| Requirement | Recommended Pattern |
|-------------|---------------------|
| Transform data in sequence | Pipeline |
| Parallelize the same operation | Fan-out/Fan-in |
| Limit the number of goroutines | Worker Pool / Semaphore |
| Stop everything on error | errgroup |
| Limit API call frequency | Rate Limiter |
| Distribute one input to many | Tee |
| Read from multiple sources sequentially | Bridge |
| Cancellable reads | Or-Done |

---

## 12. Anti-Patterns

### Anti-Pattern 1: Unbounded Goroutine Creation

```go
// BAD: spawns an unlimited number of goroutines per request
func handler(w http.ResponseWriter, r *http.Request) {
	for _, item := range getItems() { // 100,000 items
		go process(item) // goroutine explosion, OOM
	}
}

// GOOD: limit with a Worker Pool
func handler(w http.ResponseWriter, r *http.Request) {
	items := getItems()
	sem := make(chan struct{}, 100) // Max 100 concurrent
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

### Anti-Pattern 2: Forgetting to Close a Channel

```go
// BAD: not closing the channel -> receiver blocks forever on range
func produce(ch chan<- int) {
	for i := 0; i < 10; i++ {
		ch <- i
	}
	// close(ch) is missing
}

// GOOD: ensure close with defer
func produce(ch chan<- int) {
	defer close(ch)
	for i := 0; i < 10; i++ {
		ch <- i
	}
}
```

### Anti-Pattern 3: Asymmetric Close Between Send and Receive

```go
// BAD: closing the channel on the receiving side
func consumer(ch <-chan int) {
	for v := range ch {
		process(v)
	}
	close(ch) // Compile error (cannot close a receive-only channel)
}

// BAD: closing the same channel from multiple goroutines
func multipleProducers(ch chan<- int) {
	for i := 0; i < 3; i++ {
		go func(id int) {
			ch <- id
			close(ch) // panic: close of closed channel
		}(i)
	}
}

// GOOD: a single owner is responsible for closing
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
		close(ch) // Close only once after the WaitGroup completes
	}()
}
```

### Anti-Pattern 4: Busy-Wait in a select Statement

```go
// BAD: busy-wait via the default branch
func waitForResult(ch <-chan int) int {
	for {
		select {
		case v := <-ch:
			return v
		default:
			// A busy loop that consumes 100% CPU
		}
	}
}

// GOOD: block without a default branch (or use time.After for timeout)
func waitForResult(ch <-chan int, timeout time.Duration) (int, error) {
	select {
	case v := <-ch:
		return v, nil
	case <-time.After(timeout):
		return 0, fmt.Errorf("timeout waiting for result")
	}
}
```

### Anti-Pattern 5: Goroutine Leak

```go
// BAD: a goroutine sending to a channel with no receiver leaks
func leakyFunction() <-chan int {
	ch := make(chan int)
	go func() {
		result := heavyComputation()
		ch <- result // If no receiver exists, blocks forever -> goroutine leak
	}()
	return ch
}

// When the caller discards the result
func caller() {
	_ = leakyFunction() // Discarding the channel -> goroutine leaks forever
}

// GOOD: make it cancellable with a context
func safeFunction(ctx context.Context) <-chan int {
	ch := make(chan int, 1) // Buffer of 1 so the sender does not block
	go func() {
		result := heavyComputation()
		select {
		case <-ctx.Done():
			return // Exit if cancelled
		case ch <- result:
		}
	}()
	return ch
}
```

---

## 13. Practical Use Cases

### Use Case 1: Image Processing Pipeline

```go
package main

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

// Image is an image to be processed
type Image struct {
	ID       int
	Filename string
	Data     []byte
	Width    int
	Height   int
}

// ProcessedImage is a processed image
type ProcessedImage struct {
	Original  Image
	Thumbnail []byte
	Optimized []byte
	Duration  time.Duration
}

// download is a stage that downloads images
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
				// Download processing (simulated)
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

// resize is a stage that generates thumbnails (parallelized with Fan-out)
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
					// Resize processing (simulated)
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

	// Generate image URLs
	urls := make(chan string)
	go func() {
		defer close(urls)
		for i := 0; i < 20; i++ {
			urls <- fmt.Sprintf("https://example.com/image-%d.jpg", i)
		}
	}()

	// Pipeline: download -> resize (4 workers)
	images := download(ctx, urls)
	results := resize(ctx, images, 4)

	// Collect results
	count := 0
	for result := range results {
		count++
		log.Printf("Processed %s in %v", result.Original.Filename, result.Duration)
	}
	fmt.Printf("Total: %d images processed\n", count)
}
```

### Use Case 2: Microservice Data Aggregation

```go
package main

import (
	"context"
	"fmt"
	"time"

	"golang.org/x/sync/errgroup"
)

// UserProfile is a user profile
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

// getUserProfile aggregates data from multiple services
func getUserProfile(ctx context.Context, userID int) (*UserProfile, error) {
	profile := &UserProfile{UserID: userID}

	g, ctx := errgroup.WithContext(ctx)

	// Fetch basic user information
	g.Go(func() error {
		name, email, err := fetchUserInfo(ctx, userID)
		if err != nil {
			return fmt.Errorf("user info: %w", err)
		}
		profile.Name = name
		profile.Email = email
		return nil
	})

	// Fetch order history
	g.Go(func() error {
		orders, err := fetchOrders(ctx, userID)
		if err != nil {
			return fmt.Errorf("orders: %w", err)
		}
		profile.Orders = orders
		return nil
	})

	// Fetch reviews
	g.Go(func() error {
		reviews, err := fetchReviews(ctx, userID)
		if err != nil {
			return fmt.Errorf("reviews: %w", err)
		}
		profile.Reviews = reviews
		return nil
	})

	// Fetch points balance
	g.Go(func() error {
		points, err := fetchPoints(ctx, userID)
		if err != nil {
			return fmt.Errorf("points: %w", err)
		}
		profile.Points = points
		return nil
	})

	// Check premium status
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

// The following are mock service calls
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

### Q1: How do I decide the number of workers in a Worker Pool?

For CPU-bound work, use `runtime.NumCPU()` as a baseline. For I/O-bound work, match the acceptable concurrency of external resources (DB connection pool size, API rate limits, etc.). Find the optimal value through benchmarking. As a general guideline:

- **CPU-bound**: `runtime.NumCPU()` or less
- **I/O-bound (local disk)**: `runtime.NumCPU() * 2`
- **I/O-bound (network)**: 50 to 200 (depends on the limits of the external service)
- **Mixed workload**: Identify the optimal value through benchmarking

### Q2: Should I use errgroup or WaitGroup?

Use errgroup when error handling is required; use WaitGroup for simple completion waiting. errgroup also integrates easily with Context, and in modern Go code errgroup is recommended in most cases. Specific advantages of errgroup include concurrency limiting via `SetLimit` and non-blocking submission via `TryGo`.

### Q3: What happens if there is a slow stage in a Pipeline pattern?

The slowest stage becomes the bottleneck. Countermeasures include (1) parallelizing the slow stage with Fan-out, (2) absorbing temporary speed differences with buffered channels, and (3) improving throughput with batch processing. Use pprof/trace to identify bottlenecks.

### Q4: How do I detect goroutine leaks?

(1) Monitor `runtime.NumGoroutine()` periodically. (2) In tests, use the `goleak` package (`go.uber.org/goleak`). (3) Check the `/debug/pprof/goroutine` endpoint of pprof. (4) Use `context.WithCancel` appropriately to ensure that all goroutines terminate on cancellation.

### Q5: How can I guarantee the order of Fan-out results?

Use an indexed result struct and sort all results after collection. Alternatively, pre-allocate an array mapped by index and have each worker write directly to its own index (`results[i] = ...` pattern).

### Q6: How are channels prioritized in a select statement?

Go's select statement has no priority (when multiple cases are ready simultaneously, one is chosen at random). When prioritization is needed, use nested select statements:

```go
// Prioritize checking ctx.Done()
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

### Q7: How do I decide the buffer size of a channel?

- **Unbuffered (0)**: Send and receive are synchronized. Use when strict step-by-step control is required
- **Small buffer (1 to 10)**: Absorbs minor speed differences between stages
- **Medium buffer (10 to 100)**: Effective for Pipelines with many I/O waits
- **Large buffer (100+)**: When accepting bursty input. Be mindful of memory consumption

---


## FAQ

### Q1: What is the most important point to focus on when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing code and observing how it behaves.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to advanced topics. We recommend solidly understanding the fundamental concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in real-world development?

The knowledge from this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Concept | Key Points |
|---------|------------|
| Pipeline | Connect stages with channels. Data flow is explicit |
| Fan-out/Fan-in | Distribute processing and aggregate results |
| Worker Pool | Limit the number of goroutines for stable operation |
| errgroup | Standard pattern for concurrent processing with errors |
| Semaphore | Limit concurrency using a buffered channel |
| Or-Done | Helper for cancellation-aware channel reading |
| Tee | Distribute one input to two outputs |
| Bridge | Flatten a channel of channels |
| Rate Limiter | Control throughput to protect external resources |

---

## Recommended Next Reads

- [03-context.md](./03-context.md) -- Cancellation control with Context
- [../02-web/00-net-http.md](../02-web/00-net-http.md) -- Concurrency in HTTP servers
- [../03-tools/02-profiling.md](../03-tools/02-profiling.md) -- Profiling concurrent code

---

## References

1. **Go Blog, "Go Concurrency Patterns: Pipelines and cancellation"** -- https://go.dev/blog/pipelines
2. **Go Blog, "Advanced Go Concurrency Patterns"** -- https://go.dev/blog/io2013-talk-concurrency
3. **golang.org/x/sync/errgroup** -- https://pkg.go.dev/golang.org/x/sync/errgroup
4. **golang.org/x/time/rate** -- https://pkg.go.dev/golang.org/x/time/rate
5. **golang.org/x/sync/semaphore** -- https://pkg.go.dev/golang.org/x/sync/semaphore
6. **Katherine Cox-Buday, "Concurrency in Go"** -- O'Reilly Media
