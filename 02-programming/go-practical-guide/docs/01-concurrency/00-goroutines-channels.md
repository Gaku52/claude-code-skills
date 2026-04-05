# Goroutines and Channels -- The Foundation of Go Concurrent Programming

> Goroutines are lightweight threads, channels are type-safe communication paths, and together they form the core of Go's concurrency model based on CSP.

---

## What You Will Learn in This Chapter

1. **goroutine** -- Lightweight concurrent execution via the go statement
2. **channel** -- Safe data passing and the select statement
3. **sync.WaitGroup** -- Waiting for goroutines to complete
4. **Patterns and Practice** -- Practical use of goroutines and channels in real applications


## Prerequisites

Before reading this guide, the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts

---

## 1. Goroutine Basics

### Code Example 1: Starting a goroutine

```go
func main() {
    go func() {
        fmt.Println("Executing in a goroutine")
    }()

    go sayHello("World")

    time.Sleep(100 * time.Millisecond) // Wait (in real code, use WaitGroup)
}

func sayHello(name string) {
    fmt.Printf("Hello, %s!\n", name)
}
```

### How goroutines Work Internally

A goroutine is a lightweight execution unit multiplexed on top of OS threads. The Go runtime adopts an M:N scheduling model, running M goroutines on N OS threads.

```go
// Demo illustrating goroutine characteristics
package main

import (
    "fmt"
    "runtime"
    "sync"
)

func main() {
    // GOMAXPROCS controls the number of OS threads
    fmt.Printf("Logical CPUs: %d\n", runtime.NumCPU())
    fmt.Printf("GOMAXPROCS: %d\n", runtime.GOMAXPROCS(0))

    var wg sync.WaitGroup
    const N = 100000

    // Starting 100,000 goroutines works fine
    for i := 0; i < N; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            // Each goroutine starts with about a 2KB stack
            // and can grow up to 1GB as needed
            _ = id
        }(i)
    }

    wg.Wait()
    fmt.Printf("All %d goroutines completed\n", N)
    fmt.Printf("Current goroutine count: %d\n", runtime.NumGoroutine())
}
```

### The GMP Model of the goroutine Scheduler

```
G = Goroutine (execution unit)
M = Machine  (OS thread)
P = Processor (logical processor, configured via GOMAXPROCS)

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

Work stealing:
  P0's Local Queue is empty → steal half from P1's Local Queue
  → Also pull from the Global Queue
```

### Code Example: Observing goroutine Scheduling

```go
package main

import (
    "fmt"
    "runtime"
    "time"
)

func main() {
    // Set GOMAXPROCS to 1 to observe cooperative scheduling
    runtime.GOMAXPROCS(1)

    go func() {
        for i := 0; i < 5; i++ {
            fmt.Printf("goroutine A: %d\n", i)
            // Without calling runtime.Gosched(),
            // this goroutine may monopolize the CPU
            runtime.Gosched() // Explicitly yield control to the scheduler
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

## 2. Channel Basics

### Code Example 2: Unbuffered Channels (Synchronous Channels)

```go
func main() {
    ch := make(chan int)    // Unbuffered channel

    go func() {
        ch <- 42            // Send (blocks until received)
    }()

    value := <-ch           // Receive (blocks until sent)
    fmt.Println(value)      // 42
}
```

The essence of an unbuffered channel lies in its "rendezvous" semantics. A send and a receive are guaranteed to occur simultaneously. This lets you create a reliable synchronization point between two goroutines.

```go
// Application of rendezvous: reliable notification between goroutines
func main() {
    done := make(chan struct{}) // struct{} is zero-byte and memory-efficient

    go func() {
        fmt.Println("Processing...")
        time.Sleep(time.Second)
        fmt.Println("Processing complete")
        done <- struct{}{} // Completion notification
    }()

    <-done // Wait for completion
    fmt.Println("Main goroutine also exiting")
}
```

### Code Example 3: Buffered Channels

```go
func main() {
    ch := make(chan string, 3) // Buffer size 3

    ch <- "a" // Does not block (buffer has space)
    ch <- "b"
    ch <- "c"
    // ch <- "d" // Would block here (buffer is full)

    fmt.Println(<-ch) // "a" (FIFO)
    fmt.Println(<-ch) // "b"
}
```

### Guidelines for Buffer Size Design

```go
// Buffer size 0: synchronous communication (rendezvous)
// Sender blocks until a receiver is ready
done := make(chan struct{})

// Buffer size 1: signaling
// Can buffer one notification
signal := make(chan os.Signal, 1)

// Buffer size N: pipeline / work queue
// Absorbs speed differences between producers and consumers
jobs := make(chan Job, 100)

// Criteria for choosing buffer size:
//   0: when synchronization is required
//   1: for notification use (e.g., signal.Notify)
//   N: to absorb throughput differences between producer/consumer
//   Oversized buffers waste memory and
//   may delay the discovery of problems
```

### Code Example: Channel Direction Constraints

```go
// Ensure type safety with send-only and receive-only channels
func producer(out chan<- int) {
    // chan<- int is send-only. Attempting to receive causes a compile error
    for i := 0; i < 10; i++ {
        out <- i * i
    }
    close(out)
}

func consumer(in <-chan int) {
    // <-chan int is receive-only. Attempting to send causes a compile error
    for v := range in {
        fmt.Println(v)
    }
}

func main() {
    ch := make(chan int, 5)

    go producer(ch)  // chan int is implicitly converted to chan<- int
    consumer(ch)     // chan int is implicitly converted to <-chan int
}
```

---

## 3. The select Statement

### Code Example 4: select Statement Basics

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

### Code Example: Applied select Patterns

```go
// Operations with timeout
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

// Non-blocking operation (select with default)
func tryReceive(ch <-chan int) (int, bool) {
    select {
    case v := <-ch:
        return v, true
    default:
        return 0, false // Return immediately if no data on the channel
    }
}

// Combining periodic polling with signal reception
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
            fmt.Println("Monitoring stopped:", ctx.Err())
            return
        }
    }
}
```

### Random Selection and Fairness in select

```go
// When multiple channels are ready simultaneously,
// select picks one at random (fairness guarantee)
func demonstrateFairness() {
    ch1 := make(chan string, 100)
    ch2 := make(chan string, 100)

    // Populate both channels
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

    // Result is about 100:100 (not exact since it is random)
    fmt.Printf("ch1: %d, ch2: %d\n", countA, countB)
}
```

---

## 4. WaitGroup and Waiting for Completion

### Code Example 5: WaitGroup Basics

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

    wg.Wait() // Wait for all goroutines to complete
}
```

### WaitGroup Pitfalls

```go
// NG: Calling wg.Add inside the goroutine
// wg.Wait() may execute before Add
func badPattern() {
    var wg sync.WaitGroup
    for i := 0; i < 10; i++ {
        go func(id int) {
            wg.Add(1)   // ← NG: Add is called after the goroutine starts
            defer wg.Done()
            process(id)
        }(i)
    }
    wg.Wait() // Wait may run before Add
}

// OK: Call wg.Add before starting the goroutine
func goodPattern() {
    var wg sync.WaitGroup
    for i := 0; i < 10; i++ {
        wg.Add(1) // ← OK: Add before starting the goroutine
        go func(id int) {
            defer wg.Done()
            process(id)
        }(i)
    }
    wg.Wait()
}
```

### Code Example: WaitGroup + Error Collection

```go
// Process concurrently with WaitGroup while also collecting errors
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

## 5. Closing Channels and range

### Code Example 6: Closing a Channel and range

```go
func generate(n int) <-chan int {
    ch := make(chan int)
    go func() {
        defer close(ch) // Close after sending is complete
        for i := 0; i < n; i++ {
            ch <- i * i
        }
    }()
    return ch
}

func main() {
    for v := range generate(5) { // Loop until the channel is closed
        fmt.Println(v) // 0, 1, 4, 9, 16
    }
}
```

### Safe Handling of close

```go
// Receiving from a closed channel
func main() {
    ch := make(chan int, 3)
    ch <- 1
    ch <- 2
    close(ch)

    // Method 1: the ok pattern
    v, ok := <-ch
    fmt.Println(v, ok)  // 1 true
    v, ok = <-ch
    fmt.Println(v, ok)  // 2 true
    v, ok = <-ch
    fmt.Println(v, ok)  // 0 false (closed, zero value)

    // Method 2: range (recommended)
    ch2 := make(chan int, 3)
    ch2 <- 10
    ch2 <- 20
    close(ch2)
    for v := range ch2 {
        fmt.Println(v) // 10, 20
    }
}

// Closing when there are multiple senders
type FanIn struct {
    ch   chan int
    once sync.Once
}

func (f *FanIn) Close() {
    f.once.Do(func() {
        close(f.ch) // Prevent double-close with sync.Once
    })
}
```

---

## 6. Practical goroutine and channel Patterns

### Pattern 1: Fan-out / Fan-in

```go
// Fan-out: distribute from one channel to multiple goroutines
// Fan-in:  merge multiple channels into one channel

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
    // Create the input channel
    input := make(chan int, 100)
    go func() {
        defer close(input)
        for i := 0; i < 1000; i++ {
            input <- i
        }
    }()

    // Fan-out: distribute across 5 workers
    outputs := fanOut(input, 5)

    // Fan-in: merge the results
    results := fanIn(outputs...)

    // Consume the results
    for result := range results {
        fmt.Println(result)
    }
}
```

### Pattern 2: Pipeline

```go
// The pipeline pattern chains stages together
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

    // Build the pipeline: generate -> square -> filter
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

### Pattern 3: Semaphore (Concurrency Limiting)

```go
// Use a buffered channel as a semaphore
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

// Usage: HTTP requests with up to 10 concurrent
func fetchConcurrently(ctx context.Context, urls []string) []Result {
    sem := NewSemaphore(10) // Up to 10 concurrent
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

### Pattern 4: Concurrency via errgroup

```go
import "golang.org/x/sync/errgroup"

// errgroup = WaitGroup + error aggregation + context integration
func processOrders(ctx context.Context, orders []Order) error {
    g, ctx := errgroup.WithContext(ctx)

    // Concurrency limit (Go 1.20+)
    g.SetLimit(5)

    for _, order := range orders {
        order := order // Local copy required before Go 1.21
        g.Go(func() error {
            // ctx is automatically linked
            // If any goroutine returns an error,
            // ctx is canceled and the other goroutines also stop
            return processOrder(ctx, order)
        })
    }

    return g.Wait() // Returns the first error
}

// Running multiple kinds of processing concurrently with errgroup
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

### Pattern 5: Or-Done Channel

```go
// Use the result of whichever of multiple operations completes first
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

    // Return the first successful result
    var lastErr error
    for i := 0; i < len(fns); i++ {
        r := <-ch
        if r.err == nil {
            cancel() // Cancel the other goroutines
            return r.val, nil
        }
        lastErr = r.err
    }

    return "", fmt.Errorf("all attempts failed, last error: %w", lastErr)
}

// Usage: get the result from the fastest DNS server
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

### Pattern 6: Ticker and Periodic Processing

```go
// Periodic background processing
func startPeriodicTask(ctx context.Context, interval time.Duration, task func(context.Context) error) {
    ticker := time.NewTicker(interval)
    defer ticker.Stop()

    // Run once right after startup as well
    if err := task(ctx); err != nil {
        log.Printf("Initial run error: %v", err)
    }

    for {
        select {
        case <-ticker.C:
            if err := task(ctx); err != nil {
                log.Printf("Periodic task error: %v", err)
            }
        case <-ctx.Done():
            log.Println("Periodic task stopped")
            return
        }
    }
}

// Usage
func main() {
    ctx, cancel := context.WithCancel(context.Background())

    go startPeriodicTask(ctx, 30*time.Second, func(ctx context.Context) error {
        return cleanupExpiredSessions(ctx)
    })

    go startPeriodicTask(ctx, 5*time.Minute, func(ctx context.Context) error {
        return reportMetrics(ctx)
    })

    // Stop on SIGTERM
    sigCh := make(chan os.Signal, 1)
    signal.Notify(sigCh, syscall.SIGTERM, syscall.SIGINT)
    <-sigCh
    cancel()
    time.Sleep(time.Second) // Wait for cleanup
}
```

---

## 7. Detecting and Preventing goroutine Leaks

### Typical Cases of goroutine Leaks

```go
// Case 1: Sending to a channel that is never received from
func leakySearch(query string) string {
    ch := make(chan string)
    go func() { ch <- searchAPI1(query) }()
    go func() { ch <- searchAPI2(query) }()
    return <-ch // Only one is received. The other goroutine leaks
}

// Fix: buffered channel
func safeSearch(query string) string {
    ch := make(chan string, 2) // All goroutines can send
    go func() { ch <- searchAPI1(query) }()
    go func() { ch <- searchAPI2(query) }()
    return <-ch
}

// Case 2: Generating a channel that is never consumed
func leakyProducer() <-chan int {
    ch := make(chan int)
    go func() {
        i := 0
        for {
            ch <- i // Blocks forever once there is no consumer
            i++
        }
    }()
    return ch
}

// Fix: cancel via context
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
                return // Guaranteed termination on cancellation
            }
        }
    }()
    return ch
}

// Case 3: Blocking forever waiting on a lock
func leakyLock() {
    var mu sync.Mutex
    mu.Lock()
    go func() {
        mu.Lock()   // Deadlock: the parent goroutine never unlocks
        defer mu.Unlock()
        doWork()
    }()
    // mu.Unlock() is missing
}
```

### Methods for Detecting goroutine Leaks

```go
// Method 1: monitor with runtime.NumGoroutine()
func monitorGoroutines(ctx context.Context) {
    ticker := time.NewTicker(10 * time.Second)
    defer ticker.Stop()

    baseline := runtime.NumGoroutine()

    for {
        select {
        case <-ticker.C:
            current := runtime.NumGoroutine()
            if current > baseline*2 {
                log.Printf("WARNING: Suspected goroutine leak: baseline=%d, current=%d",
                    baseline, current)
                // Dump the stack trace
                buf := make([]byte, 1<<20)
                n := runtime.Stack(buf, true)
                log.Printf("Stack trace:\n%s", buf[:n])
            }
        case <-ctx.Done():
            return
        }
    }
}

// Method 2: leak detection in tests (go.uber.org/goleak)
import "go.uber.org/goleak"

func TestMain(m *testing.M) {
    goleak.VerifyTestMain(m)
}

func TestNoLeak(t *testing.T) {
    defer goleak.VerifyNone(t)

    ctx, cancel := context.WithCancel(context.Background())
    ch := safeProducer(ctx)

    // Consume a few
    for i := 0; i < 10; i++ {
        <-ch
    }

    cancel() // Ensure the goroutine stops
}
```

---

## 8. ASCII Diagrams

### Figure 1: Relationship Between goroutines and OS Threads (M:N Scheduling)

```
┌─────────────── Go Runtime ──────────────────┐
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

### Figure 2: Unbuffered vs Buffered Channels

```
Unbuffered (make(chan int)):
  Sender ──[sync]──> Receiver
  Send blocks until the receiver is ready

  Sender    Channel    Receiver
    │                    │
    ├──── send ────>     │ (blocked)
    │              ├──── recv
    │              │
    │   (sync complete)  │

Buffered (make(chan int, 3)):
  Sender ──[buf]──[buf]──[buf]──> Receiver
  Blocks only when the buffer is full

  Sender    Buffer[3]    Receiver
    │      [_][_][_]       │
    ├─send─>[A][_][_]      │
    ├─send─>[A][B][_]      │
    │       [A][B][_]──recv─┤  → A
    │       [_][B][_]       │
```

### Figure 3: How the select Statement Works

```
select {
case msg := <-ch1:     ┐
case msg := <-ch2:     ├── Execute the ready case
case ch3 <- value:     │   If multiple are ready, pick randomly
default:               ┘   default avoids blocking
}

       ┌─ ch1 ready? ──YES──> run case 1
       │
select─┼─ ch2 ready? ──YES──> run case 2
       │
       ├─ ch3 ready? ──YES──> run case 3
       │
       └─ none ready ──────> run default
                              (block if no default)
```

### Figure 4: Fan-out / Fan-in Pattern

```
Fan-out (distribute from one source to multiple workers):

            ┌──> Worker1 ──┐
  Source ───┼──> Worker2 ──┼──> Merged Output
            ├──> Worker3 ──┤
            └──> Worker4 ──┘

Pipeline (chain of stages):

  Generate ──> Transform ──> Filter ──> Consume
    (chan)       (chan)        (chan)

Or-Done (adopt the first result):

  Query DNS A ──┐
  Query DNS B ──┼──> First Result
  Query DNS C ──┘
```

### Figure 5: goroutine Lifecycle

```
┌─────────────────────────────────────────────┐
│             goroutine Lifecycle              │
│                                             │
│  ┌──────────┐    ┌──────────┐               │
│  │ Runnable │───>│ Running  │               │
│  │ (waiting)│    │(executing)│              │
│  └──────────┘    └────┬─────┘               │
│       ▲               │                     │
│       │               ├──> I/O wait          │
│       │               │    → Waiting state   │
│       │               │    → Runnable on I/O completion│
│       │               │                     │
│       │               ├──> channel wait      │
│       │               │    → Waiting state   │
│       │               │    → Runnable on send/recv │
│       │               │                     │
│       │               ├──> preemption        │
│       │               │    → back to Runnable│
│       │               │                     │
│       └───────────────┘                     │
│                       │                     │
│                       ▼                     │
│               ┌──────────┐                  │
│               │   Dead   │                  │
│               │(terminated)│                │
│               └──────────┘                  │
└─────────────────────────────────────────────┘
```

---

## 9. Comparison Tables

### Table 1: Kinds of Channels

| Kind | Syntax | Send blocks when | Receive blocks when | Use case |
|------|--------|------------------|---------------------|----------|
| Unbuffered | `make(chan T)` | No receiver present | No sender present | Synchronous communication |
| Buffered | `make(chan T, n)` | Buffer full | Buffer empty | Asynchronous communication |
| Send-only | `chan<- T` | Same as above | Compile error | API constraint |
| Receive-only | `<-chan T` | Compile error | Same as above | API constraint |

### Table 2: goroutine vs OS Thread vs async/await

| Item | goroutine | OS Thread | async/await (JS) |
|------|-----------|-----------|------------------|
| Initial stack size | ~2KB | ~1MB | N/A |
| Context switch cost | Low (user space) | High (kernel) | Low (event loop) |
| Practical concurrency | Hundreds of thousands to millions | Thousands | Tens of thousands |
| Scheduling | Go runtime (M:N) | OS | Event loop |
| Memory model | happens-before | OS-dependent | Single-threaded |
| Blocking I/O | Threads added automatically | Thread occupied | Not supported (different mechanism needed) |

### Table 3: Guidelines for Choosing Concurrency Patterns

| Pattern | Use case | Complexity | Number of goroutines |
|---------|----------|------------|---------------------|
| WaitGroup | Wait for all to complete | Low | Known |
| errgroup | Concurrent processing with errors | Low | Known |
| Fan-out/Fan-in | Parallel processing and result aggregation | Medium | Number of workers |
| Pipeline | Stage-based processing | Medium | Number of stages |
| Semaphore | Concurrency limit | Low | Limited |
| Or-Done | Adopt fastest result | Medium | Number of candidates |
| Worker Pool | Continuous job processing | Medium | Fixed |

### Table 4: Results of Channel Operations

| Operation | nil channel | Closed | Buffer empty/full | Normal |
|-----------|-------------|--------|-------------------|--------|
| Send `ch <-` | Blocks forever | **panic** | Blocks (full) | Sends |
| Receive `<-ch` | Blocks forever | Zero value, false | Blocks (empty) | Receives |
| close | **panic** | **panic** | Remaining data still receivable | Closes |
| len | 0 | Items in buffer | Items in buffer | Items in buffer |
| cap | 0 | Buffer size | Buffer size | Buffer size |

---

## 10. Anti-Patterns

### Anti-Pattern 1: goroutine Leak

```go
// BAD: The channel is never received from, so the goroutine blocks forever
func leakySearch(query string) string {
    ch := make(chan string)
    go func() { ch <- searchAPI1(query) }()
    go func() { ch <- searchAPI2(query) }()
    return <-ch // Only one is received; the other goroutine leaks
}

// GOOD: Use context cancellation or a buffered channel
func safeSearch(ctx context.Context, query string) string {
    ch := make(chan string, 2) // Buffer allows all goroutines to send
    go func() { ch <- searchAPI1(query) }()
    go func() { ch <- searchAPI2(query) }()
    return <-ch
}
```

### Anti-Pattern 2: Capturing Loop Variables (before Go 1.21)

```go
// BAD (before Go 1.21): loop variable is shared
for _, url := range urls {
    go func() {
        fetch(url) // All goroutines reference the last url
    }()
}

// GOOD: pass as a parameter (not needed from Go 1.22 onward)
for _, url := range urls {
    url := url // Local copy
    go func() {
        fetch(url)
    }()
}
```

### Anti-Pattern 3: Synchronizing with time.Sleep

```go
// BAD: Wait for goroutine completion with time.Sleep
func badSync() {
    go processData()
    time.Sleep(5 * time.Second) // It is not guaranteed to finish in 5 seconds
}

// GOOD: Synchronize with WaitGroup or a channel
func goodSync() {
    done := make(chan struct{})
    go func() {
        defer close(done)
        processData()
    }()
    <-done // Reliably wait for completion
}
```

### Anti-Pattern 4: Overusing Channels

```go
// BAD: Using a channel for a simple counter
type Counter struct {
    ch chan int
}
func (c *Counter) Inc() {
    c.ch <- 1
}
func (c *Counter) Value() int {
    return <-c.ch
}

// GOOD: Use atomic for a simple counter
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

## Practical Exercises

### Exercise 1: Basic Implementation

Implement code that satisfies the following requirements.

**Requirements:**
- Validate input data
- Implement proper error handling
- Write test code as well

```python
# Exercise 1: Basic implementation template
class Exercise1:
    """Exercise for basic implementation patterns"""

    def __init__(self):
        self.data = []

    def validate_input(self, value):
        """Validate input value"""
        if value is None:
            raise ValueError("Input value is None")
        return True

    def process(self, value):
        """Main data processing logic"""
        self.validate_input(value)
        self.data.append(value)
        return self.data

    def get_results(self):
        """Get processing results"""
        return {
            'count': len(self.data),
            'data': self.data
        }

# Tests
def test_exercise1():
    ex = Exercise1()
    assert ex.process(1) == [1]
    assert ex.process(2) == [1, 2]
    assert ex.get_results()['count'] == 2

    try:
        ex.process(None)
        assert False, "Should have raised an exception"
    except ValueError:
        pass

    print("All tests passed!")

test_exercise1()
```

### Exercise 2: Advanced Patterns

Extend the basic implementation by adding the following features.

```python
# Exercise 2: Advanced patterns
from typing import List, Dict, Optional
from datetime import datetime

class AdvancedExercise:
    """Exercise for advanced patterns"""

    def __init__(self, max_size: int = 100):
        self._items: List[Dict] = []
        self._max_size = max_size
        self._created_at = datetime.now()

    def add(self, key: str, value: any) -> bool:
        """Add item (with size limit)"""
        if len(self._items) >= self._max_size:
            return False
        self._items.append({
            'key': key,
            'value': value,
            'timestamp': datetime.now().isoformat()
        })
        return True

    def find(self, key: str) -> Optional[Dict]:
        """Search by key"""
        for item in reversed(self._items):
            if item['key'] == key:
                return item
        return None

    def remove(self, key: str) -> bool:
        """Remove by key"""
        for i, item in enumerate(self._items):
            if item['key'] == key:
                self._items.pop(i)
                return True
        return False

    def stats(self) -> Dict:
        """Get statistics"""
        return {
            'total_items': len(self._items),
            'max_size': self._max_size,
            'usage_percent': len(self._items) / self._max_size * 100,
            'uptime': str(datetime.now() - self._created_at)
        }

# Tests
def test_advanced():
    ex = AdvancedExercise(max_size=3)
    assert ex.add("a", 1) == True
    assert ex.add("b", 2) == True
    assert ex.add("c", 3) == True
    assert ex.add("d", 4) == False  # Size limit
    assert ex.find("b")['value'] == 2
    assert ex.remove("b") == True
    assert ex.find("b") is None
    stats = ex.stats()
    assert stats['total_items'] == 2
    print("All advanced tests passed!")

test_advanced()
```

### Exercise 3: Performance Optimization

Improve the performance of the following code.

```python
# Exercise 3: Performance optimization
import time
from functools import lru_cache

# Before optimization (O(n^2))
def slow_search(data: list, target: int) -> int:
    """Inefficient search"""
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] + data[j] == target:
                return (i, j)
    return (-1, -1)

# After optimization (O(n))
def fast_search(data: list, target: int) -> tuple:
    """Efficient search using a hash map"""
    seen = {}
    for i, num in enumerate(data):
        complement = target - num
        if complement in seen:
            return (seen[complement], i)
        seen[num] = i
    return (-1, -1)

# Benchmark
def benchmark():
    import random
    data = list(range(5000))
    random.shuffle(data)
    target = data[100] + data[4000]

    start = time.time()
    result1 = slow_search(data, target)
    slow_time = time.time() - start

    start = time.time()
    result2 = fast_search(data, target)
    fast_time = time.time() - start

    print(f"Inefficient version: {slow_time:.4f}s")
    print(f"Efficient version:   {fast_time:.6f}s")
    print(f"Speedup: {slow_time/fast_time:.0f}x")

benchmark()
```

**Key points:**
- Be aware of algorithmic complexity
- Choose appropriate data structures
- Measure the effect with benchmarks

---

## Troubleshooting

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| Initialization error | Configuration file issue | Verify config file path and format |
| Timeout | Network latency / insufficient resources | Adjust timeout values, add retry logic |
| Out of memory | Growing data volume | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Check executing user's permissions, review settings |
| Data inconsistency | Concurrent processing conflict | Introduce locking, manage transactions |

### Debugging Steps

1. **Check error messages**: Read the stack trace and identify where it occurs
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Form hypotheses**: List possible causes
4. **Verify step by step**: Use logging and debuggers to verify hypotheses
5. **Fix and regression test**: After fixing, run related tests too

```python
# Debugging utility
import logging
import traceback
from functools import wraps

# Logger setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """Decorator that logs a function's input and output"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Call: {func.__name__}(args={args}, kwargs={kwargs})")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Return: {func.__name__} -> {result}")
            return result
        except Exception as e:
            logger.error(f"Exception: {func.__name__}: {e}")
            logger.error(traceback.format_exc())
            raise
    return wrapper

@debug_decorator
def process_data(items):
    """Data processing (debug target)"""
    if not items:
        raise ValueError("Empty data")
    return [item * 2 for item in items]
```

### Diagnosing Performance Issues

Steps for diagnosing performance problems:

1. **Identify the bottleneck**: Measure with profiling tools
2. **Check memory usage**: Check for memory leaks
3. **Check I/O waits**: Check disk and network I/O status
4. **Check concurrent connections**: Check connection pool state

| Problem type | Diagnostic tool | Countermeasure |
|--------------|-----------------|----------------|
| CPU load | cProfile, py-spy | Algorithm improvement, parallelization |
| Memory leak | tracemalloc, objgraph | Properly release references |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexing, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

The criteria for making technology choices are summarized below.

| Criterion | When to prioritize | When it can be compromised |
|-----------|-------------------|----------------------------|
| Performance | Real-time processing, large-scale data | Admin panels, batch processing |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Services expected to grow | Internal tools, fixed user base |
| Security | Personal info, financial data | Public data, internal use |
| Development speed | MVP, time-to-market | Quality focus, mission-critical |

### Choosing an Architecture Pattern

```
┌─────────────────────────────────────────────────┐
│           Architecture Selection Flow             │
├─────────────────────────────────────────────────┤
│                                                 │
│  1) What is your team size?                     │
│    ├─ Small (1-5) → Monolith                    │
│    └─ Large (10+) → Go to 2)                    │
│                                                 │
│  2) Deployment frequency?                       │
│    ├─ Weekly or less → Monolith + modularization │
│    └─ Daily/multiple → Go to 3)                 │
│                                                 │
│  3) Inter-team independence?                    │
│    ├─ High → Microservices                      │
│    └─ Moderate → Modular monolith               │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Trade-off Analysis

Technical decisions always involve trade-offs. Analyze them from the following perspectives:

**1. Short-term vs long-term cost**
- A short-term fast approach can become long-term technical debt
- Conversely, over-engineering is costly short-term and can delay the project

**2. Consistency vs flexibility**
- A unified tech stack has a low learning cost
- Adopting a variety of technologies allows best-fit choices but increases operational cost

**3. Level of abstraction**
- High abstraction is highly reusable but can make debugging harder
- Low abstraction is intuitive but tends to duplicate code

```python
# Template for recording design decisions
class ArchitectureDecisionRecord:
    """Creation of an ADR (Architecture Decision Record)"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """Describe background and problem"""
        self.context = context
        return self

    def set_decision(self, decision: str):
        """Describe the decision"""
        self.decision = decision
        return self

    def add_consequence(self, consequence: str, positive: bool = True):
        """Add a consequence"""
        self.consequences.append({
            'description': consequence,
            'type': 'positive' if positive else 'negative'
        })
        return self

    def add_alternative(self, name: str, reason_rejected: str):
        """Add a rejected alternative"""
        self.alternatives.append({
            'name': name,
            'reason_rejected': reason_rejected
        })
        return self

    def to_markdown(self) -> str:
        """Output in Markdown format"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## Background\n{self.context}\n\n"
        md += f"## Decision\n{self.decision}\n\n"
        md += "## Consequences\n"
        for c in self.consequences:
            icon = "[+]" if c['type'] == 'positive' else "[!]"
            md += f"- {icon} {c['description']}\n"
        md += "\n## Rejected Alternatives\n"
        for a in self.alternatives:
            md += f"- **{a['name']}**: {a['reason_rejected']}\n"
        return md
```

---

## Real-World Application Scenarios

### Scenario 1: MVP Development at a Startup

**Situation:** You need to release a product quickly with limited resources.

**Approach:**
- Choose a simple architecture
- Focus on the minimum necessary features
- Automated tests only for the critical path
- Introduce monitoring early

**Lessons learned:**
- Don't strive for perfection (YAGNI principle)
- Obtain user feedback early
- Manage technical debt consciously

### Scenario 2: Modernizing a Legacy System

**Situation:** Gradually renewing a system that has been running for more than 10 years.

**Approach:**
- Migrate gradually with the Strangler Fig pattern
- If no tests exist, write Characterization Tests first
- Use an API gateway to let old and new systems coexist
- Migrate data in phases

| Phase | Work | Duration | Risk |
|-------|------|----------|------|
| 1. Investigation | Analyze current state, map dependencies | 2-4 weeks | Low |
| 2. Foundation | Build CI/CD, test environment | 4-6 weeks | Low |
| 3. Migration start | Migrate peripheral features first | 3-6 months | Medium |
| 4. Core migration | Migrate core features | 6-12 months | High |
| 5. Completion | Retire the old system | 2-4 weeks | Medium |

### Scenario 3: Development with a Large Team

**Situation:** 50+ engineers working on the same product.

**Approach:**
- Clarify boundaries with domain-driven design
- Assign ownership per team
- Manage shared libraries with an Inner Source approach
- Design API-first to minimize inter-team dependencies

```python
# Defining API contracts between teams
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class APIContract:
    """API contract between teams"""
    endpoint: str
    method: str
    owner_team: str
    consumers: List[str]
    sla_ms: int  # Response time SLA
    priority: Priority

    def validate_sla(self, actual_ms: int) -> bool:
        """Check SLA compliance"""
        return actual_ms <= self.sla_ms

    def to_openapi(self) -> dict:
        """Output in OpenAPI format"""
        return {
            'path': self.endpoint,
            'method': self.method,
            'x-owner': self.owner_team,
            'x-consumers': self.consumers,
            'x-sla-ms': self.sla_ms
        }

# Usage
contracts = [
    APIContract(
        endpoint="/api/v1/users",
        method="GET",
        owner_team="user-team",
        consumers=["order-team", "notification-team"],
        sla_ms=200,
        priority=Priority.HIGH
    ),
    APIContract(
        endpoint="/api/v1/orders",
        method="POST",
        owner_team="order-team",
        consumers=["payment-team", "inventory-team"],
        sla_ms=500,
        priority=Priority.CRITICAL
    )
]
```

### Scenario 4: A Performance-Critical System

**Situation:** A system that requires millisecond-level response times.

**Optimization points:**
1. Caching strategy (L1: in-memory, L2: Redis, L3: CDN)
2. Use of asynchronous processing
3. Connection pooling
4. Query optimization and index design

| Optimization | Effect | Implementation cost | When to apply |
|--------------|--------|---------------------|---------------|
| In-memory cache | High | Low | Frequently accessed data |
| CDN | High | Low | Static content |
| Asynchronous processing | Medium | Medium | I/O-heavy processing |
| DB optimization | High | High | When queries are slow |
| Code optimization | Low-Medium | High | When CPU-bound |
---

## 11. FAQ

### Q1: How many goroutines can you start?

Theoretically, millions. Each goroutine starts with about a 2KB stack, which grows dynamically as needed. However, CPU-bound work will not exceed the parallelism of `runtime.GOMAXPROCS` (which defaults to the number of CPUs). For I/O-bound work, starting many goroutines makes sense. Even 1 million goroutines can be started with about 2GB of memory, but scheduling overhead increases.

### Q2: Should you use channels or Mutex?

Rule of thumb: "Use a channel to transfer ownership of data; use a Mutex to protect shared state." The Go Proverb says "Don't communicate by sharing memory; share memory by communicating," but a Mutex is appropriate for simple counters or caches. Channels feel natural for pipelines and event notifications. Note that when performance matters, channels are slower than Mutex (they use a Mutex internally as well).

### Q3: What happens if you send to a closed channel?

A panic occurs. The "sender" should close the channel, and the "receiver" detects it via range or the ok check. When there are multiple senders, protect closing with sync.Once. The receiver should not close the channel.

### Q4: Should you change GOMAXPROCS?

Usually not. It defaults to the number of CPU cores and is optimal for most workloads. However, in container environments (Docker/Kubernetes), the host CPU count may be visible. Using the `uber-go/automaxprocs` package automatically adjusts it to the CPUs allocated to the container.

### Q5: How do you stop a goroutine from outside?

Go provides no way to forcibly stop a goroutine from outside. Instead, use a design pattern where the goroutine cooperatively checks the cancellation signal from a `context.Context`. Inside the goroutine, periodically check the `ctx.Done()` channel and return if canceled. This is an intentional design decision that guarantees safe resource cleanup.

### Q6: What are nil channels used for?

Sends and receives on a nil channel block forever. This can be used to dynamically disable a specific case in a select statement. For example, if you want to temporarily stop receiving from a channel, setting that channel variable to nil means its case will no longer be selected by select.

```go
func dynamicSelect(ch1, ch2 <-chan int) {
    for ch1 != nil || ch2 != nil {
        select {
        case v, ok := <-ch1:
            if !ok {
                ch1 = nil // Disable ch1
                continue
            }
            fmt.Println("ch1:", v)
        case v, ok := <-ch2:
            if !ok {
                ch2 = nil // Disable ch2
                continue
            }
            fmt.Println("ch2:", v)
        }
    }
}
```

### Q7: Should you use for-range over channel or for-select?

`for v := range ch` is optimal for receiving from a single channel, and the loop terminates automatically when the channel is closed. On the other hand, `for-select` is used when waiting on multiple channels simultaneously, or when you need to detect context cancellation at the same time. Even with a single channel, choose for-select if you need context cancellation.

---


## FAQ

### Q1: What is the most important point to focus on when learning this topic?

Gaining practical experience is the most important thing. Understanding deepens not just through theory, but by actually writing and running code to see how things work.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to advanced topics. We recommend solidly understanding the fundamental concepts explained in this guide before moving on to the next step.

### Q3: How is this applied in real-world development?

The knowledge from this topic is frequently applied in day-to-day development work. It becomes especially important during code reviews and architecture design.

---

## Summary

| Concept | Key points |
|---------|------------|
| goroutine | Start with `go f()`. Lightweight (~2KB), M:N scheduling |
| GMP model | Three-layer structure of Goroutine-Machine-Processor |
| channel | Type-safe communication path. Two kinds: buffered/unbuffered |
| Direction constraints | `chan<-` send-only, `<-chan` receive-only |
| select | Wait on multiple channels. Non-deterministic selection |
| WaitGroup | Wait for a group of goroutines to complete |
| errgroup | WaitGroup + error aggregation + context integration |
| close | Close a channel. Done by the sender |
| range over channel | Loop-receive until closed |
| Fan-out/Fan-in | Parallel distribution and result aggregation |
| Pipeline | Processing flow of chained stages |
| Semaphore | Limit concurrent executions with a channel |

---

## Recommended Next Reads

- [01-sync-primitives.md](./01-sync-primitives.md) -- Synchronization primitives like Mutex/atomic
- [02-concurrency-patterns.md](./02-concurrency-patterns.md) -- Concurrency patterns such as Fan-out/Fan-in
- [03-context.md](./03-context.md) -- Cancellation control with Context

---

## References

1. **Go Blog, "Share Memory By Communicating"** -- https://go.dev/blog/codelab-share
2. **Go Blog, "Go Concurrency Patterns"** -- https://go.dev/blog/concurrency-patterns
3. **Go Blog, "Advanced Go Concurrency Patterns"** -- https://go.dev/blog/io2013-talk-concurrency
4. **Go Blog, "Go Concurrency Patterns: Pipelines and cancellation"** -- https://go.dev/blog/pipelines
5. **Hoare, C.A.R. (1978) "Communicating Sequential Processes"** -- https://www.cs.cmu.edu/~crary/819-f09/Hoare78.pdf
6. **golang.org/x/sync/errgroup** -- https://pkg.go.dev/golang.org/x/sync/errgroup
7. **uber-go/goleak** -- https://github.com/uber-go/goleak
