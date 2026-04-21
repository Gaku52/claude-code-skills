# Go Profiling Guide

> Use pprof and trace to identify bottlenecks in Go applications and optimize performance

## What You'll Learn in This Chapter

1. **pprof** techniques for CPU, memory, and goroutine profiling
2. **runtime/trace** for visualizing goroutine scheduling and latency
3. **Benchmark integration** — how to acquire profiles from tests and run an optimization cycle
4. **Mutex/Block profiling** — analyzing lock contention and blocking operations
5. **Continuous profiling** — strategies for constant monitoring in production


## Prerequisites

Reading this guide is easier if you have the following knowledge:

- Basic programming knowledge
- Understanding of related fundamental concepts
- Understanding the content of [Go Generics Guide](./01-generics.md)

---

## 1. Overview of Go Profiling

### Classification of Profiling Tools

```
+----------------------------------------------------------+
|                  Go Profiling System                     |
+----------------------------------------------------------+
|                                                          |
|  +-----------------+  +------------------+  +-----------+|
|  | CPU Profile     |  | Memory Profile   |  | Trace     ||
|  | Where time is   |  | Where memory is  |  | When what ||
|  | being spent     |  | being allocated  |  | happened  ||
|  +-----------------+  +------------------+  +-----------+|
|         |                     |                   |      |
|         v                     v                   v      |
|  go tool pprof         go tool pprof       go tool trace |
|                                                          |
|  +-----------------+  +------------------+               |
|  | Goroutine Prof  |  | Block Profile    |               |
|  | Check the       |  | Analyze lock     |               |
|  | state of        |  | waits            |               |
|  | goroutines      |  |                  |               |
|  +-----------------+  +------------------+               |
|                                                          |
|  +-----------------+  +------------------+               |
|  | Mutex Profile   |  | Threadcreate     |               |
|  | Analyze mutex   |  | Track OS thread  |               |
|  | contention      |  | creation         |               |
|  +-----------------+  +------------------+               |
+----------------------------------------------------------+
```

### Choosing How to Acquire a Profile

```
Want to take a profile
        |
        +-- Production server (always running)
        |       |
        |       v
        |   net/http/pprof (HTTP endpoint)
        |
        +-- Tests / benchmarks
        |       |
        |       v
        |   go test -cpuprofile / -memprofile
        |
        +-- Short-lived programs (CLI, etc.)
        |       |
        |       v
        |   runtime/pprof (start/stop within the program)
        |
        +-- Continuous monitoring
                |
                v
            Pyroscope / Parca / Google Cloud Profiler
```

### Basic Profiling Flow

```
+----------------------------------------------------------+
|  Performance Optimization Cycle                          |
+----------------------------------------------------------+
|                                                          |
|  1. Measure                                              |
|     |  Quantify the current state via benchmarks /      |
|     |  load tests                                       |
|     v                                                    |
|  2. Profile                                              |
|     |  Identify hotspots with pprof                     |
|     v                                                    |
|  3. Analyze                                              |
|     |  Understand causes via flame graphs / call graphs |
|     v                                                    |
|  4. Optimize                                             |
|     |  Improve only the bottleneck                      |
|     v                                                    |
|  5. Verify                                               |
|     |  Quantitatively confirm the effect via benchmarks |
|     v                                                    |
|  6. Go back to 1 (repeat if improvement is insufficient)|
+----------------------------------------------------------+
```

---

## 2. net/http/pprof — Profiling HTTP Servers

### Code Example 1: Adding pprof Endpoints

```go
package main

import (
    "log"
    "net/http"
    _ "net/http/pprof" // Side-effect import registers endpoints
)

func main() {
    // Application routing
    mux := http.NewServeMux()
    mux.HandleFunc("/api/users", handleUsers)

    // pprof is registered on DefaultServeMux, so start a
    // dedicated pprof server on a separate port (recommended for prod)
    go func() {
        log.Println("pprof server: http://localhost:6060/debug/pprof/")
        log.Fatal(http.ListenAndServe(":6060", nil))
    }()

    // Application server
    log.Fatal(http.ListenAndServe(":8080", mux))
}
```

### Code Example 2: Registering pprof on a Custom mux

```go
package main

import (
    "net/http"
    "net/http/pprof"
    "log"
)

func main() {
    // mux for the application
    appMux := http.NewServeMux()
    appMux.HandleFunc("/api/users", handleUsers)

    // Dedicated mux for pprof (with authentication)
    debugMux := http.NewServeMux()
    debugMux.HandleFunc("/debug/pprof/", pprof.Index)
    debugMux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
    debugMux.HandleFunc("/debug/pprof/profile", pprof.Profile)
    debugMux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
    debugMux.HandleFunc("/debug/pprof/trace", pprof.Trace)

    // Apply authentication middleware
    protectedDebug := basicAuth(debugMux, "admin", "secret-password")

    go func() {
        log.Println("pprof server (auth required): http://localhost:6060/debug/pprof/")
        log.Fatal(http.ListenAndServe("127.0.0.1:6060", protectedDebug))
    }()

    log.Fatal(http.ListenAndServe(":8080", appMux))
}

func basicAuth(next http.Handler, username, password string) http.Handler {
    return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        user, pass, ok := r.BasicAuth()
        if !ok || user != username || pass != password {
            w.Header().Set("WWW-Authenticate", `Basic realm="pprof"`)
            http.Error(w, "Unauthorized", http.StatusUnauthorized)
            return
        }
        next.ServeHTTP(w, r)
    })
}
```

### List of pprof Endpoints

| Endpoint | Content | How to Retrieve |
|--------------|------|---------|
| `/debug/pprof/` | Profile list page | Access directly in a browser |
| `/debug/pprof/profile?seconds=30` | CPU profile (30 seconds) | `go tool pprof URL` |
| `/debug/pprof/heap` | Heap memory profile | `go tool pprof URL` |
| `/debug/pprof/allocs` | Cumulative memory allocations | `go tool pprof URL` |
| `/debug/pprof/goroutine` | Goroutine stack traces | `go tool pprof URL` |
| `/debug/pprof/block` | Blocking operation profile | `go tool pprof URL` |
| `/debug/pprof/mutex` | Mutex contention profile | `go tool pprof URL` |
| `/debug/pprof/threadcreate` | OS thread creation profile | `go tool pprof URL` |
| `/debug/pprof/trace?seconds=5` | Execution trace (5 seconds) | `go tool trace` |

### Query Parameters for pprof Endpoints

| Parameter | Applies to | Description | Example |
|-----------|--------|------|-----|
| `seconds` | profile, trace | Sampling period (seconds) | `?seconds=60` |
| `debug` | goroutine, heap, etc. | Text output mode (0=binary, 1=text, 2=detailed) | `?debug=2` |
| `gc` | heap | Run GC before profiling (1=run) | `?gc=1` |

---

## 3. CPU Profiling

### Code Example 3: Using go tool pprof

```bash
# Acquire a CPU profile (30-second sampling)
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=30

# Analyze a saved profile
go tool pprof cpu.prof

# Display in Web UI (opens browser)
go tool pprof -http=:8081 cpu.prof

# Focus on a specific function
go tool pprof -focus=handleRequest cpu.prof

# Exclude a specific function
go tool pprof -ignore=runtime cpu.prof

# Text output (for CI)
go tool pprof -text cpu.prof

# Show the diff between two profiles
go tool pprof -diff_base=before.prof after.prof
```

### pprof Interactive Mode

```bash
(pprof) top10
Showing nodes accounting for 4.5s, 90% of 5s total
      flat  flat%   sum%  cum   cum%
      2.0s 40.00% 40.00%  2.0s 40.00%  runtime.memmove
      1.0s 20.00% 60.00%  1.5s 30.00%  encoding/json.(*decodeState).object
      0.5s 10.00% 70.00%  0.5s 10.00%  runtime.mallocgc
      ...

(pprof) list encoding/json.(*decodeState).object
# Display the cost of each line with source code

(pprof) web
# Display the SVG call graph in a browser

(pprof) peek handleRequest
# Display callers/callees of handleRequest

(pprof) tree
# Display in call tree format

(pprof) disasm handleRequest
# Display profile with assembly code
```

### The Difference Between flat and cum

```
+----------------------------------------------------------+
|  Understanding flat vs cum                                |
+----------------------------------------------------------+
|                                                          |
|  func A() {          flat of A = 1s (A's own work)       |
|    doWork() // 1s    cum of A  = 4s (A + B + C total)    |
|    B()      // 3s                                        |
|  }                                                       |
|                                                          |
|  func B() {          flat of B = 1s (B's own work)       |
|    doWork() // 1s    cum of B  = 3s (B + C total)        |
|    C()      // 2s                                        |
|  }                                                       |
|                                                          |
|  func C() {          flat of C = 2s (C's own work)       |
|    doWork() // 2s    cum of C  = 2s (C only)             |
|  }                                                       |
|                                                          |
|  In the top command:                                     |
|  High flat  -> the function itself is heavy              |
|  High cum   -> the function's callees are heavy          |
|  Big gap between flat and cum -> cause is downstream     |
+----------------------------------------------------------+
```

### Code Example 4: Acquiring a CPU Profile from Within the Program

```go
package main

import (
    "flag"
    "log"
    "os"
    "runtime/pprof"
)

var cpuprofile = flag.String("cpuprofile", "", "Output destination for CPU profile")
var memprofile = flag.String("memprofile", "", "Output destination for memory profile")

func main() {
    flag.Parse()

    // Start CPU profile
    if *cpuprofile != "" {
        f, err := os.Create(*cpuprofile)
        if err != nil {
            log.Fatal(err)
        }
        defer f.Close()

        if err := pprof.StartCPUProfile(f); err != nil {
            log.Fatal(err)
        }
        defer pprof.StopCPUProfile()
    }

    // Work to be profiled
    doHeavyWork()

    // Acquire memory profile
    if *memprofile != "" {
        f, err := os.Create(*memprofile)
        if err != nil {
            log.Fatal(err)
        }
        defer f.Close()

        // Run GC to capture the latest memory state
        runtime.GC()
        if err := pprof.WriteHeapProfile(f); err != nil {
            log.Fatal(err)
        }
    }
}
```

### Code Example 5: HTTP Server with Profiling (Conditionally Enabled)

```go
package main

import (
    "log"
    "net/http"
    "os"
    "runtime"
)

func main() {
    // Enable pprof via an environment variable
    if os.Getenv("ENABLE_PPROF") == "true" {
        // Enable Block/Mutex profiling
        runtime.SetBlockProfileRate(1)
        runtime.SetMutexProfileFraction(1)

        // Adjust the sampling rate for memory profiling
        // Default: once every 512KB
        // For more detail: runtime.MemProfileRate = 1 (record every allocation)
        // Production: runtime.MemProfileRate = 524288 (default)

        go func() {
            import _ "net/http/pprof"
            log.Println("pprof enabled on :6060")
            log.Fatal(http.ListenAndServe("127.0.0.1:6060", nil))
        }()
    }

    // Start the application
    srv := &http.Server{Addr: ":8080", Handler: appRouter()}
    log.Fatal(srv.ListenAndServe())
}
```

---

## 4. Memory Profiling

### Code Example 6: Acquiring and Analyzing a Heap Profile

```bash
# Acquire a heap profile
go tool pprof http://localhost:6060/debug/pprof/heap

# Cumulative allocations (total since program start)
go tool pprof -alloc_space http://localhost:6060/debug/pprof/allocs

# Only memory currently in use
go tool pprof -inuse_space http://localhost:6060/debug/pprof/heap

# Number of allocations (object count)
go tool pprof -alloc_objects http://localhost:6060/debug/pprof/allocs

# Number of objects currently in use
go tool pprof -inuse_objects http://localhost:6060/debug/pprof/heap

# Display flame graph in Web UI
go tool pprof -http=:8081 http://localhost:6060/debug/pprof/heap
```

### Comparison of Memory Profile Modes

| Mode | Flag | Measured | Use Case |
|--------|--------|---------|------|
| inuse_space | `-inuse_space` | Amount of memory currently in use | Detecting memory leaks |
| inuse_objects | `-inuse_objects` | Number of objects currently in use | Investigating GC pressure |
| alloc_space | `-alloc_space` | Cumulative allocation size | Identifying hot paths |
| alloc_objects | `-alloc_objects` | Cumulative number of allocations | Identifying frequent allocation sites |

### Memory Profile Analysis Flow

```
+-------------------+     +-------------------+     +-------------------+
| Acquire heap      | --> | Identify          | --> | Identify          |
| profile           |     | allocation sites  |     | hotspots          |
|                   |     | with top / list   |     |                   |
+-------------------+     +-------------------+     +-------------------+
                                                            |
                                                            v
+-------------------+     +-------------------+     +-------------------+
| Apply the         | <-- | Consider          | <-- | sync.Pool?        |
| optimization,     |     | remediation       |     | Preallocate?      |
| verify via        |     | Buffer reuse?     |     |                   |
| benchmarks        |     |                   |     |                   |
+-------------------+     +-------------------+     +-------------------+
```

### Code Example 7: Patterns for Detecting Memory Leaks

```go
// Patterns prone to memory leaks and how to address them
package main

import (
    "context"
    "log"
    "os"
    "runtime"
    "runtime/pprof"
    "time"
)

// NG: goroutine leak
func leakyFunction() {
    for i := 0; i < 1000; i++ {
        go func() {
            ch := make(chan int)
            <-ch // Blocks forever -> goroutine leak
        }()
    }
}

// OK: cancellable via context
func safeFunction(ctx context.Context) {
    for i := 0; i < 1000; i++ {
        go func() {
            ch := make(chan int)
            select {
            case v := <-ch:
                process(v)
            case <-ctx.Done():
                return // Exit cleanly
            }
        }()
    }
}

// Monitor the number of goroutines
func monitorGoroutines() {
    ticker := time.NewTicker(10 * time.Second)
    for range ticker.C {
        log.Printf("goroutine count: %d", runtime.NumGoroutine())
    }
}

// Write the goroutine profile to a file
func dumpGoroutineProfile() {
    f, _ := os.Create("goroutine.prof")
    defer f.Close()
    pprof.Lookup("goroutine").WriteTo(f, 1)
}
```

### Code Example 8: Comparing Memory Leak Snapshots

```go
package main

import (
    "fmt"
    "net/http"
    "os"
    "runtime"
    "runtime/pprof"
    "time"
)

// Compare heap profiles at two points in time to detect leaks
func detectMemoryLeak() {
    // Take snapshot 1
    runtime.GC()
    f1, _ := os.Create("heap_before.prof")
    pprof.WriteHeapProfile(f1)
    f1.Close()

    // Apply load
    runLoad()

    // Wait a while and run GC
    time.Sleep(30 * time.Second)
    runtime.GC()
    time.Sleep(5 * time.Second) // Wait for GC to complete

    // Take snapshot 2
    f2, _ := os.Create("heap_after.prof")
    pprof.WriteHeapProfile(f2)
    f2.Close()

    // Diff analysis
    // go tool pprof -diff_base=heap_before.prof heap_after.prof
    fmt.Println("Run: go tool pprof -http=:8081 -diff_base=heap_before.prof heap_after.prof")
}

// Check memory usage with runtime.ReadMemStats
func printMemStats() {
    var m runtime.MemStats
    runtime.ReadMemStats(&m)

    fmt.Printf("Alloc      = %v MiB\n", m.Alloc/1024/1024)
    fmt.Printf("TotalAlloc = %v MiB\n", m.TotalAlloc/1024/1024)
    fmt.Printf("Sys        = %v MiB\n", m.Sys/1024/1024)
    fmt.Printf("NumGC      = %v\n", m.NumGC)
    fmt.Printf("HeapObjects= %v\n", m.HeapObjects)
    fmt.Printf("HeapInuse  = %v MiB\n", m.HeapInuse/1024/1024)
    fmt.Printf("StackInuse = %v MiB\n", m.StackInuse/1024/1024)
}
```

### Code Example 9: Slice Memory Leak Patterns

```go
// NG: Referencing part of a large slice -> the entire underlying array is not GC'd
func getFirstThree(data []byte) []byte {
    return data[:3]
    // The entire underlying array of data is retained (100MB -> 100MB held)
}

// OK: Copy to break the reference
func getFirstThree(data []byte) []byte {
    result := make([]byte, 3)
    copy(result, data[:3])
    return result
    // data becomes eligible for GC
}

// NG: Case where append leaves excessive capacity
func filterLarge(items []Item) []Item {
    // Filter from a slice of 10000 down to 10
    // But the underlying array still holds capacity for 10000
    var result []Item
    for _, item := range items {
        if item.IsImportant() {
            result = append(result, item)
        }
    }
    return result
}

// OK: Trim capacity as needed
func filterLarge(items []Item) []Item {
    var result []Item
    for _, item := range items {
        if item.IsImportant() {
            result = append(result, item)
        }
    }
    // Trim capacity to match length
    return slices.Clip(result) // Go 1.21+ (= result[:len(result):len(result)])
}
```

---

## 5. Mutex / Block Profiling

### Code Example 10: Mutex Profiling

```go
package main

import (
    "log"
    "net/http"
    _ "net/http/pprof"
    "runtime"
    "sync"
    "time"
)

func main() {
    // Enable Mutex profiling
    // Argument: n -> sample once every n mutex contentions
    // 1 = record every contention (for development)
    // 5 = record once every 5 (for production)
    runtime.SetMutexProfileFraction(5)

    // Enable Block profiling
    // Argument: threshold in nanoseconds
    // 1 = record every blocking event
    // 1000000 = record only blocks of 1ms or longer
    runtime.SetBlockProfileRate(1)

    go func() {
        log.Fatal(http.ListenAndServe(":6060", nil))
    }()

    // Workload that causes Mutex contention
    var mu sync.Mutex
    var counter int

    for i := 0; i < 100; i++ {
        go func() {
            for {
                mu.Lock()
                counter++
                time.Sleep(time.Millisecond)
                mu.Unlock()
            }
        }()
    }

    select {}
}
```

```bash
# Acquire a Mutex profile
go tool pprof http://localhost:6060/debug/pprof/mutex

# Acquire a Block profile
go tool pprof http://localhost:6060/debug/pprof/block

# Inspect in interactive mode
(pprof) top
(pprof) list main.main.func2
(pprof) web
```

### Code Example 11: Analyzing RWMutex Contention

```go
package main

import (
    "fmt"
    "runtime"
    "sync"
    "time"
)

// When reads dominate, RWMutex is more efficient
type Cache struct {
    mu    sync.RWMutex
    items map[string]string
}

func NewCache() *Cache {
    return &Cache{items: make(map[string]string)}
}

func (c *Cache) Get(key string) (string, bool) {
    c.mu.RLock()         // Read lock (allows concurrent reads)
    defer c.mu.RUnlock()
    v, ok := c.items[key]
    return v, ok
}

func (c *Cache) Set(key, value string) {
    c.mu.Lock()          // Write lock (exclusive)
    defer c.mu.Unlock()
    c.items[key] = value
}

// When to use Mutex vs RWMutex
//
// Mutex:
//   - Roughly equal read/write ratio
//   - Simple implementation
//   - Short lock hold time (RWMutex overhead becomes relatively large)
//
// RWMutex:
//   - Overwhelmingly read-heavy (90%+ reads)
//   - Read processing takes time
//   - Concurrent reads provide significant benefit
//
// sync.Map:
//   - Keys are stable (added but not removed)
//   - Overwhelmingly read-heavy
//   - Each goroutine accesses different keys

func benchmarkMutexVsRWMutex() {
    cache := NewCache()
    // Preload
    for i := 0; i < 1000; i++ {
        cache.Set(fmt.Sprintf("key_%d", i), fmt.Sprintf("value_%d", i))
    }

    start := time.Now()
    var wg sync.WaitGroup

    // 95% reads, 5% writes
    for i := 0; i < 100; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            for j := 0; j < 10000; j++ {
                key := fmt.Sprintf("key_%d", j%1000)
                if j%20 == 0 { // 5% writes
                    cache.Set(key, fmt.Sprintf("new_%d", j))
                } else { // 95% reads
                    cache.Get(key)
                }
            }
        }(i)
    }

    wg.Wait()
    fmt.Printf("Duration: %v\n", time.Since(start))
}
```

### Flow for Visualizing Lock Contention

```
+----------------------------------------------------------+
|  Workflow for investigating lock contention              |
+----------------------------------------------------------+
|                                                          |
|  1. Acquire Mutex profile                                |
|     go tool pprof http://localhost:6060/debug/pprof/mutex |
|     |                                                    |
|     v                                                    |
|  2. Identify contended locations with top                |
|     (pprof) top                                          |
|     -> functions with high contentions/delay             |
|     |                                                    |
|     v                                                    |
|  3. Check specific code lines with list                  |
|     (pprof) list MyFunction                              |
|     |                                                    |
|     v                                                    |
|  4. Consider remediations                                |
|     +--> Finer lock granularity (per struct field)       |
|     +--> Switch to RWMutex                               |
|     +--> Replace with sync.Map / atomic                  |
|     +--> Shorten lock hold time                          |
|     +--> Sharding (split into multiple Mutexes)          |
+----------------------------------------------------------+
```

---

## 6. runtime/trace — Execution Traces

### Code Example 12: Acquiring and Analyzing a Trace

```go
package main

import (
    "os"
    "runtime/trace"
)

func main() {
    f, err := os.Create("trace.out")
    if err != nil {
        panic(err)
    }
    defer f.Close()

    // Start the trace
    if err := trace.Start(f); err != nil {
        panic(err)
    }
    defer trace.Stop()

    // Work to be traced
    doWork()
}
```

```bash
# Visualize the trace (opens in a browser)
go tool trace trace.out
```

### Code Example 13: Custom Tasks and Regions

```go
package main

import (
    "context"
    "runtime/trace"
)

func processOrder(ctx context.Context, orderID string) error {
    // Create a task (grouped in the trace UI)
    ctx, task := trace.NewTask(ctx, "processOrder")
    defer task.End()

    // Regions (phases within the task)
    trace.WithRegion(ctx, "validate", func() {
        validateOrder(ctx, orderID)
    })

    trace.WithRegion(ctx, "payment", func() {
        processPayment(ctx, orderID)
    })

    trace.WithRegion(ctx, "shipping", func() {
        createShipment(ctx, orderID)
    })

    // Log events (viewable in the trace UI)
    trace.Log(ctx, "orderID", orderID)

    return nil
}

// Tracing in an HTTP handler
func handleOrder(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context()
    ctx, task := trace.NewTask(ctx, "handleOrder")
    defer task.End()

    trace.WithRegion(ctx, "decode", func() {
        // Decode the request
    })

    trace.WithRegion(ctx, "process", func() {
        // Business logic
    })

    trace.WithRegion(ctx, "respond", func() {
        // Send the response
    })
}
```

### Comparison Table: trace vs pprof

| Item | pprof | trace |
|------|-------|-------|
| Purpose | Identify CPU/memory hotspots | Time-series event analysis |
| Granularity | Function-level statistics | Goroutine-level events |
| Overhead | Low (sampling) | High (records every event) |
| Good for | Wanting to know "what is slow" | Wanting to know "why it is slow" |
| Visualization | Call graph, flame graph | Timeline, goroutine analysis |
| Capture time | 30 seconds to several minutes | A few seconds to 10 seconds recommended |
| GC analysis | Not possible | Detailed GC events visible |
| Network | Not possible | Network waits visible |
| Scheduler | Not possible | P/G/M relationships visible |

### What You Can See with trace

```
+----------------------------------------------------------+
| go tool trace Timeline View                              |
+----------------------------------------------------------+
|                                                          |
| Proc 0  |===G1====|  |==G3==|      |====G1====|         |
| Proc 1  |==G2====|      |==G4==|  |===G2===|            |
| Proc 2    |=G5=|  |===G6===|        |==G5==|            |
| Proc 3      |==G7==|  |=====G8=====|                    |
|                                                          |
| Network |---wait---|  |---wait------|                    |
| GC      |          |GC|             |GC|                 |
|                                                          |
| Time -> 0ms    50ms    100ms    150ms    200ms           |
+----------------------------------------------------------+
  G=goroutine, GC=garbage collection
```

### Code Example 14: GC Analysis via trace

```go
package main

import (
    "fmt"
    "os"
    "runtime"
    "runtime/debug"
    "runtime/trace"
    "time"
)

func main() {
    // Retrieve GC statistics
    var stats debug.GCStats
    debug.ReadGCStats(&stats)
    fmt.Printf("GC count: %d\n", stats.NumGC)
    fmt.Printf("Last GC: %v\n", stats.LastGC)
    fmt.Printf("Total GC time: %v\n", stats.PauseTotal)

    // Configure GC percentage
    // GOGC=100 is the default (GC when heap doubles)
    // GOGC=50 raises GC frequency (latency-focused)
    // GOGC=200 lowers GC frequency (throughput-focused)
    oldGOGC := debug.SetGCPercent(100)
    fmt.Printf("Previous GOGC: %d\n", oldGOGC)

    // Configure memory limit (Go 1.19+)
    // GOMEMLIMIT=1GiB or set from the program
    debug.SetMemoryLimit(1 << 30) // 1 GiB

    // Observe GC behavior with tracing
    f, _ := os.Create("gc_trace.out")
    defer f.Close()
    trace.Start(f)
    defer trace.Stop()

    // Work that allocates a lot of memory
    allocateAndRelease()

    // Check GC timing with: go tool trace gc_trace.out
}

func allocateAndRelease() {
    for i := 0; i < 100; i++ {
        data := make([]byte, 10*1024*1024) // 10MB
        _ = data
        time.Sleep(10 * time.Millisecond)
    }
}
```

### GC Tracing via the GODEBUG Environment Variable

```bash
# Print GC timing and duration to stderr
GODEBUG=gctrace=1 ./myapp

# Example output:
# gc 1 @0.012s 2%: 0.019+0.85+0.003 ms clock, 0.076+0.20/0.75/0+0.012 ms cpu, 4->4->0 MB, 4 MB goal, 0 MB stacks, 0 MB globals, 4 P
#
# How to read:
# gc 1         -> 1st GC
# @0.012s      -> 0.012 seconds after program start
# 2%           -> GC consumed 2% of CPU time
# 0.019+0.85+0.003 ms -> STW sweep start + concurrent + STW mark termination
# 4->4->0 MB   -> heap before GC -> heap after GC -> live data
# 4 MB goal     -> next GC trigger size

# Detailed scheduler info
GODEBUG=schedtrace=1000 ./myapp
# Outputs scheduler state every 1000ms
```

---

## 7. Benchmark-Integrated Profiling

### Code Example 15: Acquiring Profiles from Benchmarks

```bash
# Benchmark with CPU profile
go test -bench=BenchmarkSerialize -cpuprofile=cpu.prof -count=5

# Benchmark with memory profile
go test -bench=BenchmarkSerialize -memprofile=mem.prof -count=5

# Benchmark with trace
go test -bench=BenchmarkSerialize -trace=trace.out

# Benchmark with Block profile
go test -bench=BenchmarkSerialize -blockprofile=block.prof

# Benchmark with Mutex profile
go test -bench=BenchmarkSerialize -mutexprofile=mutex.prof

# Analyze the profiles
go tool pprof cpu.prof
go tool pprof mem.prof
go tool pprof -http=:8081 cpu.prof
```

### Code Example 16: A Memory Allocation Optimization Cycle

```go
// Before optimization
func ConcatStrings(strs []string) string {
    result := ""
    for _, s := range strs {
        result += s // Allocates a new string each time
    }
    return result
}

func BenchmarkConcatStrings(b *testing.B) {
    strs := make([]string, 1000)
    for i := range strs {
        strs[i] = "hello"
    }
    b.ResetTimer()
    b.ReportAllocs()
    for i := 0; i < b.N; i++ {
        ConcatStrings(strs)
    }
}
// BenchmarkConcatStrings    500   2145678 ns/op   5308416 B/op   999 allocs/op

// After optimization: strings.Builder
func ConcatStringsOptimized(strs []string) string {
    var b strings.Builder
    size := 0
    for _, s := range strs {
        size += len(s)
    }
    b.Grow(size) // Pre-allocate capacity
    for _, s := range strs {
        b.WriteString(s)
    }
    return b.String()
}
// BenchmarkConcatStringsOpt  50000   28456 ns/op   5120 B/op   1 allocs/op
//                                    75x faster            999x reduction
```

### Code Example 17: Reducing Allocations with sync.Pool

```go
package main

import (
    "bytes"
    "encoding/json"
    "sync"
    "testing"
)

// Buffer reuse using sync.Pool
var bufPool = sync.Pool{
    New: func() interface{} {
        return new(bytes.Buffer)
    },
}

// Version without Pool
func marshalJSON(v interface{}) ([]byte, error) {
    var buf bytes.Buffer // Allocates every time
    enc := json.NewEncoder(&buf)
    if err := enc.Encode(v); err != nil {
        return nil, err
    }
    return buf.Bytes(), nil
}

// Version with Pool
func marshalJSONPooled(v interface{}) ([]byte, error) {
    buf := bufPool.Get().(*bytes.Buffer)
    buf.Reset()
    defer bufPool.Put(buf)

    enc := json.NewEncoder(buf)
    if err := enc.Encode(v); err != nil {
        return nil, err
    }

    // Copy before returning to Pool (buf will be returned to the Pool)
    result := make([]byte, buf.Len())
    copy(result, buf.Bytes())
    return result, nil
}

func BenchmarkMarshalJSON(b *testing.B) {
    data := map[string]interface{}{"name": "Alice", "age": 30, "active": true}
    b.ReportAllocs()
    for i := 0; i < b.N; i++ {
        marshalJSON(data)
    }
}

func BenchmarkMarshalJSONPooled(b *testing.B) {
    data := map[string]interface{}{"name": "Alice", "age": 30, "active": true}
    b.ReportAllocs()
    for i := 0; i < b.N; i++ {
        marshalJSONPooled(data)
    }
}

// Typical results:
// BenchmarkMarshalJSON-8          500000   3200 ns/op   768 B/op   3 allocs/op
// BenchmarkMarshalJSONPooled-8    800000   1800 ns/op   256 B/op   2 allocs/op
```

### Code Example 18: Optimization via Preallocation

```go
package main

import "testing"

// NG: The underlying array is reallocated on every append
func collectItemsSlow(n int) []int {
    var result []int
    for i := 0; i < n; i++ {
        result = append(result, i*2)
    }
    return result
}

// OK: Pre-allocate capacity
func collectItemsFast(n int) []int {
    result := make([]int, 0, n)
    for i := 0; i < n; i++ {
        result = append(result, i*2)
    }
    return result
}

// Even faster: direct index assignment
func collectItemsFastest(n int) []int {
    result := make([]int, n)
    for i := 0; i < n; i++ {
        result[i] = i * 2
    }
    return result
}

func BenchmarkCollectSlow(b *testing.B) {
    for i := 0; i < b.N; i++ {
        collectItemsSlow(10000)
    }
}

func BenchmarkCollectFast(b *testing.B) {
    for i := 0; i < b.N; i++ {
        collectItemsFast(10000)
    }
}

func BenchmarkCollectFastest(b *testing.B) {
    for i := 0; i < b.N; i++ {
        collectItemsFastest(10000)
    }
}

// Typical results:
// BenchmarkCollectSlow-8      10000   152000 ns/op   386048 B/op   20 allocs/op
// BenchmarkCollectFast-8      50000    28000 ns/op    81920 B/op    1 allocs/op
// BenchmarkCollectFastest-8   50000    25000 ns/op    81920 B/op    1 allocs/op
```

---

## 8. How to Read a Flame Graph

### Flame Graph Structure

```
+----------------------------------------------------------+
|  How to read a flame graph                                |
+----------------------------------------------------------+
|                                                          |
|  X-axis = share of CPU time (wider = more time spent)    |
|  Y-axis = call stack depth (upward = callees)            |
|                                                          |
|  +---------------------------------------------------+   |
|  |                    main.main                       |   |
|  +---------------------------------------------------+   |
|  |         main.handleRequest          | main.other  |   |
|  +-------------------------------------+-------------+   |
|  | main.processData  | main.queryDB   |              |   |
|  +-------------------+----------------+              |   |
|  | json.Unmarshal    | sql.Query      |              |   |
|  +-------------------+----------------+              |   |
|                                                          |
|  -> json.Unmarshal and sql.Query are the main bottlenecks |
|  -> handleRequest accounts for 75% of the total          |
+----------------------------------------------------------+
```

### Key Points to Look For in Flame Graphs

```
1. Wide frames
   -> Functions consuming a lot of CPU time
   -> Prime candidates for optimization

2. Deep call stacks
   -> Deep call hierarchy (refactoring candidates)
   -> Consider inlining when many indirect calls exist

3. The share of runtime.*
   -> Large runtime.mallocgc -> excessive allocations
   -> Large runtime.gcBgMarkWorker -> high GC load
   -> runtime.futex / runtime.notesleep -> lock waits

4. The same function appearing in multiple places
   -> Called from different call paths
   -> Optimizing a hot shared function yields big gains
```

### Code Example 19: Tool for Comparing Benchmark Results

```bash
# Use benchstat to statistically compare benchmark results
# Install
go install golang.org/x/perf/cmd/benchstat@latest

# Benchmark before optimization
go test -bench=. -count=10 -benchmem > before.txt

# Benchmark after optimization
go test -bench=. -count=10 -benchmem > after.txt

# Compare
benchstat before.txt after.txt

# Example output:
# name           old time/op    new time/op    delta
# Serialize-8    2.15ms ± 3%    0.85ms ± 2%   -60.47%  (p=0.000 n=10+10)
#
# name           old alloc/op   new alloc/op   delta
# Serialize-8    5.30MB ± 0%    0.01MB ± 0%   -99.81%  (p=0.000 n=10+10)
#
# name           old allocs/op  new allocs/op  delta
# Serialize-8     999 ± 0%       1 ± 0%       -99.90%  (p=0.000 n=10+10)
```

---

## 9. Continuous Profiling

### The Need for Continuous Profiling

```
+----------------------------------------------------------+
|  Traditional profiling vs continuous profiling           |
+----------------------------------------------------------+
|                                                          |
|  Traditional:                                             |
|  - Acquire profiles only after problems occur            |
|  - Miss hard-to-reproduce problems                       |
|  - Hard to grasp dev vs production performance gaps      |
|                                                          |
|  Continuous:                                              |
|  - Constantly collect profiles                           |
|  - Analyze trends over time (detect regressions)         |
|  - Easy comparison before/after deploys                  |
|  - Can capture low-frequency problems                    |
+----------------------------------------------------------+
```

### Code Example 20: Continuous Profiling with Pyroscope

```go
package main

import (
    "log"
    "net/http"
    "os"

    "github.com/grafana/pyroscope-go"
)

func main() {
    // Pyroscope configuration
    pyroscope.Start(pyroscope.Config{
        ApplicationName: "myapp",
        ServerAddress:   os.Getenv("PYROSCOPE_SERVER"), // e.g. http://pyroscope:4040
        Logger:          pyroscope.StandardLogger,

        // Select which profile types to collect
        ProfileTypes: []pyroscope.ProfileType{
            pyroscope.ProfileCPU,
            pyroscope.ProfileAllocObjects,
            pyroscope.ProfileAllocSpace,
            pyroscope.ProfileInuseObjects,
            pyroscope.ProfileInuseSpace,
            pyroscope.ProfileGoroutines,
            pyroscope.ProfileMutexCount,
            pyroscope.ProfileMutexDuration,
            pyroscope.ProfileBlockCount,
            pyroscope.ProfileBlockDuration,
        },

        // Enable filtering by tags
        Tags: map[string]string{
            "env":     os.Getenv("APP_ENV"),
            "version": version,
            "region":  os.Getenv("AWS_REGION"),
        },
    })

    // Tag a specific piece of work
    pyroscope.TagWrapper(context.Background(), pyroscope.Labels(
        "handler", "processOrder",
        "orderType", "premium",
    ), func(ctx context.Context) {
        processOrder(ctx)
    })

    http.ListenAndServe(":8080", router())
}
```

### Code Example 21: Integration with Google Cloud Profiler

```go
package main

import (
    "log"

    "cloud.google.com/go/profiler"
)

func main() {
    // Google Cloud Profiler configuration
    cfg := profiler.Config{
        Service:        "myapp",
        ServiceVersion: version,
        ProjectID:      "my-gcp-project",
        // MutexProfiling: true,  // Enable Mutex profiling
    }

    if err := profiler.Start(cfg); err != nil {
        log.Printf("Failed to start Cloud Profiler: %v", err)
        // Profiler startup failure should not stop the application
    }

    // Start the application
    startServer()
}
```

### Comparison of Continuous Profiling Tools

| Tool | Provider | Price | Characteristics |
|--------|--------|------|------|
| Pyroscope | Grafana | OSS / Cloud | Grafana integration, rich Go SDK |
| Parca | Polar Signals | OSS | eBPF-based, low overhead |
| Cloud Profiler | Google | Included in GCP usage | GCP integration, easy setup |
| Datadog Profiler | Datadog | Paid | APM integration, rich analytics |
| pprof + custom collection | - | Free | Flexible but high operational cost |

---

## 10. Practical Optimization Patterns

### Code Example 22: HTTP Response Streaming Optimization

```go
package main

import (
    "encoding/json"
    "net/http"
    "sync"
)

// NG: Build the entire response in memory
func handleUsersNG(w http.ResponseWriter, r *http.Request) {
    users, err := db.GetAllUsers() // Loads all users into memory
    if err != nil {
        http.Error(w, err.Error(), 500)
        return
    }
    json.NewEncoder(w).Encode(users) // Encodes a massive JSON at once
}

// OK: Write out incrementally via streaming
func handleUsersOK(w http.ResponseWriter, r *http.Request) {
    w.Header().Set("Content-Type", "application/json")
    w.Write([]byte("["))

    rows, err := db.QueryUsers(r.Context())
    if err != nil {
        http.Error(w, err.Error(), 500)
        return
    }
    defer rows.Close()

    enc := json.NewEncoder(w)
    first := true
    for rows.Next() {
        var user User
        if err := rows.Scan(&user); err != nil {
            break
        }
        if !first {
            w.Write([]byte(","))
        }
        first = false
        enc.Encode(user)
    }
    w.Write([]byte("]"))
}
```

### Code Example 23: Pre-sizing Maps

```go
package main

import "testing"

// NG: No size specified -> rehashing happens multiple times
func createMapSlow(n int) map[string]int {
    m := make(map[string]int) // Small initial bucket count
    for i := 0; i < n; i++ {
        m[fmt.Sprintf("key_%d", i)] = i
    }
    return m
}

// OK: Size specified up front -> avoids rehashing
func createMapFast(n int) map[string]int {
    m := make(map[string]int, n) // Pre-reserves needed bucket count
    for i := 0; i < n; i++ {
        m[fmt.Sprintf("key_%d", i)] = i
    }
    return m
}

// Example benchmark results (n=10000):
// BenchmarkMapSlow-8    5000   312000 ns/op   687432 B/op   172 allocs/op
// BenchmarkMapFast-8    8000   198000 ns/op   473440 B/op    12 allocs/op
```

### Code Example 24: Optimizing String Operations

```go
package main

import (
    "fmt"
    "strconv"
    "strings"
    "testing"
)

// NG: fmt.Sprintf uses reflection and is slow
func formatUserSlow(name string, age int) string {
    return fmt.Sprintf("Name: %s, Age: %d", name, age)
}

// OK: strings.Builder + strconv is faster
func formatUserFast(name string, age int) string {
    var b strings.Builder
    b.Grow(20 + len(name)) // Pre-allocate required size
    b.WriteString("Name: ")
    b.WriteString(name)
    b.WriteString(", Age: ")
    b.WriteString(strconv.Itoa(age))
    return b.String()
}

// OK: For a small number of concatenations, the + operator is fine
func formatUserSimple(name string, age int) string {
    return "Name: " + name + ", Age: " + strconv.Itoa(age)
}

// Example benchmark results:
// BenchmarkFormatSlow-8      5000000    280 ns/op   64 B/op   2 allocs/op
// BenchmarkFormatFast-8     15000000     85 ns/op   48 B/op   1 allocs/op
// BenchmarkFormatSimple-8   12000000     95 ns/op   48 B/op   1 allocs/op
```

### Code Example 25: Optimization via Concrete-Type Interface Assertions

```go
package main

import (
    "io"
    "os"
)

// Optimization leveraging the io.WriterTo interface
// Many standard library types implement WriterTo
func copyData(dst io.Writer, src io.Reader) (int64, error) {
    // io.Copy internally checks for WriterTo / ReaderFrom:
    // - If src implements WriterTo, it calls src.WriteTo(dst)
    // - If dst implements ReaderFrom, it calls dst.ReadFrom(src)
    // - If neither, copies via an intermediate buffer
    return io.Copy(dst, src)
}

// Specify buffer size (for large files)
func copyLargeFile(dst io.Writer, src io.Reader) (int64, error) {
    // Use a larger buffer instead of the default 32KB
    buf := make([]byte, 1024*1024) // 1MB buffer
    return io.CopyBuffer(dst, src, buf)
}

// Use a type assertion to pick the optimal path
type Flusher interface {
    Flush() error
}

func writeWithFlush(w io.Writer, data []byte) error {
    if _, err := w.Write(data); err != nil {
        return err
    }
    // Flush if the writer implements Flusher
    if f, ok := w.(Flusher); ok {
        return f.Flush()
    }
    return nil
}
```

---

## 11. Anti-Patterns

### Anti-Pattern 1: Exposing pprof on a Public Port in Production

```go
// NG: pprof on a public port in production
import _ "net/http/pprof"

func main() {
    // pprof is accessible from the outside -> security risk
    http.ListenAndServe(":8080", nil)
}

// OK: Run pprof on a separate port, restricted to the internal network
func main() {
    go func() {
        // Localhost only, or the internal network only
        log.Fatal(http.ListenAndServe("127.0.0.1:6060", nil))
    }()
    http.ListenAndServe(":8080", appHandler)
}
```

### Anti-Pattern 2: Speculative Optimization Without Profiling

```go
// NG: Optimizing by guessing "this part must be slow"
// -> Wastes time on places that aren't actually bottlenecks

// OK: Profile-driven optimization cycle
// 1. Run benchmarks
// 2. Acquire a profile
// 3. Identify hotspots
// 4. Implement improvements
// 5. Verify the effect with benchmarks
// 6. Go back to 1
```

### Anti-Pattern 3: Misusing sync.Pool

```go
// NG: Using sync.Pool for objects that are too small
var intPool = sync.Pool{
    New: func() interface{} {
        v := 0
        return &v // int pointers are so small that Pool overhead dominates
    },
}

// NG: Using an object from the Pool without initializing it
var bufPool = sync.Pool{
    New: func() interface{} {
        return &bytes.Buffer{}
    },
}

func process() {
    buf := bufPool.Get().(*bytes.Buffer)
    defer bufPool.Put(buf)
    // Forgot buf.Reset() -> leftover data from the previous use
    buf.WriteString("new data")
}

// OK: Use Pool with appropriately sized objects and always initialize
var bufPool = sync.Pool{
    New: func() interface{} {
        return bytes.NewBuffer(make([]byte, 0, 4096))
    },
}

func process() {
    buf := bufPool.Get().(*bytes.Buffer)
    buf.Reset() // Always reset
    defer bufPool.Put(buf)
    buf.WriteString("new data")
}
```

### Anti-Pattern 4: Collecting a Trace for Too Long

```go
// NG: Collect a 60-second trace -> enormous data, UI freezes
// go tool trace trace_60s.out -> browser crashes

// OK: Limit traces to a short window (1-5 seconds)
// curl "http://localhost:6060/debug/pprof/trace?seconds=3" > trace.out
// go tool trace trace.out

// To trace a specific operation, control it from within the program
func traceOperation(ctx context.Context) error {
    f, _ := os.CreateTemp("", "trace_*.out")
    defer f.Close()

    trace.Start(f)
    defer trace.Stop()

    // Target operation (should complete quickly)
    return doOperation(ctx)
}
```

### Anti-Pattern 5: Running Production with MemProfileRate Set to 1

```go
// NG: Records every allocation (significant performance impact)
func init() {
    runtime.MemProfileRate = 1 // Records every allocation
}

// OK: Use the default value in production
// The default for runtime.MemProfileRate is 524288 (512KB)
// Adjust only if necessary
func init() {
    if os.Getenv("DETAILED_MEMPROFILE") == "true" {
        runtime.MemProfileRate = 1 // Only during debugging
    }
    // Otherwise use the default (sample once per 512KB)
}
```

---

## FAQ

### Q1. Is the overhead of profiling acceptable in production?

Merely having the `net/http/pprof` endpoints present adds virtually zero overhead. CPU profiling only samples while requests are in flight, typically causing a 1-5% impact. Memory profiling is controlled by `runtime.MemProfileRate`, sampling once every 512KB by default. Block/Mutex profiling is controlled via `SetBlockProfileRate` / `SetMutexProfileFraction`, and a low sampling rate is recommended in production.

### Q2. How do I read a flame graph?

A flame graph uses the X-axis for the share of CPU time and the Y-axis for the call stack depth. Wide frames are bottlenecks. Functions higher up are callees. You can view it via the Flame Graph tab of `go tool pprof -http=:8081 cpu.prof`. A wide `runtime.mallocgc` indicates excessive allocations; a wide `runtime.gcBgMarkWorker` indicates high GC load.

### Q3. How do I detect goroutine leaks?

Periodically log `runtime.NumGoroutine()` and watch for an upward trend. Check goroutine stack traces at `/debug/pprof/goroutine?debug=1`; if many goroutines share the same stack trace, a leak is likely. Integrating the `goleak` package (`go.uber.org/goleak`) into tests lets you detect unfinished goroutines at the end of a test.

```go
// Goroutine leak detection test using goleak
func TestMain(m *testing.M) {
    goleak.VerifyTestMain(m)
}

// When used per-test
func TestSomething(t *testing.T) {
    defer goleak.VerifyNone(t)
    // test code
}
```

### Q4. What views are available in the pprof Web UI?

The Web UI launched by `go tool pprof -http=:8081 profile.prof` offers the following views:
- **Top**: Ranking of CPU/memory consumption by function
- **Graph**: Call graph (call relationships between functions)
- **Flame Graph**: Flame graph (width = cost, height = call stack depth)
- **Peek**: Callers and callees of a specific function
- **Source**: Cost displayed over the source code
- **Disasm**: Cost displayed over the assembly code

### Q5. How should I choose between GOGC and GOMEMLIMIT?

`GOGC` triggers GC based on heap growth rate (default 100 = GC when heap doubles). `GOMEMLIMIT` (Go 1.19+) sets a memory upper bound and aggressively runs GC as it nears the limit. In container environments, setting `GOMEMLIMIT` to 80-90% of the container's memory limit is recommended. You can also combine both.

```bash
# Example recommended settings for a container environment
# Container memory limit: 1GB
GOMEMLIMIT=900MiB  # 90% of the memory limit
GOGC=100            # default (combined with GOMEMLIMIT)
```

### Q6. Why do benchmark results vary every run?

Causes include CPU thermal throttling, interference from other processes, and OS scheduling. To get stable results: (1) run `-count=10` multiple times and use `benchstat` for statistical processing, (2) pin CPUs with `taskset` / `cpuset`, (3) disable turbo boost, (4) minimize other processes. CI environments are noisy, so local measurement is recommended.

### Q7. How do I check the results of Escape Analysis?

```bash
# Check escapes to the heap
go build -gcflags="-m" ./...

# More detailed info
go build -gcflags="-m -m" ./...

# Example output:
# ./main.go:15:6: can inline NewUser
# ./main.go:20:10: &User{...} escapes to heap
# -> confirms that "&User{...}" is allocated on the heap
```

Values allocated on the stack don't burden the GC, so they are faster. Common causes of escaping to the heap: (1) returning a pointer, (2) assigning to an interface, (3) captured by a closure, (4) size too large (typically 64KB+).

---

## Summary

| Concept | Key Point |
|------|------|
| net/http/pprof | Acquire profiles via HTTP endpoints |
| go tool pprof | Tool for analyzing and visualizing profiles |
| CPU profile | Identify CPU consumption time per function |
| Heap profile | Identify memory allocation hotspots |
| goroutine profile | Detect goroutine leaks |
| Mutex/Block profile | Analyze lock contention and blocking operations |
| runtime/trace | Visualize time-series events |
| -bench + -cpuprofile | Integrate benchmarks and profiles |
| b.ReportAllocs() | Measure allocation counts |
| sync.Pool | Reduce allocations via object reuse |
| benchstat | Statistical comparison of benchmark results |
| GOGC / GOMEMLIMIT | Control GC behavior |
| Continuous profiling | Pyroscope / Parca / Cloud Profiler |
| Escape Analysis | Check heap allocations with `go build -gcflags="-m"` |

---

## Guides to Read Next

- **03-tools/03-deployment.md** — Deployment: Docker, cross-compilation
- **03-tools/04-best-practices.md** — Best Practices: Effective Go
- **02-web/04-testing.md** — Testing: table-driven tests, testify, httptest

---

## References

1. **Go Blog — Profiling Go Programs** https://go.dev/blog/pprof
2. **Go Official — runtime/pprof package** https://pkg.go.dev/runtime/pprof
3. **Go Official — runtime/trace package** https://pkg.go.dev/runtime/trace
4. **Julia Evans — A Practical Guide to pprof** https://jvns.ca/blog/2017/09/24/profiling-go-with-pprof/
5. **Go Official — runtime/debug package** https://pkg.go.dev/runtime/debug
6. **Pyroscope Official Documentation** https://pyroscope.io/docs/
7. **Google Cloud Profiler** https://cloud.google.com/profiler/docs
8. **benchstat tool** https://pkg.go.dev/golang.org/x/perf/cmd/benchstat
