# Synchronization Primitives -- Mutex, RWMutex, Once, Pool, atomic

> The sync package provides synchronization primitives such as Mutex, RWMutex, Once, and Pool, while sync/atomic offers lock-free atomic operations.

---

## What You Will Learn in This Chapter

1. **Mutex / RWMutex** -- Exclusive control and read/write locks
2. **sync.Once / sync.Pool** -- One-time initialization and object reuse
3. **sync/atomic** -- Lock-free atomic operations
4. **sync.Cond / sync.Map** -- Condition variables and concurrency-safe maps
5. **Practical patterns** -- Leveraging synchronization primitives in production code


## Prerequisites

Before reading this guide, the following knowledge will help deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Understanding the content of [Goroutines and Channels -- The Foundation of Go Concurrent Programming](./00-goroutines-channels.md)

---

## 1. Mutex

### Code Example 1: Basics of sync.Mutex

```go
type SafeCounter struct {
    mu sync.Mutex
    v  map[string]int
}

func (c *SafeCounter) Inc(key string) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.v[key]++
}

func (c *SafeCounter) Value(key string) int {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.v[key]
}
```

### How Mutex Works Internally

Go's Mutex has two modes: Normal mode and Starvation mode.

```
Normal mode:
  Newly arriving goroutines attempt to acquire the lock
  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ Waiting  │────>│ Spinning │────>│ Acquired │
  │ goroutine│     │ (CAS)    │     │ Lock     │
  └──────────┘     └──────────┘     └──────────┘
  → A newly arrived goroutine can acquire the lock before
    goroutines already waiting (good performance but not fair)

Starvation mode:
  Entered when a goroutine has been waiting for more than 1ms
  ┌──────────┐     ┌──────────┐
  │ FIFO     │────>│ Acquired │
  │ Queue    │     │ Lock     │
  └──────────┘     └──────────┘
  → The lock is granted in strict FIFO order
  → No spinning; fairness is guaranteed
  → Returns to normal mode when the wait queue is empty
    or the last waiter's wait time is < 1ms
```

### Code Example: A Mutex-Protected Cache

```go
type TTLCache struct {
    mu      sync.Mutex
    items   map[string]cacheItem
    ttl     time.Duration
    cleanCh chan struct{}
}

type cacheItem struct {
    value     interface{}
    expiresAt time.Time
}

func NewTTLCache(ttl time.Duration) *TTLCache {
    c := &TTLCache{
        items:   make(map[string]cacheItem),
        ttl:     ttl,
        cleanCh: make(chan struct{}),
    }
    go c.cleanup()
    return c
}

func (c *TTLCache) Set(key string, value interface{}) {
    c.mu.Lock()
    defer c.mu.Unlock()
    c.items[key] = cacheItem{
        value:     value,
        expiresAt: time.Now().Add(c.ttl),
    }
}

func (c *TTLCache) Get(key string) (interface{}, bool) {
    c.mu.Lock()
    defer c.mu.Unlock()

    item, ok := c.items[key]
    if !ok {
        return nil, false
    }
    if time.Now().After(item.expiresAt) {
        delete(c.items, key)
        return nil, false
    }
    return item.value, true
}

func (c *TTLCache) Delete(key string) {
    c.mu.Lock()
    defer c.mu.Unlock()
    delete(c.items, key)
}

func (c *TTLCache) Len() int {
    c.mu.Lock()
    defer c.mu.Unlock()
    return len(c.items)
}

func (c *TTLCache) cleanup() {
    ticker := time.NewTicker(c.ttl / 2)
    defer ticker.Stop()

    for {
        select {
        case <-ticker.C:
            c.mu.Lock()
            now := time.Now()
            for key, item := range c.items {
                if now.After(item.expiresAt) {
                    delete(c.items, key)
                }
            }
            c.mu.Unlock()
        case <-c.cleanCh:
            return
        }
    }
}

func (c *TTLCache) Close() {
    close(c.cleanCh)
}
```

---

## 2. RWMutex

### Code Example 2: sync.RWMutex

```go
type Cache struct {
    mu   sync.RWMutex
    data map[string]string
}

func (c *Cache) Get(key string) (string, bool) {
    c.mu.RLock()         // Read lock (multiple readers allowed simultaneously)
    defer c.mu.RUnlock()
    v, ok := c.data[key]
    return v, ok
}

func (c *Cache) Set(key, value string) {
    c.mu.Lock()          // Write lock (exclusive)
    defer c.mu.Unlock()
    c.data[key] = value
}
```

### Detailed Behavior of RWMutex

```go
// Read/write behavior of RWMutex
//
// Read lock (RLock):
//   - Acquired immediately if no write lock is held
//   - Multiple goroutines can hold RLock simultaneously
//   - If a write lock is waiting, new RLocks must wait
//     (to prevent writer starvation)
//
// Write lock (Lock):
//   - Waits until all RLocks are released
//   - Once acquired, no other RLock or Lock can be obtained

type ConfigManager struct {
    mu     sync.RWMutex
    config map[string]string
}

func NewConfigManager() *ConfigManager {
    return &ConfigManager{
        config: make(map[string]string),
    }
}

// Reads can happen concurrently
func (cm *ConfigManager) Get(key string) string {
    cm.mu.RLock()
    defer cm.mu.RUnlock()
    return cm.config[key]
}

// GetAll returns a snapshot of the config
func (cm *ConfigManager) GetAll() map[string]string {
    cm.mu.RLock()
    defer cm.mu.RUnlock()

    // Return a copy (to prevent modification outside the lock)
    snapshot := make(map[string]string, len(cm.config))
    for k, v := range cm.config {
        snapshot[k] = v
    }
    return snapshot
}

// Writes are exclusive
func (cm *ConfigManager) Set(key, value string) {
    cm.mu.Lock()
    defer cm.mu.Unlock()
    cm.config[key] = value
}

// Bulk update (transactional-style update)
func (cm *ConfigManager) Update(updates map[string]string) {
    cm.mu.Lock()
    defer cm.mu.Unlock()

    for k, v := range updates {
        cm.config[k] = v
    }
}
```

### RWMutex vs Mutex Benchmark

```go
// Benchmark to measure performance difference based on read ratio
func BenchmarkMutexRead(b *testing.B) {
    var mu sync.Mutex
    data := map[string]int{"key": 42}

    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            mu.Lock()
            _ = data["key"]
            mu.Unlock()
        }
    })
}

func BenchmarkRWMutexRead(b *testing.B) {
    var mu sync.RWMutex
    data := map[string]int{"key": 42}

    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            mu.RLock()
            _ = data["key"]
            mu.RUnlock()
        }
    })
}

// Approximate results (8-core machine):
// BenchmarkMutexRead-8      20000000    60 ns/op
// BenchmarkRWMutexRead-8    50000000    25 ns/op
//
// → When reads are 95%+, RWMutex is advantageous
// → When writes are 50%+, Mutex can be better
//   (due to RWMutex overhead)
```

---

## 3. sync.Once

### Code Example 3: Basics of sync.Once

```go
var (
    instance *Database
    once     sync.Once
)

func GetDB() *Database {
    once.Do(func() {
        // Executed only once even when called from multiple goroutines
        instance = &Database{
            conn: connectDB(),
        }
    })
    return instance
}
```

### OnceFunc / OnceValue / OnceValues in Go 1.21+

```go
// Convenient helpers added in Go 1.21

// sync.OnceFunc: returns a wrapper that runs the function only once
cleanup := sync.OnceFunc(func() {
    fmt.Println("Running cleanup")
    db.Close()
})

cleanup() // Outputs "Running cleanup"
cleanup() // Nothing happens (second call is not executed)

// sync.OnceValue: returns a wrapper that computes a value only once
getConfig := sync.OnceValue(func() *Config {
    fmt.Println("Loading config...")
    cfg, err := loadConfig("config.yaml")
    if err != nil {
        panic(err) // Panics are re-raised on subsequent calls
    }
    return cfg
})

cfg := getConfig() // Outputs "Loading config..."
cfg = getConfig()  // Returns the cached value

// sync.OnceValues: version that returns a value and an error
loadCert := sync.OnceValues(func() (*tls.Certificate, error) {
    return tls.LoadX509KeyPair("cert.pem", "key.pem")
})

cert, err := loadCert() // First call: reads the files
cert, err = loadCert()  // Second call: returns the cached result
```

### Error-Handling Patterns with Once

```go
// Pattern for safely handling initialization errors with sync.Once
type LazyDB struct {
    once sync.Once
    db   *sql.DB
    err  error
}

func (l *LazyDB) Get() (*sql.DB, error) {
    l.once.Do(func() {
        l.db, l.err = sql.Open("postgres", os.Getenv("DATABASE_URL"))
        if l.err != nil {
            return
        }
        l.err = l.db.Ping()
    })
    return l.db, l.err
}

// Note: Once.Do considers the call "done" even if it panics
// If you want to retry on error, you need a different approach

// Retryable initialization (sync.Once cannot be used)
type RetryableInit struct {
    mu       sync.Mutex
    db       *sql.DB
    initDone bool
}

func (r *RetryableInit) Get() (*sql.DB, error) {
    r.mu.Lock()
    defer r.mu.Unlock()

    if r.initDone {
        return r.db, nil
    }

    db, err := sql.Open("postgres", os.Getenv("DATABASE_URL"))
    if err != nil {
        return nil, err // Can be retried on the next call
    }

    if err := db.Ping(); err != nil {
        db.Close()
        return nil, err // Can be retried on the next call
    }

    r.db = db
    r.initDone = true
    return r.db, nil
}
```

---

## 4. sync.Pool

### Code Example 4: Basics of sync.Pool

```go
var bufPool = sync.Pool{
    New: func() interface{} {
        return new(bytes.Buffer)
    },
}

func processRequest(data []byte) string {
    buf := bufPool.Get().(*bytes.Buffer)
    defer func() {
        buf.Reset()
        bufPool.Put(buf) // Return to the pool
    }()

    buf.Write(data)
    return buf.String()
}
```

### Practical Uses of sync.Pool

```go
// Pool of JSON encoders
var encoderPool = sync.Pool{
    New: func() interface{} {
        return &bytes.Buffer{}
    },
}

func respondJSON(w http.ResponseWriter, data interface{}) error {
    buf := encoderPool.Get().(*bytes.Buffer)
    defer func() {
        buf.Reset()
        encoderPool.Put(buf)
    }()

    if err := json.NewEncoder(buf).Encode(data); err != nil {
        return err
    }

    w.Header().Set("Content-Type", "application/json")
    _, err := w.Write(buf.Bytes())
    return err
}

// Slice pool (fixed-size buffer)
var slicePool = sync.Pool{
    New: func() interface{} {
        s := make([]byte, 0, 4096)
        return &s
    },
}

func processData(input []byte) []byte {
    bufPtr := slicePool.Get().(*[]byte)
    buf := (*bufPtr)[:0] // Reset the length, keep the capacity
    defer func() {
        *bufPtr = buf[:0]
        slicePool.Put(bufPtr)
    }()

    // Use buf for processing
    buf = append(buf, input...)
    // ... processing ...

    // Copy and return the result (copying is required when taking data out of the pool)
    result := make([]byte, len(buf))
    copy(result, buf)
    return result
}

// A log formatter using sync.Pool
type LogFormatter struct {
    pool sync.Pool
}

func NewLogFormatter() *LogFormatter {
    return &LogFormatter{
        pool: sync.Pool{
            New: func() interface{} {
                return &strings.Builder{}
            },
        },
    }
}

func (f *LogFormatter) Format(level, msg string, fields map[string]interface{}) string {
    sb := f.pool.Get().(*strings.Builder)
    defer func() {
        sb.Reset()
        f.pool.Put(sb)
    }()

    sb.WriteString(time.Now().Format(time.RFC3339))
    sb.WriteString(" [")
    sb.WriteString(level)
    sb.WriteString("] ")
    sb.WriteString(msg)

    for k, v := range fields {
        sb.WriteString(" ")
        sb.WriteString(k)
        sb.WriteString("=")
        fmt.Fprintf(sb, "%v", v)
    }

    return sb.String()
}
```

### Caveats of sync.Pool

```
┌──────────────────────────────────────────────────────┐
│                Characteristics of sync.Pool          │
│                                                      │
│  Appropriate uses:                                   │
│    - Temporary objects frequently allocated/freed    │
│    - Buffers, encoders, formatters                   │
│    - Goal is to reduce GC pressure                   │
│                                                      │
│  Inappropriate uses:                                 │
│    - Using as a cache                                │
│      → Objects can be reclaimed by GC without notice │
│    - Connection pools (DB, HTTP)                     │
│      → Cannot manage connection state                │
│    - Long-lived objects                              │
│      → Contradicts the purpose of Pool               │
│                                                      │
│  Cautions:                                           │
│    - Always initialize the result of Pool.Get()      │
│    - Reset the object before calling Pool.Put()      │
│    - Do not let objects from the pool escape         │
│    - Verify the benefits with benchmarks first       │
└──────────────────────────────────────────────────────┘
```

---

## 5. sync/atomic

### Code Example 5: Basics of atomic (Go 1.19+ type-safe API)

```go
type AtomicCounter struct {
    count atomic.Int64  // Go 1.19+
}

func (c *AtomicCounter) Inc() {
    c.count.Add(1)
}

func (c *AtomicCounter) Value() int64 {
    return c.count.Load()
}

// Safely read/write arbitrary values with atomic.Value
var config atomic.Value // *Config

func UpdateConfig(cfg *Config) {
    config.Store(cfg)
}

func GetConfig() *Config {
    return config.Load().(*Config)
}
```

### Detailed Usage of atomic

```go
// Type-safe atomic types in Go 1.19+
type Metrics struct {
    RequestCount  atomic.Int64
    ErrorCount    atomic.Int64
    ActiveConns   atomic.Int32
    BytesReceived atomic.Uint64
    IsHealthy     atomic.Bool
}

func (m *Metrics) RecordRequest(success bool, bytes uint64) {
    m.RequestCount.Add(1)
    m.BytesReceived.Add(bytes)
    if !success {
        m.ErrorCount.Add(1)
    }
}

func (m *Metrics) ConnectionOpened() {
    m.ActiveConns.Add(1)
}

func (m *Metrics) ConnectionClosed() {
    m.ActiveConns.Add(-1)
}

func (m *Metrics) Snapshot() map[string]interface{} {
    return map[string]interface{}{
        "requests":       m.RequestCount.Load(),
        "errors":         m.ErrorCount.Load(),
        "active_conns":   m.ActiveConns.Load(),
        "bytes_received": m.BytesReceived.Load(),
        "healthy":        m.IsHealthy.Load(),
    }
}
```

### Compare-And-Swap (CAS) Pattern

```go
// Lock-free stack using CAS
type LockFreeStack struct {
    top atomic.Pointer[node]
}

type node struct {
    value int
    next  *node
}

func (s *LockFreeStack) Push(value int) {
    newNode := &node{value: value}
    for {
        oldTop := s.top.Load()
        newNode.next = oldTop
        // CAS: replace with newNode only if top is still oldTop
        if s.top.CompareAndSwap(oldTop, newNode) {
            return
        }
        // Retry on failure (another goroutine modified it first)
    }
}

func (s *LockFreeStack) Pop() (int, bool) {
    for {
        oldTop := s.top.Load()
        if oldTop == nil {
            return 0, false
        }
        // CAS: replace with next only if top is still oldTop
        if s.top.CompareAndSwap(oldTop, oldTop.next) {
            return oldTop.value, true
        }
        // Retry on failure
    }
}

// Config hot reload using atomic.Value
type HotConfig struct {
    value atomic.Value
}

func NewHotConfig(initial *AppConfig) *HotConfig {
    hc := &HotConfig{}
    hc.value.Store(initial)
    return hc
}

func (hc *HotConfig) Get() *AppConfig {
    return hc.value.Load().(*AppConfig)
}

func (hc *HotConfig) Reload(newConfig *AppConfig) {
    hc.value.Store(newConfig)
    // Readers will get the new config on the next Load()
    // No locking needed; reads are always non-blocking
}

// Hot reload combined with file watching
func (hc *HotConfig) WatchFile(ctx context.Context, path string) error {
    ticker := time.NewTicker(5 * time.Second)
    defer ticker.Stop()

    var lastModified time.Time

    for {
        select {
        case <-ticker.C:
            info, err := os.Stat(path)
            if err != nil {
                log.Printf("Error checking config file: %v", err)
                continue
            }

            if info.ModTime().After(lastModified) {
                cfg, err := loadAppConfig(path)
                if err != nil {
                    log.Printf("Error loading config: %v", err)
                    continue
                }
                hc.Reload(cfg)
                lastModified = info.ModTime()
                log.Printf("Config reload complete: %s", path)
            }

        case <-ctx.Done():
            return ctx.Err()
        }
    }
}
```

---

## 6. sync.Map

### Code Example 6: Basics of sync.Map

```go
var cache sync.Map

func main() {
    // Store
    cache.Store("key1", "value1")
    cache.Store("key2", "value2")

    // Load
    if v, ok := cache.Load("key1"); ok {
        fmt.Println(v.(string))
    }

    // LoadOrStore: retrieve if present, otherwise store
    actual, loaded := cache.LoadOrStore("key3", "value3")
    fmt.Println(actual, loaded) // "value3" false

    // Range: iterate over all entries
    cache.Range(func(key, value any) bool {
        fmt.Printf("%s: %s\n", key, value)
        return true // return false to stop iteration
    })
}
```

### When sync.Map Is and Isn't Appropriate

```go
// When sync.Map is appropriate:
//
// 1. Keys are stable (additions are common, deletions are rare)
// 2. Reads vastly outnumber writes
// 3. Different goroutines access different keys (key-level distribution)

// Case 1: Routing table (set at startup, read at runtime)
var routeHandlers sync.Map

func registerRoute(pattern string, handler http.Handler) {
    routeHandlers.Store(pattern, handler)
}

func findHandler(pattern string) (http.Handler, bool) {
    v, ok := routeHandlers.Load(pattern)
    if !ok {
        return nil, false
    }
    return v.(http.Handler), true
}

// Case 2: Goroutine-local storage
var goroutineData sync.Map

func processWithID(id int) {
    goroutineData.Store(id, &ProcessState{
        StartTime: time.Now(),
    })
    defer goroutineData.Delete(id)

    // Processing...
}

// When sync.Map is inappropriate → use map + RWMutex
//
// 1. Frequent writes
// 2. Frequent iteration over all entries (Range)
// 3. Type safety is required

// Type-safe generic map (generics + RWMutex)
type SafeMap[K comparable, V any] struct {
    mu   sync.RWMutex
    data map[K]V
}

func NewSafeMap[K comparable, V any]() *SafeMap[K, V] {
    return &SafeMap[K, V]{
        data: make(map[K]V),
    }
}

func (m *SafeMap[K, V]) Get(key K) (V, bool) {
    m.mu.RLock()
    defer m.mu.RUnlock()
    v, ok := m.data[key]
    return v, ok
}

func (m *SafeMap[K, V]) Set(key K, value V) {
    m.mu.Lock()
    defer m.mu.Unlock()
    m.data[key] = value
}

func (m *SafeMap[K, V]) Delete(key K) {
    m.mu.Lock()
    defer m.mu.Unlock()
    delete(m.data, key)
}

func (m *SafeMap[K, V]) Len() int {
    m.mu.RLock()
    defer m.mu.RUnlock()
    return len(m.data)
}

func (m *SafeMap[K, V]) Range(fn func(K, V) bool) {
    m.mu.RLock()
    defer m.mu.RUnlock()
    for k, v := range m.data {
        if !fn(k, v) {
            break
        }
    }
}
```

---

## 7. sync.Cond

### Using Condition Variables

```go
// sync.Cond makes goroutines wait until a condition is satisfied.
// It enables "broadcast notification" which is hard to achieve with channels.

type BoundedQueue struct {
    mu       sync.Mutex
    notEmpty *sync.Cond
    notFull  *sync.Cond
    items    []interface{}
    maxSize  int
}

func NewBoundedQueue(maxSize int) *BoundedQueue {
    q := &BoundedQueue{
        items:   make([]interface{}, 0, maxSize),
        maxSize: maxSize,
    }
    q.notEmpty = sync.NewCond(&q.mu)
    q.notFull = sync.NewCond(&q.mu)
    return q
}

func (q *BoundedQueue) Put(item interface{}) {
    q.mu.Lock()
    defer q.mu.Unlock()

    // Wait while the queue is full
    for len(q.items) >= q.maxSize {
        q.notFull.Wait() // mu.Unlock() → wait → mu.Lock()
    }

    q.items = append(q.items, item)
    q.notEmpty.Signal() // Wake up one waiting goroutine
}

func (q *BoundedQueue) Take() interface{} {
    q.mu.Lock()
    defer q.mu.Unlock()

    // Wait while the queue is empty
    for len(q.items) == 0 {
        q.notEmpty.Wait()
    }

    item := q.items[0]
    q.items = q.items[1:]
    q.notFull.Signal()
    return item
}

// Example of Broadcast: notifying all waiters
type ReadyGate struct {
    mu    sync.Mutex
    cond  *sync.Cond
    ready bool
}

func NewReadyGate() *ReadyGate {
    g := &ReadyGate{}
    g.cond = sync.NewCond(&g.mu)
    return g
}

func (g *ReadyGate) Wait() {
    g.mu.Lock()
    defer g.mu.Unlock()
    for !g.ready {
        g.cond.Wait()
    }
}

func (g *ReadyGate) Open() {
    g.mu.Lock()
    defer g.mu.Unlock()
    g.ready = true
    g.cond.Broadcast() // Wake up all waiting goroutines
}

// Usage example
func main() {
    gate := NewReadyGate()

    // Multiple workers wait for the gate to open
    for i := 0; i < 10; i++ {
        go func(id int) {
            gate.Wait()
            fmt.Printf("Worker %d: started\n", id)
        }(i)
    }

    time.Sleep(time.Second)
    fmt.Println("Opening gate...")
    gate.Open() // All workers start simultaneously
}
```

---

## 8. Advanced Use of sync.WaitGroup

```go
// WaitGroup + semaphore: concurrency-limited parallel processing
func processWithLimit(items []Item, maxConcurrency int) error {
    var wg sync.WaitGroup
    sem := make(chan struct{}, maxConcurrency)
    errCh := make(chan error, len(items))

    for _, item := range items {
        wg.Add(1)
        go func(it Item) {
            defer wg.Done()

            sem <- struct{}{}        // Acquire semaphore
            defer func() { <-sem }() // Release semaphore

            if err := process(it); err != nil {
                errCh <- err
            }
        }(item)
    }

    wg.Wait()
    close(errCh)

    // Aggregate errors
    var errs []error
    for err := range errCh {
        errs = append(errs, err)
    }
    if len(errs) > 0 {
        return fmt.Errorf("%d processing errors: %v", len(errs), errs[0])
    }
    return nil
}

// WaitGroup + progress reporting
type ProgressTracker struct {
    total     int
    completed atomic.Int64
    wg        sync.WaitGroup
}

func NewProgressTracker(total int) *ProgressTracker {
    pt := &ProgressTracker{total: total}
    pt.wg.Add(total)
    return pt
}

func (pt *ProgressTracker) Done() {
    pt.completed.Add(1)
    pt.wg.Done()
}

func (pt *ProgressTracker) Wait() {
    pt.wg.Wait()
}

func (pt *ProgressTracker) Progress() float64 {
    return float64(pt.completed.Load()) / float64(pt.total) * 100
}

// Usage example
func processFiles(files []string) {
    pt := NewProgressTracker(len(files))

    // Progress reporting goroutine
    go func() {
        ticker := time.NewTicker(time.Second)
        defer ticker.Stop()
        for {
            select {
            case <-ticker.C:
                fmt.Printf("Progress: %.1f%%\n", pt.Progress())
                if pt.Progress() >= 100 {
                    return
                }
            }
        }
    }()

    for _, file := range files {
        go func(f string) {
            defer pt.Done()
            processFile(f)
        }(file)
    }

    pt.Wait()
    fmt.Println("All files processed")
}
```

---

## 9. Practical Patterns: Combining Multiple Synchronization Primitives

### Pattern 1: Sharded Map

```go
// Sharding to improve the performance of a concurrent map with many keys
const numShards = 32

type ShardedMap[V any] struct {
    shards [numShards]struct {
        mu    sync.RWMutex
        items map[string]V
    }
}

func NewShardedMap[V any]() *ShardedMap[V] {
    sm := &ShardedMap[V]{}
    for i := range sm.shards {
        sm.shards[i].items = make(map[string]V)
    }
    return sm
}

func (sm *ShardedMap[V]) shard(key string) int {
    h := fnv.New32a()
    h.Write([]byte(key))
    return int(h.Sum32()) % numShards
}

func (sm *ShardedMap[V]) Get(key string) (V, bool) {
    s := &sm.shards[sm.shard(key)]
    s.mu.RLock()
    defer s.mu.RUnlock()
    v, ok := s.items[key]
    return v, ok
}

func (sm *ShardedMap[V]) Set(key string, value V) {
    s := &sm.shards[sm.shard(key)]
    s.mu.Lock()
    defer s.mu.Unlock()
    s.items[key] = value
}

func (sm *ShardedMap[V]) Delete(key string) {
    s := &sm.shards[sm.shard(key)]
    s.mu.Lock()
    defer s.mu.Unlock()
    delete(s.items, key)
}

func (sm *ShardedMap[V]) Len() int {
    total := 0
    for i := range sm.shards {
        sm.shards[i].mu.RLock()
        total += len(sm.shards[i].items)
        sm.shards[i].mu.RUnlock()
    }
    return total
}
```

### Pattern 2: Singleton with Lazy Init

```go
// Generic singleton pattern using generics
type Singleton[T any] struct {
    once     sync.Once
    value    T
    initFunc func() T
}

func NewSingletonT any T) *Singleton[T] {
    return &Singleton[T]{initFunc: init}
}

func (s *Singleton[T]) Get() T {
    s.once.Do(func() {
        s.value = s.initFunc()
    })
    return s.value
}

// Usage example
var dbSingleton = NewSingleton(func() *sql.DB {
    db, err := sql.Open("postgres", os.Getenv("DATABASE_URL"))
    if err != nil {
        log.Fatal(err)
    }
    db.SetMaxOpenConns(25)
    db.SetMaxIdleConns(5)
    return db
})

func handler(w http.ResponseWriter, r *http.Request) {
    db := dbSingleton.Get()
    // Use db...
}
```

### Pattern 3: Rate Limiter (Token Bucket)

```go
// Token bucket rate limiter using atomic
type TokenBucket struct {
    tokens     atomic.Int64
    maxTokens  int64
    refillRate int64 // Refill amount per second
    lastRefill atomic.Int64
}

func NewTokenBucket(maxTokens, refillRate int64) *TokenBucket {
    tb := &TokenBucket{
        maxTokens:  maxTokens,
        refillRate: refillRate,
    }
    tb.tokens.Store(maxTokens)
    tb.lastRefill.Store(time.Now().UnixNano())
    return tb
}

func (tb *TokenBucket) refill() {
    now := time.Now().UnixNano()
    last := tb.lastRefill.Load()
    elapsed := float64(now-last) / float64(time.Second)

    if elapsed < 0.001 { // Ignore intervals less than 1ms
        return
    }

    if tb.lastRefill.CompareAndSwap(last, now) {
        newTokens := int64(elapsed * float64(tb.refillRate))
        if newTokens > 0 {
            current := tb.tokens.Load()
            updated := current + newTokens
            if updated > tb.maxTokens {
                updated = tb.maxTokens
            }
            tb.tokens.Store(updated)
        }
    }
}

func (tb *TokenBucket) Allow() bool {
    tb.refill()
    for {
        current := tb.tokens.Load()
        if current <= 0 {
            return false
        }
        if tb.tokens.CompareAndSwap(current, current-1) {
            return true
        }
    }
}
```

---

## 10. ASCII Diagrams

### Diagram 1: Mutex vs RWMutex

```
Mutex (sync.Mutex):
  G1: [===Lock===]
  G2:             [===Lock===]
  G3:                         [===Lock===]
  → All access is serialized

RWMutex (sync.RWMutex):
  G1(R): [==RLock==]
  G2(R): [==RLock==]  ← Reads can happen concurrently
  G3(R): [==RLock==]
  G4(W):              [===Lock===]  ← Writes are exclusive
  G5(R):                           [==RLock==]

  → Improves throughput when reads are frequent

RWMutex writer starvation prevention:
  G1(R): [==RLock==]
  G4(W):              waiting → [===Lock===]
  G5(R):              waiting──────────────>[==RLock==]
  → G5(R) arriving while G4(W) is waiting is also made to wait
```

### Diagram 2: sync.Pool Lifecycle

```
┌──────────────────────────────────────┐
│            sync.Pool                 │
│  ┌──────┐ ┌──────┐ ┌──────┐        │
│  │ buf1 │ │ buf2 │ │ buf3 │ pool    │
│  └──┬───┘ └──────┘ └──────┘        │
│     │                                │
│  Get()  ───> retrieve buf1           │
│              (calls New() if pool is empty) │
│                                      │
│  Put(buf1) ──> return buf1 to pool   │
│                                      │
│  * Objects in the pool may be        │
│    reclaimed during GC               │
└──────────────────────────────────────┘

Internal structure of Pool:
  ┌─────────────────────────────────────┐
  │  P0 (Processor)                     │
  │  ┌─────────────┐  ┌──────────────┐ │
  │  │ private     │  │ shared       │ │
  │  │ (only one)  │  │ (lock-free)  │ │
  │  │ ┌───┐      │  │ ┌───┐┌───┐  │ │
  │  │ │buf│      │  │ │buf││buf│  │ │
  │  │ └───┘      │  │ └───┘└───┘  │ │
  │  └─────────────┘  └──────────────┘ │
  │                                     │
  │  Get: private → shared → other P's shared → New()
  │  Put: private (if empty) → shared   │
  └─────────────────────────────────────┘
```

### Diagram 3: atomic operations vs Mutex

```
atomic.Add:
  Indivisible operation at the CPU instruction level
  ┌─────┐
  │ CAS │  Compare-And-Swap
  │instr│  Read + compare + write in a single instruction
  └─────┘
  → No lock required, fastest

Mutex:
  ┌──────────┐
  │ Lock()   │ ← Spin lock or OS scheduler
  │ operation│
  │ Unlock() │
  └──────────┘
  → Context switching overhead

Performance comparison (approximate):
  atomic.Int64.Add:     ~5ns/op
  sync.Mutex + op:      ~25ns/op  (no contention)
  sync.Mutex + op:      ~100ns/op (high contention)
  sync.RWMutex (Read):  ~15ns/op  (no contention)
```

### Diagram 4: Architecture of ShardedMap

```
Key "user:123" → hash → shard 7

┌─────────────────────────────────────────┐
│              ShardedMap                  │
│                                         │
│  Shard[0]   Shard[1]   ...   Shard[31] │
│  ┌───────┐  ┌───────┐       ┌───────┐ │
│  │RWMutex│  │RWMutex│       │RWMutex│ │
│  │┌─────┐│  │┌─────┐│       │┌─────┐│ │
│  ││ map ││  ││ map ││       ││ map ││ │
│  │└─────┘│  │└─────┘│       │└─────┘│ │
│  └───────┘  └───────┘       └───────┘ │
│                                         │
│  → Each shard has its own independent lock │
│  → Access to different shards can be concurrent │
│  → With 32 shards, up to 32x throughput theoretically │
└─────────────────────────────────────────┘
```

---

## 11. Comparison Tables

### Table 1: Selection Guide for Synchronization Primitives

| Primitive | Purpose | Cost | Thread-safe |
|-----------|---------|------|-------------|
| sync.Mutex | Simple exclusive control | Medium | Yes |
| sync.RWMutex | Read-heavy workloads | Medium | Yes |
| sync.Once | One-time initialization | Low | Yes |
| sync.OnceValue | One-time computation (returns value) | Low | Yes |
| sync.Pool | Object reuse | Low | Yes |
| sync.Map | Map with specific access patterns | Medium | Yes |
| sync.Cond | Condition waiting, broadcasting | Medium | Yes |
| atomic.Int64 | Simple counters | Lowest | Yes |
| atomic.Value | Atomic read/write of any value | Low | Yes |
| atomic.Pointer | CAS operations on pointers | Low | Yes |
| channel | Transferring data ownership | Medium to high | Yes |

### Table 2: sync.Map vs map+Mutex vs ShardedMap

| Item | sync.Map | map + RWMutex | ShardedMap |
|------|----------|---------------|------------|
| Read performance | Very high | High | Very high |
| Write performance | Low to medium | Medium | High |
| Best for | Stable keys, read-heavy | Frequent writes | Many keys, high concurrency |
| Type safety | `any` (type assertion needed) | Type-safe with generics | Type-safe with generics |
| GC pressure | Somewhat high | Low | Low |
| Implementation complexity | Simplest | Simple | Moderate |
| Recommendation | Limited scenarios | Generally recommended | When high performance is needed |

### Table 3: Comparison of Lock Acquisition Strategies

| Strategy | Mechanism | CPU usage | Latency | Use case |
|----------|-----------|-----------|---------|----------|
| Spin lock | Repeated CAS in a loop | High | Low | Short-lived lock holding |
| Mutex (Go) | Spin then semaphore | Adaptive | Medium | General-purpose |
| Channel | Runtime scheduler | Low | Medium to high | Message passing |
| atomic CAS | Single CPU instruction | Lowest | Lowest | Updating a single variable |

---

## 12. Anti-Patterns

### Anti-pattern 1: Copying a Mutex

```go
// BAD: Copying a struct that contains a Mutex
type Counter struct {
    mu sync.Mutex
    n  int
}

func (c Counter) Value() int { // Value receiver → the struct is copied!
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.n
}

// GOOD: Use a pointer receiver
func (c *Counter) Value() int {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.n
}

// Detectable by go vet: copies lock value
```

### Anti-pattern 2: Lock Granularity Too Coarse

```go
// BAD: Locking the entire function
func (s *Service) ProcessOrder(order *Order) error {
    s.mu.Lock()
    defer s.mu.Unlock()

    validated := validate(order)        // Does not require the lock
    enriched := enrichData(validated)    // Does not require the lock
    s.orders[order.ID] = enriched       // Only this requires the lock
    return nil
}

// GOOD: Lock only where needed
func (s *Service) ProcessOrder(order *Order) error {
    validated := validate(order)
    enriched := enrichData(validated)

    s.mu.Lock()
    s.orders[order.ID] = enriched
    s.mu.Unlock()
    return nil
}
```

### Anti-pattern 3: Deadlock

```go
// BAD: Deadlock due to inconsistent lock ordering
func transfer(from, to *Account, amount int) {
    from.mu.Lock()   // goroutine1: A → B order
    to.mu.Lock()     // goroutine2: B → A order (deadlock!)
    // ...
}

// GOOD: Enforce a consistent lock order
func transfer(from, to *Account, amount int) {
    // Lock the one with the smaller ID first (consistent order)
    first, second := from, to
    if from.ID > to.ID {
        first, second = to, from
    }
    first.mu.Lock()
    second.mu.Lock()
    defer first.mu.Unlock()
    defer second.mu.Unlock()

    from.Balance -= amount
    to.Balance += amount
}
```

### Anti-pattern 4: Mixing atomic and Mutex

```go
// BAD: Mixing atomic and Mutex on the same data
type Counter struct {
    mu    sync.Mutex
    count int64
}

func (c *Counter) Inc() {
    atomic.AddInt64(&c.count, 1) // Updated with atomic
}

func (c *Counter) Reset() {
    c.mu.Lock() // Protected by Mutex
    c.count = 0
    c.mu.Unlock()
}
// → atomic and Mutex protections conflict, causing potential data races

// GOOD: Use a consistent synchronization mechanism
type Counter struct {
    count atomic.Int64
}

func (c *Counter) Inc()       { c.count.Add(1) }
func (c *Counter) Reset()     { c.count.Store(0) }
func (c *Counter) Value() int64 { return c.count.Load() }
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
        """Add an item (with size limit)"""
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
| Initialization error | Faulty config file | Check config file path and format |
| Timeout | Network latency / insufficient resources | Adjust timeout values, add retry logic |
| Out of memory | Increasing data volume | Introduce batch processing, implement pagination |
| Permission error | Insufficient access rights | Check the executing user's permissions, review settings |
| Data inconsistency | Concurrent processing conflicts | Introduce locking mechanisms, manage transactions |

### Debugging Steps

1. **Check error messages**: Read the stack trace and identify where the problem occurred
2. **Establish reproduction steps**: Reproduce the error with minimal code
3. **Form hypotheses**: List possible causes
4. **Verify incrementally**: Use logging or a debugger to test hypotheses
5. **Fix and regression test**: After the fix, also test related areas

```python
# Debugging utility
import logging
import traceback
from functools import wraps

# Logger configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def debug_decorator(func):
    """Decorator that logs function inputs and outputs"""
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
    """Process data (the debugging target)"""
    if not items:
        raise ValueError("Empty data")
    return [item * 2 for item in items]
```

### Diagnosing Performance Issues

Steps to diagnose performance issues when they occur:

1. **Identify bottlenecks**: Measure with profiling tools
2. **Check memory usage**: Check for memory leaks
3. **Check I/O waits**: Check disk and network I/O status
4. **Check concurrent connection counts**: Check connection pool state

| Problem type | Diagnostic tool | Countermeasure |
|--------------|-----------------|-----------------|
| CPU load | cProfile, py-spy | Improve algorithms, parallelize |
| Memory leak | tracemalloc, objgraph | Properly release references |
| I/O bottleneck | strace, iostat | Async I/O, caching |
| DB latency | EXPLAIN, slow query log | Indexes, query optimization |

---

## Design Decision Guide

### Selection Criteria Matrix

Criteria to consider when making technical choices:

| Criterion | When to prioritize | When it can be compromised |
|-----------|-------------------|----------------------------|
| Performance | Real-time processing, large data | Admin dashboards, batch jobs |
| Maintainability | Long-term operation, team development | Prototypes, short-term projects |
| Scalability | Services expected to grow | Internal tools, fixed user base |
| Security | Personal information, financial data | Public data, internal use |
| Development speed | MVP, time-to-market | Quality-critical, mission-critical |

### Choosing Architecture Patterns

```
┌─────────────────────────────────────────────────┐
│         Architecture Selection Flow              │
├─────────────────────────────────────────────────┤
│                                                 │
│  1. What's the team size?                       │
│    ├─ Small (1-5) → Monolith                    │
│    └─ Large (10+) → Go to 2                     │
│                                                 │
│  2. How often do you deploy?                    │
│    ├─ Weekly or less → Monolith + modular split │
│    └─ Daily / multiple times → Go to 3          │
│                                                 │
│  3. How independent are the teams?              │
│    ├─ Highly → Microservices                    │
│    └─ Moderately → Modular monolith             │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Analyzing Trade-offs

Technical decisions always involve trade-offs. Analyze them from the following perspectives:

**1. Short-term vs long-term cost**
- A fast short-term approach may become technical debt in the long run
- Conversely, over-engineering raises short-term costs and can delay the project

**2. Consistency vs flexibility**
- A unified tech stack has a lower learning curve
- Adopting diverse technologies allows picking the right tool for the job, but increases operational cost

**3. Level of abstraction**
- High abstraction increases reusability, but can make debugging harder
- Low abstraction is intuitive, but code duplication tends to emerge

```python
# Template for recording design decisions
class ArchitectureDecisionRecord:
    """Creating an ADR (Architecture Decision Record)"""

    def __init__(self, title: str):
        self.title = title
        self.context = ""
        self.decision = ""
        self.consequences = []
        self.alternatives = []

    def set_context(self, context: str):
        """Describe the background and problem"""
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
        """Output as Markdown"""
        md = f"# ADR: {self.title}\n\n"
        md += f"## Context\n{self.context}\n\n"
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

## 13. FAQ

### Q1: What happens if the func in sync.Once panics?

`sync.Once` considers the call "done" even if it panics the first time it runs. It will not be called again. In Go 1.21, `sync.OnceFunc`/`sync.OnceValue` were added, making panic re-raising and error handling easier. `OnceFunc` re-raises the same panic on subsequent calls if a panic occurred.

### Q2: Can sync.Pool be used as a cache?

No. Objects in `sync.Pool` can be reclaimed at any time by the GC. Use `map+Mutex` or a dedicated library (such as groupcache) for caches. Pool should be limited to reusing temporary objects (e.g., buffers). By specification, all objects in the pool are reclaimed after two consecutive GC cycles.

### Q3: Which is faster, atomic or Mutex?

For simple integer operations, atomic is orders of magnitude faster (no locking required). Benchmarks show that atomic is more than 5x faster than Mutex under low contention. However, atomic is limited to operations on a single value. A Mutex is required to maintain consistency across multiple fields.

### Q4: Should I use sync.Cond or a channel?

In most cases, a channel is recommended. sync.Cond is advantageous in only three cases: (1) when broadcast notification is needed (similar to channel close, but repeatable), (2) when waiting on complex conditions is required (condition checks inside a for loop), and (3) when integration with existing Mutex-based code is needed.

### Q5: Can go vet or -race detect synchronization problems?

`go vet` finds statically detectable problems such as Mutex copies. `go test -race` enables the data race detector, which detects data races at runtime. However, `-race` only detects races on code paths that are executed, so test coverage matters. In production, `-race` is usually disabled because of its performance penalty (2-10x slowdown).

### Q6: What happens if the same goroutine holds RLock on RWMutex and then calls Lock?

It deadlocks. Go's RWMutex is not reentrant. If the same goroutine calls Lock while holding RLock, it will wait for the RLock to be released—but since the same goroutine is holding it, it will wait forever. This is a design constraint, and you must carefully manage lock acquisition order.

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

| Concept | Key Points |
|---------|------------|
| Mutex | The basis of exclusive control. Always use defer Unlock() |
| RWMutex | Improves performance for read-heavy workloads. Has writer starvation prevention |
| Once | Safely runs initialization only once. Go 1.21+ adds OnceValue |
| Pool | Reuses temporary objects to reduce GC pressure. Not a cache |
| atomic | Fast lock-free value operations. Supports CAS patterns |
| sync.Map | Concurrency-safe map for specific patterns. In general, prefer map+RWMutex |
| sync.Cond | Condition variable. Use when broadcast notification is needed |
| ShardedMap | Map for large key counts in high-concurrency environments. Independent locks per shard |

---

## Recommended Next Reads

- [02-concurrency-patterns.md](./02-concurrency-patterns.md) -- Concurrency patterns
- [03-context.md](./03-context.md) -- Context
- [../03-tools/02-profiling.md](../03-tools/02-profiling.md) -- Profiling

---

## References

1. **Go Standard Library: sync** -- https://pkg.go.dev/sync
2. **Go Standard Library: sync/atomic** -- https://pkg.go.dev/sync/atomic
3. **Go Memory Model** -- https://go.dev/ref/mem
4. **Go Blog: "Introducing the Go Race Detector"** -- https://go.dev/blog/race-detector
5. **Bryan C. Mills: "Rethinking Classical Concurrency Patterns"** -- GopherCon 2018
