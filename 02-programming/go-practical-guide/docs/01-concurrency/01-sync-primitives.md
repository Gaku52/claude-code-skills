# 同期プリミティブ -- Mutex, RWMutex, Once, Pool, atomic

> syncパッケージはMutex・RWMutex・Once・Poolなどの同期プリミティブを提供し、sync/atomicはロックフリーのアトミック操作を実現する。

---

## この章で学ぶこと

1. **Mutex / RWMutex** -- 排他制御と読み書きロック
2. **sync.Once / sync.Pool** -- 一度だけの初期化とオブジェクトの再利用
3. **sync/atomic** -- ロックフリーなアトミック操作
4. **sync.Cond / sync.Map** -- 条件変数と並行安全マップ
5. **実践パターン** -- 本番コードでの同期プリミティブ活用

---

## 1. Mutex

### コード例 1: sync.Mutex の基本

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

### Mutex の内部動作

Go の Mutex は2つのモードを持つ: 通常モード（Normal mode）と飢餓モード（Starvation mode）。

```
通常モード (Normal mode):
  新しく到着した goroutine がロック獲得を試みる
  ┌──────────┐     ┌──────────┐     ┌──────────┐
  │ Waiting  │────>│ Spinning │────>│ Acquired │
  │ goroutine│     │ (CAS)    │     │ Lock     │
  └──────────┘     └──────────┘     └──────────┘
  → 新規 goroutine が待機中の goroutine より先に
    ロックを取得できる（性能は良いが公平ではない）

飢餓モード (Starvation mode):
  待機時間が 1ms を超えた goroutine がいる場合に遷移
  ┌──────────┐     ┌──────────┐
  │ FIFO     │────>│ Acquired │
  │ Queue    │     │ Lock     │
  └──────────┘     └──────────┘
  → 厳密な FIFO 順序でロックを付与
  → スピンなし、公平性を保証
  → 待機キューが空 or 最後の待機者の待ち時間 < 1ms で
    通常モードに戻る
```

### コード例: Mutex で保護されたキャッシュ

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

### コード例 2: sync.RWMutex

```go
type Cache struct {
    mu   sync.RWMutex
    data map[string]string
}

func (c *Cache) Get(key string) (string, bool) {
    c.mu.RLock()         // 読み取りロック（複数同時可）
    defer c.mu.RUnlock()
    v, ok := c.data[key]
    return v, ok
}

func (c *Cache) Set(key, value string) {
    c.mu.Lock()          // 書き込みロック（排他）
    defer c.mu.Unlock()
    c.data[key] = value
}
```

### RWMutex の詳細な動作

```go
// RWMutex の読み取り/書き込みの動作
//
// 読み取りロック (RLock):
//   - 書き込みロックが保持されていなければ即座に取得
//   - 複数の goroutine が同時に RLock を保持可能
//   - 書き込みロックが待機中の場合、新しい RLock は待機する
//     （書き込み側の飢餓を防ぐ）
//
// 書き込みロック (Lock):
//   - 全ての RLock が解放されるまで待機
//   - 取得後は他の RLock も Lock も取得不可

type ConfigManager struct {
    mu     sync.RWMutex
    config map[string]string
}

func NewConfigManager() *ConfigManager {
    return &ConfigManager{
        config: make(map[string]string),
    }
}

// 読み取りは複数同時可能
func (cm *ConfigManager) Get(key string) string {
    cm.mu.RLock()
    defer cm.mu.RUnlock()
    return cm.config[key]
}

// GetAll は設定のスナップショットを返す
func (cm *ConfigManager) GetAll() map[string]string {
    cm.mu.RLock()
    defer cm.mu.RUnlock()

    // コピーを返す（ロック外での変更を防ぐ）
    snapshot := make(map[string]string, len(cm.config))
    for k, v := range cm.config {
        snapshot[k] = v
    }
    return snapshot
}

// 書き込みは排他
func (cm *ConfigManager) Set(key, value string) {
    cm.mu.Lock()
    defer cm.mu.Unlock()
    cm.config[key] = value
}

// バルク更新（トランザクション的な更新）
func (cm *ConfigManager) Update(updates map[string]string) {
    cm.mu.Lock()
    defer cm.mu.Unlock()

    for k, v := range updates {
        cm.config[k] = v
    }
}
```

### RWMutex vs Mutex のベンチマーク

```go
// ベンチマークで読み取り割合による性能差を測定
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

// 結果の目安（8コアマシン）:
// BenchmarkMutexRead-8      20000000    60 ns/op
// BenchmarkRWMutexRead-8    50000000    25 ns/op
//
// → 読み取りが95%以上の場合、RWMutexが有利
// → 書き込みが50%以上の場合、Mutexの方が良いことがある
//   （RWMutexのオーバーヘッドのため）
```

---

## 3. sync.Once

### コード例 3: sync.Once の基本

```go
var (
    instance *Database
    once     sync.Once
)

func GetDB() *Database {
    once.Do(func() {
        // 複数goroutineから呼ばれても1度だけ実行
        instance = &Database{
            conn: connectDB(),
        }
    })
    return instance
}
```

### Go 1.21+ の OnceFunc / OnceValue / OnceValues

```go
// Go 1.21 で追加された便利なヘルパー

// sync.OnceFunc: 関数を1回だけ実行するラッパーを返す
cleanup := sync.OnceFunc(func() {
    fmt.Println("クリーンアップ実行")
    db.Close()
})

cleanup() // "クリーンアップ実行" が出力
cleanup() // 何も起きない（2回目は実行されない）

// sync.OnceValue: 値を1回だけ計算するラッパーを返す
getConfig := sync.OnceValue(func() *Config {
    fmt.Println("設定読み込み中...")
    cfg, err := loadConfig("config.yaml")
    if err != nil {
        panic(err) // panicは再呼び出し時にも再送出される
    }
    return cfg
})

cfg := getConfig() // "設定読み込み中..." が出力
cfg = getConfig()  // キャッシュされた値を返す

// sync.OnceValues: 値とエラーを返す版
loadCert := sync.OnceValues(func() (*tls.Certificate, error) {
    return tls.LoadX509KeyPair("cert.pem", "key.pem")
})

cert, err := loadCert() // 初回: ファイルを読み込む
cert, err = loadCert()  // 2回目: キャッシュされた結果を返す
```

### Once のエラーハンドリングパターン

```go
// sync.Once で初期化エラーを安全に扱うパターン
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

// 注意: Once.Do は panic しても「完了」とみなす
// エラーの場合はリトライしたければ別のアプローチが必要

// リトライ可能な初期化（sync.Once は使えない）
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
        return nil, err // 次回の呼び出しで再試行可能
    }

    if err := db.Ping(); err != nil {
        db.Close()
        return nil, err // 次回の呼び出しで再試行可能
    }

    r.db = db
    r.initDone = true
    return r.db, nil
}
```

---

## 4. sync.Pool

### コード例 4: sync.Pool の基本

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
        bufPool.Put(buf) // プールに返却
    }()

    buf.Write(data)
    return buf.String()
}
```

### sync.Pool の実践的な使用例

```go
// JSON エンコーダのプール
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

// スライスのプール（固定サイズバッファ）
var slicePool = sync.Pool{
    New: func() interface{} {
        s := make([]byte, 0, 4096)
        return &s
    },
}

func processData(input []byte) []byte {
    bufPtr := slicePool.Get().(*[]byte)
    buf := (*bufPtr)[:0] // 長さをリセット、容量は保持
    defer func() {
        *bufPtr = buf[:0]
        slicePool.Put(bufPtr)
    }()

    // buf を使って処理
    buf = append(buf, input...)
    // ... 処理 ...

    // 結果をコピーして返す（プール外に持ち出す場合はコピー必須）
    result := make([]byte, len(buf))
    copy(result, buf)
    return result
}

// sync.Pool を使ったログフォーマッタ
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

### sync.Pool の注意点

```
┌──────────────────────────────────────────────────────┐
│                sync.Pool の特性                       │
│                                                      │
│  ✅ 適切な使い方:                                     │
│    - 頻繁に割り当て・解放される一時オブジェクト       │
│    - バッファ、エンコーダ、フォーマッタ               │
│    - GC負荷の軽減が目的                              │
│                                                      │
│  ❌ 不適切な使い方:                                   │
│    - キャッシュとして使う                             │
│      → GCで予告なく回収される                        │
│    - 接続プール（DB接続、HTTP接続）                   │
│      → 接続の状態管理ができない                      │
│    - 長寿命のオブジェクト                            │
│      → Pool の目的に反する                           │
│                                                      │
│  ⚠️ 注意事項:                                       │
│    - Pool.Get() の戻り値は必ず初期化してから使う     │
│    - Pool.Put() の前にオブジェクトをリセットする     │
│    - Pool から取得したオブジェクトを外に持ち出さない │
│    - ベンチマークで効果を確認してから導入する        │
└──────────────────────────────────────────────────────┘
```

---

## 5. sync/atomic

### コード例 5: atomic の基本（Go 1.19+ 型安全API）

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

// atomic.Value で任意の値を安全に読み書き
var config atomic.Value // *Config

func UpdateConfig(cfg *Config) {
    config.Store(cfg)
}

func GetConfig() *Config {
    return config.Load().(*Config)
}
```

### atomic の詳細な使い方

```go
// Go 1.19+ の型安全な atomic 型
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

### Compare-And-Swap (CAS) パターン

```go
// CAS を使ったロックフリーなスタック
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
        // CAS: top が oldTop のままなら newNode に置き換え
        if s.top.CompareAndSwap(oldTop, newNode) {
            return
        }
        // 失敗した場合はリトライ（他の goroutine が先に変更した）
    }
}

func (s *LockFreeStack) Pop() (int, bool) {
    for {
        oldTop := s.top.Load()
        if oldTop == nil {
            return 0, false
        }
        // CAS: top が oldTop のままなら next に置き換え
        if s.top.CompareAndSwap(oldTop, oldTop.next) {
            return oldTop.value, true
        }
        // 失敗した場合はリトライ
    }
}

// atomic.Value による設定のホットリロード
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
    // 読み取り側は次の Load() で新しい設定を取得
    // ロック不要、読み取りは常にノンブロッキング
}

// ファイル監視と組み合わせたホットリロード
func (hc *HotConfig) WatchFile(ctx context.Context, path string) error {
    ticker := time.NewTicker(5 * time.Second)
    defer ticker.Stop()

    var lastModified time.Time

    for {
        select {
        case <-ticker.C:
            info, err := os.Stat(path)
            if err != nil {
                log.Printf("設定ファイル確認エラー: %v", err)
                continue
            }

            if info.ModTime().After(lastModified) {
                cfg, err := loadAppConfig(path)
                if err != nil {
                    log.Printf("設定読み込みエラー: %v", err)
                    continue
                }
                hc.Reload(cfg)
                lastModified = info.ModTime()
                log.Printf("設定リロード完了: %s", path)
            }

        case <-ctx.Done():
            return ctx.Err()
        }
    }
}
```

---

## 6. sync.Map

### コード例 6: sync.Map の基本

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

    // LoadOrStore: 既存なら取得、なければ格納
    actual, loaded := cache.LoadOrStore("key3", "value3")
    fmt.Println(actual, loaded) // "value3" false

    // Range: 全要素を走査
    cache.Range(func(key, value any) bool {
        fmt.Printf("%s: %s\n", key, value)
        return true // falseで中断
    })
}
```

### sync.Map が適するケースと適さないケース

```go
// sync.Map が適するケース:
//
// 1. キーが安定している（追加はあるが削除は少ない）
// 2. 読み取りが圧倒的に多い
// 3. キーごとにアクセスするgoroutineが異なる（キーの分散）

// ケース1: ルーティングテーブル（起動時に設定、ランタイムで読み取り）
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

// ケース2: goroutine-local なストレージ
var goroutineData sync.Map

func processWithID(id int) {
    goroutineData.Store(id, &ProcessState{
        StartTime: time.Now(),
    })
    defer goroutineData.Delete(id)

    // 処理...
}

// sync.Map が適さないケース → map + RWMutex を使う
//
// 1. 頻繁な書き込みがある
// 2. 全要素の走査（Range）が頻繁
// 3. 型安全性が必要

// 型安全な汎用マップ（ジェネリクス + RWMutex）
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

### 条件変数の使い方

```go
// sync.Cond は条件が満たされるまで goroutine を待機させる
// チャネルでは実現しにくい「ブロードキャスト通知」が可能

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

    // キューが満杯の間は待機
    for len(q.items) >= q.maxSize {
        q.notFull.Wait() // mu.Unlock() → 待機 → mu.Lock()
    }

    q.items = append(q.items, item)
    q.notEmpty.Signal() // 1つの待機goroutineを起こす
}

func (q *BoundedQueue) Take() interface{} {
    q.mu.Lock()
    defer q.mu.Unlock()

    // キューが空の間は待機
    for len(q.items) == 0 {
        q.notEmpty.Wait()
    }

    item := q.items[0]
    q.items = q.items[1:]
    q.notFull.Signal()
    return item
}

// Broadcast の使用例: 全待機者への通知
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
    g.cond.Broadcast() // 全ての待機 goroutine を起こす
}

// 使用例
func main() {
    gate := NewReadyGate()

    // 複数のワーカーがゲートの開放を待つ
    for i := 0; i < 10; i++ {
        go func(id int) {
            gate.Wait()
            fmt.Printf("Worker %d: started\n", id)
        }(i)
    }

    time.Sleep(time.Second)
    fmt.Println("Opening gate...")
    gate.Open() // 全ワーカーが一斉に開始
}
```

---

## 8. sync.WaitGroup の高度な使い方

```go
// WaitGroup + セマフォ: 同時実行数制限付き並行処理
func processWithLimit(items []Item, maxConcurrency int) error {
    var wg sync.WaitGroup
    sem := make(chan struct{}, maxConcurrency)
    errCh := make(chan error, len(items))

    for _, item := range items {
        wg.Add(1)
        go func(it Item) {
            defer wg.Done()

            sem <- struct{}{}        // セマフォ取得
            defer func() { <-sem }() // セマフォ解放

            if err := process(it); err != nil {
                errCh <- err
            }
        }(item)
    }

    wg.Wait()
    close(errCh)

    // エラーの集約
    var errs []error
    for err := range errCh {
        errs = append(errs, err)
    }
    if len(errs) > 0 {
        return fmt.Errorf("処理エラー %d件: %v", len(errs), errs[0])
    }
    return nil
}

// WaitGroup + 進捗報告
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

// 使用例
func processFiles(files []string) {
    pt := NewProgressTracker(len(files))

    // 進捗報告 goroutine
    go func() {
        ticker := time.NewTicker(time.Second)
        defer ticker.Stop()
        for {
            select {
            case <-ticker.C:
                fmt.Printf("進捗: %.1f%%\n", pt.Progress())
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
    fmt.Println("全ファイル処理完了")
}
```

---

## 9. 実践パターン: 複数の同期プリミティブの組み合わせ

### パターン1: Sharded Map（シャーディングマップ）

```go
// 大量のキーを持つ並行マップの性能を向上させるシャーディング
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

### パターン2: Singleton with Lazy Init

```go
// ジェネリクスを使った汎用 Singleton パターン
type Singleton[T any] struct {
    once     sync.Once
    value    T
    initFunc func() T
}

func NewSingleton[T any](init func() T) *Singleton[T] {
    return &Singleton[T]{initFunc: init}
}

func (s *Singleton[T]) Get() T {
    s.once.Do(func() {
        s.value = s.initFunc()
    })
    return s.value
}

// 使用例
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
    // db を使用...
}
```

### パターン3: Rate Limiter (Token Bucket)

```go
// atomic を使ったトークンバケットレートリミッタ
type TokenBucket struct {
    tokens     atomic.Int64
    maxTokens  int64
    refillRate int64 // 1秒あたりの補充数
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

    if elapsed < 0.001 { // 1ms未満は無視
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

## 10. ASCII図解

### 図1: Mutex vs RWMutex

```
Mutex (sync.Mutex):
  G1: [===Lock===]
  G2:             [===Lock===]
  G3:                         [===Lock===]
  → 全アクセスが直列化

RWMutex (sync.RWMutex):
  G1(R): [==RLock==]
  G2(R): [==RLock==]  ← 読み取り同時可
  G3(R): [==RLock==]
  G4(W):              [===Lock===]  ← 書き込みは排他
  G5(R):                           [==RLock==]

  → 読み取りが多い場合にスループット向上

RWMutex の書き込み飢餓防止:
  G1(R): [==RLock==]
  G4(W):              待機 → [===Lock===]
  G5(R):              待機──────────────>[==RLock==]
  → G4(W) が待機中に到着した G5(R) も待機させる
```

### 図2: sync.Pool のライフサイクル

```
┌──────────────────────────────────────┐
│            sync.Pool                 │
│  ┌──────┐ ┌──────┐ ┌──────┐        │
│  │ buf1 │ │ buf2 │ │ buf3 │ プール  │
│  └──┬───┘ └──────┘ └──────┘        │
│     │                                │
│  Get()  ───> buf1 を取得             │
│              (プール空なら New() 呼出) │
│                                      │
│  Put(buf1) ──> buf1 をプールに返却    │
│                                      │
│  ※ GC時にプール内のオブジェクトは     │
│    回収される可能性がある             │
└──────────────────────────────────────┘

Pool の内部構造:
  ┌─────────────────────────────────────┐
  │  P0 (Processor)                     │
  │  ┌─────────────┐  ┌──────────────┐ │
  │  │ private     │  │ shared       │ │
  │  │ (1つだけ)   │  │ (ロックフリー)│ │
  │  │ ┌───┐      │  │ ┌───┐┌───┐  │ │
  │  │ │buf│      │  │ │buf││buf│  │ │
  │  │ └───┘      │  │ └───┘└───┘  │ │
  │  └─────────────┘  └──────────────┘ │
  │                                     │
  │  Get: private → shared → 他Pのshared → New()
  │  Put: private（空なら）→ shared     │
  └─────────────────────────────────────┘
```

### 図3: atomic操作 vs Mutex

```
atomic.Add:
  CPU命令レベルで不可分操作
  ┌─────┐
  │ CAS │  Compare-And-Swap
  │命令  │  1命令で読み取り+比較+書き込み
  └─────┘
  → ロック不要、最速

Mutex:
  ┌──────────┐
  │ Lock()   │ ← スピンロック or OSスケジューラ
  │ 操作     │
  │ Unlock() │
  └──────────┘
  → コンテキストスイッチのオーバーヘッド

性能比較（目安）:
  atomic.Int64.Add:     ~5ns/op
  sync.Mutex + 操作:    ~25ns/op  (競合なし)
  sync.Mutex + 操作:    ~100ns/op (高競合)
  sync.RWMutex (Read):  ~15ns/op  (競合なし)
```

### 図4: ShardedMap のアーキテクチャ

```
キー "user:123" → hash → shard 7

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
│  → 各シャードが独立したロックを持つ      │
│  → 異なるシャードへのアクセスは並行可能  │
│  → 32シャードで理論上32倍のスループット │
└─────────────────────────────────────────┘
```

---

## 11. 比較表

### 表1: 同期プリミティブの選択指針

| プリミティブ | 用途 | コスト | スレッドセーフ |
|-------------|------|--------|-------------|
| sync.Mutex | 単純な排他制御 | 中 | はい |
| sync.RWMutex | 読み多・書き少 | 中 | はい |
| sync.Once | 1回だけ初期化 | 低 | はい |
| sync.OnceValue | 1回だけ計算（値返却） | 低 | はい |
| sync.Pool | オブジェクト再利用 | 低 | はい |
| sync.Map | 特定パターンのmap | 中 | はい |
| sync.Cond | 条件待ち・ブロードキャスト | 中 | はい |
| atomic.Int64 | 単純なカウンタ | 最低 | はい |
| atomic.Value | 任意の値のアトミック読み書き | 低 | はい |
| atomic.Pointer | ポインタのCAS操作 | 低 | はい |
| channel | データの所有権移転 | 中〜高 | はい |

### 表2: sync.Map vs map+Mutex vs ShardedMap

| 項目 | sync.Map | map + RWMutex | ShardedMap |
|------|----------|---------------|------------|
| 読み取り性能 | 非常に高速 | 高速 | 非常に高速 |
| 書き込み性能 | 低〜中 | 中 | 高速 |
| 適する場面 | キーが安定、読み取り主体 | 頻繁な書き込み | 大量キー、高並行 |
| 型安全性 | `any` (型アサーション必要) | ジェネリクスで型安全 | ジェネリクスで型安全 |
| GC負荷 | やや高い | 低い | 低い |
| 実装の複雑さ | 最も簡単 | 簡単 | 中程度 |
| 推奨度 | 限定的な場面 | 一般的に推奨 | 高性能が必要な場面 |

### 表3: ロック取得の戦略比較

| 戦略 | 仕組み | CPU使用 | レイテンシ | 用途 |
|------|--------|---------|-----------|------|
| スピンロック | ループで繰り返しCAS | 高い | 低い | 短時間のロック保持 |
| Mutex (Go) | スピン→セマフォ | 適応的 | 中 | 汎用 |
| チャネル | ランタイムスケジューラ | 低い | 中〜高 | メッセージパッシング |
| atomic CAS | CPU命令1つ | 最低 | 最低 | 単一変数の更新 |

---

## 12. アンチパターン

### アンチパターン 1: Mutexのコピー

```go
// BAD: Mutexを含む構造体をコピー
type Counter struct {
    mu sync.Mutex
    n  int
}

func (c Counter) Value() int { // 値レシーバ → コピーされる！
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.n
}

// GOOD: ポインタレシーバを使う
func (c *Counter) Value() int {
    c.mu.Lock()
    defer c.mu.Unlock()
    return c.n
}

// go vet で検出可能: copies lock value
```

### アンチパターン 2: ロックの粒度が大きすぎる

```go
// BAD: 関数全体をロック
func (s *Service) ProcessOrder(order *Order) error {
    s.mu.Lock()
    defer s.mu.Unlock()

    validated := validate(order)        // ロック不要な処理
    enriched := enrichData(validated)    // ロック不要な処理
    s.orders[order.ID] = enriched       // これだけロック必要
    return nil
}

// GOOD: 必要な箇所だけロック
func (s *Service) ProcessOrder(order *Order) error {
    validated := validate(order)
    enriched := enrichData(validated)

    s.mu.Lock()
    s.orders[order.ID] = enriched
    s.mu.Unlock()
    return nil
}
```

### アンチパターン 3: デッドロック

```go
// BAD: ロック順序の不一致によるデッドロック
func transfer(from, to *Account, amount int) {
    from.mu.Lock()   // goroutine1: A → B の順
    to.mu.Lock()     // goroutine2: B → A の順（デッドロック！）
    // ...
}

// GOOD: ロック順序を統一する
func transfer(from, to *Account, amount int) {
    // ID の小さい方を先にロック（一貫した順序）
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

### アンチパターン 4: atomic と Mutex の混在

```go
// BAD: 同じデータに atomic と Mutex を混在させる
type Counter struct {
    mu    sync.Mutex
    count int64
}

func (c *Counter) Inc() {
    atomic.AddInt64(&c.count, 1) // atomic で更新
}

func (c *Counter) Reset() {
    c.mu.Lock() // Mutex で保護
    c.count = 0
    c.mu.Unlock()
}
// → atomic と Mutex の保護が競合し、データ競合の可能性

// GOOD: 一貫した同期メカニズムを使う
type Counter struct {
    count atomic.Int64
}

func (c *Counter) Inc()       { c.count.Add(1) }
func (c *Counter) Reset()     { c.count.Store(0) }
func (c *Counter) Value() int64 { return c.count.Load() }
```

---

## 13. FAQ

### Q1: sync.Onceのfuncがpanicしたらどうなるか？

`sync.Once`は一度実行されたらpanicしても「完了」とみなす。再呼び出しされない。Go 1.21で`sync.OnceFunc`/`sync.OnceValue`が追加され、panicの再送出やエラーハンドリングが容易になった。`OnceFunc`はpanicが発生した場合、次の呼び出しでも同じpanicを再送出する。

### Q2: sync.Poolはキャッシュとして使えるか？

使えない。`sync.Pool`のオブジェクトはGC時にいつでも回収される可能性がある。キャッシュには`map+Mutex`や専用ライブラリ（groupcache等）を使う。Poolは一時オブジェクトの再利用（バッファ等）に限定する。2回連続のGCでプール内のオブジェクトは全て回収される仕様である。

### Q3: atomicとMutexのどちらが速いか？

単純な整数操作ならatomicが桁違いに速い（ロック不要）。ベンチマーク結果では、低競合時でatomicはMutexの5倍以上速い。ただしatomicは単一の値に対する操作に限られる。複数フィールドの整合性を保つにはMutexが必要。

### Q4: sync.Cond と channel のどちらを使うべきか？

ほとんどのケースではchannelが推奨される。sync.Condが有利なのは、(1) ブロードキャスト通知が必要（チャネルのclose相当だが繰り返し可能）、(2) 複雑な条件での待機が必要（for ループ内の条件チェック）、(3) 既存のMutexベースのコードとの統合が必要、の3ケースに限られる。

### Q5: go vet や -race フラグで同期の問題を検出できるか？

`go vet` はMutexのコピーなど静的に検出可能な問題を発見する。`go test -race` はデータ競合検出器を有効にし、実行時にデータ競合を検出する。ただし `-race` は実行されたコードパスでのみ検出するため、テストカバレッジが重要。本番では `-race` はパフォーマンスペナルティ（2-10倍の遅延）があるため通常は無効にする。

### Q6: RWMutex で RLock 中に同じ goroutine で Lock を取ると何が起きるか？

デッドロックする。Go の RWMutex はリエントラント（再入可能）ではない。同じ goroutine が RLock を保持した状態で Lock を呼ぶと、RLock が解放されるのを待つが、同じ goroutine が保持しているため永久に待機する。これは設計上の制約であり、ロックの取得順序を注意深く管理する必要がある。

---

## まとめ

| 概念 | 要点 |
|------|------|
| Mutex | 排他制御の基本。defer Unlock()を常に使う |
| RWMutex | 読み取り多の場合に性能向上。書き込み飢餓防止あり |
| Once | 初期化を1回だけ安全に実行。Go 1.21+ で OnceValue 追加 |
| Pool | 一時オブジェクトの再利用でGC負荷低減。キャッシュではない |
| atomic | ロックフリーの高速な値操作。CAS パターンも可能 |
| sync.Map | 特定パターン向けの並行安全map。一般的には map+RWMutex |
| sync.Cond | 条件変数。ブロードキャスト通知が必要な場面で使う |
| ShardedMap | 高並行環境での大量キーマップ。シャードごとに独立ロック |

---

## 次に読むべきガイド

- [02-concurrency-patterns.md](./02-concurrency-patterns.md) -- 並行パターン
- [03-context.md](./03-context.md) -- Context
- [../03-tools/02-profiling.md](../03-tools/02-profiling.md) -- プロファイリング

---

## 参考文献

1. **Go Standard Library: sync** -- https://pkg.go.dev/sync
2. **Go Standard Library: sync/atomic** -- https://pkg.go.dev/sync/atomic
3. **Go Memory Model** -- https://go.dev/ref/mem
4. **Go Blog: "Introducing the Go Race Detector"** -- https://go.dev/blog/race-detector
5. **Bryan C. Mills: "Rethinking Classical Concurrency Patterns"** -- GopherCon 2018
