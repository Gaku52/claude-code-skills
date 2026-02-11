# 同期プリミティブ -- Mutex, RWMutex, Once, Pool, atomic

> syncパッケージはMutex・RWMutex・Once・Poolなどの同期プリミティブを提供し、sync/atomicはロックフリーのアトミック操作を実現する。

---

## この章で学ぶこと

1. **Mutex / RWMutex** -- 排他制御と読み書きロック
2. **sync.Once / sync.Pool** -- 一度だけの初期化とオブジェクトの再利用
3. **sync/atomic** -- ロックフリーなアトミック操作

---

## 1. Mutex

### コード例 1: sync.Mutex

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

### コード例 3: sync.Once

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

### コード例 4: sync.Pool

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

### コード例 5: sync/atomic

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

### コード例 6: sync.Map

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

---

## 2. ASCII図解

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
```

---

## 3. 比較表

### 表1: 同期プリミティブの選択指針

| プリミティブ | 用途 | コスト | スレッドセーフ |
|-------------|------|--------|-------------|
| sync.Mutex | 単純な排他制御 | 中 | はい |
| sync.RWMutex | 読み多・書き少 | 中 | はい |
| sync.Once | 1回だけ初期化 | 低 | はい |
| sync.Pool | オブジェクト再利用 | 低 | はい |
| sync.Map | 特定パターンのmap | 中 | はい |
| atomic.Int64 | 単純なカウンタ | 最低 | はい |
| channel | データの所有権移転 | 中〜高 | はい |

### 表2: sync.Map vs map+Mutex

| 項目 | sync.Map | map + RWMutex |
|------|----------|---------------|
| 読み取り性能 | 非常に高速 | 高速 |
| 書き込み性能 | 低〜中 | 中 |
| 適する場面 | キーが安定、読み取り主体 | 頻繁な書き込み |
| 型安全性 | `any` (型アサーション必要) | ジェネリクスで型安全 |
| GC負荷 | やや高い | 低い |
| 推奨度 | 限定的な場面で使用 | 一般的に推奨 |

---

## 4. アンチパターン

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

---

## 5. FAQ

### Q1: sync.Onceのfuncがpanicしたらどうなるか？

`sync.Once`は一度実行されたらpanicしても「完了」とみなす。再呼び出しされない。Go 1.21で`sync.OnceFunc`/`sync.OnceValue`が追加され、panicの再送出やエラーハンドリングが容易になった。

### Q2: sync.Poolはキャッシュとして使えるか？

使えない。`sync.Pool`のオブジェクトはGC時にいつでも回収される可能性がある。キャッシュには`map+Mutex`や専用ライブラリ（groupcache等）を使う。Poolは一時オブジェクトの再利用（バッファ等）に限定する。

### Q3: atomicとMutexのどちらが速いか？

単純な整数操作ならatomicが桁違いに速い（ロック不要）。ただしatomicは単一の値に対する操作に限られる。複数フィールドの整合性を保つにはMutexが必要。

---

## まとめ

| 概念 | 要点 |
|------|------|
| Mutex | 排他制御の基本。defer Unlock()を常に使う |
| RWMutex | 読み取り多の場合に性能向上 |
| Once | 初期化を1回だけ安全に実行 |
| Pool | 一時オブジェクトの再利用でGC負荷低減 |
| atomic | ロックフリーの高速な値操作 |
| sync.Map | 特定パターン向けの並行安全map |

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
