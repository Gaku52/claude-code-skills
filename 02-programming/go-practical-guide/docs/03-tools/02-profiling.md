# Go プロファイリングガイド

> pprof、traceを使ってGoアプリケーションのボトルネックを特定し、パフォーマンスを最適化する

## この章で学ぶこと

1. **pprof** を使ったCPU・メモリ・goroutineプロファイリングの手法
2. **runtime/trace** によるgoroutineスケジューリングとレイテンシの可視化
3. **ベンチマーク連携** — テストからプロファイルを取得して最適化サイクルを回す方法
4. **Mutex/Block プロファイリング** — ロック競合とブロッキング操作の分析
5. **継続的プロファイリング** — 本番環境での常時監視戦略

---

## 1. Goプロファイリングの全体像

### プロファイリングツールの分類

```
+----------------------------------------------------------+
|                  Go プロファイリング体系                    |
+----------------------------------------------------------+
|                                                          |
|  +-----------------+  +------------------+  +-----------+|
|  | CPU Profile     |  | Memory Profile   |  | Trace     ||
|  | どこで時間を    |  | どこでメモリを   |  | いつ何が  ||
|  | 消費しているか  |  | 確保しているか   |  | 起きたか  ||
|  +-----------------+  +------------------+  +-----------+|
|         |                     |                   |      |
|         v                     v                   v      |
|  go tool pprof         go tool pprof       go tool trace |
|                                                          |
|  +-----------------+  +------------------+               |
|  | Goroutine Prof  |  | Block Profile    |               |
|  | goroutine の    |  | ロック待ちの     |               |
|  | 状態を確認      |  | 分析             |               |
|  +-----------------+  +------------------+               |
|                                                          |
|  +-----------------+  +------------------+               |
|  | Mutex Profile   |  | Threadcreate     |               |
|  | mutex の競合    |  | OSスレッド生成   |               |
|  | を分析          |  | の追跡           |               |
|  +-----------------+  +------------------+               |
+----------------------------------------------------------+
```

### プロファイル取得方法の選択

```
プロファイルを取りたい
        |
        +-- 本番サーバー（常時稼働）
        |       |
        |       v
        |   net/http/pprof (HTTPエンドポイント)
        |
        +-- テスト・ベンチマーク
        |       |
        |       v
        |   go test -cpuprofile / -memprofile
        |
        +-- 短命プログラム（CLI等）
        |       |
        |       v
        |   runtime/pprof (プログラム内で開始/停止)
        |
        +-- 継続的モニタリング
                |
                v
            Pyroscope / Parca / Google Cloud Profiler
```

### プロファイリングの基本フロー

```
+----------------------------------------------------------+
|  パフォーマンス最適化サイクル                                |
+----------------------------------------------------------+
|                                                          |
|  1. 計測 (Measure)                                       |
|     |  ベンチマーク / 負荷テストで現状を数値化              |
|     v                                                    |
|  2. プロファイル (Profile)                                |
|     |  pprof でホットスポットを特定                        |
|     v                                                    |
|  3. 分析 (Analyze)                                       |
|     |  フレームグラフ・コールグラフで原因を理解             |
|     v                                                    |
|  4. 最適化 (Optimize)                                    |
|     |  ボトルネック箇所のみを改善                          |
|     v                                                    |
|  5. 検証 (Verify)                                        |
|     |  ベンチマークで効果を定量的に確認                    |
|     v                                                    |
|  6. 1に戻る（改善が不十分なら繰り返す）                   |
+----------------------------------------------------------+
```

---

## 2. net/http/pprof — HTTPサーバーのプロファイリング

### コード例1: pprof エンドポイントの追加

```go
package main

import (
    "log"
    "net/http"
    _ "net/http/pprof" // 副作用インポートでエンドポイント登録
)

func main() {
    // アプリケーションのルーティング
    mux := http.NewServeMux()
    mux.HandleFunc("/api/users", handleUsers)

    // pprofは DefaultServeMux に登録されるため
    // 別ポートで pprof 専用サーバーを起動（本番推奨）
    go func() {
        log.Println("pprof server: http://localhost:6060/debug/pprof/")
        log.Fatal(http.ListenAndServe(":6060", nil))
    }()

    // アプリケーションサーバー
    log.Fatal(http.ListenAndServe(":8080", mux))
}
```

### コード例2: カスタム mux での pprof 登録

```go
package main

import (
    "net/http"
    "net/http/pprof"
    "log"
)

func main() {
    // アプリケーション用 mux
    appMux := http.NewServeMux()
    appMux.HandleFunc("/api/users", handleUsers)

    // pprof 専用 mux（認証付き）
    debugMux := http.NewServeMux()
    debugMux.HandleFunc("/debug/pprof/", pprof.Index)
    debugMux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
    debugMux.HandleFunc("/debug/pprof/profile", pprof.Profile)
    debugMux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
    debugMux.HandleFunc("/debug/pprof/trace", pprof.Trace)

    // 認証ミドルウェアを適用
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

### pprofエンドポイント一覧

| エンドポイント | 内容 | 取得方法 |
|--------------|------|---------|
| `/debug/pprof/` | プロファイル一覧ページ | ブラウザで直接アクセス |
| `/debug/pprof/profile?seconds=30` | CPUプロファイル（30秒間） | `go tool pprof URL` |
| `/debug/pprof/heap` | ヒープメモリプロファイル | `go tool pprof URL` |
| `/debug/pprof/allocs` | メモリアロケーション累積 | `go tool pprof URL` |
| `/debug/pprof/goroutine` | goroutine スタックトレース | `go tool pprof URL` |
| `/debug/pprof/block` | ブロッキング操作プロファイル | `go tool pprof URL` |
| `/debug/pprof/mutex` | ミューテックス競合プロファイル | `go tool pprof URL` |
| `/debug/pprof/threadcreate` | OSスレッド生成プロファイル | `go tool pprof URL` |
| `/debug/pprof/trace?seconds=5` | 実行トレース（5秒間） | `go tool trace` |

### pprofエンドポイントのクエリパラメータ

| パラメータ | 適用先 | 説明 | 例 |
|-----------|--------|------|-----|
| `seconds` | profile, trace | サンプリング期間（秒） | `?seconds=60` |
| `debug` | goroutine, heap等 | テキスト出力モード（0=バイナリ, 1=テキスト, 2=詳細） | `?debug=2` |
| `gc` | heap | プロファイル前にGCを実行（1=実行） | `?gc=1` |

---

## 3. CPU プロファイリング

### コード例3: go tool pprof の操作

```bash
# CPUプロファイル取得（30秒間サンプリング）
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=30

# 保存済みプロファイルの分析
go tool pprof cpu.prof

# Web UI で表示（ブラウザが開く）
go tool pprof -http=:8081 cpu.prof

# 特定の関数にフォーカス
go tool pprof -focus=handleRequest cpu.prof

# 特定の関数を除外
go tool pprof -ignore=runtime cpu.prof

# テキスト出力（CI向け）
go tool pprof -text cpu.prof

# 2つのプロファイルの差分を表示
go tool pprof -diff_base=before.prof after.prof
```

### pprof インタラクティブモード

```bash
(pprof) top10
Showing nodes accounting for 4.5s, 90% of 5s total
      flat  flat%   sum%  cum   cum%
      2.0s 40.00% 40.00%  2.0s 40.00%  runtime.memmove
      1.0s 20.00% 60.00%  1.5s 30.00%  encoding/json.(*decodeState).object
      0.5s 10.00% 70.00%  0.5s 10.00%  runtime.mallocgc
      ...

(pprof) list encoding/json.(*decodeState).object
# ソースコード付きで各行のコストを表示

(pprof) web
# SVG のコールグラフをブラウザで表示

(pprof) peek handleRequest
# handleRequest を呼び出している/呼び出されている関数を表示

(pprof) tree
# コールツリー形式で表示

(pprof) disasm handleRequest
# アセンブリコード付きのプロファイル表示
```

### flat vs cum の違い

```
+----------------------------------------------------------+
|  flat vs cum の理解                                       |
+----------------------------------------------------------+
|                                                          |
|  func A() {          A の flat = 1s (A自身の処理)        |
|    doWork() // 1s    A の cum  = 4s (A + B + C の合計)   |
|    B()      // 3s                                        |
|  }                                                       |
|                                                          |
|  func B() {          B の flat = 1s (B自身の処理)        |
|    doWork() // 1s    B の cum  = 3s (B + C の合計)       |
|    C()      // 2s                                        |
|  }                                                       |
|                                                          |
|  func C() {          C の flat = 2s (C自身の処理)        |
|    doWork() // 2s    C の cum  = 2s (C のみ)             |
|  }                                                       |
|                                                          |
|  top コマンドで:                                          |
|  flat が高い → その関数自体が重い                         |
|  cum が高い → その関数の呼び出し先が重い                  |
|  flat と cum の差が大きい → 下流に原因がある              |
+----------------------------------------------------------+
```

### コード例4: プログラム内でCPUプロファイルを取得

```go
package main

import (
    "flag"
    "log"
    "os"
    "runtime/pprof"
)

var cpuprofile = flag.String("cpuprofile", "", "CPUプロファイルの出力先")
var memprofile = flag.String("memprofile", "", "メモリプロファイルの出力先")

func main() {
    flag.Parse()

    // CPUプロファイル開始
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

    // プロファイル対象の処理
    doHeavyWork()

    // メモリプロファイル取得
    if *memprofile != "" {
        f, err := os.Create(*memprofile)
        if err != nil {
            log.Fatal(err)
        }
        defer f.Close()

        // GC を実行して最新のメモリ状態を取得
        runtime.GC()
        if err := pprof.WriteHeapProfile(f); err != nil {
            log.Fatal(err)
        }
    }
}
```

### コード例5: プロファイル付きの HTTP サーバー（条件付き有効化）

```go
package main

import (
    "log"
    "net/http"
    "os"
    "runtime"
)

func main() {
    // 環境変数で pprof を有効化
    if os.Getenv("ENABLE_PPROF") == "true" {
        // Block/Mutex プロファイリングを有効化
        runtime.SetBlockProfileRate(1)
        runtime.SetMutexProfileFraction(1)

        // メモリプロファイリングのサンプリングレートを調整
        // デフォルト: 512KB ごとに1回
        // より詳細にする場合: runtime.MemProfileRate = 1 (全アロケーション記録)
        // 本番環境: runtime.MemProfileRate = 524288 (デフォルト)

        go func() {
            import _ "net/http/pprof"
            log.Println("pprof enabled on :6060")
            log.Fatal(http.ListenAndServe("127.0.0.1:6060", nil))
        }()
    }

    // アプリケーション起動
    srv := &http.Server{Addr: ":8080", Handler: appRouter()}
    log.Fatal(srv.ListenAndServe())
}
```

---

## 4. メモリプロファイリング

### コード例6: ヒーププロファイルの取得と分析

```bash
# ヒーププロファイル取得
go tool pprof http://localhost:6060/debug/pprof/heap

# アロケーション累積（プログラム開始からの総量）
go tool pprof -alloc_space http://localhost:6060/debug/pprof/allocs

# 現在使用中のメモリのみ
go tool pprof -inuse_space http://localhost:6060/debug/pprof/heap

# アロケーション回数（オブジェクト数）
go tool pprof -alloc_objects http://localhost:6060/debug/pprof/allocs

# 現在使用中のオブジェクト数
go tool pprof -inuse_objects http://localhost:6060/debug/pprof/heap

# Web UI でフレームグラフ表示
go tool pprof -http=:8081 http://localhost:6060/debug/pprof/heap
```

### メモリプロファイルのモード比較

| モード | フラグ | 計測対象 | 用途 |
|--------|--------|---------|------|
| inuse_space | `-inuse_space` | 現在使用中のメモリ量 | メモリリーク検出 |
| inuse_objects | `-inuse_objects` | 現在使用中のオブジェクト数 | GC圧力の調査 |
| alloc_space | `-alloc_space` | 累積アロケーション量 | ホットパスの特定 |
| alloc_objects | `-alloc_objects` | 累積アロケーション回数 | アロケーション多発箇所の特定 |

### メモリプロファイルの分析フロー

```
+-------------------+     +-------------------+     +-------------------+
| ヒーププロファイル  | --> | top / list で     | --> | ホットスポット     |
| 取得               |     | アロケーション    |     | 特定              |
|                   |     | 箇所を特定         |     |                   |
+-------------------+     +-------------------+     +-------------------+
                                                            |
                                                            v
+-------------------+     +-------------------+     +-------------------+
| 最適化適用        | <-- | 改善策検討        | <-- | sync.Pool?        |
| ベンチマークで検証 |     | バッファ再利用?   |     | プリアロケート?   |
+-------------------+     +-------------------+     +-------------------+
```

### コード例7: メモリリークの検出パターン

```go
// メモリリークしやすいパターンと対策
package main

import (
    "context"
    "log"
    "os"
    "runtime"
    "runtime/pprof"
    "time"
)

// NG: goroutine リーク
func leakyFunction() {
    for i := 0; i < 1000; i++ {
        go func() {
            ch := make(chan int)
            <-ch // 永遠にブロック → goroutine リーク
        }()
    }
}

// OK: コンテキストでキャンセル可能
func safeFunction(ctx context.Context) {
    for i := 0; i < 1000; i++ {
        go func() {
            ch := make(chan int)
            select {
            case v := <-ch:
                process(v)
            case <-ctx.Done():
                return // クリーンに終了
            }
        }()
    }
}

// goroutine 数の監視
func monitorGoroutines() {
    ticker := time.NewTicker(10 * time.Second)
    for range ticker.C {
        log.Printf("goroutine count: %d", runtime.NumGoroutine())
    }
}

// goroutine プロファイルをファイルに書き出し
func dumpGoroutineProfile() {
    f, _ := os.Create("goroutine.prof")
    defer f.Close()
    pprof.Lookup("goroutine").WriteTo(f, 1)
}
```

### コード例8: メモリリークのスナップショット比較

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

// 2つの時点のヒーププロファイルを比較してリークを検出
func detectMemoryLeak() {
    // スナップショット1を取得
    runtime.GC()
    f1, _ := os.Create("heap_before.prof")
    pprof.WriteHeapProfile(f1)
    f1.Close()

    // 負荷をかける
    runLoad()

    // 一定時間待ってGCを実行
    time.Sleep(30 * time.Second)
    runtime.GC()
    time.Sleep(5 * time.Second) // GCが完了するのを待つ

    // スナップショット2を取得
    f2, _ := os.Create("heap_after.prof")
    pprof.WriteHeapProfile(f2)
    f2.Close()

    // 差分分析
    // go tool pprof -diff_base=heap_before.prof heap_after.prof
    fmt.Println("Run: go tool pprof -http=:8081 -diff_base=heap_before.prof heap_after.prof")
}

// runtime.ReadMemStats でメモリ使用状況を確認
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

### コード例9: スライスのメモリリークパターン

```go
// NG: 大きなスライスの一部を参照 → 元の配列全体がGCされない
func getFirstThree(data []byte) []byte {
    return data[:3]
    // data の底層配列全体が保持される（100MB → 100MB保持）
}

// OK: コピーして参照を切る
func getFirstThree(data []byte) []byte {
    result := make([]byte, 3)
    copy(result, data[:3])
    return result
    // data はGC対象になる
}

// NG: append でキャパシティが過大に残るケース
func filterLarge(items []Item) []Item {
    // 10000 件のスライスから 10 件にフィルタ
    // しかし底層配列は 10000 件分のキャパシティを保持
    var result []Item
    for _, item := range items {
        if item.IsImportant() {
            result = append(result, item)
        }
    }
    return result
}

// OK: 必要に応じてキャパシティを切り詰める
func filterLarge(items []Item) []Item {
    var result []Item
    for _, item := range items {
        if item.IsImportant() {
            result = append(result, item)
        }
    }
    // キャパシティを長さに合わせて切り詰め
    return slices.Clip(result) // Go 1.21+ (= result[:len(result):len(result)])
}
```

---

## 5. Mutex / Block プロファイリング

### コード例10: Mutex プロファイリング

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
    // Mutex プロファイリングを有効化
    // 引数: n → n回のmutex競合ごとに1回サンプリング
    // 1 = 全ての競合を記録（開発用）
    // 5 = 5回に1回記録（本番用）
    runtime.SetMutexProfileFraction(5)

    // Block プロファイリングを有効化
    // 引数: ナノ秒単位の閾値
    // 1 = 全てのブロッキングイベントを記録
    // 1000000 = 1ms以上のブロッキングのみ記録
    runtime.SetBlockProfileRate(1)

    go func() {
        log.Fatal(http.ListenAndServe(":6060", nil))
    }()

    // Mutex 競合が発生するワークロード
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
# Mutex プロファイルの取得
go tool pprof http://localhost:6060/debug/pprof/mutex

# Block プロファイルの取得
go tool pprof http://localhost:6060/debug/pprof/block

# インタラクティブモードで確認
(pprof) top
(pprof) list main.main.func2
(pprof) web
```

### コード例11: RWMutex の競合分析

```go
package main

import (
    "fmt"
    "runtime"
    "sync"
    "time"
)

// 読み取りが多い場合は RWMutex の方が効率的
type Cache struct {
    mu    sync.RWMutex
    items map[string]string
}

func NewCache() *Cache {
    return &Cache{items: make(map[string]string)}
}

func (c *Cache) Get(key string) (string, bool) {
    c.mu.RLock()         // 読み取りロック（並行可能）
    defer c.mu.RUnlock()
    v, ok := c.items[key]
    return v, ok
}

func (c *Cache) Set(key, value string) {
    c.mu.Lock()          // 書き込みロック（排他的）
    defer c.mu.Unlock()
    c.items[key] = value
}

// Mutex vs RWMutex の使い分け
//
// Mutex:
//   - 読み書きの比率が同程度
//   - 実装がシンプル
//   - ロック保持時間が短い場合（RWMutex のオーバーヘッドが相対的に大きくなる）
//
// RWMutex:
//   - 読み取りが圧倒的に多い（90%以上が読み取り）
//   - 読み取り処理に時間がかかる場合
//   - 並行読み取りの恩恵が大きい場合
//
// sync.Map:
//   - キーが安定（追加されるが削除されない）
//   - 読み取りが圧倒的に多い
//   - goroutine ごとにアクセスするキーが異なる

func benchmarkMutexVsRWMutex() {
    cache := NewCache()
    // プリロード
    for i := 0; i < 1000; i++ {
        cache.Set(fmt.Sprintf("key_%d", i), fmt.Sprintf("value_%d", i))
    }

    start := time.Now()
    var wg sync.WaitGroup

    // 95% 読み取り、5% 書き込み
    for i := 0; i < 100; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            for j := 0; j < 10000; j++ {
                key := fmt.Sprintf("key_%d", j%1000)
                if j%20 == 0 { // 5% 書き込み
                    cache.Set(key, fmt.Sprintf("new_%d", j))
                } else { // 95% 読み取り
                    cache.Get(key)
                }
            }
        }(i)
    }

    wg.Wait()
    fmt.Printf("Duration: %v\n", time.Since(start))
}
```

### Lock 競合の可視化フロー

```
+----------------------------------------------------------+
|  Lock 競合の調査フロー                                     |
+----------------------------------------------------------+
|                                                          |
|  1. Mutex プロファイル取得                                 |
|     go tool pprof http://localhost:6060/debug/pprof/mutex |
|     |                                                    |
|     v                                                    |
|  2. top で競合が多い箇所を特定                             |
|     (pprof) top                                          |
|     → contentions/delay が高い関数                        |
|     |                                                    |
|     v                                                    |
|  3. list で具体的なコード行を確認                          |
|     (pprof) list MyFunction                              |
|     |                                                    |
|     v                                                    |
|  4. 改善策の検討                                          |
|     +--> ロック粒度を細かくする（構造体フィールドごと）    |
|     +--> RWMutex に変更                                  |
|     +--> sync.Map / atomic に置き換え                    |
|     +--> ロック保持時間を短縮                              |
|     +--> シャーディング（複数のMutex に分割）             |
+----------------------------------------------------------+
```

---

## 6. runtime/trace — 実行トレース

### コード例12: トレースの取得と分析

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

    // トレース開始
    if err := trace.Start(f); err != nil {
        panic(err)
    }
    defer trace.Stop()

    // トレース対象の処理
    doWork()
}
```

```bash
# トレースの可視化（ブラウザで開く）
go tool trace trace.out
```

### コード例13: カスタムタスクとリージョン

```go
package main

import (
    "context"
    "runtime/trace"
)

func processOrder(ctx context.Context, orderID string) error {
    // タスクを作成（trace UI でグルーピングされる）
    ctx, task := trace.NewTask(ctx, "processOrder")
    defer task.End()

    // リージョン（タスク内のフェーズ）
    trace.WithRegion(ctx, "validate", func() {
        validateOrder(ctx, orderID)
    })

    trace.WithRegion(ctx, "payment", func() {
        processPayment(ctx, orderID)
    })

    trace.WithRegion(ctx, "shipping", func() {
        createShipment(ctx, orderID)
    })

    // ログイベント（trace UI で確認可能）
    trace.Log(ctx, "orderID", orderID)

    return nil
}

// HTTP ハンドラでのトレース
func handleOrder(w http.ResponseWriter, r *http.Request) {
    ctx := r.Context()
    ctx, task := trace.NewTask(ctx, "handleOrder")
    defer task.End()

    trace.WithRegion(ctx, "decode", func() {
        // リクエストのデコード
    })

    trace.WithRegion(ctx, "process", func() {
        // ビジネスロジック
    })

    trace.WithRegion(ctx, "respond", func() {
        // レスポンスの送信
    })
}
```

### trace vs pprof 比較表

| 項目 | pprof | trace |
|------|-------|-------|
| 目的 | CPU/メモリのホットスポット特定 | 時系列のイベント分析 |
| 粒度 | 関数レベルの統計 | goroutineレベルのイベント |
| オーバーヘッド | 低（サンプリング） | 高（全イベント記録） |
| 適したケース | 「何が遅い」を知りたい | 「なぜ遅い」を知りたい |
| 可視化 | コールグラフ、フレームグラフ | タイムライン、goroutine解析 |
| 取得時間 | 30秒〜数分 | 数秒〜10秒推奨 |
| GC分析 | 不可 | GC イベントの詳細が見える |
| ネットワーク | 不可 | ネットワーク待ちが見える |
| スケジューラ | 不可 | P/G/M の関係が見える |

### trace で見えるもの

```
+----------------------------------------------------------+
| go tool trace タイムライン表示                              |
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
| 時間 →  0ms    50ms    100ms    150ms    200ms           |
+----------------------------------------------------------+
  G=goroutine, GC=ガベージコレクション
```

### コード例14: trace による GC 分析

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
    // GC の統計情報を取得
    var stats debug.GCStats
    debug.ReadGCStats(&stats)
    fmt.Printf("GC回数: %d\n", stats.NumGC)
    fmt.Printf("最後のGC: %v\n", stats.LastGC)
    fmt.Printf("GC合計時間: %v\n", stats.PauseTotal)

    // GC パーセンテージの設定
    // GOGC=100 はデフォルト（ヒープが倍になったらGC）
    // GOGC=50 はGC頻度を上げる（レイテンシ重視）
    // GOGC=200 はGC頻度を下げる（スループット重視）
    oldGOGC := debug.SetGCPercent(100)
    fmt.Printf("Previous GOGC: %d\n", oldGOGC)

    // メモリ制限の設定（Go 1.19+）
    // GOMEMLIMIT=1GiB またはプログラムから設定
    debug.SetMemoryLimit(1 << 30) // 1 GiB

    // トレース付きでGC挙動を観察
    f, _ := os.Create("gc_trace.out")
    defer f.Close()
    trace.Start(f)
    defer trace.Stop()

    // メモリを大量に確保する処理
    allocateAndRelease()

    // go tool trace gc_trace.out で GC タイミングを確認
}

func allocateAndRelease() {
    for i := 0; i < 100; i++ {
        data := make([]byte, 10*1024*1024) // 10MB
        _ = data
        time.Sleep(10 * time.Millisecond)
    }
}
```

### GODEBUG 環境変数によるGCトレース

```bash
# GC のタイミングと所要時間を標準エラーに出力
GODEBUG=gctrace=1 ./myapp

# 出力例:
# gc 1 @0.012s 2%: 0.019+0.85+0.003 ms clock, 0.076+0.20/0.75/0+0.012 ms cpu, 4->4->0 MB, 4 MB goal, 0 MB stacks, 0 MB globals, 4 P
#
# 読み方:
# gc 1         → 1回目のGC
# @0.012s      → プログラム開始から0.012秒後
# 2%           → GCがCPU時間の2%を消費
# 0.019+0.85+0.003 ms → STW sweep start + concurrent + STW mark termination
# 4->4->0 MB   → GC前ヒープ -> GC後ヒープ -> ライブデータ
# 4 MB goal     → 次のGCトリガーサイズ

# スケジューラの詳細情報
GODEBUG=schedtrace=1000 ./myapp
# 1000ms ごとにスケジューラの状態を出力
```

---

## 7. ベンチマーク連携プロファイリング

### コード例15: ベンチマークからプロファイル取得

```bash
# CPUプロファイル付きベンチマーク
go test -bench=BenchmarkSerialize -cpuprofile=cpu.prof -count=5

# メモリプロファイル付きベンチマーク
go test -bench=BenchmarkSerialize -memprofile=mem.prof -count=5

# トレース付きベンチマーク
go test -bench=BenchmarkSerialize -trace=trace.out

# Block プロファイル付きベンチマーク
go test -bench=BenchmarkSerialize -blockprofile=block.prof

# Mutex プロファイル付きベンチマーク
go test -bench=BenchmarkSerialize -mutexprofile=mutex.prof

# プロファイル分析
go tool pprof cpu.prof
go tool pprof mem.prof
go tool pprof -http=:8081 cpu.prof
```

### コード例16: メモリアロケーション最適化のサイクル

```go
// 最適化前
func ConcatStrings(strs []string) string {
    result := ""
    for _, s := range strs {
        result += s // 毎回新しい文字列を確保
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

// 最適化後: strings.Builder
func ConcatStringsOptimized(strs []string) string {
    var b strings.Builder
    size := 0
    for _, s := range strs {
        size += len(s)
    }
    b.Grow(size) // 事前にキャパシティ確保
    for _, s := range strs {
        b.WriteString(s)
    }
    return b.String()
}
// BenchmarkConcatStringsOpt  50000   28456 ns/op   5120 B/op   1 allocs/op
//                                    75x高速化            999x削減
```

### コード例17: sync.Pool によるアロケーション削減

```go
package main

import (
    "bytes"
    "encoding/json"
    "sync"
    "testing"
)

// sync.Pool を使ったバッファ再利用
var bufPool = sync.Pool{
    New: func() interface{} {
        return new(bytes.Buffer)
    },
}

// Pool なし版
func marshalJSON(v interface{}) ([]byte, error) {
    var buf bytes.Buffer // 毎回アロケーション
    enc := json.NewEncoder(&buf)
    if err := enc.Encode(v); err != nil {
        return nil, err
    }
    return buf.Bytes(), nil
}

// Pool あり版
func marshalJSONPooled(v interface{}) ([]byte, error) {
    buf := bufPool.Get().(*bytes.Buffer)
    buf.Reset()
    defer bufPool.Put(buf)

    enc := json.NewEncoder(buf)
    if err := enc.Encode(v); err != nil {
        return nil, err
    }

    // Pool に返す前にコピー（bufはPoolに返却されるため）
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

// 典型的な結果:
// BenchmarkMarshalJSON-8          500000   3200 ns/op   768 B/op   3 allocs/op
// BenchmarkMarshalJSONPooled-8    800000   1800 ns/op   256 B/op   2 allocs/op
```

### コード例18: プリアロケーションによる最適化

```go
package main

import "testing"

// NG: append のたびに底層配列が再割り当て
func collectItemsSlow(n int) []int {
    var result []int
    for i := 0; i < n; i++ {
        result = append(result, i*2)
    }
    return result
}

// OK: 事前にキャパシティを確保
func collectItemsFast(n int) []int {
    result := make([]int, 0, n)
    for i := 0; i < n; i++ {
        result = append(result, i*2)
    }
    return result
}

// さらに高速: インデックス直接代入
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

// 典型的な結果:
// BenchmarkCollectSlow-8      10000   152000 ns/op   386048 B/op   20 allocs/op
// BenchmarkCollectFast-8      50000    28000 ns/op    81920 B/op    1 allocs/op
// BenchmarkCollectFastest-8   50000    25000 ns/op    81920 B/op    1 allocs/op
```

---

## 8. フレームグラフの読み方

### フレームグラフの構造

```
+----------------------------------------------------------+
|  フレームグラフの読み方                                     |
+----------------------------------------------------------+
|                                                          |
|  横軸 = CPU消費時間の割合（広いほど時間を使っている）       |
|  縦軸 = コールスタックの深さ（上に行くほど呼び出し先）      |
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
|  → json.Unmarshal と sql.Query が主なボトルネック         |
|  → handleRequest が全体の75%を占めている                  |
+----------------------------------------------------------+
```

### フレームグラフで見るべきポイント

```
1. 幅の広いフレーム
   → CPU時間を多く消費している関数
   → まずここを最適化候補にする

2. 深いコールスタック
   → 呼び出し階層が深い（リファクタリング候補）
   → 間接呼び出しが多い場合はインライン化を検討

3. runtime.* の占める割合
   → runtime.mallocgc が大きい → アロケーションが多い
   → runtime.gcBgMarkWorker が大きい → GC負荷が高い
   → runtime.futex / runtime.notesleep → ロック待ち

4. 同じ関数が複数箇所に出現
   → 異なるコールパスから呼ばれている
   → ホットな共有関数の最適化効果が大きい
```

### コード例19: ベンチマーク結果の比較ツール

```bash
# benchstat でベンチマーク結果を統計的に比較
# インストール
go install golang.org/x/perf/cmd/benchstat@latest

# 最適化前のベンチマーク
go test -bench=. -count=10 -benchmem > before.txt

# 最適化後のベンチマーク
go test -bench=. -count=10 -benchmem > after.txt

# 比較
benchstat before.txt after.txt

# 出力例:
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

## 9. 継続的プロファイリング

### 継続的プロファイリングの必要性

```
+----------------------------------------------------------+
|  従来のプロファイリング vs 継続的プロファイリング            |
+----------------------------------------------------------+
|                                                          |
|  従来:                                                    |
|  - 問題が発生してからプロファイルを取得                    |
|  - 再現が困難な問題を見逃す                               |
|  - 開発環境と本番環境のパフォーマンス差を把握しにくい      |
|                                                          |
|  継続的:                                                  |
|  - 常時プロファイルを収集                                  |
|  - 時系列で傾向を分析（リグレッション検知）               |
|  - デプロイ前後の比較が容易                               |
|  - 低頻度の問題も捕捉可能                                 |
+----------------------------------------------------------+
```

### コード例20: Pyroscope を使った継続的プロファイリング

```go
package main

import (
    "log"
    "net/http"
    "os"

    "github.com/grafana/pyroscope-go"
)

func main() {
    // Pyroscope の設定
    pyroscope.Start(pyroscope.Config{
        ApplicationName: "myapp",
        ServerAddress:   os.Getenv("PYROSCOPE_SERVER"), // 例: http://pyroscope:4040
        Logger:          pyroscope.StandardLogger,

        // プロファイルの種類を選択
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

        // タグでフィルタリング可能にする
        Tags: map[string]string{
            "env":     os.Getenv("APP_ENV"),
            "version": version,
            "region":  os.Getenv("AWS_REGION"),
        },
    })

    // 特定の処理にタグを付ける
    pyroscope.TagWrapper(context.Background(), pyroscope.Labels(
        "handler", "processOrder",
        "orderType", "premium",
    ), func(ctx context.Context) {
        processOrder(ctx)
    })

    http.ListenAndServe(":8080", router())
}
```

### コード例21: Google Cloud Profiler との統合

```go
package main

import (
    "log"

    "cloud.google.com/go/profiler"
)

func main() {
    // Google Cloud Profiler の設定
    cfg := profiler.Config{
        Service:        "myapp",
        ServiceVersion: version,
        ProjectID:      "my-gcp-project",
        // MutexProfiling: true,  // Mutex プロファイリングを有効化
    }

    if err := profiler.Start(cfg); err != nil {
        log.Printf("Cloud Profiler の起動に失敗: %v", err)
        // プロファイラの起動失敗はアプリケーション停止の理由にしない
    }

    // アプリケーション起動
    startServer()
}
```

### 継続的プロファイリングツール比較

| ツール | 提供元 | 価格 | 特徴 |
|--------|--------|------|------|
| Pyroscope | Grafana | OSS / Cloud | Grafana との統合、Go SDK が充実 |
| Parca | Polar Signals | OSS | eBPF ベース、低オーバーヘッド |
| Cloud Profiler | Google | GCP利用料に含む | GCP 統合、設定が簡単 |
| Datadog Profiler | Datadog | 有料 | APM との統合、豊富な分析機能 |
| pprof + 自前収集 | - | 無料 | 柔軟だが運用コスト高 |

---

## 10. 実践的な最適化パターン

### コード例22: HTTP レスポンスのストリーミング最適化

```go
package main

import (
    "encoding/json"
    "net/http"
    "sync"
)

// NG: レスポンス全体をメモリに構築
func handleUsersNG(w http.ResponseWriter, r *http.Request) {
    users, err := db.GetAllUsers() // 全ユーザーをメモリに読み込み
    if err != nil {
        http.Error(w, err.Error(), 500)
        return
    }
    json.NewEncoder(w).Encode(users) // 巨大な JSON を一括エンコード
}

// OK: ストリーミングで段階的に書き出し
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

### コード例23: マップの事前サイズ指定

```go
package main

import "testing"

// NG: サイズ指定なし → rehash が複数回発生
func createMapSlow(n int) map[string]int {
    m := make(map[string]int) // 初期バケット数が小さい
    for i := 0; i < n; i++ {
        m[fmt.Sprintf("key_%d", i)] = i
    }
    return m
}

// OK: 事前にサイズ指定 → rehash を回避
func createMapFast(n int) map[string]int {
    m := make(map[string]int, n) // 必要なバケット数を事前確保
    for i := 0; i < n; i++ {
        m[fmt.Sprintf("key_%d", i)] = i
    }
    return m
}

// ベンチマーク結果例（n=10000）:
// BenchmarkMapSlow-8    5000   312000 ns/op   687432 B/op   172 allocs/op
// BenchmarkMapFast-8    8000   198000 ns/op   473440 B/op    12 allocs/op
```

### コード例24: 文字列操作の最適化

```go
package main

import (
    "fmt"
    "strconv"
    "strings"
    "testing"
)

// NG: fmt.Sprintf は反射を使うため遅い
func formatUserSlow(name string, age int) string {
    return fmt.Sprintf("Name: %s, Age: %d", name, age)
}

// OK: strings.Builder + strconv で高速化
func formatUserFast(name string, age int) string {
    var b strings.Builder
    b.Grow(20 + len(name)) // 必要なサイズを事前確保
    b.WriteString("Name: ")
    b.WriteString(name)
    b.WriteString(", Age: ")
    b.WriteString(strconv.Itoa(age))
    return b.String()
}

// OK: 文字列結合が少ない場合は + 演算子でも十分
func formatUserSimple(name string, age int) string {
    return "Name: " + name + ", Age: " + strconv.Itoa(age)
}

// ベンチマーク結果例:
// BenchmarkFormatSlow-8      5000000    280 ns/op   64 B/op   2 allocs/op
// BenchmarkFormatFast-8     15000000     85 ns/op   48 B/op   1 allocs/op
// BenchmarkFormatSimple-8   12000000     95 ns/op   48 B/op   1 allocs/op
```

### コード例25: インターフェースの具体型アサーションによる最適化

```go
package main

import (
    "io"
    "os"
)

// io.WriterTo インターフェースを活用した最適化
// 標準ライブラリの多くの型が WriterTo を実装している
func copyData(dst io.Writer, src io.Reader) (int64, error) {
    // io.Copy は内部で WriterTo / ReaderFrom をチェックする
    // - src が WriterTo を実装 → src.WriteTo(dst) を呼ぶ
    // - dst が ReaderFrom を実装 → dst.ReadFrom(src) を呼ぶ
    // - どちらもない → 中間バッファを介してコピー
    return io.Copy(dst, src)
}

// バッファサイズの指定（大きなファイルの場合）
func copyLargeFile(dst io.Writer, src io.Reader) (int64, error) {
    // 32KB のデフォルトバッファではなく、より大きなバッファを使用
    buf := make([]byte, 1024*1024) // 1MB バッファ
    return io.CopyBuffer(dst, src, buf)
}

// 型アサーションで最適パスを選択
type Flusher interface {
    Flush() error
}

func writeWithFlush(w io.Writer, data []byte) error {
    if _, err := w.Write(data); err != nil {
        return err
    }
    // Flusher を実装している場合はフラッシュ
    if f, ok := w.(Flusher); ok {
        return f.Flush()
    }
    return nil
}
```

---

## 11. アンチパターン

### アンチパターン1: 本番環境でpprofを公開ポートに露出

```go
// NG: 本番で公開ポートにpprof
import _ "net/http/pprof"

func main() {
    // pprofが外部からアクセス可能 → セキュリティリスク
    http.ListenAndServe(":8080", nil)
}

// OK: pprofは別ポート＋内部ネットワークのみ
func main() {
    go func() {
        // localhost のみ、または内部ネットワークのみ
        log.Fatal(http.ListenAndServe("127.0.0.1:6060", nil))
    }()
    http.ListenAndServe(":8080", appHandler)
}
```

### アンチパターン2: プロファイリングなしの推測的最適化

```go
// NG: 「ここが遅いはず」と推測で最適化
// → 実際にはボトルネックではない箇所に時間を浪費

// OK: プロファイルに基づく最適化サイクル
// 1. ベンチマーク実行
// 2. プロファイル取得
// 3. ホットスポット特定
// 4. 改善実装
// 5. ベンチマークで効果検証
// 6. 1に戻る
```

### アンチパターン3: sync.Pool の誤用

```go
// NG: 小さすぎるオブジェクトに sync.Pool を使う
var intPool = sync.Pool{
    New: func() interface{} {
        v := 0
        return &v // int のポインタはアロケーションが小さすぎてPoolのオーバーヘッドが上回る
    },
}

// NG: Pool から取得したオブジェクトを初期化せずに使う
var bufPool = sync.Pool{
    New: func() interface{} {
        return &bytes.Buffer{}
    },
}

func process() {
    buf := bufPool.Get().(*bytes.Buffer)
    defer bufPool.Put(buf)
    // buf.Reset() を忘れている → 前回のデータが残っている
    buf.WriteString("new data")
}

// OK: 適切なサイズのオブジェクトでPoolを使い、必ず初期化
var bufPool = sync.Pool{
    New: func() interface{} {
        return bytes.NewBuffer(make([]byte, 0, 4096))
    },
}

func process() {
    buf := bufPool.Get().(*bytes.Buffer)
    buf.Reset() // 必ずリセット
    defer bufPool.Put(buf)
    buf.WriteString("new data")
}
```

### アンチパターン4: トレースを長時間取得する

```go
// NG: トレースを60秒間取得 → データが膨大になりUIが固まる
// go tool trace trace_60s.out → ブラウザがクラッシュ

// OK: トレースは短時間（1〜5秒）に限定
// curl "http://localhost:6060/debug/pprof/trace?seconds=3" > trace.out
// go tool trace trace.out

// 特定の操作をトレースしたい場合はプログラム内で制御
func traceOperation(ctx context.Context) error {
    f, _ := os.CreateTemp("", "trace_*.out")
    defer f.Close()

    trace.Start(f)
    defer trace.Stop()

    // トレース対象の操作（短時間で完了するもの）
    return doOperation(ctx)
}
```

### アンチパターン5: MemProfileRate を 1 にして本番運用

```go
// NG: 全アロケーションを記録（パフォーマンスへの影響大）
func init() {
    runtime.MemProfileRate = 1 // 全てのアロケーションを記録
}

// OK: 本番環境ではデフォルト値を使う
// runtime.MemProfileRate のデフォルトは 524288 (512KB)
// 必要に応じて調整
func init() {
    if os.Getenv("DETAILED_MEMPROFILE") == "true" {
        runtime.MemProfileRate = 1 // デバッグ時のみ
    }
    // それ以外はデフォルト (512KB ごとに1回サンプリング)
}
```

---

## FAQ

### Q1. プロファイリングのオーバーヘッドは本番環境で許容できるか？

`net/http/pprof` のエンドポイントが存在するだけではオーバーヘッドはほぼゼロ。CPUプロファイルはリクエスト時のみサンプリングが走り、通常1-5%程度の影響。メモリプロファイルは `runtime.MemProfileRate` で制御でき、デフォルトでは512KBごとに1回サンプリング。Block/Mutex プロファイルは `SetBlockProfileRate` / `SetMutexProfileFraction` で制御し、本番では低いサンプリングレートを推奨。

### Q2. フレームグラフはどう読むか？

フレームグラフは横軸がCPU消費時間の割合、縦軸がコールスタックの深さを表す。幅の広いフレームがボトルネック。上に行くほど呼び出し先の関数。`go tool pprof -http=:8081 cpu.prof` のFlame Graph タブで確認可能。`runtime.mallocgc` が広い場合はアロケーション過多、`runtime.gcBgMarkWorker` が広い場合はGC負荷が高いことを示す。

### Q3. goroutineリークの検出方法は？

`runtime.NumGoroutine()` を定期的にログ出力し、増加傾向がないか監視する。`/debug/pprof/goroutine?debug=1` でgoroutineのスタックトレースを確認し、同じスタックトレースのgoroutineが大量にある場合はリークの可能性が高い。`goleak` パッケージ（`go.uber.org/goleak`）をテストに組み込むことで、テスト終了時に未完了のgoroutineを検出できる。

```go
// goleak を使ったgoroutineリーク検出テスト
func TestMain(m *testing.M) {
    goleak.VerifyTestMain(m)
}

// 個別のテストで使う場合
func TestSomething(t *testing.T) {
    defer goleak.VerifyNone(t)
    // テストコード
}
```

### Q4. pprof の Web UI で利用できるビューは？

`go tool pprof -http=:8081 profile.prof` で起動するWeb UIには以下のビューがある。
- **Top**: 関数ごとのCPU/メモリ消費ランキング
- **Graph**: コールグラフ（関数間の呼び出し関係）
- **Flame Graph**: フレームグラフ（横幅=コスト、縦=コールスタック深さ）
- **Peek**: 特定関数の呼び出し元・呼び出し先
- **Source**: ソースコード上でのコスト表示
- **Disasm**: アセンブリコード上でのコスト表示

### Q5. GOGC と GOMEMLIMIT の使い分けは？

`GOGC` はヒープの成長率に基づいてGCをトリガーする（デフォルト100 = ヒープが倍になったらGC）。`GOMEMLIMIT`（Go 1.19+）はメモリの上限を設定し、上限に近づくとGCを積極的に実行する。コンテナ環境では `GOMEMLIMIT` をコンテナのメモリ制限の80-90%に設定するのが推奨。両方を組み合わせて使うことも可能。

```bash
# コンテナ環境での推奨設定例
# コンテナメモリ制限: 1GB
GOMEMLIMIT=900MiB  # メモリ制限の90%
GOGC=100            # デフォルト（GOMEMLIMITと組み合わせ）
```

### Q6. ベンチマークの結果が毎回異なるのはなぜか？

CPUのサーマルスロットリング、他のプロセスの影響、OSのスケジューリングなどが原因。安定した結果を得るには: (1) `-count=10` で複数回実行し `benchstat` で統計処理、(2) `taskset` / `cpuset` でCPUを固定、(3) ターボブーストを無効化、(4) 他のプロセスを最小限にする。CI環境ではノイズが大きいため、ローカルでの計測を推奨。

### Q7. Escape Analysis の結果を確認する方法は？

```bash
# ヒープへの escape を確認
go build -gcflags="-m" ./...

# より詳細な情報
go build -gcflags="-m -m" ./...

# 出力例:
# ./main.go:15:6: can inline NewUser
# ./main.go:20:10: &User{...} escapes to heap
# → "&User{...}" がヒープに割り当てられることがわかる
```

スタックに割り当てられるとGCの負荷がかからないため高速。ヒープに escape する主な原因: (1) ポインタを返す、(2) インターフェースに代入、(3) クロージャで参照、(4) サイズが大きすぎる（通常64KB以上）。

---

## まとめ

| 概念 | 要点 |
|------|------|
| net/http/pprof | HTTPエンドポイントでプロファイル取得 |
| go tool pprof | プロファイルの分析・可視化ツール |
| CPU profile | 関数ごとのCPU消費時間を特定 |
| Heap profile | メモリアロケーションのホットスポット特定 |
| goroutine profile | goroutineリーク検出 |
| Mutex/Block profile | ロック競合とブロッキング操作の分析 |
| runtime/trace | 時系列イベントの可視化 |
| -bench + -cpuprofile | ベンチマークとプロファイルの連携 |
| b.ReportAllocs() | アロケーション数の計測 |
| sync.Pool | オブジェクト再利用でアロケーション削減 |
| benchstat | ベンチマーク結果の統計的比較 |
| GOGC / GOMEMLIMIT | GCの挙動制御 |
| 継続的プロファイリング | Pyroscope / Parca / Cloud Profiler |
| Escape Analysis | `go build -gcflags="-m"` でヒープ割り当てを確認 |

---

## 次に読むべきガイド

- **03-tools/03-deployment.md** — デプロイ：Docker、クロスコンパイル
- **03-tools/04-best-practices.md** — ベストプラクティス：Effective Go
- **02-web/04-testing.md** — テスト：table-driven tests、testify、httptest

---

## 参考文献

1. **Go Blog — Profiling Go Programs** https://go.dev/blog/pprof
2. **Go公式 — runtime/pprof パッケージ** https://pkg.go.dev/runtime/pprof
3. **Go公式 — runtime/trace パッケージ** https://pkg.go.dev/runtime/trace
4. **Julia Evans — A Practical Guide to pprof** https://jvns.ca/blog/2017/09/24/profiling-go-with-pprof/
5. **Go公式 — runtime/debug パッケージ** https://pkg.go.dev/runtime/debug
6. **Pyroscope 公式ドキュメント** https://pyroscope.io/docs/
7. **Google Cloud Profiler** https://cloud.google.com/profiler/docs
8. **benchstat ツール** https://pkg.go.dev/golang.org/x/perf/cmd/benchstat
