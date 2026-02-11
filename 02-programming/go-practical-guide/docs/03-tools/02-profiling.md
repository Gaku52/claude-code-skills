# Go プロファイリングガイド

> pprof、traceを使ってGoアプリケーションのボトルネックを特定し、パフォーマンスを最適化する

## この章で学ぶこと

1. **pprof** を使ったCPU・メモリ・goroutineプロファイリングの手法
2. **runtime/trace** によるgoroutineスケジューリングとレイテンシの可視化
3. **ベンチマーク連携** — テストからプロファイルを取得して最適化サイクルを回す方法

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
                |
                v
            runtime/pprof (プログラム内で開始/停止)
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

### pprofエンドポイント一覧

| エンドポイント | 内容 |
|--------------|------|
| `/debug/pprof/` | プロファイル一覧ページ |
| `/debug/pprof/profile?seconds=30` | CPUプロファイル（30秒間） |
| `/debug/pprof/heap` | ヒープメモリプロファイル |
| `/debug/pprof/allocs` | メモリアロケーション累積 |
| `/debug/pprof/goroutine` | goroutine スタックトレース |
| `/debug/pprof/block` | ブロッキング操作プロファイル |
| `/debug/pprof/mutex` | ミューテックス競合プロファイル |
| `/debug/pprof/trace?seconds=5` | 実行トレース（5秒間） |

---

## 3. CPU プロファイリング

### コード例2: go tool pprof の操作

```bash
# CPUプロファイル取得（30秒間サンプリング）
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=30

# 保存済みプロファイルの分析
go tool pprof cpu.prof

# Web UI で表示（ブラウザが開く）
go tool pprof -http=:8081 cpu.prof
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
```

### コード例3: プログラム内でCPUプロファイルを取得

```go
package main

import (
    "os"
    "runtime/pprof"
    "log"
)

func main() {
    // CPUプロファイル開始
    f, err := os.Create("cpu.prof")
    if err != nil {
        log.Fatal(err)
    }
    defer f.Close()

    if err := pprof.StartCPUProfile(f); err != nil {
        log.Fatal(err)
    }
    defer pprof.StopCPUProfile()

    // プロファイル対象の処理
    doHeavyWork()
}
```

---

## 4. メモリプロファイリング

### コード例4: ヒーププロファイルの取得と分析

```bash
# ヒーププロファイル取得
go tool pprof http://localhost:6060/debug/pprof/heap

# アロケーション累積（プログラム開始からの総量）
go tool pprof -alloc_space http://localhost:6060/debug/pprof/allocs

# 現在使用中のメモリのみ
go tool pprof -inuse_space http://localhost:6060/debug/pprof/heap
```

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

### コード例5: メモリリークの検出パターン

```go
// メモリリークしやすいパターンと対策
package main

import (
    "runtime"
    "runtime/pprof"
    "os"
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

---

## 5. runtime/trace — 実行トレース

### コード例6: トレースの取得と分析

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

### trace vs pprof 比較表

| 項目 | pprof | trace |
|------|-------|-------|
| 目的 | CPU/メモリのホットスポット特定 | 時系列のイベント分析 |
| 粒度 | 関数レベルの統計 | goroutineレベルのイベント |
| オーバーヘッド | 低（サンプリング） | 高（全イベント記録） |
| 適したケース | 「何が遅い」を知りたい | 「なぜ遅い」を知りたい |
| 可視化 | コールグラフ、フレームグラフ | タイムライン、goroutine解析 |
| 取得時間 | 30秒〜数分 | 数秒〜10秒推奨 |

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

---

## 6. ベンチマーク連携プロファイリング

### コード例7: ベンチマークからプロファイル取得

```bash
# CPUプロファイル付きベンチマーク
go test -bench=BenchmarkSerialize -cpuprofile=cpu.prof -count=5

# メモリプロファイル付きベンチマーク
go test -bench=BenchmarkSerialize -memprofile=mem.prof -count=5

# トレース付きベンチマーク
go test -bench=BenchmarkSerialize -trace=trace.out

# プロファイル分析
go tool pprof cpu.prof
go tool pprof mem.prof
```

### コード例8: メモリアロケーション最適化のサイクル

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

---

## 7. アンチパターン

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

---

## FAQ

### Q1. プロファイリングのオーバーヘッドは本番環境で許容できるか？

`net/http/pprof` のエンドポイントが存在するだけではオーバーヘッドはほぼゼロ。CPUプロファイルはリクエスト時のみサンプリングが走り、通常1-5%程度の影響。メモリプロファイルは `runtime.MemProfileRate` で制御でき、デフォルトでは512KBごとに1回サンプリング。

### Q2. フレームグラフはどう読むか？

フレームグラフは横軸がCPU消費時間の割合、縦軸がコールスタックの深さを表す。幅の広いフレームがボトルネック。上に行くほど呼び出し先の関数。`go tool pprof -http=:8081 cpu.prof` のFlame Graph タブで確認可能。

### Q3. goroutineリークの検出方法は？

`runtime.NumGoroutine()` を定期的にログ出力し、増加傾向がないか監視する。`/debug/pprof/goroutine?debug=1` でgoroutineのスタックトレースを確認し、同じスタックトレースのgoroutineが大量にある場合はリークの可能性が高い。

---

## まとめ

| 概念 | 要点 |
|------|------|
| net/http/pprof | HTTPエンドポイントでプロファイル取得 |
| go tool pprof | プロファイルの分析・可視化ツール |
| CPU profile | 関数ごとのCPU消費時間を特定 |
| Heap profile | メモリアロケーションのホットスポット特定 |
| goroutine profile | goroutineリーク検出 |
| runtime/trace | 時系列イベントの可視化 |
| -bench + -cpuprofile | ベンチマークとプロファイルの連携 |
| b.ReportAllocs() | アロケーション数の計測 |

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
