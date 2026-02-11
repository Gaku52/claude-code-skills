# Goroutine と Channel -- Go並行プログラミングの基盤

> goroutineは軽量スレッド、channelは型安全な通信路であり、CSPモデルに基づくGoの並行処理の中核を成す。

---

## この章で学ぶこと

1. **goroutine** -- go文による軽量並行実行の仕組み
2. **channel** -- データの安全な受け渡しとselect文
3. **sync.WaitGroup** -- goroutineの完了待ち合わせ

---

## 1. goroutine の基本

### コード例 1: goroutine の起動

```go
func main() {
    go func() {
        fmt.Println("goroutine で実行")
    }()

    go sayHello("World")

    time.Sleep(100 * time.Millisecond) // 待機（実際はWaitGroupを使う）
}

func sayHello(name string) {
    fmt.Printf("Hello, %s!\n", name)
}
```

### コード例 2: channel の基本操作

```go
func main() {
    ch := make(chan int)    // バッファなしチャネル

    go func() {
        ch <- 42            // 送信（受信されるまでブロック）
    }()

    value := <-ch           // 受信（送信されるまでブロック）
    fmt.Println(value)      // 42
}
```

### コード例 3: バッファ付きチャネル

```go
func main() {
    ch := make(chan string, 3) // バッファサイズ3

    ch <- "a" // ブロックしない（バッファに余裕あり）
    ch <- "b"
    ch <- "c"
    // ch <- "d" // ここでブロック（バッファ満杯）

    fmt.Println(<-ch) // "a" (FIFO)
    fmt.Println(<-ch) // "b"
}
```

### コード例 4: select文

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

### コード例 5: WaitGroup

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

    wg.Wait() // 全goroutineの完了を待つ
}
```

### コード例 6: チャネルのクローズとrange

```go
func generate(n int) <-chan int {
    ch := make(chan int)
    go func() {
        defer close(ch) // 送信完了後にクローズ
        for i := 0; i < n; i++ {
            ch <- i * i
        }
    }()
    return ch
}

func main() {
    for v := range generate(5) { // クローズされるまでループ
        fmt.Println(v) // 0, 1, 4, 9, 16
    }
}
```

---

## 2. ASCII図解

### 図1: goroutine と OS スレッドの関係 (M:N スケジューリング)

```
┌─────────────── Go ランタイム ───────────────┐
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

### 図2: バッファなし vs バッファ付きチャネル

```
バッファなし (make(chan int)):
  送信側 ──[同期]──> 受信側
  送信は受信側が準備できるまでブロック

  Sender    Channel    Receiver
    │                    │
    ├──── send ────>     │ (ブロック)
    │              ├──── recv
    │              │
    │    (同期完了)  │

バッファ付き (make(chan int, 3)):
  送信側 ──[buf]──[buf]──[buf]──> 受信側
  バッファが満杯のときだけブロック

  Sender    Buffer[3]    Receiver
    │      [_][_][_]       │
    ├─send─>[A][_][_]      │
    ├─send─>[A][B][_]      │
    │       [A][B][_]──recv─┤  → A
    │       [_][B][_]       │
```

### 図3: select文の動作

```
select {
case msg := <-ch1:     ┐
case msg := <-ch2:     ├── 準備できたケースを実行
case ch3 <- value:     │   複数準備完了ならランダム選択
default:               ┘   defaultはブロック回避用
}

       ┌─ ch1 ready? ──YES──> case 1 実行
       │
select─┼─ ch2 ready? ──YES──> case 2 実行
       │
       ├─ ch3 ready? ──YES──> case 3 実行
       │
       └─ none ready ──────> default実行
                              (defaultなければブロック)
```

---

## 3. 比較表

### 表1: チャネルの種類

| 種類 | 構文 | 送信ブロック条件 | 受信ブロック条件 | 用途 |
|------|------|---------------|---------------|------|
| バッファなし | `make(chan T)` | 受信側がいない | 送信側がいない | 同期的通信 |
| バッファ付き | `make(chan T, n)` | バッファ満杯 | バッファ空 | 非同期的通信 |
| 送信専用 | `chan<- T` | 同上 | コンパイルエラー | APIの制約 |
| 受信専用 | `<-chan T` | コンパイルエラー | 同上 | APIの制約 |

### 表2: goroutine vs OS Thread vs async/await

| 項目 | goroutine | OS Thread | async/await (JS) |
|------|-----------|-----------|-----------------|
| 初期スタックサイズ | ~2KB | ~1MB | N/A |
| 切替コスト | 低い (ユーザー空間) | 高い (カーネル) | 低い (イベントループ) |
| 同時数目安 | 数十万〜数百万 | 数千 | 数万 |
| スケジューリング | Go ランタイム (M:N) | OS | イベントループ |
| メモリモデル | happens-before | OS依存 | シングルスレッド |
| ブロッキングI/O | 自動的にスレッド追加 | スレッド占有 | 不可（別の仕組み必要） |

---

## 4. アンチパターン

### アンチパターン 1: goroutineリーク

```go
// BAD: チャネルから受信されず goroutine が永遠にブロック
func leakySearch(query string) string {
    ch := make(chan string)
    go func() { ch <- searchAPI1(query) }()
    go func() { ch <- searchAPI2(query) }()
    return <-ch // 1つだけ受信。もう1つの goroutine はリーク
}

// GOOD: context でキャンセル、またはバッファ付きチャネル
func safeSearch(ctx context.Context, query string) string {
    ch := make(chan string, 2) // バッファで全goroutineが送信可能
    go func() { ch <- searchAPI1(query) }()
    go func() { ch <- searchAPI2(query) }()
    return <-ch
}
```

### アンチパターン 2: ループ変数のキャプチャ (Go 1.21以前)

```go
// BAD (Go 1.21以前): ループ変数が共有される
for _, url := range urls {
    go func() {
        fetch(url) // 全goroutineが最後のurlを参照
    }()
}

// GOOD: 引数で渡す（Go 1.22以降は不要）
for _, url := range urls {
    url := url // ローカルコピー
    go func() {
        fetch(url)
    }()
}
```

---

## 5. FAQ

### Q1: goroutineはいくつまで起動できるか？

理論上は数百万。各goroutineの初期スタックは約2KBで、必要に応じて動的に拡張される。ただし、CPUバウンドの処理では`runtime.GOMAXPROCS`（デフォルト=CPU数）以上の並列度は出ない。I/Oバウンドなら大量に起動する意味がある。

### Q2: チャネルとMutexのどちらを使うべきか？

原則: 「データの所有権を移転するならチャネル、共有状態を保護するならMutex」。Go Proverbsでは「Don't communicate by sharing memory; share memory by communicating」とされるが、単純なカウンタやキャッシュにはMutexが適切。

### Q3: クローズされたチャネルに送信するとどうなるか？

panicが発生する。チャネルのクローズは「送信側」が行い、「受信側」はrangeやok判定で検知する。複数の送信者がいる場合はsync.Onceでクローズを保護する。

---

## まとめ

| 概念 | 要点 |
|------|------|
| goroutine | `go f()` で起動。軽量（~2KB）、M:Nスケジューリング |
| channel | 型安全な通信路。バッファ有/無の2種類 |
| select | 複数チャネルの待ち合わせ。非決定的選択 |
| WaitGroup | goroutine群の完了待ち |
| close | チャネルの閉鎖。送信側が行う |
| range over channel | クローズまでループ受信 |

---

## 次に読むべきガイド

- [01-sync-primitives.md](./01-sync-primitives.md) -- Mutex/atomic等の同期プリミティブ
- [02-concurrency-patterns.md](./02-concurrency-patterns.md) -- Fan-out/Fan-in等の並行パターン
- [03-context.md](./03-context.md) -- Context によるキャンセル制御

---

## 参考文献

1. **Go Blog, "Share Memory By Communicating"** -- https://go.dev/blog/codelab-share
2. **Go Blog, "Go Concurrency Patterns"** -- https://go.dev/blog/concurrency-patterns
3. **Hoare, C.A.R. (1978) "Communicating Sequential Processes"** -- https://www.cs.cmu.edu/~crary/819-f09/Hoare78.pdf
