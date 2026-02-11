# Go言語概要 -- 設計哲学とエコシステム

> Goはシンプルさ・並行性・高速コンパイルを柱に設計された、Google発の静的型付けコンパイル言語である。

---

## この章で学ぶこと

1. **Go の設計哲学** -- なぜ「少ない機能」が強みになるのか
2. **並行性モデル** -- goroutine/channel が解決する課題
3. **開発ワークフロー** -- コンパイル・テスト・デプロイの高速サイクル

---

## 1. Goの設計哲学

### コード例 1: Hello World

```go
package main

import "fmt"

func main() {
    fmt.Println("Hello, World!")
}
```

### コード例 2: 複数戻り値

```go
func divide(a, b float64) (float64, error) {
    if b == 0 {
        return 0, fmt.Errorf("division by zero")
    }
    return a / b, nil
}
```

### コード例 3: 構造体とメソッド

```go
type Server struct {
    Host string
    Port int
}

func (s Server) Address() string {
    return fmt.Sprintf("%s:%d", s.Host, s.Port)
}
```

### コード例 4: インターフェース

```go
type Writer interface {
    Write(p []byte) (n int, err error)
}

// 構造体が暗黙的にインターフェースを満たす
type FileWriter struct{}

func (fw FileWriter) Write(p []byte) (int, error) {
    return len(p), nil
}
```

### コード例 5: goroutine と channel

```go
func main() {
    ch := make(chan string)
    go func() {
        ch <- "hello from goroutine"
    }()
    msg := <-ch
    fmt.Println(msg)
}
```

---

## 2. ASCII図解

### 図1: Goのコンパイルフロー

```
┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌────────────┐
│ .go ファイル│───>│  パーサー  │───>│  型チェッカー  │───>│ ネイティブ   │
│ (ソース)   │    │  (AST)   │    │  (SSA/IR)    │    │ バイナリ    │
└──────────┘    └──────────┘    └──────────────┘    └────────────┘
          全工程が数秒で完了 (大規模プロジェクトでも)
```

### 図2: Go のメモリモデル

```
┌─────────────────────────────────────┐
│              Go ランタイム             │
│  ┌──────┐ ┌──────┐ ┌──────┐        │
│  │ G1   │ │ G2   │ │ G3   │ goroutine│
│  └──┬───┘ └──┬───┘ └──┬───┘        │
│     │        │        │             │
│  ┌──▼────────▼────────▼───┐         │
│  │     スケジューラ (M:N)    │         │
│  └──┬────────┬────────┬───┘         │
│     │        │        │             │
│  ┌──▼───┐ ┌──▼───┐ ┌──▼───┐        │
│  │ OS   │ │ OS   │ │ OS   │ スレッド │
│  │Thread│ │Thread│ │Thread│        │
│  └──────┘ └──────┘ └──────┘        │
└─────────────────────────────────────┘
```

### 図3: Go ツールチェイン

```
┌─────────────────────────────────────────┐
│           go コマンド                     │
│                                         │
│  go build   ── コンパイル                  │
│  go test    ── テスト実行                  │
│  go run     ── ビルド+実行                 │
│  go fmt     ── フォーマット                 │
│  go vet     ── 静的解析                    │
│  go mod     ── モジュール管理               │
│  go generate── コード生成                  │
│  go tool pprof ── プロファイリング          │
└─────────────────────────────────────────┘
```

---

## 3. 比較表

### 表1: Go vs 他言語 -- 設計思想比較

| 項目 | Go | Rust | Java | Python |
|------|-----|------|------|--------|
| 型システム | 静的・構造的部分型 | 静的・所有権 | 静的・名前的 | 動的 |
| メモリ管理 | GC | 所有権システム | GC | GC+参照カウント |
| 並行モデル | goroutine+channel | async/await+thread | Thread+Virtual Thread | asyncio/thread |
| コンパイル速度 | 非常に高速 | 低速 | 中程度 | N/A (インタプリタ) |
| バイナリサイズ | 中 (静的リンク) | 小〜中 | 大 (JVM必要) | N/A |
| 学習曲線 | 緩やか | 急峻 | 中程度 | 緩やか |

### 表2: Goが適する領域と不向きな領域

| 適する領域 | 不向きな領域 |
|-----------|-------------|
| マイクロサービス / API サーバー | GUI デスクトップアプリ |
| CLI ツール | 機械学習モデル構築 |
| DevOps / インフラツール | リアルタイムシステム (GCの影響) |
| ネットワークプログラミング | 複雑なジェネリクスが必要な場面 |
| データパイプライン | 動的メタプログラミング |

---

## 4. アンチパターン

### アンチパターン 1: init()の乱用

```go
// BAD: init()で複雑な初期化をする
func init() {
    db, _ = sql.Open("postgres", os.Getenv("DB_URL")) // エラー無視
    db.Ping()                                          // テスト困難
}

// GOOD: 明示的に初期化関数を呼ぶ
func NewDB(url string) (*sql.DB, error) {
    db, err := sql.Open("postgres", url)
    if err != nil {
        return nil, fmt.Errorf("db open: %w", err)
    }
    if err := db.Ping(); err != nil {
        return nil, fmt.Errorf("db ping: %w", err)
    }
    return db, nil
}
```

### アンチパターン 2: パニックをエラーハンドリング代わりに使う

```go
// BAD: panicでエラーを伝搬
func MustParse(s string) int {
    v, err := strconv.Atoi(s)
    if err != nil {
        panic(err) // ライブラリがpanicするべきではない
    }
    return v
}

// GOOD: エラーを返す
func Parse(s string) (int, error) {
    v, err := strconv.Atoi(s)
    if err != nil {
        return 0, fmt.Errorf("parse %q: %w", s, err)
    }
    return v, nil
}
```

---

## 5. FAQ

### Q1: GoにはなぜGenericsが後から追加されたのか？

Go の設計者は「シンプルさ」を最優先し、初期リリース(2009年)では意図的にジェネリクスを省いた。Go 1.18(2022年)で型パラメータが導入されたのは、10年以上の議論と設計検討の結果である。シンプルさを保ちつつ実用的な型安全性を提供する設計が見つかるまで待った、という哲学的判断。

### Q2: Goのガベージコレクタはレイテンシに影響するか？

Go の GC は低レイテンシ設計（目標: STW < 1ms）。Go 1.5 以降、コンカレント GC により大幅に改善された。ほとんどのWebサービスでは問題にならないが、マイクロ秒単位のレイテンシが必要な場合は `sync.Pool` やオブジェクトの再利用を検討する。

### Q3: Goは大規模開発に向いているか？

はい。Google 内部で数百万行規模のGoコードベースが運用されている。パッケージシステム・gofmt による統一フォーマット・高速コンパイルが大規模開発を支える。ただし、型システムの表現力ではRustやHaskellに劣る部分がある。

---

## まとめ

| 概念 | 要点 |
|------|------|
| 設計哲学 | シンプルさ・直交性・明示的なエラーハンドリング |
| 並行性 | goroutine + channel による CSP モデル |
| コンパイル | 静的リンク・高速ビルド・クロスコンパイル対応 |
| ツールチェイン | go build/test/fmt/vet が標準で統合 |
| 型システム | 構造的部分型 (structural subtyping) |
| GC | 低レイテンシ・コンカレント GC |
| エコシステム | 標準ライブラリが充実・サードパーティは go get で管理 |

---

## 次に読むべきガイド

- [01-types-and-structs.md](./01-types-and-structs.md) -- 型とstruct の詳細
- [02-error-handling.md](./02-error-handling.md) -- エラーハンドリングパターン
- [../01-concurrency/00-goroutines-channels.md](../01-concurrency/00-goroutines-channels.md) -- 並行プログラミング入門

---

## 参考文献

1. **The Go Programming Language Specification** -- https://go.dev/ref/spec
2. **Effective Go** -- https://go.dev/doc/effective_go
3. **Rob Pike, "Go Proverbs"** -- https://go-proverbs.github.io/
4. **Donovan, A. & Kernighan, B. (2015) "The Go Programming Language"** -- Addison-Wesley
