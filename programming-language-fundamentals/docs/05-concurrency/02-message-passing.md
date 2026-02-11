# メッセージパッシング

> 「共有メモリではなく、メッセージの送受信で通信する」。データ競合を根本的に回避する並行処理の設計原則。

## この章で学ぶこと

- [ ] メッセージパッシングの概念と利点を理解する
- [ ] チャネル、アクターモデルの仕組みを理解する
- [ ] 共有メモリとの使い分けを判断できる

---

## 1. 共有メモリ vs メッセージパッシング

```
共有メモリ:
  ┌─────────┐     ┌─────────┐
  │ Thread A │────→│ 共有データ │←────│ Thread B │
  └─────────┘  ↑  └─────────┘  ↑  └─────────┘
               Mutex で排他制御
  問題: デッドロック、データ競合のリスク

メッセージパッシング:
  ┌─────────┐  message  ┌─────────┐
  │ Thread A │──────────→│ Thread B │
  └─────────┘  channel  └─────────┘
  各スレッドが独自のデータを所有。通信はメッセージのみ

  Go の格言: "Do not communicate by sharing memory;
              share memory by communicating."
```

---

## 2. Go のチャネル

```go
// チャネルの基本
ch := make(chan int)       // バッファなしチャネル
ch := make(chan int, 10)   // バッファ付き（10要素）

// 送信と受信
ch <- 42         // 送信（受信者がいるまでブロック）
value := <-ch    // 受信（送信者がいるまでブロック）

// ワーカープール
func worker(id int, jobs <-chan int, results chan<- int) {
    for job := range jobs {
        results <- job * 2  // 処理して結果を返す
    }
}

func main() {
    jobs := make(chan int, 100)
    results := make(chan int, 100)

    // 3つのワーカーを起動
    for w := 0; w < 3; w++ {
        go worker(w, jobs, results)
    }

    // ジョブを送信
    for j := 0; j < 10; j++ {
        jobs <- j
    }
    close(jobs)

    // 結果を収集
    for r := 0; r < 10; r++ {
        fmt.Println(<-results)
    }
}

// パイプライン
func generate(nums ...int) <-chan int {
    out := make(chan int)
    go func() {
        for _, n := range nums {
            out <- n
        }
        close(out)
    }()
    return out
}

func square(in <-chan int) <-chan int {
    out := make(chan int)
    go func() {
        for n := range in {
            out <- n * n
        }
        close(out)
    }()
    return out
}

// パイプライン実行: generate → square → 出力
for v := range square(generate(1, 2, 3, 4, 5)) {
    fmt.Println(v)  // 1, 4, 9, 16, 25
}
```

---

## 3. Rust のチャネル

```rust
use std::sync::mpsc;  // Multi-Producer, Single-Consumer
use std::thread;

// 基本的なチャネル
let (tx, rx) = mpsc::channel();

thread::spawn(move || {
    tx.send("Hello from thread!").unwrap();
    // tx の所有権がスレッドに移動（move）
});

let received = rx.recv().unwrap();
println!("{}", received);

// 複数プロデューサー
let (tx, rx) = mpsc::channel();
for i in 0..5 {
    let tx = tx.clone();  // 送信側をクローン
    thread::spawn(move || {
        tx.send(format!("Message from thread {}", i)).unwrap();
    });
}
drop(tx);  // 元の tx を破棄（全 clone が終了すると rx がNoneを返す）

for msg in rx {
    println!("{}", msg);
}

// crossbeam チャネル（より高機能）
use crossbeam_channel::{bounded, select};

let (s1, r1) = bounded(0);  // ランデブーチャネル
let (s2, r2) = bounded(10); // バッファ付き

// select!（複数チャネルの待機）
select! {
    recv(r1) -> msg => println!("From r1: {:?}", msg),
    recv(r2) -> msg => println!("From r2: {:?}", msg),
    default(Duration::from_secs(1)) => println!("Timeout"),
}
```

---

## 4. アクターモデル

```
アクターモデル = 独立した「アクター」がメッセージで通信

  各アクター:
    - 独自の状態を持つ（外部からアクセス不可）
    - メッセージを受信して処理
    - 新しいアクターを生成できる
    - 他のアクターにメッセージを送信

  ┌──────┐  msg  ┌──────┐  msg  ┌──────┐
  │Actor1│──────→│Actor2│──────→│Actor3│
  │ state│       │ state│       │ state│
  └──────┘       └──────┘       └──────┘
  ← 各アクターは独立、メッセージのみで通信 →
```

```elixir
# Elixir: アクターモデル（Erlang VM上）
defmodule Counter do
  def start(initial \\ 0) do
    spawn(fn -> loop(initial) end)
  end

  defp loop(count) do
    receive do
      {:increment, caller} ->
        send(caller, {:count, count + 1})
        loop(count + 1)
      {:get, caller} ->
        send(caller, {:count, count})
        loop(count)
    end
  end
end

# 使用
counter = Counter.start(0)
send(counter, {:increment, self()})
send(counter, {:get, self()})
receive do
  {:count, n} -> IO.puts("Count: #{n}")  # → 1
end
```

---

## 5. 使い分け

```
メッセージパッシングが適する場面:
  ✓ 独立した処理単位が明確
  ✓ パイプライン処理
  ✓ 分散システム（ネットワーク越し）
  ✓ 障害分離が重要（1アクターの失敗が他に影響しない）

共有メモリが適する場面:
  ✓ 低レイテンシが必要
  ✓ 大量のデータを共有
  ✓ 単純なカウンタ・キャッシュ
```

---

## まとめ

| モデル | 通信方式 | 安全性 | 代表言語 |
|-------|---------|--------|---------|
| チャネル | 型付きメッセージ | 高い | Go, Rust |
| アクター | 非同期メッセージ | 最も高い | Erlang, Elixir |
| 共有メモリ | Mutex + 共有変数 | 低い | C, Java |

---

## 次に読むべきガイド
→ [[03-parallel-programming.md]] — 並列プログラミング

---

## 参考文献
1. Donovan, A. & Kernighan, B. "The Go Programming Language." Ch.8-9, 2015.
2. Armstrong, J. "Programming Erlang." 2nd Ed, 2013.
