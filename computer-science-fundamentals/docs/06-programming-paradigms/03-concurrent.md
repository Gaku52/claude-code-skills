# 並行・並列プログラミング

> 並行性は「構造」の問題であり、並列性は「実行」の問題である。——Rob Pike

## この章で学ぶこと

- [ ] 並行（Concurrency）と並列（Parallelism）の違いを説明できる
- [ ] デッドロック、競合状態の原因と対策を理解する
- [ ] async/awaitの仕組みを理解する

---

## 1. 並行 vs 並列

```
並行（Concurrency）: 複数のタスクを「論理的に」同時に管理
  → 1つのCPUコアでも実現可能（タイムスライシング）
  → 構造の問題: 「どう複数の仕事を整理するか」

並列（Parallelism）: 複数のタスクを「物理的に」同時に実行
  → 複数のCPUコアが必要
  → 実行の問題: 「どう物理的に同時実行するか」

  例えで理解:
  並行: 1人のシェフが複数の料理を交互に調理
  並列: 複数のシェフが同時に別々の料理を調理
```

---

## 2. 並行性の問題

### 2.1 競合状態とデッドロック

```python
import threading

# 競合状態（Race Condition）
counter = 0
def increment():
    global counter
    for _ in range(100000):
        counter += 1  # ❌ アトミックでない！
        # 内部: read → increment → write
        # 2つのスレッドが同時にreadすると値が失われる

# 対策: ロック
lock = threading.Lock()
def safe_increment():
    global counter
    for _ in range(100000):
        with lock:  # ✅ 排他制御
            counter += 1

# デッドロック: 2つのスレッドが互いのロック解放を待つ
# Thread A: lock1.acquire() → lock2.acquire()
# Thread B: lock2.acquire() → lock1.acquire()
# → 永遠に待ち続ける！

# デッドロック防止策:
# 1. ロック取得の順序を統一
# 2. タイムアウト付きロック
# 3. ロックフリーアルゴリズム
# 4. できるだけロックを避ける設計
```

### 2.2 並行モデルの比較

```
主要な並行モデル:

  ┌──────────────────┬──────────────────┬─────────────────┐
  │ モデル            │ 特徴             │ 言語/フレームワーク│
  ├──────────────────┼──────────────────┼─────────────────┤
  │ スレッド+ロック   │ 古典的。複雑     │ Java, C++, Python│
  │ アクターモデル    │ メッセージパッシング│ Erlang, Akka    │
  │ CSP             │ チャネル通信     │ Go (goroutine)   │
  │ async/await     │ 協調的マルチタスク │ JS, Python, Rust │
  │ STM             │ トランザクション  │ Haskell, Clojure │
  └──────────────────┴──────────────────┴─────────────────┘
```

---

## 3. async/await

```python
import asyncio

# async/await: シングルスレッドの非同期処理
async def fetch_data(url):
    # I/O待ちの間、他のタスクに制御を渡す
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def main():
    # 3つのリクエストを並行実行（並列ではない！）
    urls = ["http://api1.com", "http://api2.com", "http://api3.com"]
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    # 3つのI/O待ちが重なるので、合計時間 ≈ 最も遅い1つ分

# async/awaitの内部:
# 1. awaitでI/O開始→イベントループに制御を返す
# 2. イベントループが他のタスクを実行
# 3. I/O完了→イベントループがタスクを再開
# → シングルスレッドなのでロック不要！
# → CPU-boundタスクには不向き（GIL問題、Pythonの場合）
```

---

## 4. Go の goroutine

```go
// Go: goroutine + channel（CSPモデル）

func producer(ch chan<- int) {
    for i := 0; i < 10; i++ {
        ch <- i  // チャネルに送信
    }
    close(ch)
}

func consumer(ch <-chan int) {
    for val := range ch {  // チャネルから受信
        fmt.Println(val)
    }
}

func main() {
    ch := make(chan int, 10)  // バッファ付きチャネル
    go producer(ch)            // goroutine起動
    consumer(ch)

    // "Don't communicate by sharing memory,
    //  share memory by communicating." — Go Proverb
}
```

---

## まとめ

| 概念 | ポイント |
|------|---------|
| 並行 vs 並列 | 並行=構造、並列=実行。1コアでも並行は可能 |
| 競合状態 | 共有状態+非アトミック操作。ロックで防止 |
| デッドロック | 循環する待ち合わせ。順序統一で防止 |
| async/await | シングルスレッド非同期。I/O待ちに最適 |
| CSP/Actor | メッセージパッシング。共有状態を排除 |

---

## 次に読むべきガイド
→ [[../07-software-engineering/00-development-process.md]] — ソフトウェア開発プロセス

---

## 参考文献
1. Pike, R. "Concurrency Is Not Parallelism." 2012.
2. Goetz, B. "Java Concurrency in Practice." Addison-Wesley, 2006.
3. Armstrong, J. "Programming Erlang." Pragmatic Bookshelf, 2013.
