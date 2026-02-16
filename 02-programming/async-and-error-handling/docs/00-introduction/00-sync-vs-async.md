# 同期 vs 非同期

> 同期処理は「前の処理が終わるまで次を待つ」、非同期処理は「待ち時間に他の処理を進める」。Webアプリケーションのパフォーマンスの鍵は、I/O待ちを効率的に処理すること。

## この章で学ぶこと

- [ ] 同期処理と非同期処理の根本的な違いを理解する
- [ ] ブロッキングとノンブロッキングの意味を把握する
- [ ] なぜ非同期処理が必要かを具体的に理解する
- [ ] 各言語における同期・非同期モデルの特徴を比較する
- [ ] 実務で遭遇する典型的なシナリオと最適な選択を学ぶ

---

## 1. 同期 vs 非同期の根本概念

### 1.1 視覚的な理解

```
同期処理（Synchronous）:
  処理A ████████████████████
  処理B                     ████████████████████
  処理C                                         ████████████████████
  → 順番に実行。前が終わるまで次は待つ
  → 合計時間 = A + B + C

非同期処理（Asynchronous）:
  処理A ████──────────████
  処理B     ████──────────████
  処理C         ████──────────████
  → I/O待ち（──）の間に他の処理を進める
  → 合計時間 ≒ max(A, B, C)

具体例: 3つのAPI呼び出し（各200ms）
  同期:  200 + 200 + 200 = 600ms
  非同期: max(200, 200, 200) = 200ms（3倍速）
```

### 1.2 日常の比喩で理解する

同期処理と非同期処理の違いは、レストランの注文に例えるとわかりやすい。

```
同期処理（1人のウェイターが1テーブルずつ完全対応）:
  テーブル1: 注文受付 → 料理完成待ち → 配膳 → 会計
  テーブル2:                                     注文受付 → 料理完成待ち → 配膳 → 会計
  テーブル3:                                                                      注文受付 → ...
  → 料理を待っている間もウェイターは立ちっぱなし
  → 非常に非効率

非同期処理（1人のウェイターが複数テーブルを効率的に処理）:
  テーブル1: 注文受付 →（キッチンに渡す）→ ... → 料理できた！配膳
  テーブル2:           注文受付 →（キッチンに渡す）→ ... → 料理できた！配膳
  テーブル3:                     注文受付 →（キッチンに渡す）→ ...
  → 料理待ちの間に他のテーブルを対応
  → 1人で多くのテーブルを効率的にさばける
```

### 1.3 プログラミングにおける定義

```
同期（Synchronous）:
  - 呼び出し元が処理の完了を待ってから次に進む
  - 処理の順序が保証される
  - コードの流れが直線的で理解しやすい
  - 関数の戻り値として結果を直接受け取る

非同期（Asynchronous）:
  - 呼び出し元が処理の完了を待たずに次に進む
  - 結果は後で通知される（コールバック、Promise、イベントなど）
  - 処理の順序が非決定的になりうる
  - より複雑だが、リソースを効率的に使える
```

---

## 2. ブロッキング vs ノンブロッキング

### 2.1 基本概念

```
ブロッキングI/O:
  → I/O完了までスレッドが停止
  → スレッドはCPUを消費しないが、占有したまま

  Thread1: [リクエスト受信] → [DBクエリ... 100ms 待ち...] → [レスポンス]
  Thread2: [リクエスト受信] → [API呼出... 200ms 待ち...] → [レスポンス]
  Thread3: [リクエスト受信] → [ファイル読み... 50ms 待ち...] → [レスポンス]
  → 同時接続数 = スレッド数に制限される

ノンブロッキングI/O:
  → I/O開始後すぐに制御が戻る
  → 完了時にコールバック/イベントで通知

  Thread1: [リクエスト1] [リクエスト2] [リクエスト3] [DB結果処理] [API結果処理]
  → 1スレッドで多数のリクエストを処理可能
  → Node.js のモデル
```

### 2.2 ブロッキングI/Oの詳細

ブロッキングI/Oでは、OSのシステムコール（read, write, connect など）が完了するまでスレッドがブロックされる。

```typescript
// ブロッキングI/Oのイメージ（擬似コード）
function handleRequest(socket: Socket): void {
  // 1. リクエストを読み取り（ブロック）
  const request = socket.read(); // ← ここでスレッドが停止

  // 2. DBに問い合わせ（ブロック）
  const data = database.query("SELECT * FROM users"); // ← ここでスレッドが停止

  // 3. 外部APIを呼び出し（ブロック）
  const externalData = http.get("https://api.example.com/data"); // ← ここでスレッドが停止

  // 4. レスポンスを書き込み（ブロック）
  socket.write(buildResponse(data, externalData)); // ← ここでスレッドが停止
}
```

```java
// Java: 伝統的なブロッキングサーバー
import java.net.ServerSocket;
import java.net.Socket;
import java.io.*;

public class BlockingServer {
    public static void main(String[] args) throws IOException {
        ServerSocket serverSocket = new ServerSocket(8080);

        while (true) {
            // accept() はクライアント接続までブロック
            Socket clientSocket = serverSocket.accept();

            // 各接続に1スレッドを割り当て
            new Thread(() -> {
                try {
                    BufferedReader reader = new BufferedReader(
                        new InputStreamReader(clientSocket.getInputStream())
                    );
                    PrintWriter writer = new PrintWriter(
                        clientSocket.getOutputStream(), true
                    );

                    // readline() はデータ到着までブロック
                    String line = reader.readLine();
                    writer.println("Echo: " + line);

                    clientSocket.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }).start();
        }
        // 問題: 10,000接続 = 10,000スレッド（各1MB） = 10GBメモリ
    }
}
```

### 2.3 ノンブロッキングI/Oの詳細

```typescript
// Node.js: ノンブロッキングI/O
import * as http from 'http';
import * as fs from 'fs';

const server = http.createServer(async (req, res) => {
  // ノンブロッキング: I/O開始後すぐに制御が戻る
  // 他のリクエストを処理できる
  try {
    const data = await fs.promises.readFile('data.json', 'utf8');
    const parsed = JSON.parse(data);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(parsed));
  } catch (err) {
    res.writeHead(500);
    res.end('Internal Server Error');
  }
});

server.listen(8080);
// 1スレッドで数万の同時接続を処理可能
```

### 2.4 ブロッキングとノンブロッキングの混同に注意

「同期」と「ブロッキング」、「非同期」と「ノンブロッキング」は密接に関連しているが、厳密には別の概念である。

```
              ブロッキング           ノンブロッキング
同期          同期ブロッキング        同期ノンブロッキング
              （一般的なI/O）        （ポーリング）
非同期        非同期ブロッキング      非同期ノンブロッキング
              （select/poll）       （epoll/kqueue/IOCP）

同期ブロッキング:
  → read() を呼ぶと、データが来るまでスレッドが停止
  → 最も単純だがスケールしない

同期ノンブロッキング（ポーリング）:
  → read() を呼ぶと、データがなければ即座にEWOULDBLOCKを返す
  → アプリケーションが繰り返しチェックする必要がある
  → CPU時間の無駄が発生しやすい

非同期ノンブロッキング:
  → I/Oを依頼して即座に戻る
  → 完了時に通知を受ける
  → 最も効率的（Node.js, nginx のモデル）
```

### 2.5 OSレベルのI/O多重化

```
Linux:
  select()  → 監視できるfd数に制限（1024）
  poll()    → fd数制限なし、しかし毎回全fdをスキャン
  epoll()   → イベント駆動、高効率（Linux 2.6+）

macOS/BSD:
  kqueue()  → epoll相当、BSD系OS

Windows:
  IOCP (I/O Completion Ports) → 完了ポートモデル

Node.js の libuv:
  → OS ごとに最適な仕組みを抽象化
  → Linux: epoll, macOS: kqueue, Windows: IOCP
  → ファイルI/O: スレッドプール（デフォルト4スレッド）
  → ネットワークI/O: OS の非同期I/O
```

```c
// epoll の使い方（C言語、Linux）
#include <sys/epoll.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>

int main() {
    int epoll_fd = epoll_create1(0);

    struct epoll_event event;
    event.events = EPOLLIN;  // 読み取り可能イベントを監視
    event.data.fd = socket_fd;

    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, socket_fd, &event);

    struct epoll_event events[MAX_EVENTS];

    while (1) {
        // イベント待ち（ブロックするが、複数fdを同時に監視）
        int nfds = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);

        for (int i = 0; i < nfds; i++) {
            if (events[i].events & EPOLLIN) {
                // データが読み取り可能
                handle_read(events[i].data.fd);
            }
        }
    }
}
```

---

## 3. なぜ非同期が必要か

### 3.1 CPUサイクル vs I/O待ち時間

```
CPUサイクル vs I/O 待ち時間（概算）:

  操作                  時間            CPUサイクル換算
  ─────────────────────────────────────────────────
  L1 キャッシュ          1ns            1回
  L2 キャッシュ          4ns            4回
  L3 キャッシュ          12ns           12回
  メインメモリ           100ns          100回
  SSD ランダムリード     16μs           16,000回
  SSD シーケンシャル     50μs           50,000回
  HDD ランダムリード     4ms            4,000,000回
  ネットワーク（同一DC）  500μs          500,000回
  ネットワーク（同一国）  30ms           30,000,000回
  ネットワーク（大陸間）  150ms          150,000,000回
  TLS ハンドシェイク     250ms          250,000,000回

  → ネットワークI/O中にCPUは1.5億サイクル分「何もしていない」
  → この待ち時間を有効活用するのが非同期処理

人間の時間感覚に例えると（1CPUサイクル = 1秒とした場合）:
  L1 キャッシュ   → 1秒
  メインメモリ    → 1分40秒
  SSD リード      → 4時間半
  HDD リード      → 46日
  ネットワーク    → 4.8年（！）
```

### 3.2 具体的な効果: Webサーバーの応答時間

```typescript
// 同期的な処理（Node.jsでは非推奨）
function syncHandler(req: Request): Response {
  const user = db.getUserSync(req.userId);      // 10ms 待ち
  const orders = db.getOrdersSync(user.id);     // 15ms 待ち
  const recommendations = api.getRecsSync(user); // 50ms 待ち
  return { user, orders, recommendations };
  // 合計: 75ms（直列実行）
}

// 非同期処理（並行実行）
async function asyncHandler(req: Request): Promise<Response> {
  const user = await db.getUser(req.userId);    // 10ms
  // user を取得後、残りを並行実行
  const [orders, recommendations] = await Promise.all([
    db.getOrders(user.id),                      // 15ms ┐
    api.getRecs(user),                           // 50ms ┤ 並行
  ]);                                            //      ┘ max = 50ms
  return { user, orders, recommendations };
  // 合計: 10 + 50 = 60ms（20%高速化）
}
```

### 3.3 スループットへの影響

```
ブロッキングサーバー（スレッドプール方式）:
  スレッド数: 200（Java Tomcatのデフォルト）
  1リクエストの平均処理時間: 100ms（うちI/O待ち: 80ms）
  最大スループット: 200 / 0.1 = 2,000 req/sec

ノンブロッキングサーバー（イベントループ方式）:
  スレッド数: 1（Node.js）
  1リクエストのCPU実行時間: 20ms（I/O待ちは他の処理に使える）
  最大スループット: 1 / 0.02 = 50 req/sec（CPU律速の場合）
  ただし同時接続数に制限がない
  → 同時接続10,000でもメモリ消費が少ない

実際のベンチマーク（概算）:
  ┌────────────────────┬──────────────┬──────────────┐
  │ サーバー            │ 同時接続1,000│ 同時接続10,000│
  ├────────────────────┼──────────────┼──────────────┤
  │ Apache（prefork）   │ 5,000 req/s  │ メモリ不足    │
  │ Nginx              │ 20,000 req/s │ 18,000 req/s │
  │ Node.js            │ 15,000 req/s │ 12,000 req/s │
  │ Go net/http        │ 25,000 req/s │ 22,000 req/s │
  └────────────────────┴──────────────┴──────────────┘
  ※ 実際の数値はワークロード・ハードウェアにより大きく変動
```

### 3.4 C10K問題

```
C10K問題（The C10K Problem, 1999年 Dan Kegel提唱）:
  → 1台のサーバーで1万（10,000）の同時接続を処理できるか？

従来のアプローチ（1接続1スレッド）:
  10,000接続 × 1MB/スレッド = 10GB メモリ
  → スレッドのコンテキストスイッチが膨大
  → 実用的に不可能

解決策:
  1. イベント駆動（epoll/kqueue）+ ノンブロッキングI/O
     → Nginx, Node.js, HAProxy
  2. 軽量スレッド / コルーチン
     → Go (goroutine: ~2KB), Erlang (process: ~2KB)
  3. 非同期I/O（io_uring, IOCP）
     → 最新のLinuxカーネル (5.1+)

現在の課題: C10M問題
  → 1台で1,000万接続を処理する
  → カーネルバイパス（DPDK, XDP）、ユーザー空間ネットワーキング
```

### 3.5 リアルワールドでの非同期処理の効果

```typescript
// ECサイトの商品ページ: 同期版
async function getProductPageSync(productId: string) {
  const start = Date.now();

  const product = await getProduct(productId);           // 20ms
  const reviews = await getReviews(productId);           // 30ms
  const relatedProducts = await getRelated(productId);   // 25ms
  const inventory = await getInventory(productId);       // 15ms
  const pricing = await getPricing(productId);           // 10ms
  const seller = await getSeller(product.sellerId);      // 20ms

  console.log(`直列実行: ${Date.now() - start}ms`);
  // → 120ms
  return { product, reviews, relatedProducts, inventory, pricing, seller };
}

// ECサイトの商品ページ: 最適化版
async function getProductPageOptimized(productId: string) {
  const start = Date.now();

  // Stage 1: 依存関係のないものを並行実行
  const [product, reviews, relatedProducts, inventory, pricing] =
    await Promise.all([
      getProduct(productId),           // 20ms ┐
      getReviews(productId),           // 30ms ┤
      getRelated(productId),           // 25ms ┤ 並行
      getInventory(productId),         // 15ms ┤
      getPricing(productId),           // 10ms ┘
    ]);
  // Stage 1: max(20, 30, 25, 15, 10) = 30ms

  // Stage 2: product に依存する処理
  const seller = await getSeller(product.sellerId); // 20ms

  console.log(`最適化版: ${Date.now() - start}ms`);
  // → 50ms（58%高速化）
  return { product, reviews, relatedProducts, inventory, pricing, seller };
}
```

---

## 4. 各言語の非同期モデル

### 4.1 モデル一覧

```
┌──────────────┬───────────────────────────────┐
│ 言語         │ 非同期モデル                   │
├──────────────┼───────────────────────────────┤
│ JavaScript   │ イベントループ + Promise       │
│ Python       │ asyncio（イベントループ）      │
│ Rust         │ async/await + ランタイム(tokio)│
│ Go           │ goroutine + channel            │
│ Java         │ スレッド + CompletableFuture   │
│ Kotlin       │ coroutines                     │
│ Swift        │ structured concurrency         │
│ Elixir       │ アクターモデル（BEAM）          │
│ C#           │ Task + async/await             │
│ C++          │ std::async + co_await (C++20)  │
└──────────────┴───────────────────────────────┘

大きく3つのアプローチ:
  1. イベントループ（JS, Python）: シングルスレッド + 非同期I/O
  2. グリーンスレッド（Go, Erlang）: 軽量スレッド × 多数
  3. OS スレッド + async（Java, C#）: スレッドプール + Future
```

### 4.2 JavaScript / TypeScript

```typescript
// JavaScript: シングルスレッド + イベントループ
// ブラウザ / Node.js 共通のモデル

// 1. Promise ベース
function fetchUserData(userId: string): Promise<User> {
  return fetch(`/api/users/${userId}`)
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.json();
    });
}

// 2. async/await（Promise の構文糖）
async function fetchUserData(userId: string): Promise<User> {
  const response = await fetch(`/api/users/${userId}`);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  return response.json();
}

// 3. Node.js 固有: Worker Threads（CPU集約型用）
import { Worker, isMainThread, parentPort } from 'worker_threads';

if (isMainThread) {
  const worker = new Worker(__filename);
  worker.on('message', (result) => {
    console.log('計算結果:', result);
  });
  worker.postMessage({ data: largeArray });
} else {
  parentPort?.on('message', (msg) => {
    // CPU集約的な処理をワーカースレッドで実行
    const result = heavyComputation(msg.data);
    parentPort?.postMessage(result);
  });
}
```

### 4.3 Python

```python
import asyncio
import aiohttp

# Python: asyncio イベントループ
# GIL（Global Interpreter Lock）があるため、
# CPU並列はmultiprocessing、I/O並行はasyncioを使う

# 基本的な非同期関数
async def fetch_user(user_id: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/users/{user_id}") as resp:
            return await resp.json()

# 並行実行
async def fetch_all_users(user_ids: list[str]) -> list[dict]:
    tasks = [fetch_user(uid) for uid in user_ids]
    return await asyncio.gather(*tasks)

# 実行
async def main():
    users = await fetch_all_users(["user-1", "user-2", "user-3"])
    for user in users:
        print(user["name"])

asyncio.run(main())

# CPU集約型: multiprocessing
from concurrent.futures import ProcessPoolExecutor
import asyncio

async def cpu_intensive_async(data_list: list) -> list:
    loop = asyncio.get_event_loop()
    with ProcessPoolExecutor() as pool:
        results = await asyncio.gather(*[
            loop.run_in_executor(pool, heavy_computation, data)
            for data in data_list
        ])
    return results
```

### 4.4 Go

```go
package main

import (
    "fmt"
    "net/http"
    "sync"
    "time"
)

// Go: goroutine + channel
// goroutineは軽量スレッド（~2KB）、OSスレッドの上にランタイムがスケジューリング

// 基本的な非同期実行
func fetchURL(url string, ch chan<- string, wg *sync.WaitGroup) {
    defer wg.Done()
    resp, err := http.Get(url)
    if err != nil {
        ch <- fmt.Sprintf("Error: %s", err)
        return
    }
    defer resp.Body.Close()
    ch <- fmt.Sprintf("%s: %d", url, resp.StatusCode)
}

func main() {
    urls := []string{
        "https://api.example.com/users",
        "https://api.example.com/orders",
        "https://api.example.com/products",
    }

    ch := make(chan string, len(urls))
    var wg sync.WaitGroup

    for _, url := range urls {
        wg.Add(1)
        go fetchURL(url, ch, &wg) // goroutineで並行実行
    }

    // 全完了を待つ
    go func() {
        wg.Wait()
        close(ch)
    }()

    for result := range ch {
        fmt.Println(result)
    }
}

// select による複数チャネルの待ち受け
func fetchWithTimeout(url string, timeout time.Duration) (string, error) {
    ch := make(chan string, 1)
    errCh := make(chan error, 1)

    go func() {
        resp, err := http.Get(url)
        if err != nil {
            errCh <- err
            return
        }
        defer resp.Body.Close()
        ch <- resp.Status
    }()

    select {
    case result := <-ch:
        return result, nil
    case err := <-errCh:
        return "", err
    case <-time.After(timeout):
        return "", fmt.Errorf("timeout after %v", timeout)
    }
}
```

### 4.5 Rust

```rust
use tokio;
use reqwest;

// Rust: async/await + ランタイム（tokio）
// ゼロコスト抽象化: async関数はステートマシンにコンパイルされる
// Future は lazy: .await するまで実行されない

async fn fetch_user(user_id: &str) -> Result<User, reqwest::Error> {
    let url = format!("https://api.example.com/users/{}", user_id);
    let user: User = reqwest::get(&url)
        .await?
        .json()
        .await?;
    Ok(user)
}

// 並行実行
async fn fetch_all_data(user_id: &str) -> Result<Dashboard, AppError> {
    // tokio::join! で並行実行
    let (user, orders, notifications) = tokio::join!(
        fetch_user(user_id),
        fetch_orders(user_id),
        fetch_notifications(user_id),
    );

    Ok(Dashboard {
        user: user?,
        orders: orders?,
        notifications: notifications?,
    })
}

// tokio::spawn でバックグラウンドタスク
async fn background_processing() {
    let handle = tokio::spawn(async {
        // バックグラウンドで実行
        heavy_async_work().await
    });

    // 他の処理を続行
    do_other_work().await;

    // バックグラウンドタスクの結果を取得
    let result = handle.await.unwrap();
}

#[tokio::main]
async fn main() {
    let dashboard = fetch_all_data("user-123").await.unwrap();
    println!("{:?}", dashboard);
}
```

### 4.6 Java

```java
import java.util.concurrent.*;

// Java: CompletableFuture (Java 8+)
// 仮想スレッド (Java 21+ / Project Loom)

public class AsyncExample {

    // CompletableFuture ベース
    public CompletableFuture<Dashboard> getDashboard(String userId) {
        CompletableFuture<User> userFuture =
            CompletableFuture.supplyAsync(() -> userRepo.findById(userId));

        CompletableFuture<List<Order>> ordersFuture =
            CompletableFuture.supplyAsync(() -> orderRepo.findByUserId(userId));

        CompletableFuture<List<Notification>> notifFuture =
            CompletableFuture.supplyAsync(() -> notifRepo.findByUserId(userId));

        // 全て完了したら結合
        return CompletableFuture.allOf(userFuture, ordersFuture, notifFuture)
            .thenApply(v -> new Dashboard(
                userFuture.join(),
                ordersFuture.join(),
                notifFuture.join()
            ));
    }

    // Java 21: 仮想スレッド（Project Loom）
    public Dashboard getDashboardVirtualThreads(String userId) throws Exception {
        try (var scope = new StructuredTaskScope.ShutdownOnFailure()) {
            var userTask = scope.fork(() -> userRepo.findById(userId));
            var ordersTask = scope.fork(() -> orderRepo.findByUserId(userId));
            var notifTask = scope.fork(() -> notifRepo.findByUserId(userId));

            scope.join();
            scope.throwIfFailed();

            return new Dashboard(
                userTask.get(),
                ordersTask.get(),
                notifTask.get()
            );
        }
    }
}
```

### 4.7 C#

```csharp
using System;
using System.Net.Http;
using System.Threading.Tasks;

// C#: Task + async/await
// .NET の非同期モデルは最も成熟したものの一つ

public class AsyncService
{
    private readonly HttpClient _httpClient;

    // async/await 基本
    public async Task<Dashboard> GetDashboardAsync(string userId)
    {
        var user = await GetUserAsync(userId);

        // 並行実行
        var ordersTask = GetOrdersAsync(userId);
        var notificationsTask = GetNotificationsAsync(userId);

        await Task.WhenAll(ordersTask, notificationsTask);

        return new Dashboard
        {
            User = user,
            Orders = ordersTask.Result,
            Notifications = notificationsTask.Result
        };
    }

    // キャンセルトークン対応
    public async Task<User> GetUserAsync(
        string userId,
        CancellationToken cancellationToken = default)
    {
        var response = await _httpClient.GetAsync(
            $"/api/users/{userId}",
            cancellationToken
        );
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadFromJsonAsync<User>(
            cancellationToken: cancellationToken
        );
    }

    // ValueTask: ホットパス最適化
    public ValueTask<CachedData> GetCachedDataAsync(string key)
    {
        if (_cache.TryGetValue(key, out var cached))
        {
            // キャッシュヒット時はヒープ割り当てなし
            return new ValueTask<CachedData>(cached);
        }

        // キャッシュミス時のみ非同期処理
        return new ValueTask<CachedData>(FetchAndCacheAsync(key));
    }
}
```

---

## 5. 同期と非同期の使い分け

### 5.1 判断基準

```
同期が適切:
  ✓ CPU集約的な計算（数値計算、暗号化、画像処理）
  ✓ シンプルなスクリプト・バッチ処理
  ✓ I/Oが少ない処理
  ✓ 逐次実行が必要な処理（順序保証が必要）
  ✓ デバッグ容易性が重要な場合
  ✓ 短時間で完了する処理

非同期が適切:
  ✓ ネットワークI/O（API呼び出し、DB接続）
  ✓ ファイルI/O（大量のファイル操作）
  ✓ 多数の同時接続を処理するサーバー
  ✓ UIをブロックしたくないクライアントアプリ
  ✓ リアルタイム処理（WebSocket、チャット）
  ✓ マイクロサービス間通信

注意:
  → CPU集約的な処理を async にしても意味がない
  → イベントループをブロックしない（Node.js の鉄則）
  → 非同期のオーバーヘッド（コンテキストスイッチ、メモリ）も考慮
```

### 5.2 具体的なシナリオ別ガイド

```typescript
// シナリオ1: ファイル処理
// ✅ 大量のファイルを非同期で並行処理
async function processFiles(filePaths: string[]): Promise<void> {
  const CONCURRENCY = 10; // 同時に10ファイルまで
  const results: string[] = [];

  for (let i = 0; i < filePaths.length; i += CONCURRENCY) {
    const batch = filePaths.slice(i, i + CONCURRENCY);
    const batchResults = await Promise.all(
      batch.map(async (filePath) => {
        const content = await fs.promises.readFile(filePath, 'utf8');
        return processContent(content);
      })
    );
    results.push(...batchResults);
  }
}

// ❌ 小さなファイル1つだけなら同期でも可
// （起動スクリプト、設定読み込みなど）
const config = JSON.parse(fs.readFileSync('config.json', 'utf8'));
```

```python
# シナリオ2: Webスクレイピング
import asyncio
import aiohttp
from typing import List, Dict

# ✅ 多数のURLを非同期で並行取得
async def scrape_urls(urls: list[str]) -> list[dict]:
    semaphore = asyncio.Semaphore(20)  # 同時接続数制限

    async def fetch_one(session: aiohttp.ClientSession, url: str) -> dict:
        async with semaphore:
            async with session.get(url) as response:
                html = await response.text()
                return {"url": url, "status": response.status, "html": html}

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=True)

# 100件のURLを同時20並列で取得
# 同期: 100 × 200ms = 20秒
# 非同期: 100 / 20 × 200ms = 1秒（20倍高速）
```

```go
// シナリオ3: マイクロサービスのAPI Gateway
package main

import (
    "context"
    "net/http"
    "time"
    "encoding/json"
    "golang.org/x/sync/errgroup"
)

type AggregatedResponse struct {
    User          *User          `json:"user"`
    Orders        []Order        `json:"orders"`
    Notifications []Notification `json:"notifications"`
}

// ✅ 複数のマイクロサービスを並行呼び出し
func aggregateHandler(w http.ResponseWriter, r *http.Request) {
    ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
    defer cancel()

    userID := r.URL.Query().Get("user_id")
    var resp AggregatedResponse

    g, ctx := errgroup.WithContext(ctx)

    g.Go(func() error {
        user, err := fetchUser(ctx, userID)
        if err != nil {
            return err
        }
        resp.User = user
        return nil
    })

    g.Go(func() error {
        orders, err := fetchOrders(ctx, userID)
        if err != nil {
            return err
        }
        resp.Orders = orders
        return nil
    })

    g.Go(func() error {
        notifs, err := fetchNotifications(ctx, userID)
        if err != nil {
            return err
        }
        resp.Notifications = notifs
        return nil
    })

    if err := g.Wait(); err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    json.NewEncoder(w).Encode(resp)
}
```

### 5.3 アンチパターン

```typescript
// ❌ アンチパターン1: CPU集約処理をイベントループで実行
async function badImageProcessing(images: Buffer[]): Promise<Buffer[]> {
  // イベントループをブロックしてしまう
  return images.map(img => {
    // 重い画像処理（同期的にCPUを占有）
    return sharp(img).resize(800, 600).toBuffer(); // ← 同期API
  });
}

// ✅ CPU集約処理はWorker Threadsに委譲
import { Worker } from 'worker_threads';

async function goodImageProcessing(images: Buffer[]): Promise<Buffer[]> {
  const worker = new Worker('./image-worker.js');
  return new Promise((resolve, reject) => {
    worker.postMessage(images);
    worker.on('message', resolve);
    worker.on('error', reject);
  });
}

// ❌ アンチパターン2: 不必要な非同期化
async function unnecessary(): Promise<number> {
  return 1 + 1; // ← 同期で十分な処理を非同期にする意味がない
}

// ❌ アンチパターン3: 非同期処理の結果を無視
function fireAndForget(data: Data): void {
  saveToDatabase(data); // Promiseの結果を無視 → エラーが見えなくなる
}

// ✅ 結果を適切にハンドリング
async function properSave(data: Data): Promise<void> {
  try {
    await saveToDatabase(data);
  } catch (error) {
    logger.error('Failed to save data', error);
    throw error; // 呼び出し元に伝播
  }
}
```

---

## 6. 実務で頻出するパターン

### 6.1 タイムアウト付き非同期処理

```typescript
// タイムアウト付きfetch
async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeoutMs: number = 5000,
): Promise<Response> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
    });
    return response;
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error(`Request timeout after ${timeoutMs}ms: ${url}`);
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

// 使用例
try {
  const response = await fetchWithTimeout('https://api.example.com/data', {}, 3000);
  const data = await response.json();
} catch (error) {
  console.error('リクエスト失敗:', error.message);
}
```

### 6.2 リトライ付き非同期処理

```typescript
// 指数バックオフ付きリトライ
async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  options: {
    maxRetries?: number;
    initialDelayMs?: number;
    maxDelayMs?: number;
    backoffMultiplier?: number;
    retryableErrors?: (error: unknown) => boolean;
  } = {},
): Promise<T> {
  const {
    maxRetries = 3,
    initialDelayMs = 1000,
    maxDelayMs = 30000,
    backoffMultiplier = 2,
    retryableErrors = () => true,
  } = options;

  let lastError: unknown;
  let delay = initialDelayMs;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;

      if (attempt === maxRetries || !retryableErrors(error)) {
        throw error;
      }

      // ジッター（ランダム揺らぎ）を追加
      const jitter = delay * 0.1 * Math.random();
      const actualDelay = Math.min(delay + jitter, maxDelayMs);

      console.warn(
        `Attempt ${attempt + 1} failed, retrying in ${actualDelay}ms...`,
        error
      );

      await new Promise(resolve => setTimeout(resolve, actualDelay));
      delay *= backoffMultiplier;
    }
  }

  throw lastError;
}

// 使用例
const data = await retryWithBackoff(
  () => fetchWithTimeout('https://api.example.com/data'),
  {
    maxRetries: 3,
    initialDelayMs: 1000,
    retryableErrors: (error) => {
      // 5xx エラーのみリトライ
      return error instanceof Error && error.message.includes('5');
    },
  }
);
```

### 6.3 並行数制限（セマフォパターン）

```typescript
// セマフォ: 同時実行数を制限
class Semaphore {
  private permits: number;
  private queue: (() => void)[] = [];

  constructor(permits: number) {
    this.permits = permits;
  }

  async acquire(): Promise<void> {
    if (this.permits > 0) {
      this.permits--;
      return;
    }

    return new Promise<void>((resolve) => {
      this.queue.push(resolve);
    });
  }

  release(): void {
    const next = this.queue.shift();
    if (next) {
      next();
    } else {
      this.permits++;
    }
  }

  async use<T>(fn: () => Promise<T>): Promise<T> {
    await this.acquire();
    try {
      return await fn();
    } finally {
      this.release();
    }
  }
}

// 使用例: 同時5並列でAPI呼び出し
const semaphore = new Semaphore(5);
const urls = Array.from({ length: 100 }, (_, i) => `https://api.example.com/item/${i}`);

const results = await Promise.all(
  urls.map(url =>
    semaphore.use(async () => {
      const response = await fetch(url);
      return response.json();
    })
  )
);
```

### 6.4 キャンセル可能な非同期処理

```typescript
// AbortController を使ったキャンセル
class CancellableTask<T> {
  private controller: AbortController;
  private promise: Promise<T>;

  constructor(executor: (signal: AbortSignal) => Promise<T>) {
    this.controller = new AbortController();
    this.promise = executor(this.controller.signal);
  }

  get result(): Promise<T> {
    return this.promise;
  }

  cancel(reason?: string): void {
    this.controller.abort(reason);
  }
}

// 使用例: 検索の自動キャンセル
let currentSearch: CancellableTask<SearchResult[]> | null = null;

async function search(query: string): Promise<SearchResult[]> {
  // 前の検索をキャンセル
  currentSearch?.cancel('New search started');

  currentSearch = new CancellableTask(async (signal) => {
    const response = await fetch(`/api/search?q=${query}`, { signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  });

  return currentSearch.result;
}
```

---

## 7. パフォーマンス計測と最適化

### 7.1 非同期処理のベンチマーク方法

```typescript
// 処理時間の計測
async function benchmark<T>(
  name: string,
  fn: () => Promise<T>,
  iterations: number = 10,
): Promise<{ name: string; avg: number; min: number; max: number; p95: number }> {
  const times: number[] = [];

  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    await fn();
    times.push(performance.now() - start);
  }

  times.sort((a, b) => a - b);

  return {
    name,
    avg: times.reduce((sum, t) => sum + t, 0) / times.length,
    min: times[0],
    max: times[times.length - 1],
    p95: times[Math.floor(times.length * 0.95)],
  };
}

// 比較テスト
async function compareSyncVsAsync(): Promise<void> {
  const syncResult = await benchmark('Sequential', async () => {
    const a = await fetchA();
    const b = await fetchB();
    const c = await fetchC();
    return { a, b, c };
  });

  const asyncResult = await benchmark('Parallel', async () => {
    const [a, b, c] = await Promise.all([
      fetchA(),
      fetchB(),
      fetchC(),
    ]);
    return { a, b, c };
  });

  console.table([syncResult, asyncResult]);
  // ┌─────────────┬──────────┬──────────┬──────────┬──────────┐
  // │ name        │ avg      │ min      │ max      │ p95      │
  // ├─────────────┼──────────┼──────────┼──────────┼──────────┤
  // │ Sequential  │ 312.5ms  │ 301.2ms  │ 325.8ms  │ 321.3ms  │
  // │ Parallel    │ 105.3ms  │ 100.1ms  │ 115.2ms  │ 112.7ms  │
  // └─────────────┴──────────┴──────────┴──────────┴──────────┘
}
```

### 7.2 よくあるボトルネックと対策

```
ボトルネック1: DBコネクションプール不足
  症状: 並行リクエストが多い時にDB接続待ち
  対策: プールサイズを適切に設定（CPU cores × 2 + disk数）

ボトルネック2: 外部API のレート制限
  症状: 429 Too Many Requests
  対策: セマフォで並行数制限、レート制限ライブラリ使用

ボトルネック3: メモリリーク（Promiseの蓄積）
  症状: ヒープメモリが継続的に増加
  対策: 不要なPromise参照の解放、WeakRefの活用

ボトルネック4: イベントループのブロック
  症状: レスポンスタイムの突発的な増大
  対策: CPU処理をWorker Threadsに移動、blocked-at等で検出

ボトルネック5: DNS解決の遅延
  症状: 初回リクエストだけ遅い
  対策: DNS プリフェッチ、keep-alive接続の活用
```

---

## 8. 非同期処理のテスト手法

### 8.1 基本的なテストパターン

```typescript
import { describe, it, expect, vi } from 'vitest';

// 非同期関数のテスト
describe('fetchUserData', () => {
  // 基本テスト
  it('ユーザーデータを正常に取得できる', async () => {
    const user = await fetchUserData('user-123');
    expect(user).toEqual({
      id: 'user-123',
      name: 'テスト太郎',
    });
  });

  // エラーテスト
  it('存在しないユーザーでエラーをスロー', async () => {
    await expect(fetchUserData('nonexistent'))
      .rejects.toThrow('User not found');
  });

  // タイムアウトテスト
  it('タイムアウト時にエラーをスロー', async () => {
    vi.useFakeTimers();

    const promise = fetchWithTimeout('https://slow.api.com', {}, 3000);

    vi.advanceTimersByTime(3000);

    await expect(promise).rejects.toThrow('timeout');

    vi.useRealTimers();
  });

  // モック使用
  it('APIをモックしてテスト', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ id: '123', name: 'Test' }),
    });

    global.fetch = mockFetch;

    const result = await fetchUserData('123');
    expect(result.name).toBe('Test');
    expect(mockFetch).toHaveBeenCalledWith(
      expect.stringContaining('/users/123')
    );
  });
});
```

### 8.2 並行処理のテスト

```typescript
describe('Promise.all パターンのテスト', () => {
  it('並行実行の順序非依存性を確認', async () => {
    const results: string[] = [];

    const task1 = async () => {
      await sleep(100);
      results.push('task1');
      return 'result1';
    };

    const task2 = async () => {
      await sleep(50);
      results.push('task2');
      return 'result2';
    };

    const [r1, r2] = await Promise.all([task1(), task2()]);

    expect(r1).toBe('result1');
    expect(r2).toBe('result2');
    // task2 の方が先に完了するが、結果の順序は保持される
    expect(results).toEqual(['task2', 'task1']);
  });

  it('部分的な失敗をハンドリング', async () => {
    const results = await Promise.allSettled([
      Promise.resolve('success'),
      Promise.reject(new Error('failure')),
      Promise.resolve('success2'),
    ]);

    expect(results[0]).toEqual({ status: 'fulfilled', value: 'success' });
    expect(results[1].status).toBe('rejected');
    expect(results[2]).toEqual({ status: 'fulfilled', value: 'success2' });
  });
});
```

---

## 9. デバッグテクニック

### 9.1 非同期処理のデバッグ

```typescript
// async_hooks を使った非同期処理のトレース（Node.js）
import { AsyncLocalStorage } from 'async_hooks';

const requestStorage = new AsyncLocalStorage<{ requestId: string }>();

// リクエストIDをコンテキストとして伝播
async function handleRequest(req: Request): Promise<Response> {
  const requestId = generateRequestId();

  return requestStorage.run({ requestId }, async () => {
    logger.info(`[${requestId}] リクエスト開始`);

    const user = await getUser(req.userId);
    logger.info(`[${requestId}] ユーザー取得完了`);

    const data = await processData(user);
    logger.info(`[${requestId}] データ処理完了`);

    return new Response(JSON.stringify(data));
  });
}

// どの非同期処理からでもリクエストIDを取得可能
function getRequestId(): string {
  return requestStorage.getStore()?.requestId ?? 'unknown';
}
```

### 9.2 Unhandled Rejection の検出

```typescript
// Node.js: 未処理のPromise Rejectionを検出
process.on('unhandledRejection', (reason, promise) => {
  console.error('未処理のPromise Rejection:', reason);
  console.error('Promise:', promise);
  // 本番環境ではログに記録してアラートを送信
  logger.error('Unhandled Promise Rejection', {
    reason: reason instanceof Error ? reason.message : String(reason),
    stack: reason instanceof Error ? reason.stack : undefined,
  });
});

// ブラウザ
window.addEventListener('unhandledrejection', (event) => {
  console.error('未処理のPromise Rejection:', event.reason);
  event.preventDefault(); // デフォルトのコンソールエラーを抑制
  // エラー追跡サービスに報告
  errorTracker.captureException(event.reason);
});
```

### 9.3 非同期処理のプロファイリング

```typescript
// 非同期処理のパフォーマンスを可視化
class AsyncProfiler {
  private traces: Map<string, { start: number; end?: number }[]> = new Map();

  wrap<T>(name: string, fn: () => Promise<T>): Promise<T> {
    const entry = { start: performance.now() };

    if (!this.traces.has(name)) {
      this.traces.set(name, []);
    }
    this.traces.get(name)!.push(entry);

    return fn().finally(() => {
      entry.end = performance.now();
    });
  }

  report(): void {
    console.log('\n=== Async Performance Report ===');
    for (const [name, entries] of this.traces) {
      const durations = entries
        .filter(e => e.end !== undefined)
        .map(e => e.end! - e.start);
      const avg = durations.reduce((s, d) => s + d, 0) / durations.length;
      const max = Math.max(...durations);
      console.log(`${name}: calls=${entries.length}, avg=${avg.toFixed(1)}ms, max=${max.toFixed(1)}ms`);
    }
  }
}

// 使用例
const profiler = new AsyncProfiler();

const user = await profiler.wrap('getUser', () => getUser(userId));
const [orders, reviews] = await Promise.all([
  profiler.wrap('getOrders', () => getOrders(user.id)),
  profiler.wrap('getReviews', () => getReviews(user.id)),
]);

profiler.report();
// === Async Performance Report ===
// getUser: calls=1, avg=12.3ms, max=12.3ms
// getOrders: calls=1, avg=18.7ms, max=18.7ms
// getReviews: calls=1, avg=45.2ms, max=45.2ms
```

---

## まとめ

| 概念 | 同期 | 非同期 |
|------|------|--------|
| 実行 | 順番に待つ | 待ち時間に他を処理 |
| I/O | ブロッキング | ノンブロッキング |
| 性能 | I/O待ちで無駄 | I/O待ちを有効活用 |
| 複雑さ | シンプル | コールバック/Promise |
| 適用 | CPU集約 | I/O集約 |
| スケーラビリティ | スレッド数に制限 | 多数の同時接続対応 |
| デバッグ | 容易（スタックトレース直線的） | 困難（非同期スタックトレース） |
| メモリ | スレッドあたり~1MB | イベント/goroutineあたり~2KB |

### 判断フローチャート

```
処理の種類は？
├── CPU集約型（計算、暗号化、画像処理）
│   ├── 単一処理 → 同期
│   └── 並列計算が必要 → Worker Threads / multiprocessing
├── I/O集約型（API、DB、ファイル）
│   ├── 単発 → async/await
│   ├── 複数の独立したI/O → Promise.all / gather / join!
│   └── ストリーム → Observable / AsyncIterator / Channel
└── 混合型
    ├── I/O → 非同期
    └── CPU → ワーカーに委譲
```

---

## 次に読むべきガイド
→ [[01-concurrency-models.md]] — 並行モデル概要

---

## 参考文献
1. Kleppmann, M. "Designing Data-Intensive Applications." O'Reilly, 2017.
2. Node.js Documentation. "Don't Block the Event Loop."
3. Kegel, D. "The C10K Problem." 1999. http://www.kegel.com/c10k.html
4. Pike, R. "Concurrency Is Not Parallelism." Waza Conference, 2012.
5. Mozilla Developer Network. "Asynchronous JavaScript." MDN Web Docs.
6. Python Documentation. "asyncio - Asynchronous I/O." docs.python.org.
7. Tokio Documentation. "Tutorial." tokio.rs.
8. Microsoft. "Asynchronous programming with async and await." docs.microsoft.com.
9. OpenJDK. "JEP 444: Virtual Threads." openjdk.org.
10. Nginx Documentation. "Inside NGINX: How We Designed for Performance & Scale."
