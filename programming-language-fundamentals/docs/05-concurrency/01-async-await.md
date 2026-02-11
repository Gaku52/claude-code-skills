# async/await（非同期プログラミング）

> async/await は「I/O待ちの間に他の処理を進める」仕組み。スレッドを使わずに大量の同時接続を効率的に処理する、現代のサーバー・UIプログラミングの基盤。

## この章で学ぶこと

- [ ] 非同期処理の必要性と仕組みを理解する
- [ ] async/await の動作原理を理解する
- [ ] 各言語の非同期モデルの違いを把握する

---

## 1. なぜ非同期処理が必要か

```
同期処理（ブロッキング）:
  HTTP Request  ──▓▓▓▓▓▓▓▓▓▓▓▓── 応答待ち（CPU遊休）
  DB Query      ──────────────────▓▓▓▓▓▓── 応答待ち
  File Read     ──────────────────────────▓▓▓▓──

非同期処理（ノンブロッキング）:
  HTTP Request  ──▓▓── 送信 → 他の処理へ → ──▓▓── 応答処理
  DB Query      ────▓▓── 送信 → 他の処理へ → ▓▓── 応答処理
  File Read     ──────▓▓── 要求 → 他の処理へ ▓▓── 読取完了

  → I/O待ち時間を有効活用（CPUが遊ばない）
  → 1スレッドで数万の同時接続を処理可能
```

---

## 2. JavaScript の async/await

```javascript
// Promise: 非同期処理の結果を表すオブジェクト
const fetchUser = (id) =>
    fetch(`/api/users/${id}`)
        .then(res => res.json());

// async/await: Promise のシンタックスシュガー
async function getUser(id) {
    const res = await fetch(`/api/users/${id}`);
    const user = await res.json();
    return user;
}

// 並行実行
async function loadDashboard(userId) {
    // 逐次実行（遅い）
    const user = await fetchUser(userId);
    const posts = await fetchPosts(userId);
    const notifications = await fetchNotifications(userId);

    // 並行実行（速い）
    const [user, posts, notifications] = await Promise.all([
        fetchUser(userId),
        fetchPosts(userId),
        fetchNotifications(userId),
    ]);

    return { user, posts, notifications };
}

// エラーハンドリング
async function safeFetch(url) {
    try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
    } catch (error) {
        console.error(`Fetch failed: ${error.message}`);
        return null;
    }
}

// Promise 便利メソッド
Promise.all([p1, p2, p3])        // 全て成功で完了
Promise.allSettled([p1, p2, p3]) // 全て完了（成功/失敗問わず）
Promise.race([p1, p2, p3])      // 最初に完了したもの
Promise.any([p1, p2, p3])       // 最初に成功したもの
```

---

## 3. Python の asyncio

```python
import asyncio
import aiohttp

# async def で非同期関数を定義
async def fetch_user(session, user_id):
    async with session.get(f"/api/users/{user_id}") as resp:
        return await resp.json()

# 並行実行
async def load_dashboard(user_id):
    async with aiohttp.ClientSession() as session:
        # 並行実行（gather）
        user, posts, notifs = await asyncio.gather(
            fetch_user(session, user_id),
            fetch_posts(session, user_id),
            fetch_notifications(session, user_id),
        )
        return {"user": user, "posts": posts, "notifications": notifs}

# イベントループの実行
asyncio.run(load_dashboard(1))

# タイムアウト
async def fetch_with_timeout(url, timeout=5):
    try:
        async with asyncio.timeout(timeout):
            return await fetch(url)
    except asyncio.TimeoutError:
        print(f"Timeout: {url}")
        return None

# 非同期イテレータ
async def stream_events():
    async for event in event_source:
        await process(event)
```

---

## 4. Rust の async/await

```rust
// Rust: Future ベースの非同期（tokio ランタイム）
use tokio;

#[tokio::main]
async fn main() {
    let result = fetch_data("https://api.example.com").await;
    println!("{:?}", result);
}

async fn fetch_data(url: &str) -> Result<String, reqwest::Error> {
    let body = reqwest::get(url).await?.text().await?;
    Ok(body)
}

// 並行実行
async fn load_dashboard(id: u32) -> Dashboard {
    let (user, posts) = tokio::join!(
        fetch_user(id),
        fetch_posts(id),
    );
    Dashboard { user: user.unwrap(), posts: posts.unwrap() }
}

// select: 最初に完了したものを処理
tokio::select! {
    result = fetch_primary() => handle_primary(result),
    result = fetch_fallback() => handle_fallback(result),
    _ = tokio::time::sleep(Duration::from_secs(5)) => timeout(),
}

// Rust の Future は遅延評価（ポーリング方式）
// .await を呼ぶまで何も実行されない
// → ゼロコスト抽象化（不要なアロケーションなし）
```

---

## 5. Go の並行処理（goroutine + channel）

```go
// Go は async/await を使わない
// 代わりに goroutine + channel（CSP モデル）

// goroutine: 軽量スレッド
func fetchUser(id int, ch chan<- User) {
    user := httpGet(fmt.Sprintf("/api/users/%d", id))
    ch <- user  // チャネルに送信
}

func main() {
    ch := make(chan User)
    go fetchUser(1, ch)       // goroutine で非同期実行

    user := <-ch              // チャネルから受信（ブロッキング）
    fmt.Println(user)
}

// select: 複数チャネルの待機
select {
case user := <-userCh:
    fmt.Println("User:", user)
case err := <-errCh:
    fmt.Println("Error:", err)
case <-time.After(5 * time.Second):
    fmt.Println("Timeout")
}
```

---

## 6. 非同期ランタイムの比較

```
┌──────────────┬──────────────┬──────────────────────┐
│ 言語          │ モデル        │ 特徴                  │
├──────────────┼──────────────┼──────────────────────┤
│ JavaScript   │ イベントループ│ シングルスレッド       │
│              │ (libuv)      │ ノンブロッキングI/O    │
├──────────────┼──────────────┼──────────────────────┤
│ Python       │ イベントループ│ GIL制約あり           │
│              │ (asyncio)    │ I/O密集型に有効        │
├──────────────┼──────────────┼──────────────────────┤
│ Rust         │ ポーリング   │ ゼロコスト抽象化       │
│              │ (tokio等)    │ マルチスレッドランタイム│
├──────────────┼──────────────┼──────────────────────┤
│ Go           │ CSP          │ goroutine（M:N）      │
│              │ (ランタイム)  │ channel でメッセージ   │
├──────────────┼──────────────┼──────────────────────┤
│ Java         │ Virtual Thread│ Project Loom(21+)    │
│              │ (キャリア)    │ 軽量スレッド           │
└──────────────┴──────────────┴──────────────────────┘
```

---

## まとめ

| 言語 | 非同期構文 | ランタイム | 特徴 |
|------|----------|----------|------|
| JavaScript | async/await | イベントループ | シンプル・広く普及 |
| Python | async/await | asyncio | I/O密集型向け |
| Rust | async/await | tokio/async-std | ゼロコスト |
| Go | goroutine+chan | ランタイム内蔵 | 最もシンプル |
| Java | Virtual Threads | Loom | 既存コード互換 |

---

## 次に読むべきガイド
→ [[02-message-passing.md]] — メッセージパッシング

---

## 参考文献
1. "Asynchronous Programming in Rust." rust-lang.github.io/async-book.
2. "Node.js Event Loop." nodejs.org/en/docs/guides.
