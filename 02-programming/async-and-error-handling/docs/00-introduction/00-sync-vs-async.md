# 同期 vs 非同期

> 同期処理は「前の処理が終わるまで次を待つ」、非同期処理は「待ち時間に他の処理を進める」。Webアプリケーションのパフォーマンスの鍵は、I/O待ちを効率的に処理すること。

## この章で学ぶこと

- [ ] 同期処理と非同期処理の根本的な違いを理解する
- [ ] ブロッキングとノンブロッキングの意味を把握する
- [ ] なぜ非同期処理が必要かを具体的に理解する

---

## 1. 同期 vs 非同期

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

---

## 2. ブロッキング vs ノンブロッキング

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

---

## 3. なぜ非同期が必要か

```
CPUサイクル vs I/O 待ち時間（概算）:

  操作                  時間            CPUサイクル換算
  ─────────────────────────────────────────────────
  L1 キャッシュ          1ns            1回
  L2 キャッシュ          4ns            4回
  メインメモリ           100ns          100回
  SSD ランダムリード     16μs           16,000回
  HDD ランダムリード     4ms            4,000,000回
  ネットワーク（同一DC）  500μs          500,000回
  ネットワーク（大陸間）  150ms          150,000,000回

  → ネットワークI/O中にCPUは1.5億サイクル分「何もしていない」
  → この待ち時間を有効活用するのが非同期処理
```

### 具体的な効果

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

---

## 4. 各言語の非同期モデル

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
└──────────────┴───────────────────────────────┘

大きく3つのアプローチ:
  1. イベントループ（JS, Python）: シングルスレッド + 非同期I/O
  2. グリーンスレッド（Go, Erlang）: 軽量スレッド × 多数
  3. OS スレッド + async（Java, C#）: スレッドプール + Future
```

---

## 5. 同期と非同期の使い分け

```
同期が適切:
  ✓ CPU集約的な計算（数値計算、暗号化）
  ✓ シンプルなスクリプト
  ✓ I/Oが少ない処理
  ✓ 逐次実行が必要な処理

非同期が適切:
  ✓ ネットワークI/O（API呼び出し、DB接続）
  ✓ ファイルI/O
  ✓ 多数の同時接続を処理するサーバー
  ✓ UIをブロックしたくないクライアントアプリ

注意:
  → CPU集約的な処理を async にしても意味がない
  → イベントループをブロックしない（Node.js の鉄則）
  → 非同期のオーバーヘッド（コンテキストスイッチ、メモリ）も考慮
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

---

## 次に読むべきガイド
→ [[01-concurrency-models.md]] — 並行モデル概要

---

## 参考文献
1. Kleppmann, M. "Designing Data-Intensive Applications." O'Reilly, 2017.
2. Node.js Documentation. "Don't Block the Event Loop."
