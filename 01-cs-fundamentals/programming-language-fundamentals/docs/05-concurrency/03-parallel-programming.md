# 並列プログラミング

> 並列プログラミングは「複数のCPUコアを活用して処理を高速化する」技術。データ並列とタスク並列の2つのアプローチがある。

## この章で学ぶこと

- [ ] データ並列とタスク並列の違いを理解する
- [ ] 各言語の並列処理機能を活用できる
- [ ] 並列化の限界（アムダールの法則）を理解する

---

## 1. アムダールの法則

```
アムダールの法則:
  並列化可能な部分が P%、コア数が N の場合の高速化率

  高速化率 = 1 / ((1 - P) + P/N)

  例: 90% が並列化可能、8コアの場合
      = 1 / (0.1 + 0.9/8) = 1 / 0.2125 = 4.7倍

  重要な示唆:
    - 並列化できない部分（10%）が律速になる
    - コア数を増やしても限界がある（10%の壁）
    - 並列化する前に、逐次部分の最適化が重要

  並列化可能率 | 2コア | 4コア | 8コア | 16コア | ∞コア
  ────────────┼──────┼──────┼──────┼───────┼──────
  50%          | 1.3x | 1.6x | 1.8x | 1.9x  | 2.0x
  75%          | 1.6x | 2.3x | 3.0x | 3.4x  | 4.0x
  90%          | 1.8x | 3.1x | 4.7x | 6.0x  | 10.0x
  95%          | 1.9x | 3.5x | 5.9x | 8.4x  | 20.0x
```

---

## 2. データ並列

```
データ並列 = 同じ処理を複数のデータに同時に適用

  入力: [1, 2, 3, 4, 5, 6, 7, 8]
  処理: x * 2

  Core 1: [1,2] → [2,4]
  Core 2: [3,4] → [6,8]
  Core 3: [5,6] → [10,12]
  Core 4: [7,8] → [14,16]

  結果: [2, 4, 6, 8, 10, 12, 14, 16]
```

```rust
// Rust: rayon（データ並列ライブラリ）
use rayon::prelude::*;

// 逐次
let sum: i64 = (0..1_000_000).map(|x| x * x).sum();

// 並列（par_iter に変えるだけ！）
let sum: i64 = (0..1_000_000).into_par_iter().map(|x| x * x).sum();

// 並列ソート
let mut data = vec![5, 2, 8, 1, 9, 3];
data.par_sort();

// 並列処理の例
let results: Vec<_> = files.par_iter()
    .map(|file| process_file(file))
    .filter(|r| r.is_ok())
    .collect();
```

```python
# Python: multiprocessing（GIL を回避）
from multiprocessing import Pool
import numpy as np

def process_chunk(data):
    return [x ** 2 for x in data]

# マルチプロセスで並列処理
with Pool(processes=4) as pool:
    chunks = np.array_split(range(1_000_000), 4)
    results = pool.map(process_chunk, chunks)

# concurrent.futures（高水準API）
from concurrent.futures import ProcessPoolExecutor

with ProcessPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(heavy_task, arg) for arg in args]
    results = [f.result() for f in futures]
```

---

## 3. タスク並列

```
タスク並列 = 異なる処理を同時に実行

  Task A: データベースクエリ
  Task B: 外部API呼び出し
  Task C: ファイル読み込み
  → 3つを同時に実行して、全完了を待つ
```

```go
// Go: goroutine + errgroup
import "golang.org/x/sync/errgroup"

func loadDashboard(ctx context.Context, userID int) (*Dashboard, error) {
    g, ctx := errgroup.WithContext(ctx)
    var user User
    var posts []Post
    var stats Stats

    g.Go(func() error {
        var err error
        user, err = fetchUser(ctx, userID)
        return err
    })
    g.Go(func() error {
        var err error
        posts, err = fetchPosts(ctx, userID)
        return err
    })
    g.Go(func() error {
        var err error
        stats, err = fetchStats(ctx, userID)
        return err
    })

    if err := g.Wait(); err != nil {
        return nil, err
    }
    return &Dashboard{user, posts, stats}, nil
}
```

---

## 4. ロックフリーとアトミック操作

```rust
// Rust: アトミック操作（ロックなしで安全に共有）
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn increment() {
    COUNTER.fetch_add(1, Ordering::Relaxed);
}

fn get_count() -> u64 {
    COUNTER.load(Ordering::Relaxed)
}

// Ordering（メモリ順序）
// Relaxed:  最小保証（カウンタに十分）
// Acquire:  この操作以降の読み書きがリオーダーされない
// Release:  この操作以前の読み書きがリオーダーされない
// SeqCst:   最強保証（全スレッドで同一の順序を観測）
```

```go
// Go: sync/atomic
import "sync/atomic"

var counter int64

func increment() {
    atomic.AddInt64(&counter, 1)
}

func getCount() int64 {
    return atomic.LoadInt64(&counter)
}
```

---

## 5. 並列処理のパターン

```
MapReduce:
  Map:    データを分割して並列処理
  Reduce: 結果を集約

  [データ] → Map(分割+処理) → [中間結果] → Reduce(集約) → [最終結果]

Fork-Join:
  タスクを分割（Fork）して並列実行し、結果を合流（Join）

  ┌─ Fork ─┐         ┌─ Join ─┐
  │ Task A  │  結果A  │        │
  ├─────────┤────────→│ 合流   │──→ 最終結果
  │ Task B  │  結果B  │        │
  └─────────┘────────→│        │
                      └────────┘

Pipeline:
  処理をステージに分割し、各ステージを並列実行

  Input → [Stage1] → [Stage2] → [Stage3] → Output
           ↓ 各ステージが独立して並列動作
```

---

## まとめ

| パターン | 用途 | 代表技術 |
|---------|------|---------|
| データ並列 | 同じ処理を大量データに | Rayon, NumPy, CUDA |
| タスク並列 | 異なる処理を同時に | goroutine, tokio::join |
| MapReduce | 分散データ処理 | Hadoop, Spark |
| アトミック | ロックフリーカウンタ | atomic, sync/atomic |

---

## 次に読むべきガイド
→ [[../06-language-comparison/00-scripting-languages.md]] — 言語比較

---

## 参考文献
1. Herlihy, M. & Shavit, N. "The Art of Multiprocessor Programming." 2nd Ed, 2020.
2. "Rayon: data-parallelism library for Rust." github.com/rayon-rs/rayon.
