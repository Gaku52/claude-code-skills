# 並列プログラミング (Parallel Programming)

> 並列プログラミングは「複数のCPUコアを活用して処理を高速化する」技術である。データ並列とタスク並列の2つのアプローチを軸に、ハードウェアレベルからアプリケーションレベルまでの並列化手法を体系的に学ぶ。

## この章で学ぶこと

- [ ] データ並列とタスク並列の違いを理解し、適切に使い分けられる
- [ ] アムダールの法則とグスタフソンの法則で並列化の効果を定量的に予測できる
- [ ] 各言語（Rust, Go, Python, Java, C++）の並列処理機能を実務で活用できる
- [ ] MapReduce, Fork-Join, Pipeline などの並列パターンを設計・実装できる
- [ ] ロックフリー・アトミック操作の原理と適用範囲を理解する
- [ ] 並列プログラミング特有のバグ（データ競合、偽共有など）を回避できる


## 前提知識

このガイドを読む前に、以下の知識があると理解が深まります:

- 基本的なプログラミングの知識
- 関連する基礎概念の理解
- [メッセージパッシング](./02-message-passing.md) の内容を理解していること

---

## 1. 並列処理の基礎概念

### 1.1 並行と並列の違い（再確認）

並行（Concurrency）と並列（Parallelism）は頻繁に混同されるが、本質的に異なる概念である。

```
============================================================
  並行（Concurrency）        並列（Parallelism）
============================================================

  論理的に同時に進行         物理的に同時に実行

  1つのCPUコアでも実現可能    複数のCPUコアが必要

  ┌──────────────────┐      ┌──────────────────┐
  │ Core 0           │      │ Core 0  │ Core 1 │
  │                  │      │         │        │
  │ ┌─A─┐ ┌─B─┐     │      │ ┌─A──┐  │ ┌─B──┐│
  │ │   │ │   │     │      │ │    │  │ │    ││
  │ └───┘ └───┘     │      │ └────┘  │ └────┘│
  │ ┌─A─┐ ┌─B─┐     │      │         │        │
  │ │   │ │   │     │      │         │        │
  │ └───┘ └───┘     │      │         │        │
  │                  │      │         │        │
  │ 時分割で交互に実行 │      │ 同時に実行       │
  └──────────────────┘      └──────────────────┘

  目的: 構造の整理            目的: 速度の向上
  例: Webサーバーの設計        例: 画像処理の高速化
============================================================
```

この章では「並列」に焦点を当てる。複数の物理コアを使って、計算をどのように分割・分配し、高速化するかを学ぶ。

### 1.2 並列処理のレベル

並列処理はハードウェアからソフトウェアまで複数のレベルで実現される。

```
================================================================
  並列処理の階層構造
================================================================

  [ビットレベル並列]
    │  64ビット演算 = 32ビットの2倍の幅を同時処理
    │  例: 64ビットCPUで64ビット整数加算を1命令で実行
    v
  [命令レベル並列 (ILP)]
    │  CPU内部で複数命令を同時実行
    │  パイプライン処理、スーパースカラー実行
    │  プログラマーが直接制御することは少ない
    v
  [SIMD (Single Instruction, Multiple Data)]
    │  1つの命令で複数のデータを同時処理
    │  例: SSE/AVX で4つのfloatを同時加算
    │  コンパイラの自動ベクトル化 or 組み込み関数
    v
  [スレッドレベル並列]
    │  複数のスレッドで異なる処理を同時実行
    │  OS がスレッドを各コアに割り当て
    │  本章の主要テーマ
    v
  [プロセスレベル並列]
    │  複数のプロセスで並列実行
    │  メモリ空間が分離 = 安全だが通信コスト大
    │  Python の multiprocessing など
    v
  [分散並列]
       複数のマシンにまたがる並列処理
       ネットワーク越しの通信が必要
       MapReduce, Apache Spark など
================================================================
```

### 1.3 ハードウェアの並列性を理解する

現代のCPUアーキテクチャを理解することは、効率的な並列プログラムを書く上で不可欠である。

```
================================================================
  現代のマルチコアCPU（例: 8コア/16スレッド）
================================================================

  ┌───────────────────────────────────────────────────────┐
  │                    CPU Package                        │
  │                                                       │
  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
  │  │ Core 0  │  │ Core 1  │  │ Core 2  │  │ Core 3  │ │
  │  │ L1 32KB │  │ L1 32KB │  │ L1 32KB │  │ L1 32KB │ │
  │  │ L2 256KB│  │ L2 256KB│  │ L2 256KB│  │ L2 256KB│ │
  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘ │
  │       └──────┬─────┘            └──────┬─────┘       │
  │         L3 Cache (共有)           L3 Cache (共有)     │
  │              └─────────┬──────────┘                   │
  │                   メモリコントローラ                    │
  └────────────────────────┬──────────────────────────────┘
                           │
                      メインメモリ (DRAM)

  重要なポイント:
  - L1/L2 は各コア専用 → 他コアのデータにアクセスするとキャッシュミス
  - L3 は複数コアで共有 → 共有データのアクセスに影響
  - キャッシュラインは通常64バイト → 偽共有 (False Sharing) の原因
================================================================
```

---

## 2. アムダールの法則とグスタフソンの法則

### 2.1 アムダールの法則 (Amdahl's Law)

並列化による高速化の理論的上限を示す法則。

```
================================================================
  アムダールの法則
================================================================

  並列化可能な部分が P (0 <= P <= 1)、コア数が N の場合:

                    1
  Speedup(N) = ─────────────
                (1-P) + P/N

  ┌─────────────────────────────────────────────────┐
  │  例: プログラムの90%が並列化可能 (P = 0.9)       │
  │                                                  │
  │  N=1:   1 / (0.1 + 0.9/1)  = 1.00x              │
  │  N=2:   1 / (0.1 + 0.9/2)  = 1.82x              │
  │  N=4:   1 / (0.1 + 0.9/4)  = 3.08x              │
  │  N=8:   1 / (0.1 + 0.9/8)  = 4.71x              │
  │  N=16:  1 / (0.1 + 0.9/16) = 5.93x              │
  │  N=64:  1 / (0.1 + 0.9/64) = 8.77x              │
  │  N=inf: 1 / (0.1 + 0)      = 10.00x (理論上限)   │
  └─────────────────────────────────────────────────┘

  高速化率
  ^
  20x│                              ............ P=95%
     │                      .......
  15x│                  ....
     │              ....
  10x│         .............................──── P=90%
     │     ....
   5x│  ...        .................──────────── P=75%
     │ ..    ......
   2x│..  ...  .........─────────────────────── P=50%
   1x│─────────────────────────────────────────
     └──┬──┬──┬──┬──┬──┬──┬──┬───────────→ コア数
        1  2  4  8  16 32 64 128

  並列化可能率 │ 2コア │ 4コア │ 8コア  │ 16コア │ 無限コア
  ─────────────┼───────┼───────┼────────┼────────┼─────────
  50%          │ 1.33x │ 1.60x │ 1.78x  │ 1.88x  │ 2.00x
  75%          │ 1.60x │ 2.29x │ 3.00x  │ 3.37x  │ 4.00x
  90%          │ 1.82x │ 3.08x │ 4.71x  │ 5.93x  │ 10.00x
  95%          │ 1.90x │ 3.48x │ 5.93x  │ 8.42x  │ 20.00x
  99%          │ 1.98x │ 3.88x │ 7.48x  │ 13.91x │ 100.00x
================================================================
```

### 2.2 グスタフソンの法則 (Gustafson's Law)

アムダールの法則は「問題サイズ固定」を前提としているが、実際にはコア数を増やせば扱える問題サイズも大きくなる。グスタフソンの法則はこの観点からの評価を提供する。

```
================================================================
  グスタフソンの法則
================================================================

  スケールアップ後の高速化率:

  Speedup(N) = N - s * (N - 1)

  ここで s は逐次実行部分の比率、N はコア数

  アムダールの法則:
    問題サイズ固定 → コアを増やしても限界あり
    「強いスケーリング (Strong Scaling)」

  グスタフソンの法則:
    問題サイズをコア数に比例して拡大
    コアを増やすほど並列部分が支配的に
    「弱いスケーリング (Weak Scaling)」

  ┌─────────────────────────────────────────────────┐
  │                                                  │
  │  例: s = 0.05 (5% が逐次部分)                    │
  │                                                  │
  │  N=8:   8 - 0.05*(8-1)   =  7.65x               │
  │  N=16:  16 - 0.05*(16-1) = 15.25x               │
  │  N=64:  64 - 0.05*(64-1) = 60.85x               │
  │  N=256: 256 - 0.05*255   = 243.25x              │
  │                                                  │
  │  → 問題サイズを増やせばほぼ線形にスケール          │
  └─────────────────────────────────────────────────┘
================================================================
```

### 2.3 スケーリング効率の評価

```
================================================================
  並列化効率の定義
================================================================

              Speedup(N)
  効率 E = ──────────────
                 N

  理想:  E = 1.0 (100%) → 完璧な線形スケーリング
  良好:  E > 0.7 (70%) → 実用的な並列化
  要改善: E < 0.5 (50%) → オーバーヘッドが支配的

  効率低下の原因:
  ┌──────────────────────────────────────────────┐
  │ 1. 逐次ボトルネック: 並列化不可能な部分        │
  │ 2. 同期オーバーヘッド: ロック競合、バリア待ち   │
  │ 3. 通信コスト: データの分配と集約               │
  │ 4. 負荷不均衡: コア間のワークロード偏り         │
  │ 5. メモリ帯域: キャッシュミス、偽共有           │
  └──────────────────────────────────────────────┘
================================================================
```

---

## 3. データ並列 (Data Parallelism)

### 3.1 概念と原理

データ並列は「同じ処理を複数のデータに同時に適用する」パターンである。科学計算、画像処理、機械学習で最も一般的な並列化手法となっている。

```
================================================================
  データ並列の基本概念
================================================================

  入力データ: [a1, a2, a3, a4, a5, a6, a7, a8, ... aN]

  分割 (Partition):
  ┌──────────┬──────────┬──────────┬──────────┐
  │ Chunk 0  │ Chunk 1  │ Chunk 2  │ Chunk 3  │
  │ a1,a2,...│ a?,a?,...│ a?,a?,...│ ...,aN   │
  └────┬─────┘────┬─────┘────┬─────┘────┬─────┘
       │          │          │          │
       v          v          v          v
  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
  │ Core 0 │ │ Core 1 │ │ Core 2 │ │ Core 3 │
  │ f(x)   │ │ f(x)   │ │ f(x)   │ │ f(x)   │
  │ 同じ処理│ │ 同じ処理│ │ 同じ処理│ │ 同じ処理│
  └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘
       │          │          │          │
       v          v          v          v
  ┌──────────┬──────────┬──────────┬──────────┐
  │ Result 0 │ Result 1 │ Result 2 │ Result 3 │
  └──────────┴──────────┴──────────┴──────────┘
       │          │          │          │
       └──────────┴──────┬───┴──────────┘
                         v
                  結合 (Merge/Reduce)
                  最終結果
================================================================
```

### 3.2 Rust: rayon によるデータ並列

Rust の rayon クレートは、イテレータベースの直感的なAPIでデータ並列を実現する。

```rust
// ================================================================
// Rust: rayon によるデータ並列
// ================================================================

use rayon::prelude::*;
use std::collections::HashMap;

// --- 基本: 逐次 → 並列への変換 ---

// 逐次バージョン
fn sequential_sum(data: &[f64]) -> f64 {
    data.iter()
        .map(|x| x.powi(2))
        .sum()
}

// 並列バージョン（iter → par_iter に変更するだけ）
fn parallel_sum(data: &[f64]) -> f64 {
    data.par_iter()
        .map(|x| x.powi(2))
        .sum()
}

// --- 並列フィルタリングと集約 ---

#[derive(Debug, Clone)]
struct LogEntry {
    level: String,
    message: String,
    timestamp: u64,
}

fn analyze_logs(logs: &[LogEntry]) -> HashMap<String, usize> {
    // 並列でログレベルごとの件数を集計
    logs.par_iter()
        .fold(
            || HashMap::new(),  // 各スレッドのローカルmap
            |mut acc, entry| {
                *acc.entry(entry.level.clone()).or_insert(0) += 1;
                acc
            },
        )
        .reduce(
            || HashMap::new(),  // 初期値
            |mut a, b| {        // 2つのmapをマージ
                for (key, value) in b {
                    *a.entry(key).or_insert(0) += value;
                }
                a
            },
        )
}

// --- 並列ソートとチャンク処理 ---

fn parallel_sort_and_process(mut data: Vec<i64>) -> Vec<i64> {
    // 並列ソート（不安定ソート:高速だが同値の順序不定）
    data.par_sort_unstable();

    // 並列チャンク処理
    data.par_chunks(1024)
        .flat_map(|chunk| {
            chunk.iter()
                .filter(|&&x| x > 0)
                .map(|&x| x * 2)
                .collect::<Vec<_>>()
        })
        .collect()
}

// --- スレッドプール制御 ---

fn configure_thread_pool() {
    // カスタムスレッドプール（デフォルトは論理コア数）
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)           // スレッド数を4に制限
        .stack_size(8 * 1024 * 1024)  // スタックサイズ8MB
        .build_global()
        .unwrap();
}

// --- 実行例 ---

fn main() {
    let data: Vec<f64> = (0..10_000_000)
        .map(|x| x as f64)
        .collect();

    let result = parallel_sum(&data);
    println!("並列合計: {}", result);

    // 並列イテレータの連鎖
    let processed: Vec<String> = (0..1_000_000_u64)
        .into_par_iter()
        .filter(|&x| x % 3 == 0)
        .map(|x| format!("item_{}", x))
        .collect();
    println!("処理件数: {}", processed.len());
}
```

### 3.3 Python: multiprocessing と concurrent.futures

Python ではGIL（Global Interpreter Lock）の制約があるため、CPU密集型の並列処理には multiprocessing を使用する。

```python
# ================================================================
# Python: multiprocessing によるデータ並列
# ================================================================

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from typing import List, Tuple
import time

# --- 基本: Pool.map ---

def square(x: float) -> float:
    """CPU密集型処理の例"""
    return sum(i ** 2 for i in range(int(x)))

def basic_parallel():
    """基本的な並列処理"""
    data = list(range(100, 200))

    # 逐次実行
    results_seq = [square(x) for x in data]

    # 並列実行
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results_par = pool.map(square, data)

    assert results_seq == results_par

# --- チャンク処理でオーバーヘッド削減 ---

def process_chunk(chunk: List[float]) -> List[float]:
    """チャンク単位で処理（プロセス間通信を削減）"""
    return [x ** 2 + 2 * x + 1 for x in chunk]

def chunked_parallel():
    """チャンク分割による効率的な並列処理"""
    data = list(range(1_000_000))
    n_workers = mp.cpu_count()

    # データを均等にチャンク分割
    chunk_size = len(data) // n_workers
    chunks = [
        data[i:i + chunk_size]
        for i in range(0, len(data), chunk_size)
    ]

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(process_chunk, chunks)

    # 結果をフラット化
    flat_results = [item for chunk in results for item in chunk]
    print(f"処理件数: {len(flat_results)}")

# --- ProcessPoolExecutor と進捗追跡 ---

def heavy_computation(args: Tuple[int, List[float]]) -> dict:
    """IDとデータを受け取り、結果を辞書で返す"""
    task_id, data = args
    result = sum(x ** 2 for x in data)
    return {"task_id": task_id, "result": result}

def parallel_with_progress():
    """進捗追跡付き並列処理"""
    tasks = [
        (i, list(range(i * 1000, (i + 1) * 1000)))
        for i in range(50)
    ]

    completed = 0
    total = len(tasks)
    results = []

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(heavy_computation, task): task[0]
            for task in tasks
        }

        for future in as_completed(futures):
            task_id = futures[future]
            try:
                result = future.result(timeout=30)
                results.append(result)
                completed += 1
                print(f"進捗: {completed}/{total} "
                      f"(Task {task_id} 完了)")
            except Exception as e:
                print(f"Task {task_id} 失敗: {e}")

    return sorted(results, key=lambda r: r["task_id"])

# --- NumPy による暗黙的な並列化 ---

def numpy_parallel():
    """NumPyのベクトル化演算（内部でSIMD/マルチスレッド）"""
    # NumPyはBLAS/LAPACKを通じて暗黙的に並列化される
    a = np.random.rand(10000, 10000)
    b = np.random.rand(10000, 10000)

    # 行列積（内部で並列実行）
    c = np.dot(a, b)

    # 要素ごとの演算（SIMD最適化）
    d = np.sin(a) + np.cos(b) * np.exp(-a)

    return c, d

if __name__ == "__main__":
    basic_parallel()
    chunked_parallel()
    results = parallel_with_progress()
    print(f"全タスク完了: {len(results)} 件")
```

### 3.4 Java: parallel streams と ForkJoinPool

```java
// ================================================================
// Java: Parallel Streams と ForkJoinPool
// ================================================================

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.*;

public class DataParallelExample {

    // --- Parallel Streams ---
    public static long parallelSum(List<Long> numbers) {
        return numbers.parallelStream()
            .filter(n -> n > 0)
            .mapToLong(n -> n * n)
            .sum();
    }

    // --- カスタムForkJoinPool（並列度制御） ---
    public static <T> List<T> parallelWithCustomPool(
            List<T> data,
            java.util.function.Function<T, T> transform,
            int parallelism) throws Exception {

        ForkJoinPool pool = new ForkJoinPool(parallelism);
        try {
            return pool.submit(() ->
                data.parallelStream()
                    .map(transform)
                    .collect(Collectors.toList())
            ).get();
        } finally {
            pool.shutdown();
        }
    }

    // --- ForkJoinTask の直接利用 ---
    static class ParallelMergeSort extends RecursiveAction {
        private final int[] array;
        private final int lo, hi;
        private static final int THRESHOLD = 1024;

        ParallelMergeSort(int[] array, int lo, int hi) {
            this.array = array;
            this.lo = lo;
            this.hi = hi;
        }

        @Override
        protected void compute() {
            if (hi - lo < THRESHOLD) {
                // 小さいサイズは逐次ソート
                Arrays.sort(array, lo, hi);
                return;
            }
            int mid = lo + (hi - lo) / 2;
            // 左右を並列にソート
            invokeAll(
                new ParallelMergeSort(array, lo, mid),
                new ParallelMergeSort(array, mid, hi)
            );
            merge(array, lo, mid, hi);
        }

        private void merge(int[] arr, int lo, int mid, int hi) {
            int[] temp = Arrays.copyOfRange(arr, lo, mid);
            int i = 0, j = mid, k = lo;
            while (i < temp.length && j < hi) {
                arr[k++] = (temp[i] <= arr[j]) ? temp[i++] : arr[j++];
            }
            while (i < temp.length) arr[k++] = temp[i++];
        }
    }

    public static void main(String[] args) throws Exception {
        // Parallel Stream の例
        List<Long> numbers = LongStream.rangeClosed(1, 10_000_000)
            .boxed()
            .collect(Collectors.toList());
        long sum = parallelSum(numbers);
        System.out.println("Sum: " + sum);

        // ForkJoinPool でのソート
        int[] data = new Random().ints(10_000_000).toArray();
        ForkJoinPool pool = ForkJoinPool.commonPool();
        pool.invoke(new ParallelMergeSort(data, 0, data.length));
        System.out.println("Sorted: " + Arrays.toString(
            Arrays.copyOf(data, 10)));
    }
}
```

---

## 4. タスク並列 (Task Parallelism)

### 4.1 概念と原理

タスク並列は「異なる処理を同時に実行する」パターンである。各タスクは独立した処理を行い、結果を最後に統合する。

```
================================================================
  タスク並列の基本概念
================================================================

  逐次実行:
  ┌────────────┐ ┌────────────┐ ┌────────────┐
  │  DB Query  │→│  API Call  │→│ File Read  │
  │  200ms     │ │  300ms     │ │  100ms     │
  └────────────┘ └────────────┘ └────────────┘
  合計: 200 + 300 + 100 = 600ms

  タスク並列:
  ┌────────────┐
  │  DB Query  │──────┐
  │  200ms     │      │
  └────────────┘      │
  ┌────────────────┐  ├──→ 結果統合
  │  API Call      │  │
  │  300ms         │──┘
  └────────────────┘  │
  ┌──────────┐        │
  │File Read │────────┘
  │  100ms   │
  └──────────┘
  合計: max(200, 300, 100) = 300ms (50%短縮)
================================================================
```

### 4.2 Go: goroutine + errgroup

```go
// ================================================================
// Go: goroutine + errgroup によるタスク並列
// ================================================================

package main

import (
    "context"
    "fmt"
    "log"
    "sync"
    "time"

    "golang.org/x/sync/errgroup"
)

// --- 基本: errgroup による並列タスク ---

type Dashboard struct {
    User    User
    Posts   []Post
    Stats   Stats
    Friends []Friend
}

func loadDashboard(ctx context.Context, userID int) (*Dashboard, error) {
    g, ctx := errgroup.WithContext(ctx)
    var (
        user    User
        posts   []Post
        stats   Stats
        friends []Friend
    )

    // 4つのタスクを並列実行
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

    g.Go(func() error {
        var err error
        friends, err = fetchFriends(ctx, userID)
        return err
    })

    // 全タスクの完了を待機（1つでもエラーなら即キャンセル）
    if err := g.Wait(); err != nil {
        return nil, fmt.Errorf("dashboard load failed: %w", err)
    }

    return &Dashboard{user, posts, stats, friends}, nil
}

// --- 並列度制限付きタスク実行 ---

func processURLs(ctx context.Context, urls []string) ([]Result, error) {
    g, ctx := errgroup.WithContext(ctx)
    // 同時実行数を10に制限
    g.SetLimit(10)

    results := make([]Result, len(urls))
    var mu sync.Mutex

    for i, url := range urls {
        i, url := i, url  // ループ変数キャプチャ
        g.Go(func() error {
            result, err := fetch(ctx, url)
            if err != nil {
                return err
            }
            mu.Lock()
            results[i] = result
            mu.Unlock()
            return nil
        })
    }

    if err := g.Wait(); err != nil {
        return nil, err
    }
    return results, nil
}

// --- タイムアウト付きタスク並列 ---

func loadWithTimeout(userID int) (*Dashboard, error) {
    ctx, cancel := context.WithTimeout(
        context.Background(),
        5*time.Second,
    )
    defer cancel()

    return loadDashboard(ctx, userID)
}
```

### 4.3 Rust: tokio::join! と rayon::join

```rust
// ================================================================
// Rust: タスク並列の2つのアプローチ
// ================================================================

// --- 非同期タスク並列（I/Oバウンド） ---
use tokio;

async fn load_dashboard(user_id: u64) -> Result<Dashboard, Error> {
    // tokio::join! で複数の非同期タスクを並列実行
    let (user, posts, stats) = tokio::join!(
        fetch_user(user_id),
        fetch_posts(user_id),
        fetch_stats(user_id),
    );

    Ok(Dashboard {
        user: user?,
        posts: posts?,
        stats: stats?,
    })
}

// try_join! でエラー時に早期終了
async fn load_dashboard_with_error_handling(
    user_id: u64,
) -> Result<Dashboard, Error> {
    let (user, posts, stats) = tokio::try_join!(
        fetch_user(user_id),
        fetch_posts(user_id),
        fetch_stats(user_id),
    )?;

    Ok(Dashboard { user, posts, stats })
}

// --- CPUバウンドなタスク並列（rayon::join） ---
use rayon;

fn parallel_analysis(data: &[f64]) -> (f64, f64, Vec<f64>) {
    let (mean_result, (variance_result, sorted)) = rayon::join(
        // 左タスク: 平均計算
        || {
            let sum: f64 = data.iter().sum();
            sum / data.len() as f64
        },
        // 右タスク: さらに分割
        || rayon::join(
            // 分散計算
            || {
                let mean = data.iter().sum::<f64>() / data.len() as f64;
                data.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>() / data.len() as f64
            },
            // ソート
            || {
                let mut sorted = data.to_vec();
                sorted.par_sort_unstable_by(|a, b| {
                    a.partial_cmp(b).unwrap()
                });
                sorted
            },
        ),
    );

    (mean_result, variance_result, sorted)
}
```

### 4.4 C++: std::async と並列アルゴリズム

```cpp
// ================================================================
// C++17/20: 並列アルゴリズムと std::async
// ================================================================

#include <algorithm>
#include <execution>  // C++17 並列実行ポリシー
#include <future>
#include <numeric>
#include <vector>

// --- C++17 並列アルゴリズム ---

void parallel_algorithms_example() {
    std::vector<int> data(10'000'000);
    std::iota(data.begin(), data.end(), 0);

    // 並列ソート
    std::sort(std::execution::par,
              data.begin(), data.end());

    // 並列変換
    std::transform(std::execution::par_unseq,
                   data.begin(), data.end(),
                   data.begin(),
                    { return x * 2; });

    // 並列リデュース
    long long sum = std::reduce(
        std::execution::par,
        data.begin(), data.end(),
        0LL);

    // 並列 for_each
    std::for_each(std::execution::par,
                  data.begin(), data.end(),
                   { x += 1; });
}

// --- std::async によるタスク並列 ---

struct AnalysisResult {
    double mean;
    double variance;
    int max_val;
};

AnalysisResult parallel_analysis(const std::vector<int>& data) {
    // 3つのタスクを非同期実行
    auto mean_future = std::async(std::launch::async, [&]() {
        double sum = std::reduce(
            std::execution::par,
            data.begin(), data.end(), 0.0);
        return sum / data.size();
    });

    auto var_future = std::async(std::launch::async, [&]() {
        double mean = std::reduce(
            std::execution::par,
            data.begin(), data.end(), 0.0) / data.size();
        double sq_sum = std::transform_reduce(
            std::execution::par,
            data.begin(), data.end(),
            0.0,
            std::plus<>{},
            mean {
                return (x - mean) * (x - mean);
            });
        return sq_sum / data.size();
    });

    auto max_future = std::async(std::launch::async, [&]() {
        return *std::max_element(
            std::execution::par,
            data.begin(), data.end());
    });

    return {
        mean_future.get(),
        var_future.get(),
        max_future.get()
    };
}

// --- 実行ポリシーの比較 ---
//
// std::execution::seq       逐次実行（デフォルト）
// std::execution::par       並列実行（スレッド使用）
// std::execution::par_unseq 並列+ベクトル化（SIMD併用）
// std::execution::unseq     ベクトル化のみ（C++20）
```

---

## 5. ロックフリーとアトミック操作

### 5.1 アトミック操作の基礎

アトミック操作は「不可分操作」であり、他のスレッドから見て途中状態が観測されない操作である。ロックを使わずに安全なスレッド間通信を実現する。

```
================================================================
  アトミック操作 vs ロック
================================================================

  ロック方式:
  Thread A: [acquire lock] → [read] → [modify] → [write] → [release lock]
  Thread B: [  wait...  ] → [  wait...  ] → [acquire lock] → [read] → ...

  アトミック方式:
  Thread A: [atomic read-modify-write]    ← 1命令で完了
  Thread B: [atomic read-modify-write]    ← 待機不要

  比較:
  ┌─────────────┬────────────────┬────────────────────┐
  │ 特性         │ ロック          │ アトミック           │
  ├─────────────┼────────────────┼────────────────────┤
  │ オーバーヘッド│ 大きい(カーネル) │ 小さい(CPU命令)     │
  │ スケーラビリティ│ 低い(競合時)   │ 高い(短時間ビジー)  │
  │ 対応操作     │ 任意の操作      │ 単純な操作のみ       │
  │ デッドロック  │ あり得る        │ なし                │
  │ ABA問題      │ なし           │ あり得る             │
  │ 実装複雑度   │ 中程度          │ 高い                │
  └─────────────┴────────────────┴────────────────────┘
================================================================
```

### 5.2 メモリオーダリング

```
================================================================
  メモリオーダリング（メモリ順序制約）
================================================================

  CPUは性能のために命令の実行順序を入れ替える（リオーダー）
  マルチスレッドでは、この入れ替えが問題になることがある

  Ordering レベル（弱い → 強い）:

  ┌──────────┬──────────────────────────────────────────────┐
  │ Relaxed  │ 順序保証なし。カウンタに最適。最高性能。       │
  │          │ 他のメモリ操作との順序関係を保証しない         │
  ├──────────┼──────────────────────────────────────────────┤
  │ Acquire  │ この操作以降の読み書きが、この操作より        │
  │          │ 前にリオーダーされないことを保証              │
  │          │ ロック獲得に相当                             │
  ├──────────┼──────────────────────────────────────────────┤
  │ Release  │ この操作以前の読み書きが、この操作より        │
  │          │ 後にリオーダーされないことを保証              │
  │          │ ロック解放に相当                             │
  ├──────────┼──────────────────────────────────────────────┤
  │ AcqRel   │ Acquire + Release の両方を保証               │
  │          │ read-modify-write 操作に使用                 │
  ├──────────┼──────────────────────────────────────────────┤
  │ SeqCst   │ 全スレッドで同一の操作順序を観測              │
  │          │ 最も直感的だが最も低速                        │
  └──────────┴──────────────────────────────────────────────┘

  使い分けの指針:
  - 単純なカウンタ     → Relaxed
  - フラグ（ready通知） → Release(書き手) / Acquire(読み手)
  - 迷ったら          → SeqCst（正しさ優先）
================================================================
```

### 5.3 Rust: アトミック操作の詳細

```rust
// ================================================================
// Rust: アトミック操作の実践パターン
// ================================================================

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

// --- パターン1: アトミックカウンタ ---

struct AtomicCounter {
    count: AtomicU64,
}

impl AtomicCounter {
    fn new() -> Self {
        Self { count: AtomicU64::new(0) }
    }

    fn increment(&self) -> u64 {
        // fetch_add は加算前の値を返す
        self.count.fetch_add(1, Ordering::Relaxed)
    }

    fn get(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }
}

fn counter_example() {
    let counter = Arc::new(AtomicCounter::new());
    let mut handles = vec![];

    for _ in 0..8 {
        let counter = Arc::clone(&counter);
        handles.push(thread::spawn(move || {
            for _ in 0..100_000 {
                counter.increment();
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
    // 常に正確に 800,000 になる
    assert_eq!(counter.get(), 800_000);
}

// --- パターン2: スピンロック ---

struct SpinLock {
    locked: AtomicBool,
}

impl SpinLock {
    fn new() -> Self {
        Self { locked: AtomicBool::new(false) }
    }

    fn lock(&self) {
        // compare_exchange で false → true に変更を試みる
        while self.locked.compare_exchange_weak(
            false,              // 期待値
            true,               // 新しい値
            Ordering::Acquire,  // 成功時のOrdering
            Ordering::Relaxed,  // 失敗時のOrdering
        ).is_err() {
            // スピン待ち（ビジーウェイト）
            std::hint::spin_loop();
        }
    }

    fn unlock(&self) {
        self.locked.store(false, Ordering::Release);
    }
}

// --- パターン3: 一度だけ初期化（lazy init） ---

use std::sync::atomic::AtomicPtr;
use std::ptr;

struct LazyValue<T> {
    ptr: AtomicPtr<T>,
}

impl<T> LazyValue<T> {
    fn new() -> Self {
        Self { ptr: AtomicPtr::new(ptr::null_mut()) }
    }

    fn get_or_init(&self, init: impl FnOnce() -> T) -> &T {
        let mut p = self.ptr.load(Ordering::Acquire);
        if p.is_null() {
            let new = Box::into_raw(Box::new(init()));
            match self.ptr.compare_exchange(
                ptr::null_mut(),
                new,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => p = new,
                Err(existing) => {
                    // 他のスレッドが先に初期化した
                    unsafe { drop(Box::from_raw(new)); }
                    p = existing;
                }
            }
        }
        unsafe { &*p }
    }
}
```

### 5.4 Go: sync/atomic パッケージ

```go
// ================================================================
// Go: sync/atomic の実践パターン
// ================================================================

package main

import (
    "fmt"
    "sync"
    "sync/atomic"
)

// --- アトミックカウンタ ---

type AtomicCounter struct {
    count atomic.Int64
}

func (c *AtomicCounter) Increment() int64 {
    return c.count.Add(1)
}

func (c *AtomicCounter) Get() int64 {
    return c.count.Load()
}

// --- アトミック値（設定の動的更新） ---

type Config struct {
    MaxWorkers int
    Timeout    int
    Debug      bool
}

type ConfigHolder struct {
    config atomic.Value // *Config を格納
}

func (h *ConfigHolder) Load() *Config {
    return h.config.Load().(*Config)
}

func (h *ConfigHolder) Store(cfg *Config) {
    h.config.Store(cfg)
}

// 使用例: 設定のホットリロード
func configExample() {
    holder := &ConfigHolder{}
    holder.Store(&Config{MaxWorkers: 4, Timeout: 30, Debug: false})

    // ワーカーは常に最新の設定を参照
    var wg sync.WaitGroup
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            cfg := holder.Load()
            fmt.Printf("Workers: %d\n", cfg.MaxWorkers)
        }()
    }

    // 設定を動的に更新（ロック不要）
    holder.Store(&Config{MaxWorkers: 8, Timeout: 60, Debug: true})
    wg.Wait()
}
```

---

## 6. 並列処理のデザインパターン

### 6.1 MapReduce パターン

```
================================================================
  MapReduce パターン
================================================================

  大規模データを分割 → 並列処理 → 集約する汎用パターン

  入力データ:
  ["hello world", "hello rust", "world hello"]

  Phase 1 - Map（並列）:
  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
  │ Worker 0        │   │ Worker 1        │   │ Worker 2        │
  │ "hello world"   │   │ "hello rust"    │   │ "world hello"   │
  │ → (hello,1)     │   │ → (hello,1)     │   │ → (world,1)     │
  │   (world,1)     │   │   (rust,1)      │   │   (hello,1)     │
  └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
           │                     │                     │
           v                     v                     v

  Phase 2 - Shuffle（キーでグループ化）:
  hello → [(hello,1), (hello,1), (hello,1)]
  world → [(world,1), (world,1)]
  rust  → [(rust,1)]

  Phase 3 - Reduce（並列）:
  ┌─────────────────┐   ┌─────────────────┐   ┌────────────────┐
  │ Reducer 0       │   │ Reducer 1       │   │ Reducer 2      │
  │ hello: 1+1+1=3  │   │ world: 1+1=2    │   │ rust: 1=1      │
  └────────┬────────┘   └────────┬────────┘   └────────┬───────┘
           │                     │                     │
           v                     v                     v
  最終結果: {hello: 3, world: 2, rust: 1}
================================================================
```

```rust
// ================================================================
// Rust: MapReduce の実装
// ================================================================

use rayon::prelude::*;
use std::collections::HashMap;

fn word_count(documents: &[String]) -> HashMap<String, usize> {
    documents
        .par_iter()
        // Map: 各ドキュメントの単語をカウント
        .map(|doc| {
            let mut local_counts = HashMap::new();
            for word in doc.split_whitespace() {
                let word = word.to_lowercase();
                *local_counts.entry(word).or_insert(0) += 1;
            }
            local_counts
        })
        // Reduce: ローカルカウントをマージ
        .reduce(
            || HashMap::new(),
            |mut acc, local| {
                for (word, count) in local {
                    *acc.entry(word).or_insert(0) += count;
                }
                acc
            },
        )
}

// 汎用 MapReduce フレームワーク
fn map_reduce<T, K, V, MapFn, ReduceFn>(
    data: &[T],
    map_fn: MapFn,
    reduce_fn: ReduceFn,
) -> HashMap<K, V>
where
    T: Sync,
    K: Eq + std::hash::Hash + Send,
    V: Send,
    MapFn: Fn(&T) -> Vec<(K, V)> + Sync,
    ReduceFn: Fn(V, V) -> V + Sync + Copy,
{
    data.par_iter()
        .flat_map(|item| map_fn(item))
        .fold(
            || HashMap::new(),
            |mut acc, (key, value)| {
                acc.entry(key)
                    .and_modify(|v| *v = reduce_fn(
                        std::mem::replace(v, unsafe {
                            std::mem::zeroed()
                        }),
                        value,
                    ))
                    .or_insert(value);
                acc
            },
        )
        .reduce(
            || HashMap::new(),
            |mut a, b| {
                for (key, value) in b {
                    a.entry(key)
                        .and_modify(|v| *v = reduce_fn(
                            std::mem::replace(v, unsafe {
                                std::mem::zeroed()
                            }),
                            value,
                        ))
                        .or_insert(value);
                }
                a
            },
        )
}
```

### 6.2 Fork-Join パターン

```
================================================================
  Fork-Join パターン
================================================================

  再帰的にタスクを分割（Fork）し、結果を合流（Join）する

                       ┌─────────────┐
                       │ 問題全体     │
                       │ [1..1000]   │
                       └──────┬──────┘
                     Fork     │
                 ┌────────────┼────────────┐
                 v            v            v
          ┌──────────┐ ┌──────────┐ ┌──────────┐
          │ [1..333] │ │[334..666]│ │[667..1000]│
          └────┬─────┘ └────┬─────┘ └────┬─────┘
               │            │            │
            Fork?         Fork?        Fork?
          ┌───┴───┐    (十分小さい)    ┌───┴───┐
          v       v     → 逐次処理     v       v
       [1..166][167..333]          [667..833][834..1000]
          │       │        │          │       │
          v       v        v          v       v
        計算     計算      計算       計算     計算
          │       │        │          │       │
          └───┬───┘        │          └───┬───┘
              │            │              │
           Join          直接           Join
              │            │              │
              └────────────┼──────────────┘
                           │
                         Join
                           │
                        最終結果
================================================================
```

### 6.3 Pipeline パターン

```
================================================================
  Pipeline パターン
================================================================

  処理をステージに分割し、各ステージを並列実行する
  各ステージはチャネルで接続される

  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ Stage 1  │───→│ Stage 2  │───→│ Stage 3  │───→│ Stage 4  │
  │ 読み込み  │ ch │ パース    │ ch │ 変換     │ ch │ 書き出し  │
  └──────────┘    └──────────┘    └──────────┘    └──────────┘

  時間軸:
  t1: [S1: item1] [         ] [         ] [         ]
  t2: [S1: item2] [S2: item1] [         ] [         ]
  t3: [S1: item3] [S2: item2] [S3: item1] [         ]
  t4: [S1: item4] [S2: item3] [S3: item2] [S4: item1]
  t5: [         ] [S2: item4] [S3: item3] [S4: item2]

  → 各ステージが独立して動作し、スループットが向上
================================================================
```

```go
// ================================================================
// Go: Pipeline パターンの実装
// ================================================================

package main

import (
    "context"
    "fmt"
    "strings"
    "sync"
)

// パイプラインのステージを関数として定義

// Stage 1: データ生成
func generate(ctx context.Context, items ...string) <-chan string {
    out := make(chan string)
    go func() {
        defer close(out)
        for _, item := range items {
            select {
            case out <- item:
            case <-ctx.Done():
                return
            }
        }
    }()
    return out
}

// Stage 2: 変換（大文字化）
func toUpper(ctx context.Context, in <-chan string) <-chan string {
    out := make(chan string)
    go func() {
        defer close(out)
        for s := range in {
            select {
            case out <- strings.ToUpper(s):
            case <-ctx.Done():
                return
            }
        }
    }()
    return out
}

// Stage 3: フィルタリング
func filterLong(ctx context.Context, in <-chan string,
                minLen int) <-chan string {
    out := make(chan string)
    go func() {
        defer close(out)
        for s := range in {
            if len(s) >= minLen {
                select {
                case out <- s:
                case <-ctx.Done():
                    return
                }
            }
        }
    }()
    return out
}

// Fan-Out: 1つのチャネルを複数のワーカーで読む
func fanOut(ctx context.Context, in <-chan string,
            n int, process func(string) string) <-chan string {
    out := make(chan string)
    var wg sync.WaitGroup

    for i := 0; i < n; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            for item := range in {
                select {
                case out <- process(item):
                case <-ctx.Done():
                    return
                }
            }
        }()
    }

    go func() {
        wg.Wait()
        close(out)
    }()
    return out
}

func main() {
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()

    // パイプライン構築
    stage1 := generate(ctx, "hello", "world", "go", "pipeline")
    stage2 := toUpper(ctx, stage1)
    stage3 := filterLong(ctx, stage2, 3)

    // 結果を消費
    for result := range stage3 {
        fmt.Println(result)
    }
    // 出力: HELLO, WORLD, PIPELINE
}
```

---

## 7. 偽共有 (False Sharing) と キャッシュ最適化

### 7.1 偽共有の問題

```
================================================================
  偽共有 (False Sharing)
================================================================

  問題: 異なるコアが「異なる変数」を更新しているのに、
       同じキャッシュラインに載っているため性能が劣化する

  メモリレイアウト（キャッシュライン = 64バイト）:
  ┌─────────────────────────────────────────────────────┐
  │           1つのキャッシュライン (64 bytes)             │
  │  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
  │  │ counter_a│  │ counter_b│  │   padding         │  │
  │  │ (8 bytes)│  │ (8 bytes)│  │                   │  │
  │  └──────────┘  └──────────┘  └───────────────────┘  │
  └─────────────────────────────────────────────────────┘
       Core 0が更新      Core 1が更新
       ↓                 ↓
  Core 0が counter_a を更新
    → キャッシュライン全体が「変更済み」
    → Core 1 のキャッシュが無効化される
    → Core 1 は counter_b を読むためにメモリからリロード
    → Core 1 が counter_b を更新
    → Core 0 のキャッシュが無効化される
    → ピンポン状態 = 大幅な性能低下

  解決策: パディングで変数を別のキャッシュラインに配置
  ┌───────────────────────────────────────┐
  │ キャッシュライン 0                      │
  │ ┌──────────┐  ┌─────────────────────┐ │
  │ │ counter_a│  │    padding (56B)    │ │
  │ └──────────┘  └─────────────────────┘ │
  └───────────────────────────────────────┘
  ┌───────────────────────────────────────┐
  │ キャッシュライン 1                      │
  │ ┌──────────┐  ┌─────────────────────┐ │
  │ │ counter_b│  │    padding (56B)    │ │
  │ └──────────┘  └─────────────────────┘ │
  └───────────────────────────────────────┘
================================================================
```

### 7.2 偽共有の回避

```rust
// ================================================================
// Rust: 偽共有の回避
// ================================================================

use std::sync::atomic::{AtomicU64, Ordering};

// BAD: 偽共有が発生する
struct BadCounters {
    counter_a: AtomicU64,  // 同じキャッシュラインに載る可能性
    counter_b: AtomicU64,
}

// GOOD: パディングで分離
#[repr(C)]
struct GoodCounters {
    counter_a: AtomicU64,
    _pad_a: [u8; 56],     // 64 - 8 = 56バイトのパディング
    counter_b: AtomicU64,
    _pad_b: [u8; 56],
}

// Rust の crossbeam には CachePadded が用意されている
use crossbeam_utils::CachePadded;

struct BestCounters {
    counter_a: CachePadded<AtomicU64>,
    counter_b: CachePadded<AtomicU64>,
}

impl BestCounters {
    fn new() -> Self {
        Self {
            counter_a: CachePadded::new(AtomicU64::new(0)),
            counter_b: CachePadded::new(AtomicU64::new(0)),
        }
    }
}
```

```java
// ================================================================
// Java: @Contended による偽共有回避
// ================================================================

import java.util.concurrent.atomic.AtomicLong;

// Java 8+: @Contended アノテーション
// JVM起動時に -XX:-RestrictContended オプションが必要
public class PaddedCounters {
    // BAD: 偽共有
    static class BadCounters {
        volatile long counterA;
        volatile long counterB;
    }

    // GOOD: @Contended で自動パディング
    static class GoodCounters {
        @jdk.internal.vm.annotation.Contended
        volatile long counterA;

        @jdk.internal.vm.annotation.Contended
        volatile long counterB;
    }

    // GOOD: 手動パディング（@Contended が使えない場合）
    static class ManualPaddedCounters {
        volatile long counterA;
        long p1, p2, p3, p4, p5, p6, p7;  // 56バイトのパディング
        volatile long counterB;
        long q1, q2, q3, q4, q5, q6, q7;
    }
}
```

---

## 8. 並列処理の比較表

### 8.1 言語別並列処理機能の比較

| 特性 | Rust (rayon) | Go (goroutine) | Python (multiprocessing) | Java (ForkJoinPool) | C++ (std::execution) |
|------|-------------|----------------|-------------------------|--------------------|--------------------|
| 並列の単位 | ワークスティーリング | goroutine + OS スレッド | OSプロセス | ForkJoinTask | OS スレッド |
| データ並列API | `par_iter()` | 手動分割 | `Pool.map()` | `parallelStream()` | `std::execution::par` |
| タスク並列API | `rayon::join` | `errgroup` | `ProcessPoolExecutor` | `CompletableFuture` | `std::async` |
| メモリモデル | 所有権で安全保証 | CSPモデル | プロセス分離 | Java Memory Model | C++ Memory Model |
| データ競合防止 | コンパイル時検出 | race detector | プロセス分離 | synchronized/volatile | 手動管理 |
| GIL問題 | なし | なし | あり(CPUバウンド) | なし | なし |
| オーバーヘッド | 低い | 低い | 高い(プロセス生成) | 中程度 | 低い |
| 学習曲線 | 急(所有権) | 緩やか | 緩やか | 中程度 | 急(UB) |
| 適用領域 | システム/HPC | ネットワーク/サーバー | データサイエンス | エンタープライズ | システム/HPC |

### 8.2 並列パターンの使い分け

| パターン | 適用条件 | 典型的なユースケース | スケーラビリティ | 実装の複雑さ |
|---------|---------|-------------------|----------------|------------|
| データ並列 | 同じ処理を大量データに適用 | 画像処理、行列演算、バッチ変換 | 非常に高い | 低い |
| タスク並列 | 異なる独立した処理が存在 | ダッシュボード読込、マイクロサービス集約 | 中程度(タスク数依存) | 中程度 |
| MapReduce | 大規模データの集計・変換 | ログ解析、単語カウント、分散集計 | 非常に高い | 中程度 |
| Fork-Join | 再帰的に分割可能な問題 | マージソート、クイックソート、木の走査 | 高い | 中程度 |
| Pipeline | 多段処理のスループット向上 | ETLパイプライン、ストリーム処理、動画エンコード | 高い(ステージ数依存) | 高い |
| Producer-Consumer | 生成と消費の速度差を吸収 | ジョブキュー、イベント処理 | 中程度 | 中程度 |
| Work Stealing | 動的な負荷分散 | 不均一なタスクの並列処理 | 非常に高い | 高い |

### 8.3 並列化判断フローチャート

```
================================================================
  並列化の判断フロー
================================================================

  処理が遅い
    │
    ├─ アルゴリズムは最適か？ ─→ No ─→ まずアルゴリズムを改善
    │                                   （O(n^2) → O(n log n) など）
    v Yes
    │
    ├─ I/Oバウンドか？ ─→ Yes ─→ 非同期I/O (async/await) を検討
    │                            → 02-async-programming.md 参照
    v No (CPUバウンド)
    │
    ├─ 並列化可能率は十分か？ ─→ No (<50%) ─→ 逐次部分の最適化を優先
    │  (アムダールの法則)                       SIMD最適化を検討
    v Yes (>75%)
    │
    ├─ データは独立に分割可能か？
    │  │
    │  ├─ Yes ─→ データ並列
    │  │         ├─ Rust: rayon par_iter
    │  │         ├─ Java: parallel stream
    │  │         ├─ Python: multiprocessing.Pool
    │  │         └─ C++: std::execution::par
    │  │
    │  └─ No ─→ タスク間に依存関係がある
    │           │
    │           ├─ 再帰的に分割可能？ ─→ Fork-Join
    │           ├─ 多段処理？ ─→ Pipeline
    │           └─ 独立タスクの集合？ ─→ タスク並列
    │
    v
    性能を測定し、効率 E を確認
    │
    ├─ E > 0.7 ─→ 良好。運用開始
    ├─ 0.5 < E < 0.7 ─→ 偽共有やロック競合を調査
    └─ E < 0.5 ─→ 並列化のアプローチを再検討
================================================================
```

---

## 9. アンチパターン

### 9.1 アンチパターン1: 過度な並列化（並列化のやりすぎ）

```
================================================================
  ANTI-PATTERN: 小さすぎるタスクの並列化
================================================================

  問題:
  並列化にはオーバーヘッドがある（スレッド生成、タスク分配、
  同期、結果集約）。タスクが小さすぎると、オーバーヘッドが
  実際の計算時間を上回り、逆に遅くなる。
================================================================
```

```python
# ================================================================
# ANTI-PATTERN: 小さなタスクの過度な並列化
# ================================================================

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# BAD: 1要素ずつプロセスに分配
# プロセス生成コスト >> 計算コスト
def bad_parallel():
    with ProcessPoolExecutor(max_workers=4) as executor:
        # 各要素をプロセス間通信で送受信 → 激遅
        results = list(executor.map(
            lambda x: x * 2,  # 極めて軽い計算
            range(10000)
        ))
    return results

# GOOD: チャンク単位で分配
def good_parallel():
    def process_chunk(chunk):
        return [x * 2 for x in chunk]

    data = list(range(10000))
    n_workers = mp.cpu_count()
    chunk_size = len(data) // n_workers
    chunks = [data[i:i+chunk_size]
              for i in range(0, len(data), chunk_size)]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(process_chunk, chunks))
    return [x for chunk in results for x in chunk]

# BEST: そもそも並列化しない（この規模なら逐次が最速）
def best_sequential():
    return [x * 2 for x in range(10000)]
```

```
  並列化の経験則:
  ┌──────────────────────────────────────────────────────┐
  │ タスクの粒度    │ 推奨アプローチ                       │
  ├──────────────────────────────────────────────────────┤
  │ < 1us          │ 並列化しない（SIMD検討）              │
  │ 1us - 100us    │ データ並列（大きなチャンクで分割）     │
  │ 100us - 10ms   │ 並列化効果あり                        │
  │ > 10ms         │ 積極的に並列化                        │
  └──────────────────────────────────────────────────────┘
```

### 9.2 アンチパターン2: 共有可変状態への無防備なアクセス

```
================================================================
  ANTI-PATTERN: 共有変数への非同期アクセス
================================================================

  問題:
  複数のスレッドから共有変数にロックなしで読み書きすると、
  データ競合（Data Race）が発生し、結果が不定になる。
  さらに厄介なことに、テストでは再現しにくく、
  本番環境で稀にしか発生しないことがある。
================================================================
```

```go
// ================================================================
// ANTI-PATTERN: 共有可変状態への無防備なアクセス
// ================================================================

package main

import (
    "fmt"
    "sync"
    "sync/atomic"
)

// BAD: データ競合（go run -race で検出可能）
func badSharedState() {
    counter := 0
    var wg sync.WaitGroup
    for i := 0; i < 1000; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            counter++ // DATA RACE: 非原子的な read-modify-write
        }()
    }
    wg.Wait()
    // counter は 1000 にならない可能性がある
    fmt.Println("BAD counter:", counter)
}

// GOOD: Mutex で保護
func goodWithMutex() {
    counter := 0
    var mu sync.Mutex
    var wg sync.WaitGroup
    for i := 0; i < 1000; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            mu.Lock()
            counter++
            mu.Unlock()
        }()
    }
    wg.Wait()
    fmt.Println("GOOD counter (mutex):", counter) // 常に 1000
}

// BEST: アトミック操作（カウンタには最適）
func bestWithAtomic() {
    var counter atomic.Int64
    var wg sync.WaitGroup
    for i := 0; i < 1000; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            counter.Add(1)
        }()
    }
    wg.Wait()
    fmt.Println("BEST counter (atomic):", counter.Load()) // 常に 1000
}

// ALSO GOOD: チャネルで集約（Go のイディオム）
func goodWithChannel() {
    results := make(chan int, 1000)
    var wg sync.WaitGroup
    for i := 0; i < 1000; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            results <- 1
        }()
    }

    go func() {
        wg.Wait()
        close(results)
    }()

    counter := 0
    for v := range results {
        counter += v
    }
    fmt.Println("GOOD counter (channel):", counter) // 常に 1000
}
```

### 9.3 アンチパターン3: 不均一な負荷分散

```
================================================================
  ANTI-PATTERN: 静的分割による負荷不均衡
================================================================

  問題:
  データを均等に分割しても、各チャンクの処理時間が異なると
  最も遅いワーカーが全体の完了時間を決定してしまう。

  例: 画像の領域分割（空白領域は高速、複雑な領域は低速）

  静的分割:
  Core 0: [簡単] ■■░░░░░░  完了 → 待機...
  Core 1: [普通] ■■■■░░░░  完了 → 待機...
  Core 2: [複雑] ■■■■■■■░  完了 → 待機
  Core 3: [超複雑] ■■■■■■■■■■■■  ← ボトルネック

  動的分割（ワークスティーリング）:
  Core 0: [簡単][追加1][追加2]  ■■■■■■■░
  Core 1: [普通][追加3]         ■■■■■■░░
  Core 2: [複雑]               ■■■■■■■░
  Core 3: [超複雑の前半]        ■■■■■■■░

  → 各コアがほぼ同時に完了
================================================================
```

---

## 10. 演習問題

### 10.1 初級: 並列集計

```
================================================================
  演習1（初級）: 並列による配列集計
================================================================

  課題:
  100万個の整数配列に対して、以下を並列で計算せよ:
  1. 合計値
  2. 最大値
  3. 最小値
  4. 平均値

  要件:
  - 好きな言語を1つ選択する
  - 逐次版と並列版の両方を実装する
  - 並列版はデータ並列パターンを使用する

  ヒント:
  - Rust: rayon の par_iter() + reduce()
  - Python: multiprocessing.Pool.map() でチャンク分割
  - Go: goroutine + channel で部分結果を集約
  - Java: parallelStream() + Collectors

  検証ポイント:
  - 逐次版と並列版で同じ結果が得られることを確認
  - コア数を変えて速度を比較し、効率 E を算出
================================================================
```

### 10.2 中級: パイプライン処理

```
================================================================
  演習2（中級）: パイプラインによるログ解析
================================================================

  課題:
  大量のログファイルを以下のパイプラインで処理せよ:

  Stage 1: ファイル読み込み（行単位で出力）
  Stage 2: パース（タイムスタンプ、レベル、メッセージに分割）
  Stage 3: フィルタ（ERROR レベルのみ抽出）
  Stage 4: 集計（エラーメッセージ別の件数を出力）

  要件:
  - 各ステージを独立したgoroutine/スレッドとして実装
  - ステージ間はチャネル/キューで接続
  - バックプレッシャー機構を実装（バッファ付きチャネル）
  - キャンセル機構を実装（Context/CancellationToken）

  発展:
  - Stage 2 を Fan-Out で並列化（複数ワーカー）
  - 処理件数のリアルタイム表示を追加
  - タイムアウト付きの graceful shutdown を実装

  評価基準:
  - 各ステージが独立して動作すること
  - メモリ使用量が一定に保たれること（ストリーム処理）
  - エラーハンドリングが適切であること
================================================================
```

### 10.3 上級: 並列マージソートの最適化

```
================================================================
  演習3（上級）: 適応的な並列マージソート
================================================================

  課題:
  Fork-Join パターンを用いた並列マージソートを実装し、
  以下の最適化を施せ:

  基本実装:
  1. 配列を再帰的に半分に分割
  2. サブ配列が閾値以下なら逐次ソート（insertion sort）
  3. 分割されたタスクを並列実行（Fork）
  4. ソート済みサブ配列をマージ（Join）

  最適化要件:
  A) 閾値の自動チューニング
     - コア数とデータサイズから最適な閾値を決定
     - 閾値が小さすぎる → タスク生成オーバーヘッド
     - 閾値が大きすぎる → 並列度が不足

  B) キャッシュ最適化
     - マージ時のバッファ再利用（メモリアロケーション削減）
     - キャッシュラインを意識したデータアクセス

  C) 負荷分散
     - 再帰の深さに応じて並列/逐次を切り替え
     - ワークスティーリングの活用

  評価基準:
  - 1000万要素で、逐次版の 3倍以上の高速化（4コア以上）
  - メモリ使用量が O(n) 以下
  - 全てのコア数 (1,2,4,8) で正しい結果を出力
  - 並列化効率 E > 0.6 を達成
================================================================
```

---

## 11. 並列デバッグとプロファイリング

### 11.1 データ競合の検出

```
================================================================
  データ競合の検出ツール
================================================================

  Go: Race Detector
  ──────────────────
  $ go run -race main.go
  $ go test -race ./...

  → 実行時にデータ競合を検出し、詳細なスタックトレースを表示
  → CI/CD に組み込むのが標準的なプラクティス

  Rust: コンパイル時検出
  ──────────────────────
  - 所有権システムがデータ競合をコンパイル時に防止
  - Send/Sync トレイトで安全性を型レベルで保証
  - unsafe を使わない限り、データ競合は原理的に発生しない

  C/C++: ThreadSanitizer (TSan)
  ─────────────────────────────
  $ clang++ -fsanitize=thread -g program.cpp -o program
  $ ./program

  → LLVM ベースのデータ競合検出器
  → 実行時オーバーヘッドは約 5-15x

  Java: jcmd / JFR
  ─────────────────
  $ jcmd <pid> Thread.print    # スレッドダンプ
  $ jcmd <pid> JFR.start       # Flight Recorder 開始

  Python: 並列処理のデバッグ
  ─────────────────────────
  - multiprocessing はプロセス分離のため、データ競合は発生しにくい
  - threading では GIL があるが、I/O操作中は競合の可能性あり
  - cProfile / py-spy でプロファイリング
================================================================
```

### 11.2 性能分析の手法

```
================================================================
  並列処理の性能分析チェックリスト
================================================================

  1. 並列化前のベースライン測定
     □ 逐次実行時間を計測
     □ CPU使用率を確認（1コアのみ100%か）
     □ メモリ使用量を確認

  2. 並列化後の測定
     □ 高速化率 = 逐次時間 / 並列時間
     □ 効率 = 高速化率 / コア数
     □ CPU使用率（全コアが使われているか）

  3. ボトルネックの特定
     □ ロック競合（mutex contention）
     □ 偽共有（perf stat で L1 cache miss を確認）
     □ メモリ帯域の飽和
     □ タスクの不均衡（一部コアが遊んでいないか）
     □ GCによる一時停止（Java, Go）

  4. スケーリング測定
     □ コア数 1, 2, 4, 8, 16 での性能をプロット
     □ アムダールの法則の予測値と比較
     □ 効率が急落するポイントを特定

  推奨ツール:
  ┌───────────────┬─────────────────────────────────┐
  │ Linux         │ perf, htop, flamegraph          │
  │ macOS         │ Instruments, Activity Monitor   │
  │ Rust          │ criterion (ベンチマーク)          │
  │ Go            │ pprof, trace                    │
  │ Java          │ JFR, async-profiler             │
  │ Python        │ cProfile, py-spy                │
  │ C++           │ Valgrind (Helgrind), VTune      │
  └───────────────┴─────────────────────────────────┘
================================================================
```

---

## 12. 実世界の適用事例

### 12.1 画像処理の並列化

```rust
// ================================================================
// Rust: 画像フィルタの並列適用
// ================================================================

use rayon::prelude::*;

struct Image {
    width: usize,
    height: usize,
    pixels: Vec<u8>,  // RGBA: 4 bytes per pixel
}

impl Image {
    /// ガウシアンブラーを並列適用
    fn parallel_blur(&self, radius: usize) -> Image {
        let mut output = vec![0u8; self.pixels.len()];
        let kernel = create_gaussian_kernel(radius);

        // 行ごとに並列処理（データ並列）
        output
            .par_chunks_mut(self.width * 4)
            .enumerate()
            .for_each(|(y, row)| {
                for x in 0..self.width {
                    let (r, g, b, a) = apply_kernel(
                        &self.pixels,
                        self.width,
                        self.height,
                        x, y,
                        &kernel,
                        radius,
                    );
                    let idx = x * 4;
                    row[idx] = r;
                    row[idx + 1] = g;
                    row[idx + 2] = b;
                    row[idx + 3] = a;
                }
            });

        Image {
            width: self.width,
            height: self.height,
            pixels: output,
        }
    }

    /// 複数のフィルタをパイプラインで適用
    fn apply_filters(&self) -> Image {
        // 各フィルタは前のフィルタの結果に依存するため
        // フィルタ内をデータ並列化する
        let blurred = self.parallel_blur(3);
        let sharpened = blurred.parallel_sharpen();
        let adjusted = sharpened.parallel_brightness(1.2);
        adjusted
    }
}
```

### 12.2 Web サーバーでのバッチ処理

```go
// ================================================================
// Go: API レスポンスの並列集約
// ================================================================

package main

import (
    "context"
    "fmt"
    "sync"
    "time"
)

type Product struct {
    ID          int
    Name        string
    Price       float64
    Reviews     []Review
    Inventory   int
    Recommended []Product
}

// 並列度制限付きバッチ処理
func enrichProducts(ctx context.Context,
                    products []Product) ([]Product, error) {
    const maxConcurrency = 20
    sem := make(chan struct{}, maxConcurrency)

    var (
        mu       sync.Mutex
        enriched = make([]Product, len(products))
        errs     []error
    )

    var wg sync.WaitGroup
    for i, p := range products {
        i, p := i, p
        wg.Add(1)

        sem <- struct{}{} // セマフォ獲得

        go func() {
            defer wg.Done()
            defer func() { <-sem }() // セマフォ解放

            // 各商品に対して3つの外部呼び出しを並列実行
            result, err := enrichSingleProduct(ctx, p)

            mu.Lock()
            if err != nil {
                errs = append(errs, err)
            } else {
                enriched[i] = result
            }
            mu.Unlock()
        }()
    }

    wg.Wait()

    if len(errs) > 0 {
        return nil, fmt.Errorf("%d errors occurred", len(errs))
    }
    return enriched, nil
}

func enrichSingleProduct(ctx context.Context,
                         p Product) (Product, error) {
    ctx, cancel := context.WithTimeout(ctx, 3*time.Second)
    defer cancel()

    type reviewResult struct {
        reviews []Review
        err     error
    }
    type inventoryResult struct {
        count int
        err   error
    }
    type recommendResult struct {
        products []Product
        err      error
    }

    reviewCh := make(chan reviewResult, 1)
    inventoryCh := make(chan inventoryResult, 1)
    recommendCh := make(chan recommendResult, 1)

    go func() {
        reviews, err := fetchReviews(ctx, p.ID)
        reviewCh <- reviewResult{reviews, err}
    }()
    go func() {
        count, err := checkInventory(ctx, p.ID)
        inventoryCh <- inventoryResult{count, err}
    }()
    go func() {
        recs, err := getRecommendations(ctx, p.ID)
        recommendCh <- recommendResult{recs, err}
    }()

    rr := <-reviewCh
    ir := <-inventoryCh
    rcr := <-recommendCh

    if rr.err != nil {
        return p, rr.err
    }

    p.Reviews = rr.reviews
    p.Inventory = ir.count
    p.Recommended = rcr.products
    return p, nil
}
```

---

## 13. FAQ（よくある質問）

### Q1: 並列化はいつ検討すべきか?

**A:** 以下の条件を全て満たす場合に並列化を検討する。

1. **処理がCPUバウンド**: CPU使用率が高く、I/O待ちが少ない。I/Oバウンドなら非同期I/O（async/await）のほうが効果的である。
2. **処理時間が十分長い**: 1回の処理が数百ミリ秒以上かかる。短い処理は並列化のオーバーヘッドで逆に遅くなる。
3. **データまたはタスクが分割可能**: 処理を独立した部分に分けられる。依存関係が強い処理は並列化が困難である。
4. **逐次最適化が済んでいる**: アルゴリズム改善やメモリ最適化など、シングルスレッドでの最適化を先に行うべきである。

判断の目安として、アムダールの法則で並列化可能率を見積もり、75%以上あれば並列化の効果が期待できる。

### Q2: スレッドプールのサイズはどう決めるべきか?

**A:** タスクの特性によって異なる。

- **CPUバウンドタスク**: `スレッド数 = 物理コア数`。ハイパースレッディングの効果はワークロードに依存する。論理コア数にしても改善は限定的な場合が多い。
- **I/Oバウンドタスク**: `スレッド数 = コア数 * (1 + 待ち時間/処理時間)`。I/O待ちが長ければ多くのスレッドを使える。
- **混在タスク**: CPUバウンド用とI/Oバウンド用で別々のスレッドプールを用意するのが推奨される。

Java の `ForkJoinPool` や Rust の `rayon` はデフォルトで論理コア数のスレッドを作成するが、これは多くの場合において適切な値である。

### Q3: Pythonで本当に並列処理は可能か?

**A:** GIL（Global Interpreter Lock）の制約があるため、`threading` モジュールではCPUバウンドな真の並列実行はできない。しかし以下の方法で並列処理は実現可能である。

1. **`multiprocessing`**: プロセスを分離することでGILを回避。各プロセスが独自のPythonインタープリタとメモリ空間を持つ。プロセス間通信のオーバーヘッドがある。
2. **`concurrent.futures.ProcessPoolExecutor`**: `multiprocessing` の高水準ラッパー。使いやすいAPIを提供する。
3. **NumPy/SciPy**: 内部はC/Fortranで実装されており、BLAS/LAPACK経由で自動的にマルチスレッド並列化される。GILはネイティブコード実行中に解放される。
4. **Cython / C拡張**: GILを明示的に解放してネイティブスレッドで並列実行可能。
5. **Python 3.13+ (Free-threaded CPython)**: GILを無効化するビルドオプションが実験的に導入されている。将来的にはスレッドベースの並列処理も可能になる見込みである。

### Q4: ロックフリーデータ構造は常にロックベースより高速か?

**A:** 必ずしもそうではない。ロックフリーの利点は「デッドロックの回避」と「高競合時のスケーラビリティ」であるが、以下の場合にはロックベースのほうが優れることがある。

- **競合が少ない場合**: ロックの獲得/解放コストは非常に低い（数十ナノ秒）。競合が稀ならロックベースが単純で高速である。
- **複雑な操作が必要な場合**: ロックフリーで実現可能な操作は限定的。複数の値を一貫して更新する場合はロックが必要である。
- **CAS（Compare-And-Swap）のリトライが多い場合**: 高競合時にCASの失敗とリトライが頻発し、CPUサイクルを浪費する。

一般的な指針として、単純なカウンタやフラグにはアトミック操作を使い、複雑なデータ構造にはロック（`Mutex` / `RwLock`）を使うのが推奨される。

### Q5: GPUの並列処理との違いは何か?

**A:** CPUとGPUは並列処理のアーキテクチャが根本的に異なる。

| 特性 | CPU並列 | GPU並列 |
|------|--------|--------|
| コア数 | 4-128コア | 数千コア |
| コアの能力 | 高性能（複雑な処理が可能） | 単純（同じ処理を大量に） |
| メモリ | 大容量・低遅延 | 高帯域・高遅延 |
| 分岐処理 | 高効率 | 低効率（ワープ分岐） |
| 適用領域 | 汎用 | 行列演算、画像処理、ML |
| プログラミング | 標準言語 | CUDA, OpenCL, Metal |

GPUは「同じ簡単な処理を膨大なデータに適用する」場合（SIMT: Single Instruction, Multiple Threads）に最適であり、本章で扱うCPU並列は「複雑で多様な処理を少数のコアで高速に実行する」場合に最適である。

---


## FAQ

### Q1: このトピックを学ぶ上で最も重要なポイントは何ですか？

実践的な経験を積むことが最も重要です。理論だけでなく、実際にコードを書いて動作を確認することで理解が深まります。

### Q2: 初心者がよく陥る間違いは何ですか？

基礎を飛ばして応用に進むことです。このガイドで説明している基本概念をしっかり理解してから、次のステップに進むことをお勧めします。

### Q3: 実務ではどのように活用されていますか？

このトピックの知識は、日常的な開発業務で頻繁に活用されます。特にコードレビューやアーキテクチャ設計の際に重要になります。

---

## 14. まとめ

### 14.1 章全体の要点

| パターン | 用途 | 代表技術 | スケーラビリティ |
|---------|------|---------|----------------|
| データ並列 | 同じ処理を大量データに | Rayon, NumPy, CUDA, parallel stream | 非常に高い |
| タスク並列 | 異なる処理を同時に | goroutine, tokio::join!, std::async | タスク数に依存 |
| MapReduce | 分散データ処理 | Hadoop, Spark, rayon fold+reduce | 非常に高い |
| Fork-Join | 再帰的分割統治 | ForkJoinPool, rayon::join | 高い |
| Pipeline | 多段処理のスループット向上 | Go channel, Unix pipe | ステージ数に依存 |
| アトミック | ロックフリーカウンタ/フラグ | atomic, sync/atomic | 非常に高い |

### 14.2 設計判断の指針

```
================================================================
  並列プログラミングの原則
================================================================

  1. 計測してから最適化する
     - 推測で並列化しない
     - プロファイラでボトルネックを特定してから着手

  2. 最も単純な方法を選ぶ
     - まず逐次アルゴリズムを最適化
     - 次にデータ並列（par_iter / parallel stream）
     - ロックフリーは最後の手段

  3. 共有状態を最小化する
     - メッセージパッシングを優先
     - 不変データを活用
     - 所有権を明確にする

  4. 粒度を適切に設定する
     - タスクが小さすぎ → オーバーヘッドで遅くなる
     - タスクが大きすぎ → 負荷が偏る
     - ワークスティーリングで動的に調整

  5. 正しさを最優先する
     - 高速だが不正確なプログラムに価値はない
     - データ競合検出ツールをCIに組み込む
     - テストでは -race フラグを常に有効にする
================================================================
```

---

## 15. 次に読むべきガイド


---

## 16. 参考文献

1. Herlihy, M. & Shavit, N. "The Art of Multiprocessor Programming." 2nd Edition, Morgan Kaufmann, 2020. -- 並列プログラミングの理論と実践を包括的に解説する名著。ロックフリーデータ構造の理論的基盤を提供する。
2. McCool, M., Reinders, J. & Robison, A. "Structured Parallel Programming: Patterns for Efficient Computation." Morgan Kaufmann, 2012. -- MapReduce, Fork-Join, Pipeline などの並列パターンを体系的に整理した教科書。Intel TBBの設計者による解説。
3. Williams, A. "C++ Concurrency in Action." 2nd Edition, Manning, 2019. -- C++のメモリモデル、アトミック操作、並列アルゴリズムを詳細に解説。他言語にも応用可能な低レベルの知識を提供する。
4. "Rayon: data-parallelism library for Rust." github.com/rayon-rs/rayon -- Rustのデータ並列ライブラリ。ワークスティーリングスケジューラの参考実装としても価値がある。
5. Amdahl, G. M. "Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities." AFIPS Conference Proceedings, 1967. -- アムダールの法則の原論文。並列化の限界を数学的に定式化した歴史的文献。
6. Gustafson, J. L. "Reevaluating Amdahl's Law." Communications of the ACM, 1988. -- グスタフソンの法則を提唱した論文。アムダールの法則の前提（固定問題サイズ）を再検討し、スケールアップの観点を導入した。
7. Pike, R. "Concurrency is not Parallelism." Talk at Waza Conference, 2012. -- Goの設計者による、並行と並列の違いを明確に説明した講演。概念の理解に最適な入門資料。

---

## 次に読むべきガイド

- 同カテゴリの他のガイドを参照してください

---

## 参考文献

- [MDN Web Docs](https://developer.mozilla.org/) - Web技術のリファレンス
- [Wikipedia](https://ja.wikipedia.org/) - 技術概念の概要
