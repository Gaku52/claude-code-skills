# Parallel Programming

> Parallel programming is a technique that "leverages multiple CPU cores to accelerate processing." This guide systematically covers parallelization methods from the hardware level to the application level, centered around two approaches: data parallelism and task parallelism.

## What You Will Learn in This Chapter

- [ ] Understand the difference between data parallelism and task parallelism, and use each appropriately
- [ ] Quantitatively predict the effect of parallelization using Amdahl's Law and Gustafson's Law
- [ ] Apply parallel processing features of each language (Rust, Go, Python, Java, C++) in practice
- [ ] Design and implement parallel patterns such as MapReduce, Fork-Join, and Pipeline
- [ ] Understand the principles and applicability of lock-free and atomic operations
- [ ] Avoid bugs specific to parallel programming (data races, false sharing, etc.)


## Prerequisites

Before reading this guide, having the following knowledge will deepen your understanding:

- Basic programming knowledge
- Understanding of related foundational concepts
- Familiarity with the content of [Message Passing](./02-message-passing.md)

---

## 1. Fundamentals of Parallel Processing

### 1.1 Concurrency vs. Parallelism (Review)

Concurrency and Parallelism are frequently confused, but they are fundamentally different concepts.

```
============================================================
  Concurrency                 Parallelism
============================================================

  Logically progressing        Physically executing
  at the same time             at the same time

  Achievable with a single     Requires multiple
  CPU core                     CPU cores

  +--------------------+      +--------------------+
  | Core 0             |      | Core 0  | Core 1   |
  |                    |      |         |          |
  | +-A-+ +-B-+       |      | +-A--+  | +-B--+   |
  | |   | |   |       |      | |    |  | |    |   |
  | +---+ +---+       |      | +----+  | +----+   |
  | +-A-+ +-B-+       |      |         |          |
  | |   | |   |       |      |         |          |
  | +---+ +---+       |      |         |          |
  |                    |      |         |          |
  | Time-sliced        |      | Simultaneous       |
  | alternating exec.  |      | execution          |
  +--------------------+      +--------------------+

  Goal: Structural          Goal: Speed
        organization              improvement
  Ex: Web server design     Ex: Image processing
                                acceleration
============================================================
```

This chapter focuses on "parallelism." We will learn how to divide and distribute computation across multiple physical cores to achieve speedup.

### 1.2 Levels of Parallelism

Parallel processing is realized at multiple levels, from hardware to software.

```
================================================================
  Hierarchical Structure of Parallel Processing
================================================================

  [Bit-Level Parallelism]
    |  64-bit operations = process twice the width of 32-bit simultaneously
    |  Ex: Execute 64-bit integer addition in a single instruction on a 64-bit CPU
    v
  [Instruction-Level Parallelism (ILP)]
    |  Execute multiple instructions simultaneously within the CPU
    |  Pipeline processing, superscalar execution
    |  Rarely controlled directly by programmers
    v
  [SIMD (Single Instruction, Multiple Data)]
    |  Process multiple data elements with a single instruction
    |  Ex: Simultaneously add four floats using SSE/AVX
    |  Compiler auto-vectorization or intrinsic functions
    v
  [Thread-Level Parallelism]
    |  Execute different tasks simultaneously across multiple threads
    |  The OS assigns threads to individual cores
    |  Main topic of this chapter
    v
  [Process-Level Parallelism]
    |  Execute in parallel across multiple processes
    |  Memory space is isolated = safe but high communication cost
    |  E.g., Python's multiprocessing
    v
  [Distributed Parallelism]
       Parallel processing across multiple machines
       Requires communication over the network
       MapReduce, Apache Spark, etc.
================================================================
```

### 1.3 Understanding Hardware Parallelism

Understanding modern CPU architecture is essential for writing efficient parallel programs.

```
================================================================
  Modern Multicore CPU (e.g., 8 cores / 16 threads)
================================================================

  +-------------------------------------------------------+
  |                    CPU Package                        |
  |                                                       |
  |  +---------+  +---------+  +---------+  +---------+  |
  |  | Core 0  |  | Core 1  |  | Core 2  |  | Core 3  |  |
  |  | L1 32KB |  | L1 32KB |  | L1 32KB |  | L1 32KB |  |
  |  | L2 256KB|  | L2 256KB|  | L2 256KB|  | L2 256KB|  |
  |  +----+----+  +----+----+  +----+----+  +----+----+  |
  |       +------+-----+            +------+-----+       |
  |         L3 Cache (shared)         L3 Cache (shared)   |
  |              +---------+----------+                   |
  |                   Memory Controller                   |
  +------------------------+------------------------------+
                           |
                      Main Memory (DRAM)

  Key Points:
  - L1/L2 are private to each core -> cache miss when accessing another core's data
  - L3 is shared across multiple cores -> affects shared data access
  - Cache lines are typically 64 bytes -> cause of False Sharing
================================================================
```

---

## 2. Amdahl's Law and Gustafson's Law

### 2.1 Amdahl's Law

A law that shows the theoretical upper limit of speedup through parallelization.

```
================================================================
  Amdahl's Law
================================================================

  If the parallelizable portion is P (0 <= P <= 1) and the number of cores is N:

                    1
  Speedup(N) = -------------
                (1-P) + P/N

  +-----------------------------------------------------+
  |  Example: 90% of the program is parallelizable       |
  |           (P = 0.9)                                   |
  |                                                       |
  |  N=1:   1 / (0.1 + 0.9/1)  = 1.00x                  |
  |  N=2:   1 / (0.1 + 0.9/2)  = 1.82x                  |
  |  N=4:   1 / (0.1 + 0.9/4)  = 3.08x                  |
  |  N=8:   1 / (0.1 + 0.9/8)  = 4.71x                  |
  |  N=16:  1 / (0.1 + 0.9/16) = 5.93x                  |
  |  N=64:  1 / (0.1 + 0.9/64) = 8.77x                  |
  |  N=inf: 1 / (0.1 + 0)      = 10.00x (theoretical    |
  |                                        upper limit)   |
  +-----------------------------------------------------+

  Speedup
  ^
  20x|                              ............ P=95%
     |                      .......
  15x|                  ....
     |              ....
  10x|         .............................---- P=90%
     |     ....
   5x|  ...        .................------------ P=75%
     | ..    ......
   2x|..  ...  .........----------------------- P=50%
   1x|-----------------------------------------
     +--+--+--+--+--+--+--+--+------------> Cores
        1  2  4  8  16 32 64 128

  Parallelizable % | 2 cores | 4 cores | 8 cores | 16 cores | Inf cores
  -----------------+---------+---------+---------+----------+----------
  50%              | 1.33x   | 1.60x   | 1.78x   | 1.88x    | 2.00x
  75%              | 1.60x   | 2.29x   | 3.00x   | 3.37x    | 4.00x
  90%              | 1.82x   | 3.08x   | 4.71x   | 5.93x    | 10.00x
  95%              | 1.90x   | 3.48x   | 5.93x   | 8.42x    | 20.00x
  99%              | 1.98x   | 3.88x   | 7.48x   | 13.91x   | 100.00x
================================================================
```

### 2.2 Gustafson's Law

Amdahl's Law assumes a "fixed problem size," but in practice, increasing the number of cores also allows handling larger problem sizes. Gustafson's Law provides an evaluation from this perspective.

```
================================================================
  Gustafson's Law
================================================================

  Speedup after scaling up:

  Speedup(N) = N - s * (N - 1)

  Where s is the ratio of the sequential portion, and N is the number of cores

  Amdahl's Law:
    Fixed problem size -> limits even with more cores
    "Strong Scaling"

  Gustafson's Law:
    Problem size scales proportionally with core count
    The parallel portion becomes dominant as cores increase
    "Weak Scaling"

  +-----------------------------------------------------+
  |                                                       |
  |  Example: s = 0.05 (5% is sequential)                |
  |                                                       |
  |  N=8:   8 - 0.05*(8-1)   =  7.65x                   |
  |  N=16:  16 - 0.05*(16-1) = 15.25x                   |
  |  N=64:  64 - 0.05*(64-1) = 60.85x                   |
  |  N=256: 256 - 0.05*255   = 243.25x                  |
  |                                                       |
  |  -> Scales nearly linearly when problem size grows    |
  +-----------------------------------------------------+
================================================================
```

### 2.3 Evaluating Scaling Efficiency

```
================================================================
  Definition of Parallelization Efficiency
================================================================

              Speedup(N)
  Efficiency E = --------------
                     N

  Ideal:       E = 1.0 (100%) -> perfect linear scaling
  Good:        E > 0.7 (70%) -> practical parallelization
  Needs work:  E < 0.5 (50%) -> overhead is dominant

  Causes of efficiency degradation:
  +----------------------------------------------+
  | 1. Sequential bottleneck: non-parallelizable  |
  |    portions                                    |
  | 2. Synchronization overhead: lock contention,  |
  |    barrier waits                                |
  | 3. Communication cost: data distribution and   |
  |    aggregation                                  |
  | 4. Load imbalance: workload skew across cores  |
  | 5. Memory bandwidth: cache misses, false       |
  |    sharing                                      |
  +----------------------------------------------+
================================================================
```

---

## 3. Data Parallelism

### 3.1 Concept and Principles

Data parallelism is a pattern that "applies the same operation to multiple data elements simultaneously." It is the most common parallelization technique in scientific computing, image processing, and machine learning.

```
================================================================
  Basic Concept of Data Parallelism
================================================================

  Input data: [a1, a2, a3, a4, a5, a6, a7, a8, ... aN]

  Partition:
  +----------+----------+----------+----------+
  | Chunk 0  | Chunk 1  | Chunk 2  | Chunk 3  |
  | a1,a2,...| a?,a?,...| a?,a?,...| ...,aN   |
  +----+-----+----+-----+----+-----+----+-----+
       |          |          |          |
       v          v          v          v
  +--------+ +--------+ +--------+ +--------+
  | Core 0 | | Core 1 | | Core 2 | | Core 3 |
  | f(x)   | | f(x)   | | f(x)   | | f(x)   |
  | Same op | | Same op | | Same op | | Same op |
  +----+---+ +----+---+ +----+---+ +----+---+
       |          |          |          |
       v          v          v          v
  +----------+----------+----------+----------+
  | Result 0 | Result 1 | Result 2 | Result 3 |
  +----------+----------+----------+----------+
       |          |          |          |
       +----------+------+---+----------+
                         v
                  Merge / Reduce
                  Final Result
================================================================
```

### 3.2 Rust: Data Parallelism with rayon

Rust's rayon crate provides data parallelism through an intuitive iterator-based API.

```rust
// ================================================================
// Rust: Data Parallelism with rayon
// ================================================================

use rayon::prelude::*;
use std::collections::HashMap;

// --- Basic: Converting sequential to parallel ---

// Sequential version
fn sequential_sum(data: &[f64]) -> f64 {
    data.iter()
        .map(|x| x.powi(2))
        .sum()
}

// Parallel version (just change iter to par_iter)
fn parallel_sum(data: &[f64]) -> f64 {
    data.par_iter()
        .map(|x| x.powi(2))
        .sum()
}

// --- Parallel filtering and aggregation ---

#[derive(Debug, Clone)]
struct LogEntry {
    level: String,
    message: String,
    timestamp: u64,
}

fn analyze_logs(logs: &[LogEntry]) -> HashMap<String, usize> {
    // Aggregate counts by log level in parallel
    logs.par_iter()
        .fold(
            || HashMap::new(),  // Thread-local map
            |mut acc, entry| {
                *acc.entry(entry.level.clone()).or_insert(0) += 1;
                acc
            },
        )
        .reduce(
            || HashMap::new(),  // Initial value
            |mut a, b| {        // Merge two maps
                for (key, value) in b {
                    *a.entry(key).or_insert(0) += value;
                }
                a
            },
        )
}

// --- Parallel sorting and chunk processing ---

fn parallel_sort_and_process(mut data: Vec<i64>) -> Vec<i64> {
    // Parallel sort (unstable sort: faster but order of equal elements is undefined)
    data.par_sort_unstable();

    // Parallel chunk processing
    data.par_chunks(1024)
        .flat_map(|chunk| {
            chunk.iter()
                .filter(|&&x| x > 0)
                .map(|&x| x * 2)
                .collect::<Vec<_>>()
        })
        .collect()
}

// --- Thread pool configuration ---

fn configure_thread_pool() {
    // Custom thread pool (default is logical core count)
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)           // Limit to 4 threads
        .stack_size(8 * 1024 * 1024)  // 8MB stack size
        .build_global()
        .unwrap();
}

// --- Execution example ---

fn main() {
    let data: Vec<f64> = (0..10_000_000)
        .map(|x| x as f64)
        .collect();

    let result = parallel_sum(&data);
    println!("Parallel sum: {}", result);

    // Chaining parallel iterators
    let processed: Vec<String> = (0..1_000_000_u64)
        .into_par_iter()
        .filter(|&x| x % 3 == 0)
        .map(|x| format!("item_{}", x))
        .collect();
    println!("Processed count: {}", processed.len());
}
```

### 3.3 Python: multiprocessing and concurrent.futures

Due to the GIL (Global Interpreter Lock) constraint in Python, multiprocessing is used for CPU-intensive parallel processing.

```python
# ================================================================
# Python: Data Parallelism with multiprocessing
# ================================================================

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from typing import List, Tuple
import time

# --- Basic: Pool.map ---

def square(x: float) -> float:
    """Example of a CPU-intensive computation"""
    return sum(i ** 2 for i in range(int(x)))

def basic_parallel():
    """Basic parallel processing"""
    data = list(range(100, 200))

    # Sequential execution
    results_seq = [square(x) for x in data]

    # Parallel execution
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results_par = pool.map(square, data)

    assert results_seq == results_par

# --- Chunk processing to reduce overhead ---

def process_chunk(chunk: List[float]) -> List[float]:
    """Process in chunk units (reduces inter-process communication)"""
    return [x ** 2 + 2 * x + 1 for x in chunk]

def chunked_parallel():
    """Efficient parallel processing with chunk splitting"""
    data = list(range(1_000_000))
    n_workers = mp.cpu_count()

    # Split data into equal chunks
    chunk_size = len(data) // n_workers
    chunks = [
        data[i:i + chunk_size]
        for i in range(0, len(data), chunk_size)
    ]

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(process_chunk, chunks)

    # Flatten results
    flat_results = [item for chunk in results for item in chunk]
    print(f"Processed count: {len(flat_results)}")

# --- ProcessPoolExecutor with progress tracking ---

def heavy_computation(args: Tuple[int, List[float]]) -> dict:
    """Receives an ID and data, returns result as a dict"""
    task_id, data = args
    result = sum(x ** 2 for x in data)
    return {"task_id": task_id, "result": result}

def parallel_with_progress():
    """Parallel processing with progress tracking"""
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
                print(f"Progress: {completed}/{total} "
                      f"(Task {task_id} completed)")
            except Exception as e:
                print(f"Task {task_id} failed: {e}")

    return sorted(results, key=lambda r: r["task_id"])

# --- Implicit parallelization with NumPy ---

def numpy_parallel():
    """NumPy vectorized operations (internally uses SIMD/multithreading)"""
    # NumPy is implicitly parallelized through BLAS/LAPACK
    a = np.random.rand(10000, 10000)
    b = np.random.rand(10000, 10000)

    # Matrix multiplication (internally executed in parallel)
    c = np.dot(a, b)

    # Element-wise operations (SIMD optimized)
    d = np.sin(a) + np.cos(b) * np.exp(-a)

    return c, d

if __name__ == "__main__":
    basic_parallel()
    chunked_parallel()
    results = parallel_with_progress()
    print(f"All tasks completed: {len(results)} items")
```

### 3.4 Java: Parallel Streams and ForkJoinPool

```java
// ================================================================
// Java: Parallel Streams and ForkJoinPool
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

    // --- Custom ForkJoinPool (parallelism control) ---
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

    // --- Direct use of ForkJoinTask ---
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
                // Sequential sort for small sizes
                Arrays.sort(array, lo, hi);
                return;
            }
            int mid = lo + (hi - lo) / 2;
            // Sort left and right halves in parallel
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
        // Parallel Stream example
        List<Long> numbers = LongStream.rangeClosed(1, 10_000_000)
            .boxed()
            .collect(Collectors.toList());
        long sum = parallelSum(numbers);
        System.out.println("Sum: " + sum);

        // Sorting with ForkJoinPool
        int[] data = new Random().ints(10_000_000).toArray();
        ForkJoinPool pool = ForkJoinPool.commonPool();
        pool.invoke(new ParallelMergeSort(data, 0, data.length));
        System.out.println("Sorted: " + Arrays.toString(
            Arrays.copyOf(data, 10)));
    }
}
```

---

## 4. Task Parallelism

### 4.1 Concept and Principles

Task parallelism is a pattern that "executes different operations simultaneously." Each task performs an independent operation, and results are consolidated at the end.

```
================================================================
  Basic Concept of Task Parallelism
================================================================

  Sequential execution:
  +------------+ +------------+ +------------+
  |  DB Query  |->|  API Call  |->| File Read  |
  |  200ms     | |  300ms     | |  100ms     |
  +------------+ +------------+ +------------+
  Total: 200 + 300 + 100 = 600ms

  Task parallel:
  +------------+
  |  DB Query  |------+
  |  200ms     |      |
  +------------+      |
  +----------------+  +--> Merge results
  |  API Call      |  |
  |  300ms         |--+
  +----------------+  |
  +----------+        |
  |File Read |--------+
  |  100ms   |
  +----------+
  Total: max(200, 300, 100) = 300ms (50% reduction)
================================================================
```

### 4.2 Go: goroutine + errgroup

```go
// ================================================================
// Go: Task Parallelism with goroutine + errgroup
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

// --- Basic: Parallel tasks with errgroup ---

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

    // Execute 4 tasks in parallel
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

    // Wait for all tasks to complete (cancel immediately if any error occurs)
    if err := g.Wait(); err != nil {
        return nil, fmt.Errorf("dashboard load failed: %w", err)
    }

    return &Dashboard{user, posts, stats, friends}, nil
}

// --- Task execution with concurrency limit ---

func processURLs(ctx context.Context, urls []string) ([]Result, error) {
    g, ctx := errgroup.WithContext(ctx)
    // Limit concurrent executions to 10
    g.SetLimit(10)

    results := make([]Result, len(urls))
    var mu sync.Mutex

    for i, url := range urls {
        i, url := i, url  // Capture loop variables
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

// --- Task parallelism with timeout ---

func loadWithTimeout(userID int) (*Dashboard, error) {
    ctx, cancel := context.WithTimeout(
        context.Background(),
        5*time.Second,
    )
    defer cancel()

    return loadDashboard(ctx, userID)
}
```

### 4.3 Rust: tokio::join! and rayon::join

```rust
// ================================================================
// Rust: Two Approaches to Task Parallelism
// ================================================================

// --- Async task parallelism (I/O-bound) ---
use tokio;

async fn load_dashboard(user_id: u64) -> Result<Dashboard, Error> {
    // Execute multiple async tasks in parallel with tokio::join!
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

// Early termination on error with try_join!
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

// --- CPU-bound task parallelism (rayon::join) ---
use rayon;

fn parallel_analysis(data: &[f64]) -> (f64, f64, Vec<f64>) {
    let (mean_result, (variance_result, sorted)) = rayon::join(
        // Left task: compute mean
        || {
            let sum: f64 = data.iter().sum();
            sum / data.len() as f64
        },
        // Right task: further split
        || rayon::join(
            // Compute variance
            || {
                let mean = data.iter().sum::<f64>() / data.len() as f64;
                data.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>() / data.len() as f64
            },
            // Sort
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

### 4.4 C++: std::async and Parallel Algorithms

```cpp
// ================================================================
// C++17/20: Parallel Algorithms and std::async
// ================================================================

#include <algorithm>
#include <execution>  // C++17 parallel execution policies
#include <future>
#include <numeric>
#include <vector>

// --- C++17 Parallel Algorithms ---

void parallel_algorithms_example() {
    std::vector<int> data(10'000'000);
    std::iota(data.begin(), data.end(), 0);

    // Parallel sort
    std::sort(std::execution::par,
              data.begin(), data.end());

    // Parallel transform
    std::transform(std::execution::par_unseq,
                   data.begin(), data.end(),
                   data.begin(),
                    { return x * 2; });

    // Parallel reduce
    long long sum = std::reduce(
        std::execution::par,
        data.begin(), data.end(),
        0LL);

    // Parallel for_each
    std::for_each(std::execution::par,
                  data.begin(), data.end(),
                   { x += 1; });
}

// --- Task parallelism with std::async ---

struct AnalysisResult {
    double mean;
    double variance;
    int max_val;
};

AnalysisResult parallel_analysis(const std::vector<int>& data) {
    // Execute 3 tasks asynchronously
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

// --- Comparison of Execution Policies ---
//
// std::execution::seq       Sequential execution (default)
// std::execution::par       Parallel execution (uses threads)
// std::execution::par_unseq Parallel + vectorized (uses SIMD)
// std::execution::unseq     Vectorization only (C++20)
```

---

## 5. Lock-Free and Atomic Operations

### 5.1 Fundamentals of Atomic Operations

Atomic operations are "indivisible operations" where no intermediate state is observable from other threads. They enable safe inter-thread communication without using locks.

```
================================================================
  Atomic Operations vs. Locks
================================================================

  Lock-based approach:
  Thread A: [acquire lock] -> [read] -> [modify] -> [write] -> [release lock]
  Thread B: [  wait...  ] -> [  wait...  ] -> [acquire lock] -> [read] -> ...

  Atomic approach:
  Thread A: [atomic read-modify-write]    <- completes in a single instruction
  Thread B: [atomic read-modify-write]    <- no waiting needed

  Comparison:
  +-------------+----------------+--------------------+
  | Property    | Lock           | Atomic             |
  +-------------+----------------+--------------------+
  | Overhead    | Large (kernel) | Small (CPU instr.) |
  | Scalability | Low (on        | High (short busy   |
  |             | contention)    | spin)              |
  | Operations  | Any operation  | Simple operations  |
  | supported   |                | only               |
  | Deadlock    | Possible       | None               |
  | ABA problem | None           | Possible           |
  | Impl.       | Moderate       | High               |
  | complexity  |                |                    |
  +-------------+----------------+--------------------+
================================================================
```

### 5.2 Memory Ordering

```
================================================================
  Memory Ordering (Memory Order Constraints)
================================================================

  CPUs reorder instruction execution for performance (reordering).
  In multithreading, this reordering can become problematic.

  Ordering levels (weak -> strong):

  +----------+----------------------------------------------+
  | Relaxed  | No ordering guarantees. Optimal for          |
  |          | counters. Best performance.                   |
  |          | Does not guarantee order relative to other    |
  |          | memory operations.                            |
  +----------+----------------------------------------------+
  | Acquire  | Guarantees that reads/writes after this       |
  |          | operation are not reordered before it.        |
  |          | Equivalent to lock acquisition.               |
  +----------+----------------------------------------------+
  | Release  | Guarantees that reads/writes before this      |
  |          | operation are not reordered after it.         |
  |          | Equivalent to lock release.                   |
  +----------+----------------------------------------------+
  | AcqRel   | Guarantees both Acquire and Release.          |
  |          | Used for read-modify-write operations.        |
  +----------+----------------------------------------------+
  | SeqCst   | All threads observe the same operation order. |
  |          | Most intuitive but slowest.                   |
  +----------+----------------------------------------------+

  Selection guidelines:
  - Simple counters        -> Relaxed
  - Flags (ready signal)   -> Release (writer) / Acquire (reader)
  - When in doubt          -> SeqCst (correctness first)
================================================================
```

### 5.3 Rust: Atomic Operations in Detail

```rust
// ================================================================
// Rust: Practical Patterns for Atomic Operations
// ================================================================

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

// --- Pattern 1: Atomic counter ---

struct AtomicCounter {
    count: AtomicU64,
}

impl AtomicCounter {
    fn new() -> Self {
        Self { count: AtomicU64::new(0) }
    }

    fn increment(&self) -> u64 {
        // fetch_add returns the value before addition
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
    // Always exactly 800,000
    assert_eq!(counter.get(), 800_000);
}

// --- Pattern 2: Spinlock ---

struct SpinLock {
    locked: AtomicBool,
}

impl SpinLock {
    fn new() -> Self {
        Self { locked: AtomicBool::new(false) }
    }

    fn lock(&self) {
        // Attempt to change from false to true with compare_exchange
        while self.locked.compare_exchange_weak(
            false,              // Expected value
            true,               // New value
            Ordering::Acquire,  // Ordering on success
            Ordering::Relaxed,  // Ordering on failure
        ).is_err() {
            // Spin wait (busy wait)
            std::hint::spin_loop();
        }
    }

    fn unlock(&self) {
        self.locked.store(false, Ordering::Release);
    }
}

// --- Pattern 3: One-time initialization (lazy init) ---

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
                    // Another thread initialized first
                    unsafe { drop(Box::from_raw(new)); }
                    p = existing;
                }
            }
        }
        unsafe { &*p }
    }
}
```

### 5.4 Go: sync/atomic Package

```go
// ================================================================
// Go: Practical Patterns with sync/atomic
// ================================================================

package main

import (
    "fmt"
    "sync"
    "sync/atomic"
)

// --- Atomic counter ---

type AtomicCounter struct {
    count atomic.Int64
}

func (c *AtomicCounter) Increment() int64 {
    return c.count.Add(1)
}

func (c *AtomicCounter) Get() int64 {
    return c.count.Load()
}

// --- Atomic value (dynamic configuration updates) ---

type Config struct {
    MaxWorkers int
    Timeout    int
    Debug      bool
}

type ConfigHolder struct {
    config atomic.Value // stores *Config
}

func (h *ConfigHolder) Load() *Config {
    return h.config.Load().(*Config)
}

func (h *ConfigHolder) Store(cfg *Config) {
    h.config.Store(cfg)
}

// Usage example: hot reload of configuration
func configExample() {
    holder := &ConfigHolder{}
    holder.Store(&Config{MaxWorkers: 4, Timeout: 30, Debug: false})

    // Workers always reference the latest configuration
    var wg sync.WaitGroup
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            cfg := holder.Load()
            fmt.Printf("Workers: %d\n", cfg.MaxWorkers)
        }()
    }

    // Dynamically update configuration (no lock needed)
    holder.Store(&Config{MaxWorkers: 8, Timeout: 60, Debug: true})
    wg.Wait()
}
```

---

## 6. Design Patterns for Parallel Processing

### 6.1 MapReduce Pattern

```
================================================================
  MapReduce Pattern
================================================================

  A general-purpose pattern that splits large data -> processes in parallel -> aggregates

  Input data:
  ["hello world", "hello rust", "world hello"]

  Phase 1 - Map (parallel):
  +-----------------+   +-----------------+   +-----------------+
  | Worker 0        |   | Worker 1        |   | Worker 2        |
  | "hello world"   |   | "hello rust"    |   | "world hello"   |
  | -> (hello,1)    |   | -> (hello,1)    |   | -> (world,1)    |
  |    (world,1)    |   |    (rust,1)     |   |    (hello,1)    |
  +--------+--------+   +--------+--------+   +--------+--------+
           |                     |                     |
           v                     v                     v

  Phase 2 - Shuffle (group by key):
  hello -> [(hello,1), (hello,1), (hello,1)]
  world -> [(world,1), (world,1)]
  rust  -> [(rust,1)]

  Phase 3 - Reduce (parallel):
  +-----------------+   +-----------------+   +----------------+
  | Reducer 0       |   | Reducer 1       |   | Reducer 2      |
  | hello: 1+1+1=3  |   | world: 1+1=2   |   | rust: 1=1      |
  +--------+--------+   +--------+--------+   +--------+-------+
           |                     |                     |
           v                     v                     v
  Final result: {hello: 3, world: 2, rust: 1}
================================================================
```

```rust
// ================================================================
// Rust: MapReduce Implementation
// ================================================================

use rayon::prelude::*;
use std::collections::HashMap;

fn word_count(documents: &[String]) -> HashMap<String, usize> {
    documents
        .par_iter()
        // Map: count words in each document
        .map(|doc| {
            let mut local_counts = HashMap::new();
            for word in doc.split_whitespace() {
                let word = word.to_lowercase();
                *local_counts.entry(word).or_insert(0) += 1;
            }
            local_counts
        })
        // Reduce: merge local counts
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

// Generic MapReduce framework
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

### 6.2 Fork-Join Pattern

```
================================================================
  Fork-Join Pattern
================================================================

  Recursively split tasks (Fork) and merge results (Join)

                       +-------------+
                       | Entire      |
                       | problem     |
                       | [1..1000]   |
                       +------+------+
                     Fork     |
                 +------------+------------+
                 v            v            v
          +----------+ +----------+ +----------+
          | [1..333] | |[334..666]| |[667..1000]|
          +----+-----+ +----+-----+ +----+-----+
               |            |            |
            Fork?         Fork?        Fork?
          +---+---+    (small enough)  +---+---+
          v       v     -> sequential  v       v
       [1..166][167..333]          [667..833][834..1000]
          |       |        |          |       |
          v       v        v          v       v
        Compute  Compute  Compute   Compute  Compute
          |       |        |          |       |
          +---+---+        |          +---+---+
              |            |              |
           Join          Direct         Join
              |            |              |
              +------------+--------------+
                           |
                         Join
                           |
                      Final Result
================================================================
```

### 6.3 Pipeline Pattern

```
================================================================
  Pipeline Pattern
================================================================

  Split processing into stages and execute each stage in parallel.
  Each stage is connected via channels.

  +----------+    +----------+    +----------+    +----------+
  | Stage 1  |--->| Stage 2  |--->| Stage 3  |--->| Stage 4  |
  | Read     | ch | Parse    | ch | Transform| ch | Write    |
  +----------+    +----------+    +----------+    +----------+

  Timeline:
  t1: [S1: item1] [         ] [         ] [         ]
  t2: [S1: item2] [S2: item1] [         ] [         ]
  t3: [S1: item3] [S2: item2] [S3: item1] [         ]
  t4: [S1: item4] [S2: item3] [S3: item2] [S4: item1]
  t5: [         ] [S2: item4] [S3: item3] [S4: item2]

  -> Each stage operates independently, improving throughput
================================================================
```

```go
// ================================================================
// Go: Pipeline Pattern Implementation
// ================================================================

package main

import (
    "context"
    "fmt"
    "strings"
    "sync"
)

// Define pipeline stages as functions

// Stage 1: Data generation
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

// Stage 2: Transformation (convert to uppercase)
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

// Stage 3: Filtering
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

// Fan-Out: Multiple workers read from a single channel
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

    // Build the pipeline
    stage1 := generate(ctx, "hello", "world", "go", "pipeline")
    stage2 := toUpper(ctx, stage1)
    stage3 := filterLong(ctx, stage2, 3)

    // Consume results
    for result := range stage3 {
        fmt.Println(result)
    }
    // Output: HELLO, WORLD, PIPELINE
}
```

---

## 7. False Sharing and Cache Optimization

### 7.1 The False Sharing Problem

```
================================================================
  False Sharing
================================================================

  Problem: Different cores are updating "different variables," but
       performance degrades because they reside on the same cache line.

  Memory layout (cache line = 64 bytes):
  +-----------------------------------------------------+
  |           One cache line (64 bytes)                  |
  |  +----------+  +----------+  +-------------------+  |
  |  | counter_a|  | counter_b|  |   padding         |  |
  |  | (8 bytes)|  | (8 bytes)|  |                   |  |
  |  +----------+  +----------+  +-------------------+  |
  +-----------------------------------------------------+
       Core 0 updates      Core 1 updates
       v                   v
  Core 0 updates counter_a
    -> Entire cache line is marked "modified"
    -> Core 1's cache is invalidated
    -> Core 1 must reload from memory to read counter_b
    -> Core 1 updates counter_b
    -> Core 0's cache is invalidated
    -> Ping-pong effect = significant performance degradation

  Solution: Use padding to place variables on separate cache lines
  +---------------------------------------+
  | Cache Line 0                          |
  | +----------+  +---------------------+ |
  | | counter_a|  |    padding (56B)    | |
  | +----------+  +---------------------+ |
  +---------------------------------------+
  +---------------------------------------+
  | Cache Line 1                          |
  | +----------+  +---------------------+ |
  | | counter_b|  |    padding (56B)    | |
  | +----------+  +---------------------+ |
  +---------------------------------------+
================================================================
```

### 7.2 Avoiding False Sharing

```rust
// ================================================================
// Rust: Avoiding False Sharing
// ================================================================

use std::sync::atomic::{AtomicU64, Ordering};

// BAD: False sharing may occur
struct BadCounters {
    counter_a: AtomicU64,  // May end up on the same cache line
    counter_b: AtomicU64,
}

// GOOD: Separated with padding
#[repr(C)]
struct GoodCounters {
    counter_a: AtomicU64,
    _pad_a: [u8; 56],     // 64 - 8 = 56 bytes of padding
    counter_b: AtomicU64,
    _pad_b: [u8; 56],
}

// crossbeam for Rust provides CachePadded
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
// Java: Avoiding False Sharing with @Contended
// ================================================================

import java.util.concurrent.atomic.AtomicLong;

// Java 8+: @Contended annotation
// Requires -XX:-RestrictContended JVM option at startup
public class PaddedCounters {
    // BAD: False sharing
    static class BadCounters {
        volatile long counterA;
        volatile long counterB;
    }

    // GOOD: Automatic padding with @Contended
    static class GoodCounters {
        @jdk.internal.vm.annotation.Contended
        volatile long counterA;

        @jdk.internal.vm.annotation.Contended
        volatile long counterB;
    }

    // GOOD: Manual padding (when @Contended is unavailable)
    static class ManualPaddedCounters {
        volatile long counterA;
        long p1, p2, p3, p4, p5, p6, p7;  // 56 bytes of padding
        volatile long counterB;
        long q1, q2, q3, q4, q5, q6, q7;
    }
}
```

---

## 8. Parallel Processing Comparison Tables

### 8.1 Language-by-Language Parallel Processing Feature Comparison

| Property | Rust (rayon) | Go (goroutine) | Python (multiprocessing) | Java (ForkJoinPool) | C++ (std::execution) |
|------|-------------|----------------|-------------------------|--------------------|--------------------|
| Unit of parallelism | Work stealing | goroutine + OS threads | OS processes | ForkJoinTask | OS threads |
| Data parallel API | `par_iter()` | Manual partitioning | `Pool.map()` | `parallelStream()` | `std::execution::par` |
| Task parallel API | `rayon::join` | `errgroup` | `ProcessPoolExecutor` | `CompletableFuture` | `std::async` |
| Memory model | Safety guaranteed by ownership | CSP model | Process isolation | Java Memory Model | C++ Memory Model |
| Data race prevention | Compile-time detection | Race detector | Process isolation | synchronized/volatile | Manual management |
| GIL issue | None | None | Yes (CPU-bound) | None | None |
| Overhead | Low | Low | High (process creation) | Moderate | Low |
| Learning curve | Steep (ownership) | Gentle | Gentle | Moderate | Steep (UB) |
| Application domain | Systems/HPC | Network/servers | Data science | Enterprise | Systems/HPC |

### 8.2 Choosing the Right Parallel Pattern

| Pattern | Applicability | Typical Use Cases | Scalability | Implementation Complexity |
|---------|---------|-------------------|----------------|------------|
| Data parallelism | Same operation applied to large data | Image processing, matrix operations, batch transforms | Very high | Low |
| Task parallelism | Different independent operations exist | Dashboard loading, microservice aggregation | Moderate (depends on task count) | Moderate |
| MapReduce | Aggregation/transformation of large data | Log analysis, word count, distributed aggregation | Very high | Moderate |
| Fork-Join | Recursively divisible problems | Merge sort, quicksort, tree traversal | High | Moderate |
| Pipeline | Throughput improvement for multi-stage processing | ETL pipelines, stream processing, video encoding | High (depends on stage count) | High |
| Producer-Consumer | Absorb speed differences between production and consumption | Job queues, event processing | Moderate | Moderate |
| Work Stealing | Dynamic load balancing | Parallel processing of non-uniform tasks | Very high | High |

### 8.3 Parallelization Decision Flowchart

```
================================================================
  Parallelization Decision Flow
================================================================

  Processing is slow
    |
    +- Is the algorithm optimal? -> No -> Improve the algorithm first
    |                                     (O(n^2) -> O(n log n), etc.)
    v Yes
    |
    +- Is it I/O-bound? -> Yes -> Consider async I/O (async/await)
    |                             -> See 02-async-programming.md
    v No (CPU-bound)
    |
    +- Is the parallelizable fraction sufficient? -> No (<50%)
    |  (Amdahl's Law)                                -> Prioritize
    |                                                    sequential
    v Yes (>75%)                                         optimization;
    |                                                    consider SIMD
    +- Can the data be independently partitioned?
    |  |
    |  +- Yes -> Data parallelism
    |  |         +- Rust: rayon par_iter
    |  |         +- Java: parallel stream
    |  |         +- Python: multiprocessing.Pool
    |  |         +- C++: std::execution::par
    |  |
    |  +- No -> Dependencies exist between tasks
    |           |
    |           +- Recursively divisible? -> Fork-Join
    |           +- Multi-stage processing? -> Pipeline
    |           +- Collection of independent tasks? -> Task parallelism
    |
    v
    Measure performance and check efficiency E
    |
    +- E > 0.7 -> Good. Deploy to production
    +- 0.5 < E < 0.7 -> Investigate false sharing and lock contention
    +- E < 0.5 -> Reconsider the parallelization approach
================================================================
```

---

## 9. Anti-Patterns

### 9.1 Anti-Pattern 1: Over-Parallelization (Excessive Parallelization)

```
================================================================
  ANTI-PATTERN: Parallelizing Tasks That Are Too Small
================================================================

  Problem:
  Parallelization has overhead (thread creation, task distribution,
  synchronization, result aggregation). If tasks are too small,
  the overhead exceeds the actual computation time, making it
  slower instead.
================================================================
```

```python
# ================================================================
# ANTI-PATTERN: Excessive parallelization of small tasks
# ================================================================

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# BAD: Distributing one element at a time to processes
# Process creation cost >> computation cost
def bad_parallel():
    with ProcessPoolExecutor(max_workers=4) as executor:
        # Each element is sent/received via IPC -> extremely slow
        results = list(executor.map(
            lambda x: x * 2,  # Extremely light computation
            range(10000)
        ))
    return results

# GOOD: Distribute in chunks
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

# BEST: Don't parallelize at all (sequential is fastest at this scale)
def best_sequential():
    return [x * 2 for x in range(10000)]
```

```
  Rules of thumb for parallelization:
  +------------------------------------------------------+
  | Task granularity | Recommended approach               |
  +------------------+------------------------------------+
  | < 1us            | Don't parallelize (consider SIMD)  |
  | 1us - 100us      | Data parallelism (large chunks)    |
  | 100us - 10ms     | Parallelization is effective       |
  | > 10ms           | Aggressively parallelize           |
  +------------------------------------------------------+
```

### 9.2 Anti-Pattern 2: Unprotected Access to Shared Mutable State

```
================================================================
  ANTI-PATTERN: Unsynchronized Access to Shared Variables
================================================================

  Problem:
  Reading from and writing to shared variables from multiple threads
  without locks causes data races, producing undefined results.
  Worse still, they are difficult to reproduce in tests and may
  occur only rarely in production environments.
================================================================
```

```go
// ================================================================
// ANTI-PATTERN: Unprotected Access to Shared Mutable State
// ================================================================

package main

import (
    "fmt"
    "sync"
    "sync/atomic"
)

// BAD: Data race (detectable with go run -race)
func badSharedState() {
    counter := 0
    var wg sync.WaitGroup
    for i := 0; i < 1000; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            counter++ // DATA RACE: non-atomic read-modify-write
        }()
    }
    wg.Wait()
    // counter may not be 1000
    fmt.Println("BAD counter:", counter)
}

// GOOD: Protected with Mutex
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
    fmt.Println("GOOD counter (mutex):", counter) // Always 1000
}

// BEST: Atomic operations (optimal for counters)
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
    fmt.Println("BEST counter (atomic):", counter.Load()) // Always 1000
}

// ALSO GOOD: Aggregate via channels (Go idiom)
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
    fmt.Println("GOOD counter (channel):", counter) // Always 1000
}
```

### 9.3 Anti-Pattern 3: Uneven Load Balancing

```
================================================================
  ANTI-PATTERN: Load Imbalance from Static Partitioning
================================================================

  Problem:
  Even if data is evenly partitioned, if the processing time for
  each chunk differs, the slowest worker determines the overall
  completion time.

  Example: Image region partitioning (blank regions are fast,
           complex regions are slow)

  Static partitioning:
  Core 0: [Easy]     ##......  Done -> Waiting...
  Core 1: [Normal]   ####....  Done -> Waiting...
  Core 2: [Complex]  #######.  Done -> Waiting
  Core 3: [Very complex] ############  <- Bottleneck

  Dynamic partitioning (work stealing):
  Core 0: [Easy][Extra1][Extra2]  #######.
  Core 1: [Normal][Extra3]        ######..
  Core 2: [Complex]               #######.
  Core 3: [First half of complex] #######.

  -> All cores finish at approximately the same time
================================================================
```

---

## 10. Exercises

### 10.1 Beginner: Parallel Aggregation

```
================================================================
  Exercise 1 (Beginner): Parallel Array Aggregation
================================================================

  Task:
  For an array of 1 million integers, compute the following
  in parallel:
  1. Sum
  2. Maximum value
  3. Minimum value
  4. Average value

  Requirements:
  - Choose any one language
  - Implement both sequential and parallel versions
  - Use the data parallelism pattern for the parallel version

  Hints:
  - Rust: rayon's par_iter() + reduce()
  - Python: multiprocessing.Pool.map() with chunk splitting
  - Go: goroutines + channels to aggregate partial results
  - Java: parallelStream() + Collectors

  Verification points:
  - Confirm that sequential and parallel versions produce
    the same results
  - Compare speeds with different core counts and calculate
    efficiency E
================================================================
```

### 10.2 Intermediate: Pipeline Processing

```
================================================================
  Exercise 2 (Intermediate): Log Analysis with Pipeline
================================================================

  Task:
  Process a large log file through the following pipeline:

  Stage 1: File reading (output line by line)
  Stage 2: Parsing (split into timestamp, level, message)
  Stage 3: Filtering (extract only ERROR level)
  Stage 4: Aggregation (output counts by error message)

  Requirements:
  - Implement each stage as an independent goroutine/thread
  - Connect stages via channels/queues
  - Implement backpressure mechanism (buffered channels)
  - Implement cancellation mechanism (Context/CancellationToken)

  Extensions:
  - Parallelize Stage 2 with Fan-Out (multiple workers)
  - Add real-time display of processed counts
  - Implement graceful shutdown with timeout

  Evaluation criteria:
  - Each stage operates independently
  - Memory usage remains constant (stream processing)
  - Error handling is appropriate
================================================================
```

### 10.3 Advanced: Optimized Parallel Merge Sort

```
================================================================
  Exercise 3 (Advanced): Adaptive Parallel Merge Sort
================================================================

  Task:
  Implement a parallel merge sort using the Fork-Join pattern
  with the following optimizations:

  Basic implementation:
  1. Recursively split the array in half
  2. If sub-array is below threshold, use sequential sort
     (insertion sort)
  3. Execute divided tasks in parallel (Fork)
  4. Merge sorted sub-arrays (Join)

  Optimization requirements:
  A) Automatic threshold tuning
     - Determine optimal threshold from core count and data size
     - Threshold too small -> task creation overhead
     - Threshold too large -> insufficient parallelism

  B) Cache optimization
     - Reuse merge buffers (reduce memory allocation)
     - Cache-line-aware data access

  C) Load balancing
     - Switch between parallel/sequential based on recursion depth
     - Leverage work stealing

  Evaluation criteria:
  - Achieve 3x or greater speedup over sequential version
    for 10 million elements (4+ cores)
  - Memory usage O(n) or less
  - Produce correct results for all core counts (1,2,4,8)
  - Achieve parallelization efficiency E > 0.6
================================================================
```

---

## 11. Parallel Debugging and Profiling

### 11.1 Detecting Data Races

```
================================================================
  Data Race Detection Tools
================================================================

  Go: Race Detector
  ------------------
  $ go run -race main.go
  $ go test -race ./...

  -> Detects data races at runtime with detailed stack traces
  -> Standard practice to integrate into CI/CD

  Rust: Compile-Time Detection
  ----------------------------
  - The ownership system prevents data races at compile time
  - Send/Sync traits guarantee safety at the type level
  - Data races are theoretically impossible without using unsafe

  C/C++: ThreadSanitizer (TSan)
  -----------------------------
  $ clang++ -fsanitize=thread -g program.cpp -o program
  $ ./program

  -> LLVM-based data race detector
  -> Runtime overhead is approximately 5-15x

  Java: jcmd / JFR
  -----------------
  $ jcmd <pid> Thread.print    # Thread dump
  $ jcmd <pid> JFR.start       # Start Flight Recorder

  Python: Debugging Parallel Processing
  -------------------------------------
  - multiprocessing uses process isolation, making data races unlikely
  - threading has the GIL, but races are possible during I/O operations
  - Profile with cProfile / py-spy
================================================================
```

### 11.2 Performance Analysis Methods

```
================================================================
  Parallel Processing Performance Analysis Checklist
================================================================

  1. Baseline measurement before parallelization
     [ ] Measure sequential execution time
     [ ] Check CPU usage (is only 1 core at 100%?)
     [ ] Check memory usage

  2. Measurement after parallelization
     [ ] Speedup = sequential time / parallel time
     [ ] Efficiency = speedup / core count
     [ ] CPU usage (are all cores being utilized?)

  3. Identifying bottlenecks
     [ ] Lock contention (mutex contention)
     [ ] False sharing (check L1 cache misses with perf stat)
     [ ] Memory bandwidth saturation
     [ ] Task imbalance (are some cores idle?)
     [ ] GC pauses (Java, Go)

  4. Scaling measurement
     [ ] Plot performance for core counts 1, 2, 4, 8, 16
     [ ] Compare against Amdahl's Law predictions
     [ ] Identify the point where efficiency drops sharply

  Recommended tools:
  +---------------+---------------------------------+
  | Linux         | perf, htop, flamegraph          |
  | macOS         | Instruments, Activity Monitor   |
  | Rust          | criterion (benchmarking)        |
  | Go            | pprof, trace                    |
  | Java          | JFR, async-profiler             |
  | Python        | cProfile, py-spy                |
  | C++           | Valgrind (Helgrind), VTune      |
  +---------------+---------------------------------+
================================================================
```

---

## 12. Real-World Application Examples

### 12.1 Parallelizing Image Processing

```rust
// ================================================================
// Rust: Parallel Application of Image Filters
// ================================================================

use rayon::prelude::*;

struct Image {
    width: usize,
    height: usize,
    pixels: Vec<u8>,  // RGBA: 4 bytes per pixel
}

impl Image {
    /// Apply Gaussian blur in parallel
    fn parallel_blur(&self, radius: usize) -> Image {
        let mut output = vec![0u8; self.pixels.len()];
        let kernel = create_gaussian_kernel(radius);

        // Process each row in parallel (data parallelism)
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

    /// Apply multiple filters through a pipeline
    fn apply_filters(&self) -> Image {
        // Each filter depends on the result of the previous one,
        // so data parallelism is applied within each filter
        let blurred = self.parallel_blur(3);
        let sharpened = blurred.parallel_sharpen();
        let adjusted = sharpened.parallel_brightness(1.2);
        adjusted
    }
}
```

### 12.2 Batch Processing in a Web Server

```go
// ================================================================
// Go: Parallel Aggregation of API Responses
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

// Batch processing with concurrency limit
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

        sem <- struct{}{} // Acquire semaphore

        go func() {
            defer wg.Done()
            defer func() { <-sem }() // Release semaphore

            // Execute 3 external calls in parallel for each product
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

## 13. FAQ (Frequently Asked Questions)

### Q1: When should parallelization be considered?

**A:** Consider parallelization when all of the following conditions are met:

1. **Processing is CPU-bound**: CPU usage is high with little I/O waiting. If I/O-bound, async I/O (async/await) is more effective.
2. **Processing time is sufficiently long**: A single operation takes several hundred milliseconds or more. Short operations become slower due to parallelization overhead.
3. **Data or tasks are divisible**: Processing can be split into independent parts. Processing with strong dependencies is difficult to parallelize.
4. **Sequential optimization is complete**: Algorithm improvements and memory optimization should be done first at the single-thread level.

As a rule of thumb, estimate the parallelizable fraction using Amdahl's Law -- if it is 75% or higher, parallelization can be expected to be effective.

### Q2: How should the thread pool size be determined?

**A:** It depends on the nature of the tasks:

- **CPU-bound tasks**: `thread count = physical core count`. The effect of hyper-threading depends on the workload. Setting it to the logical core count often provides limited improvement.
- **I/O-bound tasks**: `thread count = core count * (1 + wait time / processing time)`. More threads can be used when I/O waits are long.
- **Mixed tasks**: It is recommended to prepare separate thread pools for CPU-bound and I/O-bound workloads.

Java's `ForkJoinPool` and Rust's `rayon` create threads equal to the logical core count by default, which is an appropriate value in most cases.

### Q3: Is true parallel processing really possible in Python?

**A:** Due to the GIL (Global Interpreter Lock) constraint, CPU-bound true parallel execution is not possible with the `threading` module. However, parallel processing can be achieved through the following methods:

1. **`multiprocessing`**: Bypasses the GIL by isolating processes. Each process has its own Python interpreter and memory space. There is inter-process communication overhead.
2. **`concurrent.futures.ProcessPoolExecutor`**: A high-level wrapper around `multiprocessing` that provides an easy-to-use API.
3. **NumPy/SciPy**: Implemented internally in C/Fortran and automatically parallelized via BLAS/LAPACK. The GIL is released during native code execution.
4. **Cython / C extensions**: Can explicitly release the GIL for parallel execution with native threads.
5. **Python 3.13+ (Free-threaded CPython)**: A build option to disable the GIL has been experimentally introduced. Thread-based parallel processing is expected to become possible in the future.

### Q4: Are lock-free data structures always faster than lock-based ones?

**A:** Not necessarily. The advantages of lock-free are "deadlock avoidance" and "scalability under high contention," but lock-based approaches can be superior in the following cases:

- **When contention is low**: The cost of lock acquisition/release is very low (tens of nanoseconds). When contention is rare, lock-based approaches are simpler and faster.
- **When complex operations are needed**: Operations achievable with lock-free are limited. Locks are necessary when consistently updating multiple values.
- **When CAS (Compare-And-Swap) retries are frequent**: Under high contention, CAS failures and retries become frequent, wasting CPU cycles.

As a general guideline, use atomic operations for simple counters and flags, and use locks (`Mutex` / `RwLock`) for complex data structures.

### Q5: What is the difference from GPU parallel processing?

**A:** CPUs and GPUs have fundamentally different parallel processing architectures.

| Property | CPU Parallel | GPU Parallel |
|------|--------|--------|
| Core count | 4-128 cores | Thousands of cores |
| Core capability | High performance (capable of complex operations) | Simple (same operation en masse) |
| Memory | Large capacity, low latency | High bandwidth, high latency |
| Branch processing | High efficiency | Low efficiency (warp divergence) |
| Application domain | General purpose | Matrix operations, image processing, ML |
| Programming | Standard languages | CUDA, OpenCL, Metal |

GPUs are optimal for "applying the same simple operation to enormous amounts of data" (SIMT: Single Instruction, Multiple Threads), while CPU parallelism covered in this chapter is optimal for "executing complex and diverse operations at high speed on a small number of cores."

---


## FAQ

### Q1: What is the most important point when learning this topic?

Building practical experience is the most important. Understanding deepens not just through theory, but by actually writing and running code.

### Q2: What are common mistakes beginners make?

Skipping the basics and jumping to applications. We recommend thoroughly understanding the fundamental concepts explained in this guide before moving to the next step.

### Q3: How is this knowledge applied in practice?

Knowledge of this topic is frequently applied in everyday development work. It is particularly important during code reviews and architecture design.

---

## 14. Summary

### 14.1 Key Points of the Chapter

| Pattern | Purpose | Representative Technologies | Scalability |
|---------|------|---------|----------------|
| Data parallelism | Same operation on large data | Rayon, NumPy, CUDA, parallel stream | Very high |
| Task parallelism | Different operations simultaneously | goroutine, tokio::join!, std::async | Depends on task count |
| MapReduce | Distributed data processing | Hadoop, Spark, rayon fold+reduce | Very high |
| Fork-Join | Recursive divide and conquer | ForkJoinPool, rayon::join | High |
| Pipeline | Throughput improvement for multi-stage processing | Go channel, Unix pipe | Depends on stage count |
| Atomic | Lock-free counters/flags | atomic, sync/atomic | Very high |

### 14.2 Design Decision Guidelines

```
================================================================
  Principles of Parallel Programming
================================================================

  1. Measure before optimizing
     - Don't parallelize based on guesswork
     - Identify bottlenecks with a profiler before starting

  2. Choose the simplest approach
     - First optimize the sequential algorithm
     - Then use data parallelism (par_iter / parallel stream)
     - Lock-free is a last resort

  3. Minimize shared state
     - Prefer message passing
     - Leverage immutable data
     - Make ownership explicit

  4. Set appropriate granularity
     - Tasks too small -> overhead makes it slower
     - Tasks too large -> uneven load distribution
     - Adjust dynamically with work stealing

  5. Prioritize correctness
     - A fast but incorrect program has no value
     - Integrate data race detection tools into CI
     - Always enable the -race flag in tests
================================================================
```

---

## 15. Recommended Next Reading


---

## 16. References

1. Herlihy, M. & Shavit, N. "The Art of Multiprocessor Programming." 2nd Edition, Morgan Kaufmann, 2020. -- A seminal work comprehensively covering the theory and practice of parallel programming. Provides the theoretical foundation for lock-free data structures.
2. McCool, M., Reinders, J. & Robison, A. "Structured Parallel Programming: Patterns for Efficient Computation." Morgan Kaufmann, 2012. -- A textbook that systematically organizes parallel patterns such as MapReduce, Fork-Join, and Pipeline. Written by the designers of Intel TBB.
3. Williams, A. "C++ Concurrency in Action." 2nd Edition, Manning, 2019. -- Provides detailed coverage of C++ memory model, atomic operations, and parallel algorithms. Offers low-level knowledge applicable to other languages as well.
4. "Rayon: data-parallelism library for Rust." github.com/rayon-rs/rayon -- Rust's data parallelism library. Also valuable as a reference implementation of a work-stealing scheduler.
5. Amdahl, G. M. "Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities." AFIPS Conference Proceedings, 1967. -- The original paper on Amdahl's Law. A historic document that mathematically formalized the limits of parallelization.
6. Gustafson, J. L. "Reevaluating Amdahl's Law." Communications of the ACM, 1988. -- The paper proposing Gustafson's Law. Reexamines the assumptions of Amdahl's Law (fixed problem size) and introduces the scale-up perspective.
7. Pike, R. "Concurrency is not Parallelism." Talk at Waza Conference, 2012. -- A talk by Go's designer clearly explaining the difference between concurrency and parallelism. An ideal introductory resource for understanding the concepts.

---

## Recommended Next Reading

- Please refer to other guides in the same category

---

## References

- [MDN Web Docs](https://developer.mozilla.org/) - Web technology reference
- [Wikipedia](https://en.wikipedia.org/) - Overview of technical concepts
