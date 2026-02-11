# ⚡ Python Performance & Optimization Guide

> **目的**: Pythonアプリケーションのパフォーマンスを最大化するための実践的な最適化手法、プロファイリング、ベンチマーク技術を習得する

## 📚 目次

1. [パフォーマンス測定](#パフォーマンス測定)
2. [プロファイリング](#プロファイリング)
3. [データ構造の最適化](#データ構造の最適化)
4. [アルゴリズム最適化](#アルゴリズム最適化)
5. [メモリ最適化](#メモリ最適化)
6. [並列処理・非同期処理](#並列処理非同期処理)
7. [NumPy/Pandas 最適化](#numpypandas-最適化)
8. [キャッシング戦略](#キャッシング戦略)
9. [データベース最適化](#データベース最適化)
10. [Cython・JIT コンパイル](#cythonjit-コンパイル)
11. [実践的な最適化事例](#実践的な最適化事例)

---

## パフォーマンス測定

### time モジュール

**基本的な時間計測**:
```python
import time

# 関数の実行時間を計測
start = time.time()
result = some_function()
end = time.time()
print(f"Execution time: {end - start:.4f} seconds")

# より精密な計測（time.perf_counter）
start = time.perf_counter()
result = some_function()
end = time.perf_counter()
print(f"Execution time: {end - start:.6f} seconds")
```

**デコレータで計測**:
```python
import time
from functools import wraps
from typing import Callable, Any


def timeit(func: Callable) -> Callable:
    """関数の実行時間を計測するデコレータ"""
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper


@timeit
def process_data(data: list[int]) -> int:
    """データ処理関数"""
    return sum(x ** 2 for x in data)


# 使用例
result = process_data(list(range(1000000)))
# process_data took 0.234567 seconds
```

### timeit モジュール

**コードスニペットのベンチマーク**:
```python
import timeit

# 単純な計測
execution_time = timeit.timeit(
    stmt='sum(range(100))',
    number=10000
)
print(f"Time: {execution_time:.6f} seconds")

# セットアップコード付き
execution_time = timeit.timeit(
    stmt='result = [x ** 2 for x in data]',
    setup='data = list(range(1000))',
    number=10000
)
print(f"Time: {execution_time:.6f} seconds")

# 関数のベンチマーク
def my_function():
    return sum(x ** 2 for x in range(1000))

execution_time = timeit.timeit(
    stmt='my_function()',
    globals=globals(),
    number=10000
)
print(f"Time: {execution_time:.6f} seconds")
```

**複数の実装を比較**:
```python
import timeit

def compare_implementations():
    """複数の実装を比較"""

    # リスト内包表記
    time1 = timeit.timeit(
        stmt='[x ** 2 for x in range(1000)]',
        number=10000
    )

    # map + lambda
    time2 = timeit.timeit(
        stmt='list(map(lambda x: x ** 2, range(1000)))',
        number=10000
    )

    # for ループ
    time3 = timeit.timeit(
        stmt='''
result = []
for x in range(1000):
    result.append(x ** 2)
''',
        number=10000
    )

    print(f"List comprehension: {time1:.6f}s")
    print(f"Map + lambda:       {time2:.6f}s")
    print(f"For loop:           {time3:.6f}s")
    print(f"Fastest: List comprehension ({time1:.6f}s)")


compare_implementations()
# List comprehension: 0.456789s  ← 最速
# Map + lambda:       0.567890s
# For loop:           0.678901s
```

### ベンチマークユーティリティ

```python
import time
from typing import Callable, Any
from dataclasses import dataclass
from statistics import mean, stdev


@dataclass
class BenchmarkResult:
    """ベンチマーク結果"""
    name: str
    mean_time: float
    std_dev: float
    min_time: float
    max_time: float
    iterations: int

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Mean: {self.mean_time:.6f}s\n"
            f"  Std Dev: {self.std_dev:.6f}s\n"
            f"  Min: {self.min_time:.6f}s\n"
            f"  Max: {self.max_time:.6f}s\n"
            f"  Iterations: {self.iterations}"
        )


def benchmark(
    func: Callable,
    *args: Any,
    iterations: int = 100,
    warmup: int = 10,
    **kwargs: Any
) -> BenchmarkResult:
    """関数をベンチマーク"""

    # ウォームアップ（JIT コンパイルなどのため）
    for _ in range(warmup):
        func(*args, **kwargs)

    # 計測
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)

    return BenchmarkResult(
        name=func.__name__,
        mean_time=mean(times),
        std_dev=stdev(times) if len(times) > 1 else 0.0,
        min_time=min(times),
        max_time=max(times),
        iterations=iterations
    )


# 使用例
def process_list_comprehension(n: int) -> list[int]:
    return [x ** 2 for x in range(n)]

def process_map(n: int) -> list[int]:
    return list(map(lambda x: x ** 2, range(n)))

result1 = benchmark(process_list_comprehension, 10000, iterations=1000)
result2 = benchmark(process_map, 10000, iterations=1000)

print(result1)
print("\n")
print(result2)
```

---

## プロファイリング

### cProfile

**基本的な使い方**:
```python
import cProfile
import pstats
from io import StringIO


def expensive_function():
    """重い処理をシミュレート"""
    total = 0
    for i in range(1000000):
        total += i ** 2
    return total


def main():
    result = expensive_function()
    # その他の処理...


# プロファイリング実行
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()

    # 結果を出力
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # 上位10件を表示
```

**コマンドラインから実行**:
```bash
# プロファイリング実行
python -m cProfile -o output.prof script.py

# 結果を表示
python -m pstats output.prof
# stats> sort cumulative
# stats> stats 10
```

**デコレータでプロファイリング**:
```python
import cProfile
import pstats
from functools import wraps
from typing import Callable


def profile(output_file: str | None = None):
    """プロファイリングデコレータ"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()

            result = func(*args, **kwargs)

            profiler.disable()

            if output_file:
                profiler.dump_stats(output_file)
            else:
                stats = pstats.Stats(profiler)
                stats.sort_stats('cumulative')
                stats.print_stats(20)

            return result
        return wrapper
    return decorator


@profile(output_file="my_function.prof")
def my_function():
    # 処理...
    pass
```

### line_profiler

**インストール**:
```bash
pip install line-profiler
```

**使用方法**:
```python
# script.py
@profile  # line_profiler のマジックデコレータ
def process_data(data: list[int]) -> list[int]:
    """データ処理"""
    result = []
    for item in data:
        # 各行の実行時間が計測される
        squared = item ** 2
        if squared > 100:
            result.append(squared)
    return result


def main():
    data = list(range(10000))
    result = process_data(data)


if __name__ == "__main__":
    main()
```

**実行**:
```bash
# 行単位でプロファイリング
kernprof -l -v script.py

# 出力例:
# Line #      Hits         Time  Per Hit   % Time  Line Contents
# ==============================================================
#      1                                           @profile
#      2                                           def process_data(data):
#      3         1          2.0      2.0      0.0      result = []
#      4     10000       5234.0      0.5     45.2      for item in data:
#      5     10000       3456.0      0.3     29.8          squared = item ** 2
#      6     10000       2345.0      0.2     20.2          if squared > 100:
#      7      9900        567.0      0.1      4.8              result.append(squared)
#      8         1          0.0      0.0      0.0      return result
```

### memory_profiler

**インストール**:
```bash
pip install memory-profiler
```

**使用方法**:
```python
from memory_profiler import profile


@profile
def memory_intensive_function():
    """メモリを大量に使う関数"""
    # 大きなリストを作成
    data = [i for i in range(1000000)]

    # さらに加工
    squared = [x ** 2 for x in data]

    # 辞書に変換
    result = {i: x for i, x in enumerate(squared)}

    return result


if __name__ == "__main__":
    memory_intensive_function()
```

**実行**:
```bash
# メモリプロファイリング
python -m memory_profiler script.py

# 出力例:
# Line #    Mem usage    Increment  Occurrences   Line Contents
# =============================================================
#      1     45.2 MiB     45.2 MiB           1   @profile
#      2                                         def memory_intensive_function():
#      3     83.5 MiB     38.3 MiB           1       data = [i for i in range(1000000)]
#      4    121.8 MiB     38.3 MiB           1       squared = [x ** 2 for x in data]
#      5    198.4 MiB     76.6 MiB           1       result = {i: x for i, x in enumerate(squared)}
#      6    198.4 MiB      0.0 MiB           1       return result
```

### pyinstrument

**インストール**:
```bash
pip install pyinstrument
```

**使用方法**:
```python
from pyinstrument import Profiler


def main():
    # 処理...
    pass


if __name__ == "__main__":
    profiler = Profiler()
    profiler.start()

    main()

    profiler.stop()

    # 結果を出力
    profiler.print()

    # HTML レポート出力
    with open("profile_report.html", "w") as f:
        f.write(profiler.output_html())
```

**コマンドラインから実行**:
```bash
# プロファイリング実行
pyinstrument script.py

# HTML レポート生成
pyinstrument -o report.html script.py
```

---

## データ構造の最適化

### リスト vs タプル vs セット vs 辞書

**パフォーマンス比較**:
```python
import timeit


def compare_data_structures():
    """データ構造のパフォーマンス比較"""

    # リスト
    list_creation = timeit.timeit(
        stmt='[i for i in range(1000)]',
        number=10000
    )

    list_lookup = timeit.timeit(
        stmt='999 in data',
        setup='data = list(range(1000))',
        number=10000
    )

    # タプル
    tuple_creation = timeit.timeit(
        stmt='tuple(i for i in range(1000))',
        number=10000
    )

    tuple_lookup = timeit.timeit(
        stmt='999 in data',
        setup='data = tuple(range(1000))',
        number=10000
    )

    # セット
    set_creation = timeit.timeit(
        stmt='{i for i in range(1000)}',
        number=10000
    )

    set_lookup = timeit.timeit(
        stmt='999 in data',
        setup='data = set(range(1000))',
        number=10000
    )

    # 辞書
    dict_creation = timeit.timeit(
        stmt='{i: i for i in range(1000)}',
        number=10000
    )

    dict_lookup = timeit.timeit(
        stmt='999 in data',
        setup='data = {i: i for i in range(1000)}',
        number=10000
    )

    print("Creation times:")
    print(f"  List:  {list_creation:.6f}s")
    print(f"  Tuple: {tuple_creation:.6f}s")
    print(f"  Set:   {set_creation:.6f}s")
    print(f"  Dict:  {dict_creation:.6f}s")

    print("\nLookup times:")
    print(f"  List:  {list_lookup:.6f}s")  # O(n) - 遅い
    print(f"  Tuple: {tuple_lookup:.6f}s")  # O(n) - 遅い
    print(f"  Set:   {set_lookup:.6f}s")    # O(1) - 速い!
    print(f"  Dict:  {dict_lookup:.6f}s")   # O(1) - 速い!


compare_data_structures()
```

**最適な選択**:
```python
# ❌ 遅い: リストで要素の存在チェック
def slow_check(items: list[int], target: int) -> bool:
    return target in items  # O(n)


# ✅ 速い: セットで要素の存在チェック
def fast_check(items: set[int], target: int) -> bool:
    return target in items  # O(1)


# ベンチマーク
import timeit

data_list = list(range(100000))
data_set = set(range(100000))

time_list = timeit.timeit(
    stmt='99999 in data',
    setup='from __main__ import data_list as data',
    number=10000
)

time_set = timeit.timeit(
    stmt='99999 in data',
    setup='from __main__ import data_set as data',
    number=10000
)

print(f"List lookup: {time_list:.6f}s")  # 約 0.5s
print(f"Set lookup:  {time_set:.6f}s")   # 約 0.0001s (5000倍速い!)
```

### collections モジュール

**defaultdict**:
```python
from collections import defaultdict


# ❌ 遅い: 通常の辞書
def group_by_category_slow(items: list[dict]) -> dict[str, list[dict]]:
    result = {}
    for item in items:
        category = item['category']
        if category not in result:
            result[category] = []
        result[category].append(item)
    return result


# ✅ 速い: defaultdict
def group_by_category_fast(items: list[dict]) -> dict[str, list[dict]]:
    result = defaultdict(list)
    for item in items:
        result[item['category']].append(item)
    return result
```

**Counter**:
```python
from collections import Counter


# ❌ 遅い: 手動でカウント
def count_words_slow(words: list[str]) -> dict[str, int]:
    counts = {}
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts


# ✅ 速い: Counter
def count_words_fast(words: list[str]) -> dict[str, int]:
    return Counter(words)


# ベンチマーク
words = ["apple", "banana", "apple", "cherry", "banana", "apple"] * 10000

import timeit

time_slow = timeit.timeit(
    stmt='count_words_slow(words)',
    setup='from __main__ import count_words_slow, words',
    number=100
)

time_fast = timeit.timeit(
    stmt='count_words_fast(words)',
    setup='from __main__ import count_words_fast, words',
    number=100
)

print(f"Manual count: {time_slow:.6f}s")
print(f"Counter:      {time_fast:.6f}s")  # 約2倍速い
```

**deque（両端キュー）**:
```python
from collections import deque
import timeit


# リストの先頭挿入は遅い（O(n)）
list_insert = timeit.timeit(
    stmt='data.insert(0, 1)',
    setup='data = list(range(10000))',
    number=1000
)

# deque の先頭挿入は速い（O(1)）
deque_insert = timeit.timeit(
    stmt='data.appendleft(1)',
    setup='from collections import deque; data = deque(range(10000))',
    number=1000
)

print(f"List insert at front:  {list_insert:.6f}s")  # 遅い
print(f"Deque insert at front: {deque_insert:.6f}s")  # 速い (100倍以上速い!)
```

---

## アルゴリズム最適化

### 計算量の改善

**O(n²) → O(n) への最適化**:
```python
# ❌ O(n²): ネストループ
def find_duplicates_slow(nums: list[int]) -> list[int]:
    duplicates = []
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] == nums[j] and nums[i] not in duplicates:
                duplicates.append(nums[i])
    return duplicates


# ✅ O(n): セットを使用
def find_duplicates_fast(nums: list[int]) -> list[int]:
    seen = set()
    duplicates = set()
    for num in nums:
        if num in seen:
            duplicates.add(num)
        else:
            seen.add(num)
    return list(duplicates)


# ベンチマーク
import timeit

data = list(range(1000)) * 2

time_slow = timeit.timeit(
    stmt='find_duplicates_slow(data)',
    setup='from __main__ import find_duplicates_slow, data',
    number=10
)

time_fast = timeit.timeit(
    stmt='find_duplicates_fast(data)',
    setup='from __main__ import find_duplicates_fast, data',
    number=10
)

print(f"O(n²): {time_slow:.6f}s")
print(f"O(n):  {time_fast:.6f}s")  # 1000倍以上速い!
```

**ソートアルゴリズムの選択**:
```python
import timeit
import random


def compare_sorting_algorithms(data: list[int]):
    """ソートアルゴリズムの比較"""

    # 組み込みソート（Timsort - O(n log n)）
    time_builtin = timeit.timeit(
        stmt='sorted(data)',
        setup=f'data = {data}',
        number=1000
    )

    # バブルソート（O(n²) - 遅い）
    def bubble_sort(arr):
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if arr[j] > arr[j + 1]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return arr

    time_bubble = timeit.timeit(
        stmt='bubble_sort(data.copy())',
        setup=f'from __main__ import bubble_sort; data = {data}',
        number=100
    )

    print(f"Built-in sort (Timsort): {time_builtin:.6f}s")
    print(f"Bubble sort:             {time_bubble:.6f}s")
    print(f"Speed ratio: {time_bubble / time_builtin:.1f}x slower")


# テスト
data = [random.randint(1, 100) for _ in range(100)]
compare_sorting_algorithms(data)
```

### ジェネレータで遅延評価

**メモリ効率的な処理**:
```python
# ❌ 遅い: すべてをメモリに展開
def process_all_at_once(n: int) -> int:
    squares = [x ** 2 for x in range(n)]
    evens = [x for x in squares if x % 2 == 0]
    return sum(evens)


# ✅ 速い: ジェネレータで遅延評価
def process_with_generator(n: int) -> int:
    squares = (x ** 2 for x in range(n))
    evens = (x for x in squares if x % 2 == 0)
    return sum(evens)


# メモリ使用量比較
import sys
import timeit

n = 1000000

# メモリ使用量
list_comp = [x ** 2 for x in range(n)]
gen_comp = (x ** 2 for x in range(n))

print(f"List size: {sys.getsizeof(list_comp):,} bytes")  # 約8MB
print(f"Gen size:  {sys.getsizeof(gen_comp):,} bytes")   # 約200 bytes

# 実行時間比較
time_list = timeit.timeit(
    stmt='process_all_at_once(1000000)',
    setup='from __main__ import process_all_at_once',
    number=10
)

time_gen = timeit.timeit(
    stmt='process_with_generator(1000000)',
    setup='from __main__ import process_with_generator',
    number=10
)

print(f"\nExecution time:")
print(f"List comprehension: {time_list:.6f}s")
print(f"Generator:          {time_gen:.6f}s")  # 約2倍速い
```

### itertools で効率的な処理

```python
from itertools import islice, chain, groupby


# ❌ 遅い: リストのスライス
def get_first_n_slow(data: list, n: int) -> list:
    return data[:n]


# ✅ 速い: islice（イテレータのスライス）
def get_first_n_fast(data, n: int):
    return list(islice(data, n))


# チェインで複数のイテレータを結合
def combine_iterators_slow(*iterators):
    result = []
    for iterator in iterators:
        result.extend(list(iterator))
    return result


def combine_iterators_fast(*iterators):
    return list(chain(*iterators))


# グループ化
from operator import itemgetter

data = [
    {"category": "A", "value": 10},
    {"category": "A", "value": 20},
    {"category": "B", "value": 30},
    {"category": "B", "value": 40},
]

# ソート済みデータをグループ化
data.sort(key=itemgetter("category"))

for category, group in groupby(data, key=itemgetter("category")):
    items = list(group)
    total = sum(item["value"] for item in items)
    print(f"{category}: {total}")
```

---

## メモリ最適化

### メモリ使用量の測定

```python
import sys
from typing import Any


def get_size(obj: Any) -> int:
    """オブジェクトのメモリサイズを取得"""
    return sys.getsizeof(obj)


# 各データ構造のメモリサイズ
data = list(range(1000))

print(f"List:  {get_size(data):,} bytes")
print(f"Tuple: {get_size(tuple(data)):,} bytes")
print(f"Set:   {get_size(set(data)):,} bytes")
print(f"Dict:  {get_size({i: i for i in data}):,} bytes")

# 文字列のメモリ
text = "Hello" * 1000
print(f"String: {get_size(text):,} bytes")
```

### __slots__ でメモリ削減

```python
import sys


# ❌ 通常のクラス（__dict__ を持つ）
class NormalUser:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email


# ✅ __slots__ 使用（__dict__ なし）
class OptimizedUser:
    __slots__ = ['name', 'age', 'email']

    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email


# メモリ比較
normal = NormalUser("Alice", 25, "alice@example.com")
optimized = OptimizedUser("Alice", 25, "alice@example.com")

print(f"Normal class:    {sys.getsizeof(normal)} bytes")
print(f"Optimized class: {sys.getsizeof(optimized)} bytes")
print(f"Memory saved:    {sys.getsizeof(normal) - sys.getsizeof(optimized)} bytes")

# 大量のインスタンスを作成
normal_users = [NormalUser(f"User{i}", 25, f"user{i}@example.com") for i in range(10000)]
optimized_users = [OptimizedUser(f"User{i}", 25, f"user{i}@example.com") for i in range(10000)]

normal_total = sum(sys.getsizeof(u) for u in normal_users)
optimized_total = sum(sys.getsizeof(u) for u in optimized_users)

print(f"\n10,000 instances:")
print(f"Normal:    {normal_total:,} bytes")
print(f"Optimized: {optimized_total:,} bytes")
print(f"Saved:     {normal_total - optimized_total:,} bytes ({(1 - optimized_total/normal_total)*100:.1f}%)")
```

### ジェネレータでメモリ効率化

```python
# ❌ メモリを大量に使用
def read_large_file_slow(file_path: str) -> list[str]:
    with open(file_path) as f:
        return f.readlines()  # ファイル全体をメモリに読み込み


# ✅ メモリ効率的
def read_large_file_fast(file_path: str):
    with open(file_path) as f:
        for line in f:  # 1行ずつ処理
            yield line.strip()


# 使用例
for line in read_large_file_fast("large_file.txt"):
    process_line(line)  # 1行ずつ処理（メモリ使用量が一定）
```

---

## 並列処理・非同期処理

### multiprocessing で CPU バウンド処理

```python
from multiprocessing import Pool, cpu_count
import time


def cpu_bound_task(n: int) -> int:
    """CPU を使う重い処理"""
    return sum(i * i for i in range(n))


def sequential_processing(tasks: list[int]) -> list[int]:
    """逐次処理"""
    return [cpu_bound_task(task) for task in tasks]


def parallel_processing(tasks: list[int], workers: int = None) -> list[int]:
    """並列処理"""
    if workers is None:
        workers = cpu_count()

    with Pool(processes=workers) as pool:
        return pool.map(cpu_bound_task, tasks)


# ベンチマーク
tasks = [10000000] * 8

start = time.time()
results_seq = sequential_processing(tasks)
time_seq = time.time() - start

start = time.time()
results_par = parallel_processing(tasks, workers=4)
time_par = time.time() - start

print(f"Sequential: {time_seq:.2f}s")
print(f"Parallel:   {time_par:.2f}s")
print(f"Speedup:    {time_seq / time_par:.2f}x")
```

### concurrent.futures

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time


# I/O バウンド処理（ThreadPoolExecutor）
def io_bound_task(url: str) -> dict:
    """I/O を待つ処理（API リクエストなど）"""
    time.sleep(0.1)  # I/O 待機をシミュレート
    return {"url": url, "status": 200}


def process_urls_parallel(urls: list[str], max_workers: int = 10) -> list[dict]:
    """URL を並列で処理"""
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(io_bound_task, url): url for url in urls}

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:
                print(f"{url} generated an exception: {exc}")

    return results


# CPU バウンド処理（ProcessPoolExecutor）
def cpu_bound_task(n: int) -> int:
    """CPU を使う処理"""
    return sum(i * i for i in range(n))


def process_cpu_bound_parallel(numbers: list[int], max_workers: int = 4) -> list[int]:
    """CPU バウンド処理を並列化"""
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(cpu_bound_task, numbers))


# ベンチマーク
urls = [f"https://example.com/page{i}" for i in range(50)]

start = time.time()
results_seq = [io_bound_task(url) for url in urls]
time_seq = time.time() - start

start = time.time()
results_par = process_urls_parallel(urls, max_workers=10)
time_par = time.time() - start

print(f"Sequential (I/O): {time_seq:.2f}s")
print(f"Parallel (I/O):   {time_par:.2f}s")
print(f"Speedup:          {time_seq / time_par:.2f}x")
```

### asyncio で非同期処理

```python
import asyncio
import aiohttp
import time


async def fetch_url_async(session: aiohttp.ClientSession, url: str) -> dict:
    """非同期で URL を取得"""
    async with session.get(url) as response:
        return {"url": url, "status": response.status}


async def fetch_all_urls_async(urls: list[str]) -> list[dict]:
    """複数 URL を非同期で取得"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url_async(session, url) for url in urls]
        return await asyncio.gather(*tasks)


# 同期版（比較用）
import requests


def fetch_url_sync(url: str) -> dict:
    """同期で URL を取得"""
    response = requests.get(url)
    return {"url": url, "status": response.status_code}


def fetch_all_urls_sync(urls: list[str]) -> list[dict]:
    """複数 URL を同期で取得"""
    return [fetch_url_sync(url) for url in urls]


# ベンチマーク
async def benchmark_async():
    urls = ["https://httpbin.org/delay/1"] * 10

    # 非同期版
    start = time.time()
    results = await fetch_all_urls_async(urls)
    time_async = time.time() - start

    print(f"Async:  {time_async:.2f}s")
    print(f"Result: {len(results)} URLs fetched")


# 実行
asyncio.run(benchmark_async())
# Async:  1.2s (10個のリクエストを並行実行)
# 同期版だと 10s かかる
```

---

## NumPy/Pandas 最適化

### NumPy でベクトル化

```python
import numpy as np
import timeit


# ❌ 遅い: Python のループ
def sum_of_squares_python(arr: list[float]) -> float:
    total = 0
    for x in arr:
        total += x ** 2
    return total


# ✅ 速い: NumPy のベクトル化
def sum_of_squares_numpy(arr: np.ndarray) -> float:
    return np.sum(arr ** 2)


# ベンチマーク
data_list = list(range(1000000))
data_numpy = np.array(data_list)

time_python = timeit.timeit(
    stmt='sum_of_squares_python(data)',
    setup='from __main__ import sum_of_squares_python, data_list as data',
    number=10
)

time_numpy = timeit.timeit(
    stmt='sum_of_squares_numpy(data)',
    setup='from __main__ import sum_of_squares_numpy, data_numpy as data',
    number=10
)

print(f"Python loop: {time_python:.6f}s")
print(f"NumPy:       {time_numpy:.6f}s")
print(f"Speedup:     {time_python / time_numpy:.1f}x")  # 100倍以上速い!
```

**NumPy 最適化のコツ**:
```python
import numpy as np


# ❌ 遅い: ループで要素にアクセス
def slow_processing(arr: np.ndarray) -> np.ndarray:
    result = np.zeros_like(arr)
    for i in range(len(arr)):
        result[i] = arr[i] ** 2 + 2 * arr[i] + 1
    return result


# ✅ 速い: ベクトル化
def fast_processing(arr: np.ndarray) -> np.ndarray:
    return arr ** 2 + 2 * arr + 1


# ✅ さらに速い: in-place 演算
def fastest_processing(arr: np.ndarray) -> np.ndarray:
    result = arr.copy()
    result **= 2
    result += 2 * arr
    result += 1
    return result


# ベンチマーク
data = np.arange(1000000, dtype=np.float64)

time_slow = timeit.timeit(
    stmt='slow_processing(data)',
    setup='from __main__ import slow_processing, data',
    number=10
)

time_fast = timeit.timeit(
    stmt='fast_processing(data)',
    setup='from __main__ import fast_processing, data',
    number=10
)

time_fastest = timeit.timeit(
    stmt='fastest_processing(data)',
    setup='from __main__ import fastest_processing, data',
    number=10
)

print(f"Loop:        {time_slow:.6f}s")
print(f"Vectorized:  {time_fast:.6f}s")
print(f"In-place:    {time_fastest:.6f}s")
```

### Pandas 最適化

**iterrows() を避ける**:
```python
import pandas as pd
import numpy as np
import timeit


# サンプルデータ
df = pd.DataFrame({
    'A': np.random.rand(100000),
    'B': np.random.rand(100000),
    'C': np.random.rand(100000),
})


# ❌ 最も遅い: iterrows()
def process_with_iterrows(df: pd.DataFrame) -> pd.Series:
    results = []
    for index, row in df.iterrows():
        results.append(row['A'] + row['B'] * row['C'])
    return pd.Series(results)


# ⚠️ 遅い: apply()
def process_with_apply(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda row: row['A'] + row['B'] * row['C'], axis=1)


# ✅ 速い: ベクトル化
def process_vectorized(df: pd.DataFrame) -> pd.Series:
    return df['A'] + df['B'] * df['C']


# ✅ 最速: NumPy
def process_numpy(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df['A'].values + df['B'].values * df['C'].values)


# ベンチマーク
time_iterrows = timeit.timeit(
    stmt='process_with_iterrows(df)',
    setup='from __main__ import process_with_iterrows, df',
    number=10
)

time_apply = timeit.timeit(
    stmt='process_with_apply(df)',
    setup='from __main__ import process_with_apply, df',
    number=10
)

time_vectorized = timeit.timeit(
    stmt='process_vectorized(df)',
    setup='from __main__ import process_vectorized, df',
    number=10
)

time_numpy = timeit.timeit(
    stmt='process_numpy(df)',
    setup='from __main__ import process_numpy, df',
    number=10
)

print(f"iterrows():   {time_iterrows:.6f}s")
print(f"apply():      {time_apply:.6f}s")
print(f"Vectorized:   {time_vectorized:.6f}s")
print(f"NumPy:        {time_numpy:.6f}s")
print(f"\nSpeedup (iterrows vs NumPy): {time_iterrows / time_numpy:.1f}x")
```

**カテゴリ型でメモリ削減**:
```python
import pandas as pd


# 文字列カラム
df = pd.DataFrame({
    'category': ['A', 'B', 'C', 'A', 'B'] * 100000
})

print(f"String type: {df.memory_usage(deep=True)['category']:,} bytes")

# カテゴリ型に変換
df['category'] = df['category'].astype('category')

print(f"Category type: {df.memory_usage(deep=True)['category']:,} bytes")
print(f"Memory saved: {100 - (df.memory_usage(deep=True)['category'] / 28000000 * 100):.1f}%")
```

**チャンクで大きなファイルを処理**:
```python
import pandas as pd


# ❌ メモリ不足: ファイル全体を読み込み
# df = pd.read_csv("huge_file.csv")  # メモリエラー!


# ✅ チャンクで処理
def process_large_file(file_path: str, chunk_size: int = 10000):
    """大きなファイルをチャンクで処理"""
    results = []

    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # チャンクごとに処理
        processed = chunk[chunk['value'] > 100]
        results.append(processed)

    # 結果を結合
    return pd.concat(results, ignore_index=True)


# 使用例
# result = process_large_file("huge_file.csv")
```

---

## キャッシング戦略

### functools.lru_cache

```python
from functools import lru_cache
import timeit


# ❌ キャッシュなし
def fibonacci_no_cache(n: int) -> int:
    if n < 2:
        return n
    return fibonacci_no_cache(n - 1) + fibonacci_no_cache(n - 2)


# ✅ lru_cache でメモ化
@lru_cache(maxsize=128)
def fibonacci_cached(n: int) -> int:
    if n < 2:
        return n
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)


# ベンチマーク
time_no_cache = timeit.timeit(
    stmt='fibonacci_no_cache(30)',
    setup='from __main__ import fibonacci_no_cache',
    number=1
)

time_cached = timeit.timeit(
    stmt='fibonacci_cached(30)',
    setup='from __main__ import fibonacci_cached',
    number=1
)

print(f"No cache: {time_no_cache:.6f}s")
print(f"Cached:   {time_cached:.6f}s")
print(f"Speedup:  {time_no_cache / time_cached:.0f}x")  # 100,000倍以上速い!

# キャッシュ統計
print(f"\nCache stats: {fibonacci_cached.cache_info()}")
# CacheInfo(hits=28, misses=31, maxsize=128, currsize=31)
```

**カスタムキャッシュデコレータ**:
```python
from functools import wraps
from typing import Callable, Any
import time


def timed_cache(expiry_seconds: int = 60):
    """期限付きキャッシュデコレータ"""
    def decorator(func: Callable) -> Callable:
        cache = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            now = time.time()

            # キャッシュチェック
            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < expiry_seconds:
                    print(f"Cache hit for {func.__name__}")
                    return result

            # キャッシュミス - 関数実行
            print(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            cache[key] = (result, now)

            return result

        return wrapper
    return decorator


@timed_cache(expiry_seconds=5)
def expensive_computation(x: int) -> int:
    """重い計算"""
    time.sleep(2)
    return x ** 2


# 使用例
print(expensive_computation(10))  # Cache miss - 2秒待機
print(expensive_computation(10))  # Cache hit - 即座に返す
time.sleep(6)
print(expensive_computation(10))  # Cache miss - 期限切れ
```

### Redis でキャッシュ

```bash
pip install redis
```

```python
import redis
import json
import time
from functools import wraps
from typing import Callable, Any


# Redis 接続
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)


def redis_cache(expiry_seconds: int = 3600):
    """Redis キャッシュデコレータ"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # キャッシュキー生成
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            # キャッシュチェック
            cached_value = redis_client.get(cache_key)
            if cached_value:
                print(f"Redis cache hit for {func.__name__}")
                return json.loads(cached_value)

            # キャッシュミス - 関数実行
            print(f"Redis cache miss for {func.__name__}")
            result = func(*args, **kwargs)

            # Redis に保存
            redis_client.setex(
                cache_key,
                expiry_seconds,
                json.dumps(result)
            )

            return result

        return wrapper
    return decorator


@redis_cache(expiry_seconds=60)
def get_user_data(user_id: int) -> dict:
    """ユーザーデータ取得（重い処理をシミュレート）"""
    time.sleep(2)
    return {
        "id": user_id,
        "name": f"User {user_id}",
        "email": f"user{user_id}@example.com"
    }


# 使用例
print(get_user_data(1))  # Cache miss - 2秒待機
print(get_user_data(1))  # Cache hit - 即座に返す
```

---

## データベース最適化

### SQLAlchemy 最適化

**N+1 問題の解決**:
```python
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, joinedload
import time

Base = declarative_base()


class Author(Base):
    __tablename__ = 'authors'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    books = relationship('Book', back_populates='author')


class Book(Base):
    __tablename__ = 'books'
    id = Column(Integer, primary_key=True)
    title = Column(String)
    author_id = Column(Integer, ForeignKey('authors.id'))
    author = relationship('Author', back_populates='books')


# ❌ N+1 問題
def get_books_slow(session):
    """各書籍ごとに著者を取得（N+1クエリ）"""
    books = session.query(Book).all()

    for book in books:
        print(f"{book.title} by {book.author.name}")  # 各ループでクエリ実行


# ✅ joinedload で最適化
def get_books_fast(session):
    """1クエリで書籍と著者を取得"""
    books = session.query(Book).options(joinedload(Book.author)).all()

    for book in books:
        print(f"{book.title} by {book.author.name}")  # クエリなし


# ベンチマーク
engine = create_engine('sqlite:///books.db')
Session = sessionmaker(bind=engine)
session = Session()

start = time.time()
get_books_slow(session)
time_slow = time.time() - start

start = time.time()
get_books_fast(session)
time_fast = time.time() - start

print(f"\nN+1 queries: {time_slow:.6f}s")
print(f"Optimized:   {time_fast:.6f}s")
print(f"Speedup:     {time_slow / time_fast:.1f}x")
```

**バルク挿入**:
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# ❌ 遅い: 1件ずつコミット
def insert_one_by_one(session, data: list[dict]):
    for item in data:
        user = User(**item)
        session.add(user)
        session.commit()  # 各挿入でコミット


# ✅ 速い: バルク挿入
def bulk_insert(session, data: list[dict]):
    session.bulk_insert_mappings(User, data)
    session.commit()  # 最後に1回だけコミット


# ベンチマーク
data = [{"name": f"User{i}", "email": f"user{i}@example.com"} for i in range(1000)]

start = time.time()
insert_one_by_one(session, data)
time_slow = time.time() - start

start = time.time()
bulk_insert(session, data)
time_fast = time.time() - start

print(f"One by one: {time_slow:.6f}s")
print(f"Bulk:       {time_fast:.6f}s")
print(f"Speedup:    {time_slow / time_fast:.1f}x")  # 100倍以上速い!
```

### インデックス最適化

```python
from sqlalchemy import Index


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, index=True)  # インデックス追加
    name = Column(String, index=True)  # インデックス追加
    created_at = Column(DateTime, index=True)

    # 複合インデックス
    __table_args__ = (
        Index('idx_email_name', 'email', 'name'),
        Index('idx_created_at_desc', 'created_at desc'),
    )
```

---

## Cython・JIT コンパイル

### Cython で高速化

**インストール**:
```bash
pip install cython
```

**Cython コード（fast_math.pyx）**:
```python
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

def sum_of_squares_cython(int n):
    """平方和をCythonで計算"""
    cdef long long total = 0
    cdef int i

    for i in range(n):
        total += i * i

    return total
```

**setup.py**:
```python
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("fast_math.pyx")
)
```

**ビルド**:
```bash
python setup.py build_ext --inplace
```

**使用例**:
```python
import timeit
from fast_math import sum_of_squares_cython


# Python 版
def sum_of_squares_python(n: int) -> int:
    total = 0
    for i in range(n):
        total += i * i
    return total


# ベンチマーク
n = 10000000

time_python = timeit.timeit(
    stmt='sum_of_squares_python(n)',
    setup='from __main__ import sum_of_squares_python, n',
    number=10
)

time_cython = timeit.timeit(
    stmt='sum_of_squares_cython(n)',
    setup='from fast_math import sum_of_squares_cython; from __main__ import n',
    number=10
)

print(f"Python:  {time_python:.6f}s")
print(f"Cython:  {time_cython:.6f}s")
print(f"Speedup: {time_python / time_cython:.1f}x")  # 10-100倍速い!
```

### Numba で JIT コンパイル

**インストール**:
```bash
pip install numba
```

**使用例**:
```python
from numba import jit
import numpy as np
import timeit


# ❌ 通常の Python
def sum_of_squares_python(arr):
    total = 0
    for i in range(len(arr)):
        total += arr[i] ** 2
    return total


# ✅ Numba JIT
@jit(nopython=True)
def sum_of_squares_numba(arr):
    total = 0
    for i in range(len(arr)):
        total += arr[i] ** 2
    return total


# ベンチマーク
arr = np.arange(10000000, dtype=np.int64)

# ウォームアップ（JIT コンパイル）
sum_of_squares_numba(arr)

time_python = timeit.timeit(
    stmt='sum_of_squares_python(arr)',
    setup='from __main__ import sum_of_squares_python, arr',
    number=10
)

time_numba = timeit.timeit(
    stmt='sum_of_squares_numba(arr)',
    setup='from __main__ import sum_of_squares_numba, arr',
    number=10
)

print(f"Python: {time_python:.6f}s")
print(f"Numba:  {time_numba:.6f}s")
print(f"Speedup: {time_python / time_numba:.1f}x")  # 100倍以上速い!
```

### PyPy

**PyPy のインストール**:
```bash
# PyPy のダウンロード・インストール
# https://www.pypy.org/download.html
```

**ベンチマーク**:
```python
# benchmark.py
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

result = fibonacci(35)
print(f"Result: {result}")
```

**実行比較**:
```bash
# CPython
time python benchmark.py
# real: 3.5s

# PyPy
time pypy3 benchmark.py
# real: 0.3s  (約10倍速い!)
```

---

## 実践的な最適化事例

### ケース1: API レスポンス最適化

**Before（遅い）**:
```python
from fastapi import FastAPI
from sqlalchemy.orm import Session

app = FastAPI()


@app.get("/users")
def get_users(db: Session = Depends(get_db)):
    """ユーザー一覧取得（最適化前）"""
    users = db.query(User).all()

    # N+1 問題: 各ユーザーの投稿数を取得
    result = []
    for user in users:
        result.append({
            "id": user.id,
            "name": user.name,
            "posts_count": len(user.posts)  # 各ループでクエリ実行
        })

    return result
```

**After（速い）**:
```python
from fastapi import FastAPI
from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

app = FastAPI()


@app.get("/users")
def get_users(db: Session = Depends(get_db)):
    """ユーザー一覧取得（最適化後）"""
    # 1クエリで集計
    users = (
        db.query(
            User.id,
            User.name,
            func.count(Post.id).label('posts_count')
        )
        .outerjoin(Post)
        .group_by(User.id, User.name)
        .all()
    )

    return [
        {
            "id": user.id,
            "name": user.name,
            "posts_count": user.posts_count
        }
        for user in users
    ]


# さらにキャッシュを追加
from functools import lru_cache

@lru_cache(maxsize=1)
def get_users_cached(db: Session):
    # ...同じ処理
    pass
```

### ケース2: データ処理パイプライン最適化

**Before（遅い）**:
```python
import pandas as pd


def process_sales_data_slow(file_path: str) -> pd.DataFrame:
    """売上データ処理（最適化前）"""
    # ファイル全体を読み込み
    df = pd.read_csv(file_path)

    # iterrows で処理（遅い）
    results = []
    for index, row in df.iterrows():
        if row['amount'] > 1000:
            results.append({
                'date': row['date'],
                'total': row['amount'] * row['quantity'],
                'category': row['category']
            })

    return pd.DataFrame(results)
```

**After（速い）**:
```python
import pandas as pd


def process_sales_data_fast(file_path: str) -> pd.DataFrame:
    """売上データ処理（最適化後）"""
    # 必要なカラムのみ読み込み + 型指定
    df = pd.read_csv(
        file_path,
        usecols=['date', 'amount', 'quantity', 'category'],
        dtype={
            'amount': 'float32',
            'quantity': 'int32',
            'category': 'category'
        },
        parse_dates=['date']
    )

    # ベクトル化演算
    df = df[df['amount'] > 1000].copy()
    df['total'] = df['amount'] * df['quantity']

    return df[['date', 'total', 'category']]


# さらにチャンク処理を追加
def process_sales_data_chunks(file_path: str, chunk_size: int = 10000) -> pd.DataFrame:
    """大きなファイルをチャンクで処理"""
    chunks = []

    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # チャンクごとに処理
        processed = chunk[chunk['amount'] > 1000].copy()
        processed['total'] = processed['amount'] * processed['quantity']
        chunks.append(processed[['date', 'total', 'category']])

    return pd.concat(chunks, ignore_index=True)
```

### ケース3: 並列スクレイピング

**Before（遅い）**:
```python
import requests
from bs4 import BeautifulSoup


def scrape_pages_slow(urls: list[str]) -> list[dict]:
    """ページを逐次スクレイピング"""
    results = []

    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('h1').text
        results.append({'url': url, 'title': title})

    return results


# 100ページで約100秒
```

**After（速い）**:
```python
import asyncio
import aiohttp
from bs4 import BeautifulSoup


async def scrape_page(session: aiohttp.ClientSession, url: str) -> dict:
    """ページを非同期スクレイピング"""
    async with session.get(url) as response:
        content = await response.text()
        soup = BeautifulSoup(content, 'html.parser')
        title = soup.find('h1').text if soup.find('h1') else ''
        return {'url': url, 'title': title}


async def scrape_pages_fast(urls: list[str]) -> list[dict]:
    """複数ページを並列スクレイピング"""
    async with aiohttp.ClientSession() as session:
        tasks = [scrape_page(session, url) for url in urls]
        return await asyncio.gather(*tasks)


# 100ページで約2秒（50倍速い!）
```

---

## まとめ

### パフォーマンス最適化チェックリスト

**測定・プロファイリング**:
- [ ] time/timeit でベンチマーク
- [ ] cProfile で関数レベルのプロファイリング
- [ ] line_profiler で行レベルの分析
- [ ] memory_profiler でメモリ使用量を確認

**データ構造**:
- [ ] 検索は set/dict を使用（O(1)）
- [ ] 大量のインスタンスは __slots__ 使用
- [ ] collections モジュール活用
- [ ] ジェネレータでメモリ効率化

**アルゴリズム**:
- [ ] 計算量を意識（O(n²) → O(n)）
- [ ] ソートアルゴリズムは組み込み使用
- [ ] ジェネレータで遅延評価
- [ ] itertools で効率的な処理

**NumPy/Pandas**:
- [ ] ベクトル化（ループを避ける）
- [ ] iterrows() を避ける
- [ ] カテゴリ型でメモリ削減
- [ ] チャンクで大きなファイル処理

**並列・非同期**:
- [ ] CPU バウンド: multiprocessing
- [ ] I/O バウンド: asyncio/ThreadPoolExecutor
- [ ] concurrent.futures で簡単に並列化

**キャッシング**:
- [ ] functools.lru_cache でメモ化
- [ ] Redis で分散キャッシュ
- [ ] 適切な有効期限設定

**データベース**:
- [ ] N+1 問題を解決（joinedload）
- [ ] バルク挿入
- [ ] インデックス追加
- [ ] クエリ最適化

**高度な最適化**:
- [ ] Cython で C 拡張
- [ ] Numba で JIT コンパイル
- [ ] PyPy で実行速度向上

---

## パフォーマンス最適化のベストプラクティス

### 1. 計測してから最適化

```python
# まず計測
import cProfile

profiler = cProfile.Profile()
profiler.enable()
slow_function()
profiler.disable()

# ボトルネックを特定してから最適化
```

### 2. 段階的に最適化

1. **アルゴリズム**: まず計算量を改善
2. **データ構造**: 適切なデータ構造を選択
3. **ベクトル化**: NumPy/Pandas で最適化
4. **並列化**: CPU/I/O バウンドに応じて並列化
5. **コンパイル**: Cython/Numba で最後の最適化

### 3. 可読性とのバランス

```python
# ❌ 過度な最適化
result = sum(map(lambda x: x**2, filter(lambda x: x%2==0, data)))

# ✅ 読みやすく、それなりに速い
result = sum(x**2 for x in data if x % 2 == 0)
```

---

*計測・分析・最適化のサイクルで高速なPythonアプリケーションを構築しましょう。*
