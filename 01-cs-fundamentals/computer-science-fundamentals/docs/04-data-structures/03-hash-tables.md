# ハッシュテーブル

> ハッシュテーブルは「期待O(1)の探索」を実現する、実務で最も頻繁に使用されるデータ構造である。

## この章で学ぶこと

- [ ] ハッシュ関数と衝突解決の仕組みを理解する
- [ ] ハッシュテーブルの性能特性を説明できる
- [ ] 実務での使い分け（dict/set/Map等）を理解する
- [ ] ハッシュテーブルのセキュリティ上の考慮点を把握する
- [ ] 各言語におけるハッシュテーブルの内部実装を理解する
- [ ] 分散システムにおけるハッシュ技法を習得する

## 前提知識

- 配列 → 参照: [[00-arrays-and-strings.md]]
- 計算量解析 → 参照: [[../03-algorithms/01-complexity-analysis.md]]

---

## 1. ハッシュテーブルの仕組み

### 1.1 基本構造

```
ハッシュテーブル: キー → ハッシュ関数 → インデックス → 値

  キー "Alice" → hash("Alice") = 0x7A3B...
                → 0x7A3B % 8 = 3  (テーブルサイズ8)
                → table[3] = "Alice: 100"

  ┌─────┐
  │  0  │ → (空)
  │  1  │ → ("Bob", 85)
  │  2  │ → (空)
  │  3  │ → ("Alice", 100)
  │  4  │ → ("Charlie", 92) → ("Eve", 78)  ← 衝突(チェイニング)
  │  5  │ → (空)
  │  6  │ → ("Diana", 88)
  │  7  │ → (空)
  └─────┘
```

ハッシュテーブルの基本概念は非常にシンプルである。任意のキーをハッシュ関数で整数値に変換し、その整数値をテーブルサイズで割った余りをインデックスとして使用する。この仕組みにより、キーから値への直接的なアクセスが可能となり、期待計算量O(1)の探索が実現される。

### 1.2 ハッシュ関数の設計原理

良いハッシュ関数は以下の性質を満たす必要がある。

```
良いハッシュ関数の条件:

  1. 決定性: 同じ入力に対して常に同じ出力を返す
  2. 均一分布: 出力がハッシュ空間に均等に分布する
  3. 効率性: 計算が高速である（O(1)に近い）
  4. 雪崩効果: 入力の1ビットの変化が出力の約半数のビットを変える

主要なハッシュ関数:

  Division Method:
    h(k) = k mod m
    mは2の冪を避ける（素数が望ましい）
    例: m = 997（素数）

  Multiplication Method:
    h(k) = floor(m × (k × A mod 1))
    A = (√5 - 1) / 2 ≈ 0.6180339887（黄金比）
    mが2の冪でも良好に動作

  Universal Hashing:
    h(k) = ((a × k + b) mod p) mod m
    a, bをランダムに選択
    衝突確率が1/mに制限される

  MurmurHash3:
    高速な非暗号ハッシュ関数
    Redis, Hadoop, Spark等で使用

  SipHash:
    Hash DoS耐性のあるPRF
    Python 3.4+, Rust, Perl で標準使用
```

### 1.3 文字列のハッシュ

```python
# 文字列のハッシュ関数の例

# 1. 単純な方法（均一性が低い）
def bad_hash(s, m):
    """各文字のASCIIコードの合計"""
    return sum(ord(c) for c in s) % m
# "abc" と "bca" が同じハッシュ値 → アナグラムが衝突

# 2. 多項式ハッシュ（一般的）
def polynomial_hash(s, m, base=31):
    """多項式ハッシュ: s[0]*base^(n-1) + s[1]*base^(n-2) + ... + s[n-1]"""
    h = 0
    for c in s:
        h = (h * base + ord(c)) % m
    return h
# "abc" → 31^2*97 + 31*98 + 99 = 96262
# "bca" → 31^2*98 + 31*99 + 97 = 97168  → 異なるハッシュ値

# 3. Java の String.hashCode() と同等
def java_string_hash(s):
    """s[0]*31^(n-1) + s[1]*31^(n-2) + ... + s[n-1]"""
    h = 0
    for c in s:
        h = h * 31 + ord(c)
        h &= 0xFFFFFFFF  # 32ビットに制限
    return h

# 4. FNV-1a ハッシュ（高速・均一分布）
def fnv1a_hash(data, m):
    """FNV-1a: offset_basis XOR byte → multiply by prime"""
    FNV_OFFSET = 0xcbf29ce484222325
    FNV_PRIME = 0x100000001b3
    h = FNV_OFFSET
    for byte in data.encode():
        h ^= byte
        h *= FNV_PRIME
        h &= 0xFFFFFFFFFFFFFFFF  # 64ビットに制限
    return h % m
```

### 1.4 衝突解決

```
衝突解決の2大方式:

  1. チェイニング（Separate Chaining）:
     → 同じスロットにリンクリストで格納
     → 実装が単純、ロードファクターが1を超えてもOK
     → Java HashMap, Go map

  2. オープンアドレス法:
     → 衝突時に別のスロットを探す
     → メモリ効率が良い、キャッシュフレンドリー
     → Python dict, Rust HashMap

     探索方法:
     - 線形探索: h(k)+1, h(k)+2, ...（クラスタリング問題）
     - 二次探索: h(k)+1², h(k)+2², ...
     - ダブルハッシュ: h(k)+i×h2(k)

  ロードファクター = 要素数 / テーブルサイズ
  → チェイニング: 0.75で拡張（Java）
  → オープンアドレス: 2/3で拡張（Python）
```

### 1.5 チェイニングの詳細実装

```python
class HashTableChaining:
    """チェイニング方式のハッシュテーブル"""

    def __init__(self, capacity=16, load_factor_threshold=0.75):
        self.capacity = capacity
        self.load_factor_threshold = load_factor_threshold
        self.size = 0
        self.buckets = [[] for _ in range(capacity)]

    def _hash(self, key):
        """ハッシュ値を計算"""
        return hash(key) % self.capacity

    def put(self, key, value):
        """キーと値のペアを挿入"""
        idx = self._hash(key)
        bucket = self.buckets[idx]

        # 既存キーの更新
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return

        # 新規挿入
        bucket.append((key, value))
        self.size += 1

        # ロードファクターチェック
        if self.size / self.capacity > self.load_factor_threshold:
            self._resize()

    def get(self, key, default=None):
        """キーから値を取得"""
        idx = self._hash(key)
        for k, v in self.buckets[idx]:
            if k == key:
                return v
        return default

    def remove(self, key):
        """キーを削除"""
        idx = self._hash(key)
        bucket = self.buckets[idx]
        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket.pop(i)
                self.size -= 1
                return True
        return False

    def __contains__(self, key):
        """in 演算子のサポート"""
        return self.get(key) is not None

    def _resize(self):
        """テーブルサイズを2倍に拡張"""
        old_buckets = self.buckets
        self.capacity *= 2
        self.buckets = [[] for _ in range(self.capacity)]
        self.size = 0
        for bucket in old_buckets:
            for key, value in bucket:
                self.put(key, value)

    def load_factor(self):
        """現在のロードファクターを返す"""
        return self.size / self.capacity

    def __len__(self):
        return self.size

    def __repr__(self):
        items = []
        for bucket in self.buckets:
            for k, v in bucket:
                items.append(f"{k!r}: {v!r}")
        return "{" + ", ".join(items) + "}"


# 使用例
ht = HashTableChaining()
ht.put("name", "Alice")
ht.put("age", 30)
ht.put("email", "alice@example.com")
print(ht.get("name"))   # "Alice"
print("age" in ht)      # True
print(ht.load_factor()) # 0.1875 (3/16)
ht.remove("age")
print(len(ht))          # 2
```

### 1.6 オープンアドレス法の詳細実装

```python
class HashTableOpenAddressing:
    """オープンアドレス法（線形探索）のハッシュテーブル"""

    EMPTY = object()    # 空スロットの番兵
    DELETED = object()  # 削除済みスロットの番兵（墓石）

    def __init__(self, capacity=16, load_factor_threshold=0.67):
        self.capacity = capacity
        self.load_factor_threshold = load_factor_threshold
        self.size = 0
        self.keys = [self.EMPTY] * capacity
        self.values = [None] * capacity

    def _hash(self, key):
        return hash(key) % self.capacity

    def _probe(self, key):
        """線形探索でスロットを探す"""
        idx = self._hash(key)
        first_deleted = None

        for i in range(self.capacity):
            pos = (idx + i) % self.capacity

            if self.keys[pos] is self.EMPTY:
                # 空スロットに到達 → キーは存在しない
                return (first_deleted if first_deleted is not None else pos, False)

            if self.keys[pos] is self.DELETED:
                if first_deleted is None:
                    first_deleted = pos
                continue

            if self.keys[pos] == key:
                return (pos, True)  # キーが見つかった

        # テーブルが満杯（通常はリサイズで防止）
        return (first_deleted if first_deleted is not None else -1, False)

    def put(self, key, value):
        pos, found = self._probe(key)
        if found:
            self.values[pos] = value  # 更新
        else:
            self.keys[pos] = key
            self.values[pos] = value
            self.size += 1
            if self.size / self.capacity > self.load_factor_threshold:
                self._resize()

    def get(self, key, default=None):
        pos, found = self._probe(key)
        if found:
            return self.values[pos]
        return default

    def remove(self, key):
        """削除: 墓石（DELETED）マーカーを配置"""
        pos, found = self._probe(key)
        if found:
            self.keys[pos] = self.DELETED
            self.values[pos] = None
            self.size -= 1
            return True
        return False

    def _resize(self):
        old_keys = self.keys
        old_values = self.values
        self.capacity *= 2
        self.keys = [self.EMPTY] * self.capacity
        self.values = [None] * self.capacity
        self.size = 0
        for k, v in zip(old_keys, old_values):
            if k is not self.EMPTY and k is not self.DELETED:
                self.put(k, v)


# 線形探索のクラスタリング問題の可視化
# スロット: [A][B][C][_][_][E][F][G][_][_]
#            ^^^^^^^^          ^^^^^^^^
#            クラスタ1          クラスタ2
# 新しい要素がクラスタ近くにハッシュされると、クラスタが成長
# → 探索時間が O(1) から O(n) に劣化
```

### 1.7 Robin Hood ハッシング

```python
class RobinHoodHashTable:
    """Robin Hood ハッシング: 挿入時に「貧しい」要素を優先する"""

    EMPTY = None

    def __init__(self, capacity=16):
        self.capacity = capacity
        self.size = 0
        self.keys = [self.EMPTY] * capacity
        self.values = [None] * capacity
        self.distances = [0] * capacity  # 理想位置からの距離

    def _hash(self, key):
        return hash(key) % self.capacity

    def put(self, key, value):
        if self.size / self.capacity > 0.8:
            self._resize()

        idx = self._hash(key)
        dist = 0

        while True:
            pos = (idx + dist) % self.capacity

            if self.keys[pos] is self.EMPTY:
                # 空スロットに挿入
                self.keys[pos] = key
                self.values[pos] = value
                self.distances[pos] = dist
                self.size += 1
                return

            if self.keys[pos] == key:
                # 既存キーの更新
                self.values[pos] = value
                return

            # Robin Hood: 現在の要素より探索距離が短い場合、交換
            if self.distances[pos] < dist:
                # 「裕福な」要素（短い距離）を追い出す
                key, self.keys[pos] = self.keys[pos], key
                value, self.values[pos] = self.values[pos], value
                dist, self.distances[pos] = self.distances[pos], dist

            dist += 1

    def get(self, key, default=None):
        idx = self._hash(key)
        dist = 0

        while True:
            pos = (idx + dist) % self.capacity

            if self.keys[pos] is self.EMPTY:
                return default

            if self.distances[pos] < dist:
                # この位置に到達する前に見つかるはず
                return default

            if self.keys[pos] == key:
                return self.values[pos]

            dist += 1

    def _resize(self):
        old_keys = self.keys
        old_values = self.values
        self.capacity *= 2
        self.keys = [self.EMPTY] * self.capacity
        self.values = [None] * self.capacity
        self.distances = [0] * self.capacity
        self.size = 0
        for k, v in zip(old_keys, old_values):
            if k is not self.EMPTY:
                self.put(k, v)


# Robin Hood の利点:
# - 最悪ケースの探索距離が O(log log n) に改善
# - 探索距離の分散が小さい（均一に近い）
# - Rust の HashMap が採用（2015-2021、その後 hashbrown/Swiss Table に移行）
```

### 1.8 計算量

```
ハッシュテーブルの計算量:

  ┌──────────┬──────────┬──────────┐
  │ 操作     │ 期待     │ 最悪     │
  ├──────────┼──────────┼──────────┤
  │ 挿入     │ O(1)     │ O(n)     │
  │ 検索     │ O(1)     │ O(n)     │
  │ 削除     │ O(1)     │ O(n)     │
  │ リサイズ │ O(n)     │ O(n)     │
  └──────────┴──────────┴──────────┘

  最悪ケース O(n) の条件:
  - 全てのキーが同じスロットに衝突
  - 意図的な攻撃（Hash DoS）

  対策:
  - ランダム化ハッシュ（SipHash: Python, Rust）
  - 赤黒木へのフォールバック（Java 8+ HashMap）

  リサイズの償却分析:
  - テーブルサイズが n のとき、リサイズコストは O(n)
  - リサイズ間に少なくとも n/2 回の挿入が発生
  - 償却コスト = O(n) / (n/2) = O(1)
  - つまり、個々の挿入の償却計算量は O(1)
```

---

## 2. 各言語のハッシュテーブル内部実装

### 2.1 Python dict の内部実装

```python
# Python 3.6+ の dict はコンパクト辞書（順序保持）
# 内部構造:
#   - indices: ハッシュテーブル（インデックスの配列）
#   - entries: (hash, key, value) のコンパクト配列

# 実装の概念図:
# indices = [None, 1, None, None, 0, None, None, 2]
#                  ↓                ↓              ↓
# entries = [(hash_a, "age", 30),      # index 0
#            (hash_n, "name", "Alice"), # index 1
#            (hash_e, "email", "x@y")] # index 2

# メリット:
# 1. メモリ効率: indicesは1バイトエントリ（要素数 < 256の場合）
# 2. 順序保持: entries配列は挿入順
# 3. イテレーション高速化: entries配列を順に走査

# Python dict の特殊な最適化
d = {}

# 1. Key-Sharing辞書（同じクラスのインスタンス間でキーを共有）
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# p1.__dict__ と p2.__dict__ はキー配列を共有
# → メモリ節約（特にクラスのインスタンスが大量にある場合）

# 2. 文字列の特別扱い
# 短い文字列は intern される（同一オブジェクトを再利用）
a = "hello"
b = "hello"
print(a is b)  # True（同一オブジェクト）

# 3. __hash__と__eq__のプロトコル
class CustomKey:
    def __init__(self, value):
        self.value = value

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return isinstance(other, CustomKey) and self.value == other.value

# ミュータブルなオブジェクトはハッシュ不可
# list, dict, set はキーにできない（TypeError）
```

### 2.2 Java HashMap の内部実装

```java
// Java 8+ HashMap の内部構造
// - 初期容量: 16
// - ロードファクター: 0.75
// - チェイニング方式
// - バケット内要素数 >= 8 で赤黒木に変換（Treeification）
// - バケット内要素数 <= 6 でリンクリストに戻す

// HashMap の put メソッドの流れ
public V put(K key, V value) {
    // 1. key の hashCode() を取得
    int hash = hash(key);

    // 2. 上位ビットを下位ビットに混ぜる（spread）
    // static int hash(Object key) {
    //     int h;
    //     return (key == null) ? 0 : (h = key.hashCode()) ^ (h >>> 16);
    // }

    // 3. index = hash & (capacity - 1)
    //    capacity が 2の冪なので、ビットANDで高速にmod

    // 4. バケットに挿入
    //    - 空 → 新規ノード
    //    - リンクリスト → 線形探索で末尾に追加
    //    - 赤黒木 → 木に挿入

    // 5. サイズ > threshold (capacity * loadFactor) ならリサイズ
    return putVal(hash, key, value, false, true);
}

// Java 8 以降の Treeification（赤黒木化）
// リンクリスト O(n) → 赤黒木 O(log n)
// Hash DoS 攻撃への対策

// ConcurrentHashMap（スレッドセーフ版）
// - Java 8+: CAS + synchronized（セグメントロック廃止）
// - 読み取りはロックフリー（volatile読み取り）
// - 書き込みはバケット単位のsynchronized

import java.util.concurrent.ConcurrentHashMap;
ConcurrentHashMap<String, Integer> map = new ConcurrentHashMap<>();
map.put("key", 42);
map.computeIfAbsent("key2", k -> expensiveComputation(k));
```

### 2.3 Go map の内部実装

```go
// Go の map は runtime パッケージで実装
// - バケット方式（各バケットに8つのキー/値ペアを格納）
// - 増分リサイズ（incremental resizing）
// - ランダム化されたイテレーション順序

package main

import "fmt"

func main() {
    // 基本操作
    m := make(map[string]int)
    m["Alice"] = 100
    m["Bob"] = 85

    // 存在確認（2値返し）
    if v, ok := m["Alice"]; ok {
        fmt.Printf("Alice: %d\n", v)
    }

    // 削除
    delete(m, "Bob")

    // イテレーション（順序は非決定的）
    for k, v := range m {
        fmt.Printf("%s: %d\n", k, v)
    }
}

// Go map の内部構造:
// type hmap struct {
//     count     int     // 要素数
//     flags     uint8
//     B         uint8   // バケット数 = 2^B
//     noverflow uint16  // オーバーフローバケット数
//     hash0     uint32  // ハッシュシード（ランダム化）
//     buckets   unsafe.Pointer
//     oldbuckets unsafe.Pointer // リサイズ中の旧バケット
//     ...
// }

// Go map はスレッドセーフではない
// 並行アクセスには sync.Map を使用
// import "sync"
// var m sync.Map
// m.Store("key", "value")
// v, ok := m.Load("key")
```

### 2.4 Rust HashMap の内部実装

```rust
use std::collections::HashMap;

fn main() {
    // Rust HashMap は hashbrown クレート（Swiss Table ベース）
    // - SipHash-1-3 をデフォルトハッシュ関数として使用（Hash DoS 耐性）
    // - SIMD を活用した高速探索

    let mut map = HashMap::new();
    map.insert("Alice", 100);
    map.insert("Bob", 85);

    // パターンマッチングで安全にアクセス
    match map.get("Alice") {
        Some(&score) => println!("Alice: {}", score),
        None => println!("Not found"),
    }

    // Entry API（存在しない場合のみ挿入）
    map.entry("Charlie").or_insert(90);

    // Entry API（既存値の更新）
    let count = map.entry("Alice").or_insert(0);
    *count += 10; // Alice: 110

    // イテレーション
    for (key, value) in &map {
        println!("{}: {}", key, value);
    }
}

// カスタムハッシュ関数の使用（高速だがDoS非耐性）
// use std::collections::HashMap;
// use std::hash::BuildHasherDefault;
// use ahash::AHasher;
//
// type AHashMap<K, V> = HashMap<K, V, BuildHasherDefault<AHasher>>;
// let mut map: AHashMap<String, i32> = AHashMap::default();
```

---

## 3. 実務での活用

### 3.1 各言語のハッシュテーブル

```python
# Python: dict（最も使用頻度が高いデータ構造）
d = {"name": "Alice", "age": 30}
d["email"] = "alice@example.com"  # O(1)
"name" in d  # O(1)

# Python: set（重複排除、集合演算）
s = {1, 2, 3}
s.add(4)          # O(1)
2 in s            # O(1)
s & {2, 3, 4}     # 積集合: {2, 3}
s | {4, 5}        # 和集合: {1, 2, 3, 4, 5}

# Counter（頻度カウント）
from collections import Counter
words = ["apple", "banana", "apple", "cherry", "banana", "apple"]
count = Counter(words)
# Counter({'apple': 3, 'banana': 2, 'cherry': 1})
count.most_common(2)  # [('apple', 3), ('banana', 2)]

# defaultdict（デフォルト値付き辞書）
from collections import defaultdict
graph = defaultdict(list)
graph["A"].append("B")  # KeyError なし
```

### 3.2 ハッシュテーブルの設計パターン

```python
# パターン1: メモ化（キャッシュ）
cache = {}
def expensive_computation(key):
    if key not in cache:
        cache[key] = compute(key)  # 初回のみ計算
    return cache[key]

# functools.lru_cache を使ったメモ化
from functools import lru_cache

@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# パターン2: グルーピング
from collections import defaultdict
def group_anagrams(words):
    groups = defaultdict(list)
    for word in words:
        key = tuple(sorted(word))
        groups[key].append(word)
    return list(groups.values())

# 使用例
print(group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
# [['eat', 'tea', 'ate'], ['tan', 'nat'], ['bat']]

# パターン3: Two Sum（ハッシュマップで O(n)）
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i

# パターン4: スライディングウィンドウ + ハッシュマップ
def length_of_longest_substring(s):
    """重複のない最長部分文字列の長さ"""
    char_index = {}
    max_len = 0
    start = 0

    for i, c in enumerate(s):
        if c in char_index and char_index[c] >= start:
            start = char_index[c] + 1
        char_index[c] = i
        max_len = max(max_len, i - start + 1)

    return max_len

# 使用例
print(length_of_longest_substring("abcabcbb"))  # 3 ("abc")
print(length_of_longest_substring("bbbbb"))     # 1 ("b")
print(length_of_longest_substring("pwwkew"))    # 3 ("wke")

# パターン5: 頻度マップによる判定
def is_anagram(s, t):
    """2つの文字列がアナグラムかどうか"""
    if len(s) != len(t):
        return False
    count = {}
    for c in s:
        count[c] = count.get(c, 0) + 1
    for c in t:
        count[c] = count.get(c, 0) - 1
        if count[c] < 0:
            return False
    return True

# パターン6: ハッシュマップによるグラフの表現
def build_graph(edges):
    """エッジリストから隣接リストを構築"""
    graph = defaultdict(list)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    return graph

edges = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D")]
graph = build_graph(edges)
# {'A': ['B', 'C'], 'B': ['A', 'C'], 'C': ['A', 'B', 'D'], 'D': ['C']}

# パターン7: 文字列の圧縮/展開
def word_pattern(pattern, s):
    """パターンと文字列の対応を検証（bijection）"""
    words = s.split()
    if len(pattern) != len(words):
        return False

    char_to_word = {}
    word_to_char = {}

    for c, w in zip(pattern, words):
        if c in char_to_word:
            if char_to_word[c] != w:
                return False
        else:
            if w in word_to_char:
                return False
            char_to_word[c] = w
            word_to_char[w] = c

    return True

print(word_pattern("abba", "dog cat cat dog"))  # True
print(word_pattern("abba", "dog cat cat fish")) # False
```

### 3.3 JavaScript/TypeScript でのハッシュテーブル

```typescript
// Map（順序保持、任意のキー型）
const map = new Map<string, number>();
map.set("Alice", 100);
map.set("Bob", 85);
console.log(map.get("Alice")); // 100
console.log(map.has("Charlie")); // false
console.log(map.size); // 2

// Object vs Map の使い分け
// Object: キーが文字列/Symbol のみ、プロトタイプチェーンあり
// Map: 任意のキー型、サイズ取得 O(1)、イテレーション順保証

// WeakMap（GC対応、キーがオブジェクトのみ）
const weakMap = new WeakMap<object, string>();
let obj = { id: 1 };
weakMap.set(obj, "metadata");
// obj への参照がなくなると、WeakMap のエントリも GC される

// Set
const set = new Set<number>([1, 2, 3, 2, 1]);
console.log(set.size); // 3（重複排除）
set.add(4);
set.delete(2);
console.log([...set]); // [1, 3, 4]

// 実務例: API レスポンスのキャッシュ
class APICache {
    private cache: Map<string, { data: unknown; timestamp: number }>;
    private ttlMs: number;

    constructor(ttlMs: number = 60_000) {
        this.cache = new Map();
        this.ttlMs = ttlMs;
    }

    get(key: string): unknown | null {
        const entry = this.cache.get(key);
        if (!entry) return null;

        if (Date.now() - entry.timestamp > this.ttlMs) {
            this.cache.delete(key);
            return null;
        }

        return entry.data;
    }

    set(key: string, data: unknown): void {
        this.cache.set(key, { data, timestamp: Date.now() });
    }

    clear(): void {
        this.cache.clear();
    }
}
```

---

## 4. LRU キャッシュの実装

### 4.1 OrderedDict を使った実装

```python
from collections import OrderedDict

class LRUCache:
    """Least Recently Used キャッシュ"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key not in self.cache:
            return -1
        # アクセスされたキーを末尾に移動（最近使用）
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            # 先頭（最も古い）を削除
            self.cache.popitem(last=False)

# 使用例
cache = LRUCache(3)
cache.put("a", 1)
cache.put("b", 2)
cache.put("c", 3)
print(cache.get("a"))  # 1（"a"が最近使用に）
cache.put("d", 4)      # "b"が削除される（最も古い）
print(cache.get("b"))   # -1（削除済み）
```

### 4.2 ハッシュマップ + 双方向連結リストでの実装

```python
class DLinkedNode:
    """双方向連結リストのノード"""
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCacheManual:
    """双方向連結リスト + ハッシュマップによる LRU キャッシュ"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.cache = {}  # key -> DLinkedNode

        # 番兵ノード（実装を簡潔にする）
        self.head = DLinkedNode()  # ダミーヘッド（最近使用側）
        self.tail = DLinkedNode()  # ダミーテール（最も古い側）
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        node = self.cache[key]
        self._move_to_head(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._move_to_head(node)
        else:
            node = DLinkedNode(key, value)
            self.cache[key] = node
            self._add_node(node)
            self.size += 1

            if self.size > self.capacity:
                tail = self._pop_tail()
                del self.cache[tail.key]
                self.size -= 1

    def _add_node(self, node):
        """ヘッドの直後に追加"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def _remove_node(self, node):
        """ノードを連結リストから削除"""
        prev = node.prev
        nxt = node.next
        prev.next = nxt
        nxt.prev = prev

    def _move_to_head(self, node):
        """ノードをヘッド直後に移動"""
        self._remove_node(node)
        self._add_node(node)

    def _pop_tail(self):
        """テールの直前（最も古い）を削除"""
        node = self.tail.prev
        self._remove_node(node)
        return node

# LRU キャッシュの計算量:
# get: O(1) - ハッシュマップ参照 + 連結リスト操作
# put: O(1) - ハッシュマップ挿入 + 連結リスト操作
# 空間: O(capacity)
```

### 4.3 TTL（Time-To-Live）付きキャッシュ

```python
import time
import threading
from collections import OrderedDict

class TTLCache:
    """有効期限付きキャッシュ"""

    def __init__(self, capacity: int, ttl_seconds: float):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self.cache = OrderedDict()  # key -> (value, expire_time)
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None

            value, expire_time = self.cache[key]

            if time.time() > expire_time:
                # 期限切れ
                del self.cache[key]
                return None

            # LRU: 最近使用に移動
            self.cache.move_to_end(key)
            return value

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)

            self.cache[key] = (value, time.time() + self.ttl)

            # 容量制限
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

    def cleanup(self):
        """期限切れエントリを一括削除"""
        with self.lock:
            now = time.time()
            expired_keys = [
                k for k, (_, exp) in self.cache.items()
                if now > exp
            ]
            for k in expired_keys:
                del self.cache[k]

# 使用例
cache = TTLCache(capacity=100, ttl_seconds=300)  # 5分間キャッシュ
cache.put("user:123", {"name": "Alice", "age": 30})
user = cache.get("user:123")  # 5分以内なら取得可能
```

---

## 5. 一貫性ハッシュ（Consistent Hashing）

### 5.1 基本概念

```
一貫性ハッシュ: 分散システムでのデータ分散

  従来のハッシュ:
    server = hash(key) % num_servers
    → サーバー追加/削除時に全データの再配置が必要
    → サーバーが3台→4台: 75%のデータが移動

  一貫性ハッシュ:
    サーバーとキーを同じハッシュリング上に配置
    → サーバー追加/削除時に移動するデータは1/N程度

    ハッシュリング（0 ~ 2^32-1）:
                   0
                 /   \
               S1     S3
              /         \
            K1           K4
           |               |
          K2               S2
            \             /
             K3         K5
               \       /
                S4---K6
                 2^32

    K1 → 時計回りで最初のサーバー S1 に割り当て
    K4 → S2
    K5 → S2
    K6 → S4

    S2 がダウン → K4, K5 は S3 に移動（K1, K2, K3, K6 は影響なし）
```

### 5.2 仮想ノードを使った実装

```python
import hashlib
from bisect import bisect_right

class ConsistentHash:
    """仮想ノード付き一貫性ハッシュ"""

    def __init__(self, num_replicas=150):
        self.num_replicas = num_replicas  # 仮想ノード数
        self.ring = {}       # hash -> node
        self.sorted_keys = []  # ソート済みハッシュ値

    def _hash(self, key: str) -> int:
        """MD5ハッシュで均一分布を実現"""
        digest = hashlib.md5(key.encode()).hexdigest()
        return int(digest, 16)

    def add_node(self, node: str):
        """ノード（サーバー）を追加"""
        for i in range(self.num_replicas):
            virtual_key = f"{node}:{i}"
            h = self._hash(virtual_key)
            self.ring[h] = node
            self.sorted_keys.append(h)
        self.sorted_keys.sort()

    def remove_node(self, node: str):
        """ノードを削除"""
        for i in range(self.num_replicas):
            virtual_key = f"{node}:{i}"
            h = self._hash(virtual_key)
            del self.ring[h]
            self.sorted_keys.remove(h)

    def get_node(self, key: str) -> str:
        """キーが割り当てられるノードを取得"""
        if not self.ring:
            return None

        h = self._hash(key)
        idx = bisect_right(self.sorted_keys, h)

        # リングの末尾を超えたら先頭に戻る
        if idx == len(self.sorted_keys):
            idx = 0

        return self.ring[self.sorted_keys[idx]]

    def get_nodes(self, key: str, n: int = 3) -> list:
        """レプリカ用: キーに対する n 個の異なるノードを取得"""
        if not self.ring or n > len(set(self.ring.values())):
            return list(set(self.ring.values()))

        h = self._hash(key)
        idx = bisect_right(self.sorted_keys, h)

        result = []
        seen = set()

        while len(result) < n:
            if idx == len(self.sorted_keys):
                idx = 0
            node = self.ring[self.sorted_keys[idx]]
            if node not in seen:
                result.append(node)
                seen.add(node)
            idx += 1

        return result


# 使用例: キャッシュサーバーの分散
ch = ConsistentHash(num_replicas=150)
ch.add_node("cache-server-1")
ch.add_node("cache-server-2")
ch.add_node("cache-server-3")

# キーの割り当て
print(ch.get_node("user:1001"))    # "cache-server-2"
print(ch.get_node("session:abc"))  # "cache-server-1"

# サーバー追加（影響は最小限）
ch.add_node("cache-server-4")
print(ch.get_node("user:1001"))    # 多くは同じサーバーに留まる

# レプリカの取得
print(ch.get_nodes("user:1001", n=2))  # 2つの異なるサーバー
```

### 5.3 一貫性ハッシュの実用例

```
一貫性ハッシュの実用例:

  1. Amazon DynamoDB:
     - キーをパーティションに分散
     - 仮想ノードでデータの偏りを緩和
     - 優先リストでレプリカ先を決定

  2. Apache Cassandra:
     - パーティションキーのハッシュでノードを決定
     - Murmur3Partitioner がデフォルト
     - vnodes（仮想ノード）で負荷分散

  3. Memcached / Redis Cluster:
     - クライアント側で一貫性ハッシュを実装
     - サーバーの追加/削除時のキャッシュミスを最小化

  4. CDN（Akamai等）:
     - リクエストURLをハッシュしてエッジサーバーに振り分け
     - サーバー障害時の影響を局所化

  5. ロードバランサー:
     - セッションアフィニティ（同じユーザーを同じサーバーに）
     - Nginx の upstream hash モジュール
```

---

## 6. Hash DoS 攻撃と対策

### 6.1 攻撃の原理

```
Hash DoS（ハッシュ衝突攻撃）:

  原理:
  - ハッシュ関数の出力が予測可能な場合
  - 攻撃者が同じバケットに衝突するキーを大量に生成
  - ハッシュテーブルの操作が O(1) → O(n) に劣化
  - n個のキーで O(n²) の処理時間

  歴史:
  - 2003年: Perl のハッシュ衝突攻撃が報告
  - 2011年: CCC（Chaos Communication Congress）で
            PHP, Python, Ruby, Java等への攻撃が公開
  - JSON/XMLリクエストに衝突するキーを大量に含めて送信
  - 小さなリクエストでサーバーのCPUを100%使用可能

  具体例（Python 3.2以前）:
  # 以下のキーは同じハッシュ値を持つように構成
  # {"key1": 1, "key2": 2, ..., "key100000": 100000}
  # → 挿入に O(n²) 時間がかかり、サーバーがハング
```

### 6.2 対策

```python
# 対策1: ランダム化ハッシュ（Python 3.3+）
# PYTHONHASHSEED 環境変数でシードを制御
# デフォルトはプロセス起動時にランダムシード
import sys
print(sys.hash_info)
# sys.hash_info(width=64, modulus=2305843009213693951,
#               inf=314159, nan=0, imag=1000003,
#               algorithm='siphash24', ...)

# 対策2: SipHash（暗号学的PRF）
# Python 3.4+, Rust, Perl 5.18+ で標準採用
# 128ビットの秘密鍵を使用
# Hash DoS に対して安全、かつ十分高速

# 対策3: 赤黒木へのフォールバック（Java 8+）
# HashMap のバケット内要素が8を超えると
# リンクリスト → 赤黒木に変換
# 最悪ケースが O(n) → O(log n) に改善

# 対策4: リクエスト制限
# Webアプリケーションレベルでの対策
# - リクエストサイズの制限
# - JSONパラメータ数の上限設定
# - レート制限

# 対策5: カスタムハッシュ関数の使用
import hmac
import hashlib

def secure_hash(key: str, secret: bytes) -> int:
    """HMAC ベースのセキュアハッシュ"""
    h = hmac.new(secret, key.encode(), hashlib.sha256)
    return int.from_bytes(h.digest()[:8], 'big')
```

---

## 7. ハッシュテーブルの応用データ構造

### 7.1 カウンティングブルームフィルター

```python
class CountingBloomFilter:
    """削除をサポートするBloom Filter"""

    def __init__(self, size, num_hashes):
        self.size = size
        self.num_hashes = num_hashes
        self.counters = [0] * size  # ビットではなくカウンター

    def _hashes(self, item):
        """k個のハッシュ値を生成"""
        import hashlib
        indices = []
        for i in range(self.num_hashes):
            h = hashlib.sha256(f"{item}:{i}".encode()).hexdigest()
            indices.append(int(h, 16) % self.size)
        return indices

    def add(self, item):
        for idx in self._hashes(item):
            self.counters[idx] += 1

    def remove(self, item):
        """標準Bloom Filterでは不可能な削除操作"""
        indices = self._hashes(item)
        if all(self.counters[idx] > 0 for idx in indices):
            for idx in indices:
                self.counters[idx] -= 1

    def contains(self, item):
        return all(self.counters[idx] > 0 for idx in self._hashes(item))
```

### 7.2 Cuckoo ハッシング

```python
class CuckooHashTable:
    """Cuckoo ハッシング: 2つのハッシュ関数で最悪 O(1) 検索"""

    MAX_KICKS = 500  # 最大追い出し回数

    def __init__(self, capacity=16):
        self.capacity = capacity
        self.table1 = [None] * capacity
        self.table2 = [None] * capacity
        self.size = 0

    def _hash1(self, key):
        return hash(key) % self.capacity

    def _hash2(self, key):
        return hash(key * 2654435761) % self.capacity  # 別のハッシュ

    def get(self, key):
        """O(1) 最悪ケース検索"""
        pos1 = self._hash1(key)
        if self.table1[pos1] and self.table1[pos1][0] == key:
            return self.table1[pos1][1]

        pos2 = self._hash2(key)
        if self.table2[pos2] and self.table2[pos2][0] == key:
            return self.table2[pos2][1]

        return None  # 見つからない

    def put(self, key, value):
        """挿入（Cuckoo 方式で追い出し）"""
        # 既存キーの更新チェック
        pos1 = self._hash1(key)
        if self.table1[pos1] and self.table1[pos1][0] == key:
            self.table1[pos1] = (key, value)
            return

        pos2 = self._hash2(key)
        if self.table2[pos2] and self.table2[pos2][0] == key:
            self.table2[pos2] = (key, value)
            return

        # 新規挿入
        current = (key, value)
        for _ in range(self.MAX_KICKS):
            # table1 に挿入を試みる
            pos = self._hash1(current[0])
            if self.table1[pos] is None:
                self.table1[pos] = current
                self.size += 1
                return

            # 追い出し
            current, self.table1[pos] = self.table1[pos], current

            # table2 に挿入を試みる
            pos = self._hash2(current[0])
            if self.table2[pos] is None:
                self.table2[pos] = current
                self.size += 1
                return

            # 追い出し
            current, self.table2[pos] = self.table2[pos], current

        # 追い出しループ → リサイズが必要
        self._resize()
        self.put(current[0], current[1])

    def _resize(self):
        old_items = []
        for entry in self.table1:
            if entry:
                old_items.append(entry)
        for entry in self.table2:
            if entry:
                old_items.append(entry)

        self.capacity *= 2
        self.table1 = [None] * self.capacity
        self.table2 = [None] * self.capacity
        self.size = 0

        for key, value in old_items:
            self.put(key, value)


# Cuckoo ハッシングの利点:
# - 検索が最悪 O(1)（2箇所を見るだけ）
# - Cuckoo Filter（Bloom Filter の代替）の基盤
# - 挿入の期待計算量は O(1)（追い出し回数の期待値は定数）
```

### 7.3 Swiss Table（Google の高性能ハッシュテーブル）

```
Swiss Table の仕組み:

  Rust (hashbrown), C++ (Abseil), Go の map が採用

  構造:
  ┌──────────────┐
  │ Control Bytes │  16バイトグループ（SIMD で一括比較）
  │ [H2|H2|..|H2]│  H2 = ハッシュの上位7ビット
  ├──────────────┤
  │   Slots      │  実際のキー/値ペア
  │ [KV|KV|..|KV]│
  └──────────────┘

  検索の流れ:
  1. H1 = hash の下位ビット → グループのインデックス
  2. H2 = hash の上位7ビット
  3. SIMD命令で16個のcontrol byteを一括比較
  4. マッチしたスロットのみキーを比較

  利点:
  - SIMD で 16 スロットを同時に比較
  - キャッシュライン効率が高い
  - 従来のオープンアドレス法より 2-3x 高速
  - メモリオーバーヘッドは 1バイト/エントリ のみ
```

---

## 8. ハッシュテーブルのベンチマークと性能比較

### 8.1 Python での性能測定

```python
import time
import random
import string

def benchmark_dict_operations(n):
    """dict の各操作のベンチマーク"""

    # ランダムキー生成
    keys = [''.join(random.choices(string.ascii_lowercase, k=10)) for _ in range(n)]

    # 挿入
    d = {}
    start = time.perf_counter()
    for k in keys:
        d[k] = random.randint(0, 1000000)
    insert_time = time.perf_counter() - start

    # 検索（存在するキー）
    start = time.perf_counter()
    for k in keys:
        _ = d[k]
    search_hit_time = time.perf_counter() - start

    # 検索（存在しないキー）
    missing = [k + "x" for k in keys]
    start = time.perf_counter()
    for k in missing:
        _ = d.get(k)
    search_miss_time = time.perf_counter() - start

    # 削除
    start = time.perf_counter()
    for k in keys:
        del d[k]
    delete_time = time.perf_counter() - start

    print(f"n={n:>10,}")
    print(f"  挿入:       {insert_time:.4f}s ({insert_time/n*1e6:.2f}μs/op)")
    print(f"  検索(hit):  {search_hit_time:.4f}s ({search_hit_time/n*1e6:.2f}μs/op)")
    print(f"  検索(miss): {search_miss_time:.4f}s ({search_miss_time/n*1e6:.2f}μs/op)")
    print(f"  削除:       {delete_time:.4f}s ({delete_time/n*1e6:.2f}μs/op)")

# 典型的な結果:
# n=    10,000
#   挿入:       0.0034s (0.34μs/op)
#   検索(hit):  0.0012s (0.12μs/op)
#   検索(miss): 0.0014s (0.14μs/op)
#   削除:       0.0010s (0.10μs/op)
# n= 1,000,000
#   挿入:       0.4521s (0.45μs/op)
#   検索(hit):  0.1523s (0.15μs/op)
#   検索(miss): 0.1842s (0.18μs/op)
#   削除:       0.1234s (0.12μs/op)
```

### 8.2 データ構造間の性能比較

```python
import time
from sortedcontainers import SortedDict

def compare_dict_vs_sorted(n):
    """dict vs SortedDict vs list の検索性能比較"""

    import random
    data = [(random.randint(0, n*10), random.randint(0, 1000)) for _ in range(n)]
    search_keys = [random.randint(0, n*10) for _ in range(10000)]

    # dict
    d = dict(data)
    start = time.perf_counter()
    for k in search_keys:
        _ = k in d
    dict_time = time.perf_counter() - start

    # SortedDict（平衡BST相当）
    sd = SortedDict(data)
    start = time.perf_counter()
    for k in search_keys:
        _ = k in sd
    sorted_time = time.perf_counter() - start

    # list（線形探索）
    lst = data
    start = time.perf_counter()
    for k in search_keys:
        _ = any(kk == k for kk, _ in lst)
    list_time = time.perf_counter() - start

    print(f"n={n:>10,} (10000回検索)")
    print(f"  dict:       {dict_time:.4f}s")
    print(f"  SortedDict: {sorted_time:.4f}s")
    print(f"  list:       {list_time:.4f}s")
    print(f"  dict/sorted: {sorted_time/dict_time:.1f}x遅い")
    print(f"  dict/list:   {list_time/dict_time:.1f}x遅い")

# 典型的な結果:
# n=   100,000 (10000回検索)
#   dict:       0.0009s
#   SortedDict: 0.0213s
#   list:       12.4523s
#   dict/sorted: 23.7x遅い
#   dict/list:   13836.x遅い
```

---

## 9. 実務でよく使うパターン集

### 9.1 頻度カウントと統計

```python
from collections import Counter, defaultdict

# パターン: ログ解析
def analyze_access_log(log_entries):
    """アクセスログの統計分析"""

    # エンドポイント別アクセス数
    endpoint_count = Counter(entry["path"] for entry in log_entries)

    # ステータスコード別集計
    status_count = Counter(entry["status"] for entry in log_entries)

    # 時間帯別アクセス数
    hourly_count = Counter(entry["timestamp"].hour for entry in log_entries)

    # IPアドレス別アクセス数（上位10）
    ip_count = Counter(entry["ip"] for entry in log_entries)

    return {
        "top_endpoints": endpoint_count.most_common(10),
        "status_distribution": dict(status_count),
        "peak_hour": hourly_count.most_common(1)[0],
        "top_ips": ip_count.most_common(10),
        "total_requests": len(log_entries),
        "unique_ips": len(ip_count),
        "error_rate": status_count.get(500, 0) / len(log_entries)
    }


# パターン: 集合演算による権限管理
def check_permissions(user_roles: set, required_roles: set) -> bool:
    """ユーザーの権限チェック"""
    return required_roles.issubset(user_roles)

def common_permissions(user_a_roles: set, user_b_roles: set) -> set:
    """2人のユーザーに共通する権限"""
    return user_a_roles & user_b_roles

def exclusive_permissions(user_a_roles: set, user_b_roles: set) -> set:
    """ユーザーAだけが持つ権限"""
    return user_a_roles - user_b_roles

# パターン: インデックス構築
def build_inverted_index(documents):
    """転置インデックスの構築（全文検索の基盤）"""
    index = defaultdict(set)

    for doc_id, text in enumerate(documents):
        words = text.lower().split()
        for word in words:
            # 正規化（句読点除去等）
            word = word.strip(".,!?;:")
            index[word].add(doc_id)

    return index

def search(index, query):
    """AND検索"""
    words = query.lower().split()
    if not words:
        return set()

    result = index.get(words[0], set())
    for word in words[1:]:
        result &= index.get(word, set())

    return result

# 使用例
docs = [
    "Python is a great programming language",
    "Java is also a programming language",
    "Python and Java are both popular",
    "Rust is a systems programming language"
]
idx = build_inverted_index(docs)
print(search(idx, "programming language"))  # {0, 1, 3}
print(search(idx, "Python"))                # {0, 2}
```

### 9.2 データのバリデーションと変換

```python
# パターン: スキーマバリデーション
def validate_schema(data: dict, schema: dict) -> list:
    """簡易スキーマバリデーション"""
    errors = []

    for field, rules in schema.items():
        # 必須チェック
        if rules.get("required") and field not in data:
            errors.append(f"Missing required field: {field}")
            continue

        if field not in data:
            continue

        value = data[field]

        # 型チェック
        expected_type = rules.get("type")
        if expected_type and not isinstance(value, expected_type):
            errors.append(f"Invalid type for {field}: expected {expected_type.__name__}")

        # 範囲チェック
        if "min" in rules and value < rules["min"]:
            errors.append(f"{field} must be >= {rules['min']}")
        if "max" in rules and value > rules["max"]:
            errors.append(f"{field} must be <= {rules['max']}")

        # 許可値チェック
        if "choices" in rules and value not in rules["choices"]:
            errors.append(f"{field} must be one of {rules['choices']}")

    return errors

# 使用例
schema = {
    "name": {"required": True, "type": str},
    "age": {"required": True, "type": int, "min": 0, "max": 150},
    "role": {"type": str, "choices": ["admin", "user", "guest"]},
}

errors = validate_schema(
    {"name": "Alice", "age": 30, "role": "admin"},
    schema
)
print(errors)  # []

errors = validate_schema(
    {"age": -5, "role": "superuser"},
    schema
)
print(errors)
# ['Missing required field: name', 'age must be >= 0',
#  'role must be one of ["admin", "user", "guest"]']


# パターン: データのマッピングと変換
def transform_records(records, field_mapping):
    """フィールド名のマッピングと変換"""
    transformed = []
    for record in records:
        new_record = {}
        for old_key, (new_key, converter) in field_mapping.items():
            if old_key in record:
                new_record[new_key] = converter(record[old_key])
        transformed.append(new_record)
    return transformed

# 使用例: CSVデータの変換
mapping = {
    "full_name": ("name", str.strip),
    "birth_year": ("age", lambda y: 2026 - int(y)),
    "salary_str": ("salary", lambda s: float(s.replace(",", ""))),
}

raw_data = [
    {"full_name": " Alice ", "birth_year": "1990", "salary_str": "85,000.00"},
    {"full_name": " Bob ", "birth_year": "1985", "salary_str": "92,500.50"},
]

print(transform_records(raw_data, mapping))
# [{'name': 'Alice', 'age': 36, 'salary': 85000.0},
#  {'name': 'Bob', 'age': 41, 'salary': 92500.5}]
```

### 9.3 設定管理と環境変数

```python
import os
from typing import Any, Optional

class Config:
    """ハッシュテーブルベースの設定管理"""

    def __init__(self, defaults: dict = None):
        self._config = {}
        self._defaults = defaults or {}

    def load_from_env(self, prefix: str = "APP_"):
        """環境変数から設定を読み込み"""
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                self._config[config_key] = value

    def load_from_dict(self, data: dict):
        """辞書から設定を読み込み（ネストをフラット化）"""
        def flatten(d, parent_key=""):
            for k, v in d.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                if isinstance(v, dict):
                    flatten(v, new_key)
                else:
                    self._config[new_key] = v
        flatten(data)

    def get(self, key: str, default: Any = None) -> Any:
        """設定値を取得（優先順位: 設定 > デフォルト > 引数）"""
        if key in self._config:
            return self._config[key]
        if key in self._defaults:
            return self._defaults[key]
        return default

    def get_int(self, key: str, default: int = 0) -> int:
        return int(self.get(key, default))

    def get_bool(self, key: str, default: bool = False) -> bool:
        val = self.get(key, default)
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes")
        return bool(val)

# 使用例
config = Config(defaults={
    "database.host": "localhost",
    "database.port": 5432,
    "debug": False,
})
config.load_from_dict({
    "database": {"host": "db.example.com", "name": "myapp"},
    "cache": {"ttl": 300},
})
print(config.get("database.host"))  # "db.example.com"（オーバーライド）
print(config.get("database.port"))  # 5432（デフォルト値）
print(config.get("cache.ttl"))      # 300
```

---

## 10. 実践演習

### 演習1: 基本操作（基礎）
ハッシュテーブルをゼロから実装せよ（チェイニング方式、リサイズ付き）。以下の要件を満たすこと:
- `put(key, value)`, `get(key)`, `remove(key)`, `contains(key)` の4操作
- ロードファクター0.75でリサイズ
- `__len__`, `__iter__`, `__repr__` の実装
- 衝突数をカウントする機能

### 演習2: LRUキャッシュ（応用）
OrderedDict または dict + 双方向連結リストで LRUキャッシュを実装せよ。以下の要件を満たすこと:
- `get(key)` と `put(key, value)` の両操作が O(1)
- 容量超過時に最も古いエントリを自動削除
- TTL（有効期限）のサポート
- スレッドセーフ（threading.Lock使用）

### 演習3: 一貫性ハッシュ（発展）
分散システムで使われる一貫性ハッシュ（Consistent Hashing）を実装せよ。以下の要件を満たすこと:
- 仮想ノードによる負荷分散
- ノードの追加・削除
- レプリカ取得（N個の異なるノードを返す）
- 負荷分散の均一性をテストするベンチマーク

### 演習4: 転置インデックス（応用）
全文検索エンジンの基盤となる転置インデックスを実装せよ:
- 文書の追加と検索（AND検索、OR検索）
- TF-IDF スコアリング
- 前方一致検索のサポート

### 演習5: Cuckoo Filter（発展）
Bloom Filter の改良版である Cuckoo Filter を実装せよ:
- 挿入、検索、削除の全操作をサポート
- 偽陽性率のパラメータ調整
- Bloom Filter との性能比較ベンチマーク

---

## まとめ

| 概念 | ポイント |
|------|---------|
| ハッシュ関数 | キー→整数。均一分布が理想。SipHash/MurmurHash3が実用的 |
| 衝突解決 | チェイニング or オープンアドレス。Robin Hood/Cuckoo も |
| 計算量 | 期待O(1)、最悪O(n)。償却分析でリサイズもO(1) |
| 実務 | dict/set/Map/Counter/defaultdict |
| 内部実装 | Python:コンパクト辞書、Java:赤黒木フォールバック、Rust:Swiss Table |
| セキュリティ | Hash DoS対策にSipHash。リクエスト制限も重要 |
| 分散 | 一貫性ハッシュで負荷分散。仮想ノードで均一化 |
| 順序 | Python 3.7+は挿入順保証。他は非保証 |

---

## 次に読むべきガイド
→ [[04-trees.md]] -- 木構造

---

## 参考文献
1. Cormen, T. H. "Introduction to Algorithms." Chapter 11: Hash Tables.
2. Sedgewick, R. "Algorithms." Chapter 3.4: Hash Tables.
3. Kleppmann, M. "Designing Data-Intensive Applications." Chapter 6: Partitioning.
4. Karger, D. et al. "Consistent Hashing and Random Trees." STOC 1997.
5. Aumasson, J-P., Bernstein, D. J. "SipHash: a fast short-input PRF." 2012.
6. Abseil Team. "Swiss Tables Design Notes." Google, 2017.
7. Pagh, R., Rodler, F. F. "Cuckoo Hashing." ESA 2001.
8. Fan, B. et al. "Cuckoo Filter: Practically Better Than Bloom." CoNEXT 2014.
