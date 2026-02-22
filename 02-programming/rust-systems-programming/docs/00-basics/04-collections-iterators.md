# コレクションとイテレータ -- Rustの関数型データ処理パイプライン

> Vec、HashMap等のコレクションとIterator traitを組み合わせることで、型安全かつゼロコスト抽象化されたデータ処理パイプラインを構築できる。

---

## この章で学ぶこと

1. **主要コレクション** -- Vec、HashMap、HashSet、BTreeMap 等の使い分けを理解する
2. **Iterator トレイト** -- イテレータプロトコルと遅延評価の仕組みを習得する
3. **イテレータアダプタ** -- map/filter/fold/collect 等のチェーンによるデータ変換を学ぶ
4. **実践的パイプライン** -- 複雑なデータ処理を型安全に構築する手法を身につける
5. **パフォーマンス特性** -- 各コレクションの計算量と最適な使い分けを理解する

---

## 1. Vec<T> -- 動的配列

### 例1: Vec の基本操作

```rust
fn main() {
    // 生成
    let mut v: Vec<i32> = Vec::new();
    let v2 = vec![1, 2, 3, 4, 5]; // マクロで初期化

    // 追加
    v.push(10);
    v.push(20);
    v.push(30);

    // アクセス
    println!("インデックス: {}", v[0]);         // 10 (パニックの可能性)
    println!("安全: {:?}", v.get(99));           // None

    // イテレーション
    for item in &v {
        println!("{}", item);
    }

    // 可変イテレーション
    for item in &mut v {
        *item *= 2;
    }
    println!("{:?}", v); // [20, 40, 60]

    // 便利メソッド
    println!("長さ: {}, 空?: {}", v.len(), v.is_empty());
    println!("含む?: {}", v.contains(&20));

    let last = v.pop(); // Some(60)
    println!("pop: {:?}", last);
}
```

### Vec メモリレイアウト

```
  スタック                     ヒープ
  ┌──────────────┐           ┌────┬────┬────┬────┬────┐
  │ ptr ──────────────────── │ 20 │ 40 │ 60 │    │    │
  │ len = 3      │           └────┴────┴────┴────┴────┘
  │ capacity = 5 │                used          unused
  └──────────────┘

  push で capacity を超えると、新しい領域を確保して
  全要素をコピー（償却 O(1)）
```

### 1.1 Vec の高度な操作

```rust
fn main() {
    let mut v = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];

    // ソート
    v.sort();
    println!("ソート済み: {:?}", v); // [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]

    // 重複除去（ソート済みの場合のみ有効）
    v.dedup();
    println!("重複除去: {:?}", v);   // [1, 2, 3, 4, 5, 6, 9]

    // retain: 条件を満たす要素だけ残す
    v.retain(|&x| x % 2 == 0);
    println!("偶数のみ: {:?}", v);   // [2, 4, 6]

    // extend: 他のイテレータから追加
    v.extend([8, 10, 12].iter());
    println!("拡張後: {:?}", v);     // [2, 4, 6, 8, 10, 12]

    // split_off: ベクタを2つに分割
    let tail = v.split_off(3);
    println!("前半: {:?}", v);       // [2, 4, 6]
    println!("後半: {:?}", tail);    // [8, 10, 12]

    // windows: スライディングウィンドウ
    let data = vec![1, 2, 3, 4, 5];
    for window in data.windows(3) {
        println!("ウィンドウ: {:?}", window);
    }
    // [1, 2, 3], [2, 3, 4], [3, 4, 5]

    // chunks: 固定サイズのチャンク
    for chunk in data.chunks(2) {
        println!("チャンク: {:?}", chunk);
    }
    // [1, 2], [3, 4], [5]
}
```

### 1.2 Vec のキャパシティ管理

```rust
fn main() {
    // キャパシティを事前確保（再割り当て回避）
    let mut v = Vec::with_capacity(1000);
    println!("len={}, capacity={}", v.len(), v.capacity());
    // len=0, capacity=1000

    for i in 0..1000 {
        v.push(i);
    }
    // 再割り当てが一度も発生しない

    // 余分なキャパシティを解放
    v.shrink_to_fit();
    println!("shrink後: len={}, capacity={}", v.len(), v.capacity());

    // キャパシティ成長戦略:
    // Vec は現在のキャパシティの 2倍 を確保する
    // push のコスト: 償却 O(1) （ほとんどの場合 O(1)、再割り当て時のみ O(n)）

    // ベンチマーク: with_capacity vs push のみ
    use std::time::Instant;

    let n = 1_000_000;

    let start = Instant::now();
    let mut v1 = Vec::new();
    for i in 0..n {
        v1.push(i);
    }
    let t1 = start.elapsed();

    let start = Instant::now();
    let mut v2 = Vec::with_capacity(n);
    for i in 0..n {
        v2.push(i);
    }
    let t2 = start.elapsed();

    println!("without capacity: {:?}", t1);
    println!("with capacity:    {:?}", t2);
    // with_capacity の方が高速（再割り当てなし）
}
```

### 1.3 スライス -- Vec への参照

```rust
fn sum_slice(data: &[i32]) -> i32 {
    data.iter().sum()
}

fn find_max(data: &[i32]) -> Option<&i32> {
    data.iter().max()
}

fn main() {
    let v = vec![10, 20, 30, 40, 50];

    // Vec → スライス（暗黙の変換）
    let total = sum_slice(&v);
    println!("合計: {}", total); // 150

    // 部分スライス
    let middle = &v[1..4]; // [20, 30, 40]
    println!("部分スライス: {:?}", middle);
    println!("部分合計: {}", sum_slice(middle)); // 90

    // 配列もスライスとして渡せる
    let arr = [1, 2, 3, 4, 5];
    println!("配列の合計: {}", sum_slice(&arr)); // 15

    // find_max の使用
    if let Some(max) = find_max(&v) {
        println!("最大値: {}", max); // 50
    }

    // 空のスライス
    let empty: &[i32] = &[];
    println!("空の最大値: {:?}", find_max(empty)); // None

    // スライスのバイナリサーチ（ソート済みの場合）
    let sorted = vec![1, 3, 5, 7, 9, 11, 13];
    match sorted.binary_search(&7) {
        Ok(index) => println!("7 は位置 {} にあります", index),
        Err(index) => println!("7 は見つかりません（挿入位置: {}）", index),
    }
}
```

---

## 2. HashMap<K, V>

### 例2: HashMap の基本操作

```rust
use std::collections::HashMap;

fn main() {
    let mut scores: HashMap<String, u32> = HashMap::new();

    // 挿入
    scores.insert("田中".to_string(), 85);
    scores.insert("鈴木".to_string(), 92);
    scores.insert("佐藤".to_string(), 78);

    // アクセス
    if let Some(score) = scores.get("田中") {
        println!("田中のスコア: {}", score);
    }

    // entry API (キーが存在しなければ挿入)
    scores.entry("山田".to_string()).or_insert(0);
    *scores.entry("田中".to_string()).or_insert(0) += 10;

    // イテレーション
    for (name, score) in &scores {
        println!("{}: {}", name, score);
    }

    // 単語カウントの定番パターン
    let text = "hello world hello rust hello";
    let mut word_count = HashMap::new();
    for word in text.split_whitespace() {
        *word_count.entry(word).or_insert(0) += 1;
    }
    println!("{:?}", word_count);
    // {"hello": 3, "world": 1, "rust": 1}
}
```

### 2.1 HashMap の高度なパターン

```rust
use std::collections::HashMap;

fn main() {
    // from イテレータで構築
    let teams: HashMap<&str, u32> = vec![
        ("Red", 10),
        ("Blue", 20),
        ("Green", 15),
    ].into_iter().collect();

    println!("{:?}", teams);

    // entry API の高度な使い方
    let mut cache: HashMap<String, Vec<String>> = HashMap::new();

    // or_insert_with: 遅延初期化
    cache.entry("users".to_string())
        .or_insert_with(Vec::new)
        .push("Alice".to_string());

    cache.entry("users".to_string())
        .or_insert_with(Vec::new)
        .push("Bob".to_string());

    println!("users: {:?}", cache.get("users"));
    // Some(["Alice", "Bob"])

    // and_modify + or_insert: 存在する場合は更新、しない場合は挿入
    let mut counter: HashMap<&str, i32> = HashMap::new();
    let words = vec!["hello", "world", "hello", "rust", "hello"];

    for word in &words {
        counter.entry(word)
            .and_modify(|count| *count += 1)
            .or_insert(1);
    }
    println!("カウント: {:?}", counter);

    // HashMap の結合（merge）
    let mut map1: HashMap<&str, i32> = [("a", 1), ("b", 2)].into();
    let map2: HashMap<&str, i32> = [("b", 10), ("c", 3)].into();

    for (key, value) in map2 {
        map1.entry(key)
            .and_modify(|v| *v += value)
            .or_insert(value);
    }
    println!("結合: {:?}", map1); // {"a": 1, "b": 12, "c": 3}
}
```

### 2.2 カスタムキーの HashMap

```rust
use std::collections::HashMap;

// HashMap のキーには Eq + Hash が必要
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Coordinate {
    x: i32,
    y: i32,
}

impl Coordinate {
    fn new(x: i32, y: i32) -> Self {
        Coordinate { x, y }
    }
}

fn main() {
    let mut grid: HashMap<Coordinate, char> = HashMap::new();

    grid.insert(Coordinate::new(0, 0), '.');
    grid.insert(Coordinate::new(1, 0), '#');
    grid.insert(Coordinate::new(0, 1), '.');
    grid.insert(Coordinate::new(1, 1), '#');

    // グリッドを表示
    for y in 0..2 {
        for x in 0..2 {
            let cell = grid.get(&Coordinate::new(x, y)).unwrap_or(&' ');
            print!("{}", cell);
        }
        println!();
    }

    // キーの所有権に注意
    // HashMap<String, V> のキーに &str でアクセスできる（Borrow トレイト）
    let mut names: HashMap<String, u32> = HashMap::new();
    names.insert("田中".to_string(), 85);

    // &str でアクセス可能（String は Borrow<str> を実装）
    if let Some(score) = names.get("田中") {
        println!("スコア: {}", score);
    }
}
```

### 2.3 ハッシュ関数のカスタマイズ

```rust
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

// 簡易的なハッシャー（教育目的）
#[derive(Default)]
struct SimpleHasher {
    hash: u64,
}

impl Hasher for SimpleHasher {
    fn finish(&self) -> u64 {
        self.hash
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.hash = self.hash.wrapping_mul(31).wrapping_add(byte as u64);
        }
    }
}

type SimpleHashMap<K, V> = HashMap<K, V, BuildHasherDefault<SimpleHasher>>;

fn main() {
    // デフォルトの SipHash（DoS 耐性あり、やや低速）
    let mut default_map: HashMap<String, i32> = HashMap::new();
    default_map.insert("key".to_string(), 42);

    // カスタムハッシャー
    let mut custom_map: SimpleHashMap<String, i32> = SimpleHashMap::default();
    custom_map.insert("key".to_string(), 42);

    // 高速ハッシュが必要な場合は外部クレートを使用:
    // - rustc-hash (FxHashMap): Rustコンパイラ内部で使用、高速
    // - ahash: AES命令を使った高速ハッシュ
    // - fnv: FNV-1a ハッシュ、小さなキーに最適

    println!("default: {:?}", default_map);
    println!("custom:  {:?}", custom_map);
}
```

---

## 3. その他のコレクション

```
┌────────────────┬──────────────────────────────────────┐
│ コレクション    │ 用途・特性                           │
├────────────────┼──────────────────────────────────────┤
│ Vec<T>         │ 動的配列。末尾追加O(1)、検索O(n)     │
│ VecDeque<T>    │ 両端キュー。先頭/末尾追加O(1)        │
│ LinkedList<T>  │ 双方向リスト。実務ではほぼ不要       │
│ HashMap<K,V>   │ ハッシュマップ。平均O(1)アクセス     │
│ BTreeMap<K,V>  │ Bツリーマップ。ソート順保持。O(logn) │
│ HashSet<T>     │ 重複なし集合。平均O(1)判定           │
│ BTreeSet<T>    │ ソート済み集合。O(logn)              │
│ BinaryHeap<T>  │ 最大ヒープ。優先度キュー             │
└────────────────┴──────────────────────────────────────┘
```

### 3.1 HashSet -- 集合演算

```rust
use std::collections::HashSet;

fn main() {
    let mut set_a: HashSet<i32> = [1, 2, 3, 4, 5].into();
    let set_b: HashSet<i32> = [3, 4, 5, 6, 7].into();

    // 基本操作
    set_a.insert(6);
    set_a.remove(&6);
    println!("含む 3?: {}", set_a.contains(&3)); // true

    // 集合演算
    // 和集合 (union)
    let union: HashSet<_> = set_a.union(&set_b).collect();
    println!("和集合: {:?}", union); // {1, 2, 3, 4, 5, 6, 7}

    // 積集合 (intersection)
    let intersection: HashSet<_> = set_a.intersection(&set_b).collect();
    println!("積集合: {:?}", intersection); // {3, 4, 5}

    // 差集合 (difference)
    let difference: HashSet<_> = set_a.difference(&set_b).collect();
    println!("差集合 (A-B): {:?}", difference); // {1, 2}

    // 対称差 (symmetric_difference)
    let sym_diff: HashSet<_> = set_a.symmetric_difference(&set_b).collect();
    println!("対称差: {:?}", sym_diff); // {1, 2, 6, 7}

    // 部分集合の判定
    let subset: HashSet<i32> = [3, 4].into();
    println!("subset ⊂ set_a?: {}", subset.is_subset(&set_a)); // true
    println!("set_a ⊃ subset?: {}", set_a.is_superset(&subset)); // true

    // 重複排除パターン
    let with_dups = vec![1, 2, 2, 3, 3, 3, 4, 4, 4, 4];
    let unique: HashSet<_> = with_dups.iter().collect();
    println!("ユニーク要素数: {}", unique.len()); // 4

    // 順序を保持して重複排除
    let mut seen = HashSet::new();
    let unique_ordered: Vec<_> = with_dups.iter()
        .filter(|x| seen.insert(*x))
        .collect();
    println!("順序保持して重複除去: {:?}", unique_ordered); // [1, 2, 3, 4]
}
```

### 3.2 BTreeMap -- ソート済みマップ

```rust
use std::collections::BTreeMap;

fn main() {
    let mut scores = BTreeMap::new();
    scores.insert("Charlie", 85);
    scores.insert("Alice", 92);
    scores.insert("Bob", 78);
    scores.insert("David", 95);

    // イテレーションは常にキーのソート順
    for (name, score) in &scores {
        println!("{}: {}", name, score);
    }
    // Alice: 92, Bob: 78, Charlie: 85, David: 95

    // 範囲検索
    for (name, score) in scores.range("Bob"..="David") {
        println!("範囲: {} = {}", name, score);
    }
    // Bob: 78, Charlie: 85, David: 95

    // 最小/最大キー
    if let Some((first, _)) = scores.iter().next() {
        println!("最小キー: {}", first); // Alice
    }
    if let Some((last, _)) = scores.iter().next_back() {
        println!("最大キー: {}", last);  // David
    }

    // BTreeMap は HashMap と違い Ord トレイトを要求する
    // f64 のようなキーは使えない（PartialOrd のみ）
    // ただし ordered-float クレートで対応可能
}
```

### 3.3 VecDeque -- 両端キュー

```rust
use std::collections::VecDeque;

fn main() {
    let mut deque = VecDeque::new();

    // 両端への追加
    deque.push_back(2);
    deque.push_back(3);
    deque.push_front(1);
    deque.push_front(0);
    println!("{:?}", deque); // [0, 1, 2, 3]

    // 両端からの取り出し
    println!("front: {:?}", deque.pop_front()); // Some(0)
    println!("back:  {:?}", deque.pop_back());  // Some(3)
    println!("{:?}", deque); // [1, 2]

    // FIFOキューとして使用
    let mut queue: VecDeque<&str> = VecDeque::new();
    queue.push_back("タスク1");
    queue.push_back("タスク2");
    queue.push_back("タスク3");

    while let Some(task) = queue.pop_front() {
        println!("処理中: {}", task);
    }

    // スライディングウィンドウ（固定サイズ）
    let mut window: VecDeque<i32> = VecDeque::with_capacity(3);
    let data = [1, 2, 3, 4, 5, 6, 7];

    for &value in &data {
        if window.len() == 3 {
            window.pop_front();
        }
        window.push_back(value);
        if window.len() == 3 {
            let sum: i32 = window.iter().sum();
            println!("ウィンドウ {:?} 合計: {}", window, sum);
        }
    }
}
```

### 3.4 BinaryHeap -- 優先度キュー

```rust
use std::collections::BinaryHeap;
use std::cmp::Reverse;

fn main() {
    // 最大ヒープ（デフォルト）
    let mut max_heap = BinaryHeap::new();
    max_heap.push(3);
    max_heap.push(1);
    max_heap.push(4);
    max_heap.push(1);
    max_heap.push(5);

    // pop は常に最大値を返す
    while let Some(value) = max_heap.pop() {
        print!("{} ", value); // 5 4 3 1 1
    }
    println!();

    // 最小ヒープ（Reverse ラッパー）
    let mut min_heap: BinaryHeap<Reverse<i32>> = BinaryHeap::new();
    min_heap.push(Reverse(3));
    min_heap.push(Reverse(1));
    min_heap.push(Reverse(4));

    while let Some(Reverse(value)) = min_heap.pop() {
        print!("{} ", value); // 1 3 4
    }
    println!();

    // タスクスケジューラの例
    #[derive(Debug, Eq, PartialEq)]
    struct Task {
        priority: u32,
        name: String,
    }

    impl Ord for Task {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.priority.cmp(&other.priority)
        }
    }

    impl PartialOrd for Task {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }

    let mut scheduler = BinaryHeap::new();
    scheduler.push(Task { priority: 1, name: "低優先度タスク".into() });
    scheduler.push(Task { priority: 10, name: "高優先度タスク".into() });
    scheduler.push(Task { priority: 5, name: "中優先度タスク".into() });

    while let Some(task) = scheduler.pop() {
        println!("実行: {} (優先度={})", task.name, task.priority);
    }
    // 高優先度タスク (10), 中優先度タスク (5), 低優先度タスク (1)
}
```

---

## 4. Iterator トレイト

### Iterator トレイトの定義

```rust
// 標準ライブラリの Iterator トレイト（簡略版）
trait Iterator {
    type Item;  // 関連型: イテレータが生成する要素の型

    fn next(&mut self) -> Option<Self::Item>;

    // 以下は next() に基づくデフォルト実装（75以上のメソッド）
    // fn map<B, F>(self, f: F) -> Map<Self, F> { ... }
    // fn filter<P>(self, predicate: P) -> Filter<Self, P> { ... }
    // fn fold<B, F>(self, init: B, f: F) -> B { ... }
    // fn collect<B: FromIterator<Self::Item>>(self) -> B { ... }
    // ...
}
```

### 例3: カスタムイテレータ

```rust
struct Counter {
    count: u32,
    max: u32,
}

impl Counter {
    fn new(max: u32) -> Self {
        Counter { count: 0, max }
    }
}

impl Iterator for Counter {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count < self.max {
            self.count += 1;
            Some(self.count)
        } else {
            None
        }
    }
}

fn main() {
    let counter = Counter::new(5);
    let v: Vec<u32> = counter.collect();
    println!("{:?}", v); // [1, 2, 3, 4, 5]

    // Counter は Iterator を実装しているので、全アダプタが使える
    let sum: u32 = Counter::new(10).sum();
    println!("1..10 の合計: {}", sum); // 55

    let doubled: Vec<u32> = Counter::new(5).map(|x| x * 2).collect();
    println!("2倍: {:?}", doubled); // [2, 4, 6, 8, 10]

    let evens: Vec<u32> = Counter::new(10).filter(|x| x % 2 == 0).collect();
    println!("偶数: {:?}", evens); // [2, 4, 6, 8, 10]
}
```

### 例4: フィボナッチイテレータ

```rust
struct Fibonacci {
    a: u64,
    b: u64,
}

impl Fibonacci {
    fn new() -> Self {
        Fibonacci { a: 0, b: 1 }
    }
}

impl Iterator for Fibonacci {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.a;
        let new_b = self.a.checked_add(self.b)?; // オーバーフロー時に None
        self.a = self.b;
        self.b = new_b;
        Some(value)
    }
}

fn main() {
    // 最初の10個のフィボナッチ数
    let fibs: Vec<u64> = Fibonacci::new().take(10).collect();
    println!("フィボナッチ: {:?}", fibs);
    // [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

    // 100未満のフィボナッチ数
    let small_fibs: Vec<u64> = Fibonacci::new()
        .take_while(|&x| x < 100)
        .collect();
    println!("100未満: {:?}", small_fibs);
    // [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]

    // フィボナッチ数の合計（最初の20個）
    let sum: u64 = Fibonacci::new().take(20).sum();
    println!("最初の20個の合計: {}", sum);
}
```

### イテレータの遅延評価

```
  map / filter / take は「アダプタ」: 遅延評価される
  collect / sum / for_each は「消費子」: 実際にイテレーションを駆動

  vec.iter()
    .map(|x| x * 2)      ← アダプタ (何も実行されない)
    .filter(|x| *x > 5)  ← アダプタ (何も実行されない)
    .collect::<Vec<_>>()  ← 消費子 (ここで初めて全要素を処理)

  ┌──────┐   ┌──────┐   ┌────────┐   ┌─────────┐
  │ iter │──>│ map  │──>│ filter │──>│ collect │
  │      │   │*2    │   │ >5     │   │Vec<_>   │
  └──────┘   └──────┘   └────────┘   └─────────┘
   要素1 ──── 2 ──────── skip ─────── (スキップ)
   要素2 ──── 4 ──────── skip ─────── (スキップ)
   要素3 ──── 6 ──────── pass ─────── 6 に追加
   要素4 ──── 8 ──────── pass ─────── 8 に追加
   要素5 ──── 10 ─────── pass ─────── 10 に追加

  重要: 各要素は一度にパイプライン全体を通過する（バッチ処理ではない）
  これにより中間コレクションが不要で、メモリ効率が良い
```

### 遅延評価の証明

```rust
fn main() {
    let v = vec![1, 2, 3, 4, 5];

    // アダプタだけでは何も起こらない
    let _lazy = v.iter()
        .map(|x| {
            println!("map: {}", x); // この行は実行されない！
            x * 2
        })
        .filter(|x| {
            println!("filter: {}", x); // この行も実行されない！
            *x > 5
        });
    println!("アダプタを作成しただけ。何も実行されていない。");

    // 消費子を呼ぶと実行される
    let result: Vec<_> = v.iter()
        .map(|x| {
            println!("map: {}", x);
            x * 2
        })
        .filter(|x| {
            println!("filter: {}", x);
            *x > 5
        })
        .collect();

    println!("結果: {:?}", result);
    // 出力:
    // map: 1, filter: 2     ← 要素1が map → filter を通過
    // map: 2, filter: 4     ← 要素2が map → filter を通過
    // map: 3, filter: 6     ← 要素3が map → filter を通過
    // map: 4, filter: 8
    // map: 5, filter: 10
    // 結果: [6, 8, 10]
}
```

---

## 5. イテレータアダプタ詳解

### 例5: 主要なアダプタとチェーン

```rust
fn main() {
    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // map: 各要素を変換
    let doubled: Vec<i32> = numbers.iter().map(|x| x * 2).collect();
    println!("doubled: {:?}", doubled);

    // filter: 条件に合う要素だけ
    let evens: Vec<&i32> = numbers.iter().filter(|x| *x % 2 == 0).collect();
    println!("evens: {:?}", evens);

    // fold: 畳み込み
    let sum = numbers.iter().fold(0, |acc, x| acc + x);
    println!("sum: {}", sum); // 55

    // enumerate: インデックス付き
    for (i, val) in numbers.iter().enumerate() {
        print!("[{}]={} ", i, val);
    }
    println!();

    // zip: 2つのイテレータを結合
    let names = vec!["Alice", "Bob", "Charlie"];
    let ages = vec![30, 25, 35];
    let people: Vec<_> = names.iter().zip(ages.iter()).collect();
    println!("{:?}", people); // [("Alice", 30), ("Bob", 25), ("Charlie", 35)]

    // chain: 2つのイテレータを連結
    let first = vec![1, 2, 3];
    let second = vec![4, 5, 6];
    let combined: Vec<_> = first.iter().chain(second.iter()).collect();
    println!("{:?}", combined); // [1, 2, 3, 4, 5, 6]

    // take / skip
    let first_three: Vec<_> = numbers.iter().take(3).collect();
    let after_three: Vec<_> = numbers.iter().skip(3).collect();
    println!("take(3): {:?}", first_three);
    println!("skip(3): {:?}", after_three);

    // flat_map: ネストしたイテレータをフラットに
    let words = vec!["hello world", "foo bar"];
    let chars: Vec<&str> = words.iter().flat_map(|s| s.split_whitespace()).collect();
    println!("{:?}", chars); // ["hello", "world", "foo", "bar"]
}
```

### 5.1 追加アダプタ

```rust
fn main() {
    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // take_while / skip_while: 条件ベースの取得/スキップ
    let small: Vec<_> = numbers.iter().take_while(|&&x| x < 5).collect();
    println!("take_while(<5): {:?}", small); // [1, 2, 3, 4]

    let large: Vec<_> = numbers.iter().skip_while(|&&x| x < 5).collect();
    println!("skip_while(<5): {:?}", large); // [5, 6, 7, 8, 9, 10]

    // scan: 状態を持つ map
    let running_sum: Vec<i32> = numbers.iter()
        .scan(0, |state, &x| {
            *state += x;
            Some(*state)
        })
        .collect();
    println!("累積和: {:?}", running_sum);
    // [1, 3, 6, 10, 15, 21, 28, 36, 45, 55]

    // inspect: デバッグ用（値を変更せずに観察）
    let result: Vec<_> = numbers.iter()
        .inspect(|x| print!("before filter: {} ", x))
        .filter(|&&x| x % 2 == 0)
        .inspect(|x| print!("after filter: {} ", x))
        .collect();
    println!("\n結果: {:?}", result);

    // peekable: 次の要素を消費せずに覗く
    let mut iter = numbers.iter().peekable();
    while let Some(&&next) = iter.peek() {
        if next % 3 == 0 {
            println!("3の倍数を発見: {}", iter.next().unwrap());
        } else {
            iter.next(); // スキップ
        }
    }

    // step_by: N要素ごとに取得
    let every_third: Vec<_> = numbers.iter().step_by(3).collect();
    println!("3つおき: {:?}", every_third); // [1, 4, 7, 10]

    // unzip: ペアのイテレータを2つのコレクションに分離
    let pairs = vec![(1, 'a'), (2, 'b'), (3, 'c')];
    let (nums, chars): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    println!("nums: {:?}, chars: {:?}", nums, chars);
    // nums: [1, 2, 3], chars: ['a', 'b', 'c']
}
```

### 5.2 partition と group_by パターン

```rust
fn main() {
    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // partition: 条件で2つに分割
    let (evens, odds): (Vec<_>, Vec<_>) = numbers.iter().partition(|&&x| x % 2 == 0);
    println!("偶数: {:?}", evens); // [2, 4, 6, 8, 10]
    println!("奇数: {:?}", odds);  // [1, 3, 5, 7, 9]

    // group_by パターン（HashMap を使用）
    use std::collections::HashMap;

    let words = vec!["apple", "banana", "avocado", "blueberry", "cherry", "apricot"];
    let grouped: HashMap<char, Vec<&&str>> = words.iter()
        .fold(HashMap::new(), |mut acc, word| {
            let first_char = word.chars().next().unwrap();
            acc.entry(first_char).or_default().push(word);
            acc
        });

    for (letter, group) in &grouped {
        println!("{}: {:?}", letter, group);
    }
    // a: ["apple", "avocado", "apricot"]
    // b: ["banana", "blueberry"]
    // c: ["cherry"]
}
```

### 例6: 実践的なイテレータパイプライン

```rust
#[derive(Debug)]
struct Student {
    name: String,
    score: u32,
}

fn main() {
    let students = vec![
        Student { name: "田中".into(), score: 85 },
        Student { name: "鈴木".into(), score: 92 },
        Student { name: "佐藤".into(), score: 67 },
        Student { name: "山田".into(), score: 78 },
        Student { name: "渡辺".into(), score: 95 },
    ];

    // 80点以上の学生の名前を取得し、スコア降順でソート
    let mut honor_roll: Vec<_> = students
        .iter()
        .filter(|s| s.score >= 80)
        .collect();
    honor_roll.sort_by(|a, b| b.score.cmp(&a.score));

    println!("優等生:");
    for s in &honor_roll {
        println!("  {} ({}点)", s.name, s.score);
    }

    // 平均点
    let avg = students.iter().map(|s| s.score).sum::<u32>() as f64
        / students.len() as f64;
    println!("平均: {:.1}点", avg);

    // スコア別グループ化
    use std::collections::HashMap;
    let grouped: HashMap<&str, Vec<&Student>> = students.iter().fold(
        HashMap::new(),
        |mut acc, s| {
            let grade = if s.score >= 90 { "A" }
                       else if s.score >= 80 { "B" }
                       else if s.score >= 70 { "C" }
                       else { "D" };
            acc.entry(grade).or_default().push(s);
            acc
        },
    );
    for (grade, group) in &grouped {
        let names: Vec<_> = group.iter().map(|s| s.name.as_str()).collect();
        println!("グレード{}: {:?}", grade, names);
    }
}
```

### 例7: 複雑なデータパイプライン

```rust
use std::collections::HashMap;

#[derive(Debug, Clone)]
struct Sale {
    product: String,
    category: String,
    amount: f64,
    quantity: u32,
}

fn main() {
    let sales = vec![
        Sale { product: "りんご".into(), category: "果物".into(), amount: 150.0, quantity: 3 },
        Sale { product: "バナナ".into(), category: "果物".into(), amount: 100.0, quantity: 5 },
        Sale { product: "にんじん".into(), category: "野菜".into(), amount: 80.0, quantity: 2 },
        Sale { product: "トマト".into(), category: "野菜".into(), amount: 200.0, quantity: 4 },
        Sale { product: "りんご".into(), category: "果物".into(), amount: 150.0, quantity: 2 },
        Sale { product: "ブロッコリー".into(), category: "野菜".into(), amount: 180.0, quantity: 1 },
    ];

    // カテゴリ別売上集計
    let category_totals: HashMap<&str, f64> = sales.iter()
        .fold(HashMap::new(), |mut acc, sale| {
            *acc.entry(sale.category.as_str()).or_insert(0.0) += sale.amount * sale.quantity as f64;
            acc
        });

    println!("カテゴリ別売上:");
    for (category, total) in &category_totals {
        println!("  {}: {:.0}円", category, total);
    }

    // 商品別販売数トップ3
    let mut product_quantities: HashMap<&str, u32> = HashMap::new();
    for sale in &sales {
        *product_quantities.entry(&sale.product).or_insert(0) += sale.quantity;
    }

    let mut top_products: Vec<_> = product_quantities.iter().collect();
    top_products.sort_by(|a, b| b.1.cmp(a.1));

    println!("\n販売数トップ3:");
    for (product, qty) in top_products.iter().take(3) {
        println!("  {}: {}個", product, qty);
    }

    // 単価が100円以上の果物の平均単価
    let expensive_fruits: Vec<f64> = sales.iter()
        .filter(|s| s.category == "果物" && s.amount >= 100.0)
        .map(|s| s.amount)
        .collect();

    if !expensive_fruits.is_empty() {
        let avg = expensive_fruits.iter().sum::<f64>() / expensive_fruits.len() as f64;
        println!("\n100円以上の果物の平均単価: {:.0}円", avg);
    }
}
```

---

## 6. into_iter / iter / iter_mut の違い

```
┌───────────────┬───────────────────┬────────────┬───────────────┐
│ メソッド       │ 要素の型           │ 所有権     │ コレクション   │
├───────────────┼───────────────────┼────────────┼───────────────┤
│ .iter()       │ &T                │ 借用       │ そのまま残る   │
│ .iter_mut()   │ &mut T            │ 可変借用   │ そのまま残る   │
│ .into_iter()  │ T                 │ ムーブ     │ 消費される     │
├───────────────┼───────────────────┼────────────┼───────────────┤
│ for x in &v   │ &T  (iter)        │ 借用       │ そのまま残る   │
│ for x in &mut v│ &mut T (iter_mut)│ 可変借用   │ そのまま残る   │
│ for x in v    │ T    (into_iter)  │ ムーブ     │ 消費される     │
└───────────────┴───────────────────┴────────────┴───────────────┘
```

### 6.1 各イテレータの使い分け

```rust
fn main() {
    let v = vec![String::from("hello"), String::from("world")];

    // iter(): 借用 -- コレクションはそのまま残る
    for s in v.iter() {
        println!("借用: {}", s); // s は &String
    }
    println!("v はまだ使える: {:?}", v);

    // iter_mut(): 可変借用 -- 要素を変更できる
    let mut v2 = vec![1, 2, 3, 4, 5];
    for n in v2.iter_mut() {
        *n *= 2; // 各要素を2倍に
    }
    println!("変更後: {:?}", v2); // [2, 4, 6, 8, 10]

    // into_iter(): 所有権ムーブ -- コレクションは消費される
    let v3 = vec![String::from("a"), String::from("b")];
    let uppercased: Vec<String> = v3.into_iter()
        .map(|s| s.to_uppercase()) // s は String（所有権あり）
        .collect();
    println!("大文字: {:?}", uppercased);
    // println!("{:?}", v3); // コンパイルエラー！v3 はムーブ済み

    // 参照のイテレータから所有権を得る方法
    let v4 = vec!["hello", "world"];
    let owned: Vec<String> = v4.iter()
        .map(|s| s.to_string()) // &str → String にクローン
        .collect();
    println!("所有: {:?}", owned);
    println!("元のv4: {:?}", v4); // まだ使える

    // cloned() / copied() でコピー/クローン
    let v5 = vec![1, 2, 3, 4, 5];
    let copied: Vec<i32> = v5.iter().copied().collect(); // &i32 → i32
    println!("copied: {:?}", copied);

    let v6 = vec!["a".to_string(), "b".to_string()];
    let cloned: Vec<String> = v6.iter().cloned().collect(); // &String → String
    println!("cloned: {:?}", cloned);
}
```

### 6.2 IntoIterator トレイト

```rust
// IntoIterator トレイトにより、for ループでのイテレーションが可能になる
// trait IntoIterator {
//     type Item;
//     type IntoIter: Iterator<Item = Self::Item>;
//     fn into_iter(self) -> Self::IntoIter;
// }

// カスタム型に IntoIterator を実装
struct Matrix {
    data: Vec<Vec<f64>>,
}

impl Matrix {
    fn new(data: Vec<Vec<f64>>) -> Self {
        Matrix { data }
    }
}

// 所有権ムーブ版
impl IntoIterator for Matrix {
    type Item = Vec<f64>;
    type IntoIter = std::vec::IntoIter<Vec<f64>>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

// 借用版
impl<'a> IntoIterator for &'a Matrix {
    type Item = &'a Vec<f64>;
    type IntoIter = std::slice::Iter<'a, Vec<f64>>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

fn main() {
    let matrix = Matrix::new(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]);

    // for ループで使える
    for row in &matrix {
        println!("行: {:?}", row);
    }

    // into_iter で行を消費
    for row in matrix {
        let sum: f64 = row.iter().sum();
        println!("行の合計: {}", sum);
    }
}
```

---

## 7. FromIterator と collect の仕組み

### 7.1 collect の多様な変換

```rust
use std::collections::{HashMap, HashSet, BTreeSet, VecDeque};

fn main() {
    let numbers = vec![1, 2, 3, 4, 5, 3, 2, 1];

    // Vec に collect
    let v: Vec<i32> = numbers.iter().copied().collect();
    println!("Vec: {:?}", v);

    // HashSet に collect（重複排除）
    let set: HashSet<i32> = numbers.iter().copied().collect();
    println!("HashSet: {:?}", set);

    // BTreeSet に collect（ソート済み集合）
    let bset: BTreeSet<i32> = numbers.iter().copied().collect();
    println!("BTreeSet: {:?}", bset);

    // VecDeque に collect
    let deque: VecDeque<i32> = numbers.iter().copied().collect();
    println!("VecDeque: {:?}", deque);

    // HashMap に collect（ペアのイテレータから）
    let map: HashMap<&str, i32> = vec![("a", 1), ("b", 2), ("c", 3)]
        .into_iter()
        .collect();
    println!("HashMap: {:?}", map);

    // String に collect
    let chars = vec!['H', 'e', 'l', 'l', 'o'];
    let s: String = chars.into_iter().collect();
    println!("String: {}", s);

    // Result<Vec<T>, E> に collect（エラーがあれば即座に Err）
    let strings = vec!["1", "2", "abc", "4"];
    let result: Result<Vec<i32>, _> = strings.iter()
        .map(|s| s.parse::<i32>())
        .collect();
    println!("Result: {:?}", result); // Err(ParseIntError)

    // 成功する場合
    let valid = vec!["1", "2", "3", "4"];
    let result: Result<Vec<i32>, _> = valid.iter()
        .map(|s| s.parse::<i32>())
        .collect();
    println!("Result: {:?}", result); // Ok([1, 2, 3, 4])
}
```

### 7.2 カスタム FromIterator 実装

```rust
use std::iter::FromIterator;

#[derive(Debug)]
struct Histogram {
    bins: std::collections::HashMap<i32, usize>,
}

impl FromIterator<i32> for Histogram {
    fn from_iter<I: IntoIterator<Item = i32>>(iter: I) -> Self {
        let mut bins = std::collections::HashMap::new();
        for value in iter {
            *bins.entry(value).or_insert(0) += 1;
        }
        Histogram { bins }
    }
}

impl Histogram {
    fn display(&self) {
        let mut entries: Vec<_> = self.bins.iter().collect();
        entries.sort_by_key(|&(k, _)| k);
        for (value, count) in entries {
            println!("{:>3}: {}", value, "#".repeat(*count));
        }
    }
}

fn main() {
    let data = vec![1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5];

    // collect でヒストグラムに変換
    let hist: Histogram = data.into_iter().collect();
    hist.display();
    //   1: #
    //   2: ##
    //   3: ###
    //   4: ####
    //   5: #####
}
```

---

## 8. イテレータの高度なパターン

### 8.1 再帰的フラット化

```rust
fn flatten_nested(nested: &[Vec<Vec<i32>>]) -> Vec<i32> {
    nested.iter()
        .flat_map(|inner| inner.iter())
        .flat_map(|v| v.iter())
        .copied()
        .collect()
}

fn main() {
    let nested = vec![
        vec![vec![1, 2], vec![3, 4]],
        vec![vec![5, 6], vec![7, 8]],
    ];

    let flat = flatten_nested(&nested);
    println!("フラット: {:?}", flat); // [1, 2, 3, 4, 5, 6, 7, 8]

    // Iterator::flatten() を使う方法
    let nested2 = vec![vec![1, 2, 3], vec![4, 5], vec![6]];
    let flat2: Vec<i32> = nested2.into_iter().flatten().collect();
    println!("flatten: {:?}", flat2); // [1, 2, 3, 4, 5, 6]

    // Option のイテレータを flatten
    let options = vec![Some(1), None, Some(3), None, Some(5)];
    let values: Vec<i32> = options.into_iter().flatten().collect();
    println!("flatten Option: {:?}", values); // [1, 3, 5]
}
```

### 8.2 ウィンドウ操作と隣接比較

```rust
fn main() {
    let data = vec![1, 3, 2, 5, 4, 7, 6, 8];

    // 隣接ペアの比較（windows を使用）
    let increasing_pairs: Vec<_> = data.windows(2)
        .filter(|w| w[1] > w[0])
        .map(|w| (w[0], w[1]))
        .collect();
    println!("増加ペア: {:?}", increasing_pairs);
    // [(1, 3), (2, 5), (4, 7), (6, 8)]

    // 移動平均
    let values = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0];
    let window_size = 3;
    let moving_avg: Vec<f64> = values.windows(window_size)
        .map(|w| w.iter().sum::<f64>() / window_size as f64)
        .collect();
    println!("移動平均(size=3): {:?}", moving_avg);
    // [20.0, 30.0, 40.0, 50.0, 60.0]

    // 差分系列
    let diffs: Vec<f64> = values.windows(2)
        .map(|w| w[1] - w[0])
        .collect();
    println!("差分: {:?}", diffs); // [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]

    // 連続する同じ値をグループ化（run-length encoding）
    let seq = vec![1, 1, 2, 2, 2, 3, 1, 1];
    let mut rle: Vec<(i32, usize)> = Vec::new();
    for &value in &seq {
        if let Some(last) = rle.last_mut() {
            if last.0 == value {
                last.1 += 1;
                continue;
            }
        }
        rle.push((value, 1));
    }
    println!("RLE: {:?}", rle); // [(1, 2), (2, 3), (3, 1), (1, 2)]
}
```

### 8.3 イテレータアダプタの自作

```rust
// カスタムアダプタ: 各要素とインデックスのペアを返す（enumerate の簡易版）
struct Indexed<I> {
    iter: I,
    index: usize,
}

impl<I: Iterator> Iterator for Indexed<I> {
    type Item = (usize, I::Item);

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.iter.next()?;
        let index = self.index;
        self.index += 1;
        Some((index, item))
    }
}

// 拡張トレイトでメソッドチェーンに組み込む
trait IteratorExt: Iterator + Sized {
    fn indexed(self) -> Indexed<Self> {
        Indexed { iter: self, index: 0 }
    }
}

impl<I: Iterator> IteratorExt for I {}

fn main() {
    let words = vec!["hello", "world", "rust"];
    for (i, word) in words.iter().indexed() {
        println!("[{}] = {}", i, word);
    }

    // チェーンの中で使える
    let result: Vec<_> = (0..5)
        .map(|x| x * x)
        .indexed()
        .filter(|(_, v)| *v > 4)
        .collect();
    println!("{:?}", result); // [(3, 9), (4, 16)]
}
```

---

## 9. パフォーマンスとゼロコスト抽象化

### 9.1 イテレータ vs ループのベンチマーク

```rust
fn main() {
    let n = 10_000_000;
    let data: Vec<i32> = (0..n).collect();

    use std::time::Instant;

    // forループ版
    let start = Instant::now();
    let mut sum1: i64 = 0;
    for &x in &data {
        if x % 2 == 0 {
            sum1 += (x as i64) * (x as i64);
        }
    }
    let t1 = start.elapsed();

    // イテレータ版
    let start = Instant::now();
    let sum2: i64 = data.iter()
        .filter(|&&x| x % 2 == 0)
        .map(|&x| (x as i64) * (x as i64))
        .sum();
    let t2 = start.elapsed();

    assert_eq!(sum1, sum2);
    println!("forループ:   {:?} (sum={})", t1, sum1);
    println!("イテレータ: {:?} (sum={})", t2, sum2);
    // 最適化ビルド (-O / --release) では同等の性能
    // LLVMが同一の機械語に最適化する
}
```

### 9.2 ゼロコスト抽象化の仕組み

```
  イテレータチェーン:
    data.iter().map(|x| x * 2).filter(|x| *x > 10).sum()

  コンパイラの最適化後（概念的な等価コード）:
    let mut sum = 0;
    for &x in data {
        let doubled = x * 2;
        if doubled > 10 {
            sum += doubled;
        }
    }

  中間コレクションは一切生成されない
  map/filter のクロージャはインライン化される
  ループは1回のみ実行される

  さらに SIMD 自動ベクトル化により:
  ┌──────────────────────────────────────┐
  │  手書きループ          → SIMD最適化  │
  │  イテレータチェーン    → SIMD最適化  │
  │  性能差 = ゼロ（同一の機械語）       │
  └──────────────────────────────────────┘
```

### 9.3 イテレータが高速になるケース

```rust
fn main() {
    let data: Vec<i32> = (0..1_000_000).collect();

    // イテレータの方が高速になる場合:
    // 1. 境界チェックの省略
    //    イテレータは内部で長さを追跡するため、
    //    インデックスアクセス時の境界チェックが不要

    // BAD: 毎回境界チェック
    let mut sum1 = 0i64;
    for i in 0..data.len() {
        sum1 += data[i] as i64; // data[i] で境界チェック発生
    }

    // GOOD: 境界チェック不要
    let sum2: i64 = data.iter().map(|&x| x as i64).sum();

    assert_eq!(sum1, sum2);

    // 2. 自動ベクトル化の促進
    //    イテレータはコンパイラにとって最適化しやすいパターン

    // 3. 分岐予測の最適化
    //    filter の条件分岐はコンパイラが最適配置できる
}
```

---

## 10. 比較表

### 10.1 コレクション選択ガイド

| 操作 | Vec | VecDeque | HashMap | BTreeMap | HashSet |
|------|-----|----------|---------|----------|---------|
| 末尾追加 | O(1)* | O(1)* | - | - | - |
| 先頭追加 | O(n) | O(1)* | - | - | - |
| キー検索 | O(n) | O(n) | O(1)* | O(log n) | O(1)* |
| 順序保持 | 挿入順 | 挿入順 | なし | ソート順 | なし |
| メモリ | 連続 | 連続 | ハッシュ | ツリー | ハッシュ |

*= 償却計算量

### 10.2 コレクション詳細比較

| コレクション | push/insert | remove | search | メモリ効率 | キャッシュ効率 |
|-------------|-------------|--------|--------|-----------|--------------|
| Vec | O(1)* 末尾 | O(n) | O(n) | 最高 | 最高 |
| VecDeque | O(1)* 両端 | O(n) | O(n) | 高 | 高 |
| LinkedList | O(1) | O(1) | O(n) | 低 | 低 |
| HashMap | O(1)* | O(1)* | O(1)* | 中 | 中 |
| BTreeMap | O(log n) | O(log n) | O(log n) | 中 | 中 |
| HashSet | O(1)* | O(1)* | O(1)* | 中 | 中 |
| BTreeSet | O(log n) | O(log n) | O(log n) | 中 | 中 |
| BinaryHeap | O(log n) | O(log n) | O(n) | 高 | 高 |

### 10.3 イテレータ消費メソッド

| メソッド | 戻り値 | 説明 |
|----------|--------|------|
| `collect()` | コレクション | イテレータをコレクションに変換 |
| `sum()` | 数値 | 合計を計算 |
| `product()` | 数値 | 総乗を計算 |
| `count()` | usize | 要素数をカウント |
| `any(f)` | bool | 条件を満たす要素があるか |
| `all(f)` | bool | 全要素が条件を満たすか |
| `find(f)` | Option | 条件を満たす最初の要素 |
| `position(f)` | Option | 条件を満たす最初の位置 |
| `min()` / `max()` | Option | 最小/最大値 |
| `min_by_key(f)` / `max_by_key(f)` | Option | 関数による最小/最大 |
| `for_each(f)` | () | 各要素に副作用を適用 |
| `reduce(f)` | Option | 初期値なしの畳み込み |
| `fold(init, f)` | B | 初期値ありの畳み込み |
| `last()` | Option | 最後の要素 |
| `nth(n)` | Option | n番目の要素 |
| `unzip()` | (B, C) | ペアを2つのコレクションに分離 |

### 10.4 イテレータアダプタ一覧

| アダプタ | 説明 | 例 |
|---------|------|-----|
| `map(f)` | 各要素を変換 | `.map(\|x\| x * 2)` |
| `filter(f)` | 条件に合う要素を選択 | `.filter(\|x\| *x > 0)` |
| `filter_map(f)` | 変換+フィルタ | `.filter_map(\|x\| x.parse().ok())` |
| `flat_map(f)` | 変換+フラット化 | `.flat_map(\|s\| s.chars())` |
| `flatten()` | ネストをフラット化 | `.flatten()` |
| `enumerate()` | インデックス付加 | `.enumerate()` |
| `zip(iter)` | 2つを結合 | `.zip(other.iter())` |
| `chain(iter)` | 2つを連結 | `.chain(other.iter())` |
| `take(n)` | 最初のn要素 | `.take(5)` |
| `skip(n)` | 最初のn要素をスキップ | `.skip(3)` |
| `take_while(f)` | 条件を満たす間 | `.take_while(\|x\| *x < 10)` |
| `skip_while(f)` | 条件を満たす間スキップ | `.skip_while(\|x\| *x < 5)` |
| `peekable()` | 先読み可能に | `.peekable()` |
| `scan(state, f)` | 状態付きmap | `.scan(0, \|s, x\| { ... })` |
| `inspect(f)` | デバッグ観察 | `.inspect(\|x\| println!("{}", x))` |
| `step_by(n)` | n要素ごと | `.step_by(2)` |
| `rev()` | 逆順 | `.rev()` |
| `cloned()` | &T → T (Clone) | `.cloned()` |
| `copied()` | &T → T (Copy) | `.copied()` |

---

## 11. アンチパターン

### アンチパターン1: 不必要な collect

```rust
// BAD: 中間コレクションを作る必要がない
fn sum_of_squares(v: &[i32]) -> i32 {
    let squared: Vec<i32> = v.iter().map(|x| x * x).collect(); // 無駄な Vec
    squared.iter().sum()
}

// GOOD: イテレータチェーンを直接消費
fn sum_of_squares_good(v: &[i32]) -> i32 {
    v.iter().map(|x| x * x).sum()
}
```

### アンチパターン2: インデックスベースのループ

```rust
// BAD: C スタイルのインデックスループ
fn print_all(v: &[String]) {
    for i in 0..v.len() {
        println!("{}: {}", i, v[i]); // 境界チェックが毎回走る
    }
}

// GOOD: イテレータを使う
fn print_all_good(v: &[String]) {
    for (i, item) in v.iter().enumerate() {
        println!("{}: {}", i, item); // 安全かつ高速
    }
}
```

### アンチパターン3: 不要なクローン

```rust
// BAD: 不必要にクローン
fn find_longest(strings: &[String]) -> Option<String> {
    strings.iter()
        .cloned()  // 全要素をクローン（高コスト）
        .max_by_key(|s| s.len())
}

// GOOD: 参照で処理
fn find_longest_good(strings: &[String]) -> Option<&String> {
    strings.iter()
        .max_by_key(|s| s.len())
}
```

### アンチパターン4: filter + map を分けて書く

```rust
// BAD: filter と map を別々に
fn parse_valid_numbers(items: &[&str]) -> Vec<i32> {
    items.iter()
        .filter(|s| s.parse::<i32>().is_ok())
        .map(|s| s.parse::<i32>().unwrap()) // 2回パースしている！
        .collect()
}

// GOOD: filter_map を使う
fn parse_valid_numbers_good(items: &[&str]) -> Vec<i32> {
    items.iter()
        .filter_map(|s| s.parse::<i32>().ok())
        .collect()
}
```

### アンチパターン5: collect してから len/is_empty

```rust
// BAD: 全要素を集めてからカウント
fn count_evens(v: &[i32]) -> usize {
    let evens: Vec<_> = v.iter().filter(|&&x| x % 2 == 0).collect();
    evens.len()
}

// GOOD: count() を使う
fn count_evens_good(v: &[i32]) -> usize {
    v.iter().filter(|&&x| x % 2 == 0).count()
}

// BAD: 全要素を集めてから空かチェック
fn has_evens(v: &[i32]) -> bool {
    let evens: Vec<_> = v.iter().filter(|&&x| x % 2 == 0).collect();
    !evens.is_empty()
}

// GOOD: any() を使う（最初の一致で即終了）
fn has_evens_good(v: &[i32]) -> bool {
    v.iter().any(|&x| x % 2 == 0)
}
```

---

## 12. FAQ

### Q1: `collect()` の型をどう指定しますか？

**A:** 3つの方法があります:
```rust
// 方法1: 変数の型注釈
let v: Vec<i32> = (0..10).collect();

// 方法2: ターボフィッシュ構文
let v = (0..10).collect::<Vec<i32>>();

// 方法3: 部分型注釈
let v = (0..10).collect::<Vec<_>>(); // 要素型は推論
```

### Q2: イテレータとforループ、どちらが速いですか？

**A:** 同等です。Rustのイテレータはゼロコスト抽象化であり、コンパイラが同じ機械語に最適化します。イテレータは境界チェックの省略や自動ベクトル化でむしろ高速になるケースもあります。`--release` ビルド（最適化あり）での比較が重要で、`debug` ビルドではイテレータの方が遅くなることがありますが、これはインライン化が無効なためです。

### Q3: HashMap のキーに自作の型を使うには？

**A:** `Eq + Hash` トレイトを実装する必要があります。`#[derive(PartialEq, Eq, Hash)]` で自動実装できます:
```rust
#[derive(PartialEq, Eq, Hash)]
struct Point {
    x: i32,
    y: i32,
}
```
注: `f64` は `Eq` を実装していないため、浮動小数点を含む型はHashMapのキーにできません。`ordered_float::OrderedFloat<f64>` を使うことで回避可能です。

### Q4: `iter()` と `into_iter()` はどう使い分けますか？

**A:** コレクションを後で使う必要がある場合は `iter()`（借用）を使い、もう使わない場合は `into_iter()`（所有権ムーブ）を使います。`into_iter()` はヒープ割り当てをそのまま再利用できるため、文字列のような所有権型を変換する際に効率的です:
```rust
// iter() → クローンが必要
let v = vec!["hello".to_string(), "world".to_string()];
let upper: Vec<String> = v.iter().map(|s| s.to_uppercase()).collect();
println!("{:?}", v); // まだ使える

// into_iter() → クローン不要、元のメモリを再利用
let v = vec!["hello".to_string(), "world".to_string()];
let upper: Vec<String> = v.into_iter().map(|s| s.to_uppercase()).collect();
// println!("{:?}", v); // コンパイルエラー
```

### Q5: 大量のデータを処理する場合、イテレータは十分ですか？

**A:** 単一スレッドでは十分です。並列処理が必要な場合は `rayon` クレートの `par_iter()` を使います:
```rust
use rayon::prelude::*;

let data: Vec<i32> = (0..10_000_000).collect();

// 自動的にスレッドプールで並列処理
let sum: i64 = data.par_iter()
    .map(|&x| (x as i64) * (x as i64))
    .sum();
```
`rayon` は既存のイテレータチェーンを `iter()` → `par_iter()` に変更するだけで並列化できます。

### Q6: コレクションの選択基準は？

**A:** 以下の判断基準で選択してください:

| 要件 | 推奨コレクション |
|------|----------------|
| 順序付きリスト | Vec |
| FIFO キュー | VecDeque |
| キーで高速検索 | HashMap |
| キーでソート順検索 | BTreeMap |
| 重複排除 | HashSet |
| ソート済み集合 | BTreeSet |
| 優先度キュー | BinaryHeap |
| 99% のケース | Vec |

迷ったら `Vec` を使い、ボトルネックが判明してから適切なコレクションに変更するのがRustでの一般的なアプローチです。

---

## 13. まとめ

| 概念 | 要点 |
|------|------|
| Vec<T> | 最も一般的なコレクション。連続メモリ、末尾操作O(1) |
| HashMap<K,V> | キー値ペア。O(1)検索。entry APIで挿入/更新 |
| HashSet<T> | 重複なし集合。集合演算が可能 |
| BTreeMap<K,V> | ソート順保持。範囲検索が可能 |
| VecDeque<T> | 両端キュー。先頭/末尾ともO(1) |
| BinaryHeap<T> | 優先度キュー。最大/最小を効率的に取得 |
| Iterator | next()を実装するトレイト。遅延評価 |
| アダプタ | map/filter/take等。遅延評価で連鎖可能 |
| 消費子 | collect/sum/for_each等。イテレーションを駆動 |
| ゼロコスト | イテレータは手書きループと同等の性能 |
| into_iter vs iter | 所有権を消費するか借用するかの違い |
| FromIterator | collect の変換先を決めるトレイト |
| 拡張トレイト | カスタムアダプタをメソッドチェーンに組み込む |

---

## 次に読むべきガイド

- [../01-advanced/02-closures-fn-traits.md](../01-advanced/02-closures-fn-traits.md) -- クロージャとFnトレイト
- [../01-advanced/00-lifetimes.md](../01-advanced/00-lifetimes.md) -- ライフタイム詳解
- [../02-async/02-async-patterns.md](../02-async/02-async-patterns.md) -- Stream (非同期イテレータ)

---

## 参考文献

1. **The Rust Programming Language - Ch.8 Common Collections, Ch.13 Iterators** -- https://doc.rust-lang.org/book/
2. **std::iter Module Documentation** -- https://doc.rust-lang.org/std/iter/
3. **std::collections Module Documentation** -- https://doc.rust-lang.org/std/collections/
4. **Rust by Example - Iterator** -- https://doc.rust-lang.org/rust-by-example/trait/iter.html
5. **Iterator Performance (Rust Blog)** -- https://blog.rust-lang.org/2017/02/02/Rust-1.15.html
6. **rayon: Data-parallelism library** -- https://docs.rs/rayon/
7. **Rust Performance Book** -- https://nnethercote.github.io/perf-book/
