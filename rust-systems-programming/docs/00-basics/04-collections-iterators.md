# コレクションとイテレータ -- Rustの関数型データ処理パイプライン

> Vec、HashMap等のコレクションとIterator traitを組み合わせることで、型安全かつゼロコスト抽象化されたデータ処理パイプラインを構築できる。

---

## この章で学ぶこと

1. **主要コレクション** -- Vec、HashMap、HashSet、BTreeMap 等の使い分けを理解する
2. **Iterator トレイト** -- イテレータプロトコルと遅延評価の仕組みを習得する
3. **イテレータアダプタ** -- map/filter/fold/collect 等のチェーンによるデータ変換を学ぶ

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

---

## 4. Iterator トレイト

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
```

---

## 5. イテレータアダプタ詳解

### 例4: 主要なアダプタとチェーン

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

### 例5: 実践的なイテレータパイプライン

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

---

## 7. 比較表

### 7.1 コレクション選択ガイド

| 操作 | Vec | VecDeque | HashMap | BTreeMap | HashSet |
|------|-----|----------|---------|----------|---------|
| 末尾追加 | O(1)* | O(1)* | - | - | - |
| 先頭追加 | O(n) | O(1)* | - | - | - |
| キー検索 | O(n) | O(n) | O(1)* | O(log n) | O(1)* |
| 順序保持 | 挿入順 | 挿入順 | なし | ソート順 | なし |
| メモリ | 連続 | 連続 | ハッシュ | ツリー | ハッシュ |

*= 償却計算量

### 7.2 イテレータ消費メソッド

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
| `for_each(f)` | () | 各要素に副作用を適用 |

---

## 8. アンチパターン

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

---

## 9. FAQ

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

**A:** 同等です。Rustのイテレータはゼロコスト抽象化であり、コンパイラが同じ機械語に最適化します。イテレータは境界チェックの省略や自動ベクトル化でむしろ高速になるケースもあります。

### Q3: HashMap のキーに自作の型を使うには？

**A:** `Eq + Hash` トレイトを実装する必要があります。`#[derive(PartialEq, Eq, Hash)]` で自動実装できます:
```rust
#[derive(PartialEq, Eq, Hash)]
struct Point {
    x: i32,
    y: i32,
}
```
注: `f64` は `Eq` を実装していないため、浮動小数点を含む型はHashMapのキーにできません。

---

## 10. まとめ

| 概念 | 要点 |
|------|------|
| Vec<T> | 最も一般的なコレクション。連続メモリ、末尾操作O(1) |
| HashMap<K,V> | キー値ペア。O(1)検索。entry APIで挿入/更新 |
| Iterator | next()を実装するトレイト。遅延評価 |
| アダプタ | map/filter/take等。遅延評価で連鎖可能 |
| 消費子 | collect/sum/for_each等。イテレーションを駆動 |
| ゼロコスト | イテレータは手書きループと同等の性能 |
| into_iter vs iter | 所有権を消費するか借用するかの違い |

---

## 次に読むべきガイド

- [../01-advanced/02-closures-fn-traits.md](../01-advanced/02-closures-fn-traits.md) -- クロージャとFnトレイト
- [../01-advanced/00-lifetimes.md](../01-advanced/00-lifetimes.md) -- ライフタイム詳解
- [../02-async/02-async-patterns.md](../02-async/02-async-patterns.md) -- Stream (非同期イテレータ)

---

## 参考文献

1. **The Rust Programming Language - Ch.8 Common Collections, Ch.13 Iterators** -- https://doc.rust-lang.org/book/
2. **std::iter Module Documentation** -- https://doc.rust-lang.org/std/iter/
3. **Rust by Example - Iterator** -- https://doc.rust-lang.org/rust-by-example/trait/iter.html
4. **Iterator Performance (Rust Blog)** -- https://blog.rust-lang.org/2017/02/02/Rust-1.15.html
