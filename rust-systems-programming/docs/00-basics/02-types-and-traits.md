# 型とトレイト -- Rustの型システムとポリモーフィズムの基盤

> Rustの型システムは struct/enum による代数的データ型と trait によるアドホックポリモーフィズムを基盤とし、ジェネリクスと組み合わせてゼロコスト抽象化を実現する。

---

## この章で学ぶこと

1. **struct と enum** -- 代数的データ型(直積型・直和型)の定義と活用を理解する
2. **トレイト** -- インターフェースの定義、実装、デフォルトメソッドを習得する
3. **ジェネリクスとトレイト境界** -- 型パラメータと制約による汎用コードの書き方を学ぶ

---

## 1. 基本型

### 1.1 プリミティブ型の一覧

```
┌─────────────────────────────────────────────────────┐
│                 Rust プリミティブ型                   │
├──────────┬──────────────────────────────────────────┤
│ 整数     │ i8 i16 i32 i64 i128 isize                │
│          │ u8 u16 u32 u64 u128 usize                │
├──────────┼──────────────────────────────────────────┤
│ 浮動小数 │ f32, f64                                 │
├──────────┼──────────────────────────────────────────┤
│ 論理     │ bool                                     │
├──────────┼──────────────────────────────────────────┤
│ 文字     │ char (4バイト Unicode スカラ値)            │
├──────────┼──────────────────────────────────────────┤
│ タプル   │ (T1, T2, ...) -- 異なる型の組み合わせ     │
├──────────┼──────────────────────────────────────────┤
│ 配列     │ [T; N] -- 固定長、スタック上              │
├──────────┼──────────────────────────────────────────┤
│ 参照     │ &T, &mut T                               │
├──────────┼──────────────────────────────────────────┤
│ unit     │ () -- 値なし(C の void に相当)           │
└──────────┴──────────────────────────────────────────┘
```

---

## 2. struct(構造体)

### 例1: 名前付きフィールド構造体

```rust
struct User {
    name: String,
    email: String,
    age: u32,
    active: bool,
}

fn main() {
    let user = User {
        name: String::from("田中"),
        email: String::from("tanaka@example.com"),
        age: 30,
        active: true,
    };
    println!("{} ({}歳)", user.name, user.age);
}
```

### 例2: タプル構造体とユニット構造体

```rust
// タプル構造体: フィールド名なし
struct Color(u8, u8, u8);
struct Meters(f64);

// ユニット構造体: フィールドなし(マーカー型に使用)
struct Marker;

fn main() {
    let red = Color(255, 0, 0);
    let distance = Meters(42.0);
    let _m = Marker;
    println!("R={}, 距離={}m", red.0, distance.0);
}
```

---

## 3. enum(列挙型)

### 例3: 代数的データ型としての enum

```rust
enum Shape {
    Circle { radius: f64 },
    Rectangle { width: f64, height: f64 },
    Triangle { base: f64, height: f64 },
}

impl Shape {
    fn area(&self) -> f64 {
        match self {
            Shape::Circle { radius } => std::f64::consts::PI * radius * radius,
            Shape::Rectangle { width, height } => width * height,
            Shape::Triangle { base, height } => 0.5 * base * height,
        }
    }
}

fn main() {
    let shapes = vec![
        Shape::Circle { radius: 5.0 },
        Shape::Rectangle { width: 4.0, height: 6.0 },
        Shape::Triangle { base: 3.0, height: 8.0 },
    ];
    for s in &shapes {
        println!("面積: {:.2}", s.area());
    }
}
```

### 3.1 Option と Result

```
┌────────────────────────────┬────────────────────────────────┐
│   Option<T>                │   Result<T, E>                 │
├────────────────────────────┼────────────────────────────────┤
│ enum Option<T> {           │ enum Result<T, E> {            │
│     Some(T),               │     Ok(T),                     │
│     None,                  │     Err(E),                    │
│ }                          │ }                              │
├────────────────────────────┼────────────────────────────────┤
│ 値が存在しない可能性       │ 処理が失敗する可能性           │
│ null の安全な代替           │ 例外の安全な代替               │
└────────────────────────────┴────────────────────────────────┘
```

---

## 4. impl ブロック

### 例4: メソッドと関連関数

```rust
struct Rectangle {
    width: f64,
    height: f64,
}

impl Rectangle {
    // 関連関数(コンストラクタ) -- Self を引数に取らない
    fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }

    fn square(size: f64) -> Self {
        Self { width: size, height: size }
    }

    // メソッド -- &self を引数に取る
    fn area(&self) -> f64 {
        self.width * self.height
    }

    fn perimeter(&self) -> f64 {
        2.0 * (self.width + self.height)
    }

    // 可変メソッド
    fn scale(&mut self, factor: f64) {
        self.width *= factor;
        self.height *= factor;
    }
}

fn main() {
    let mut rect = Rectangle::new(10.0, 5.0);
    println!("面積: {}", rect.area());       // 50.0
    println!("周囲: {}", rect.perimeter());  // 30.0
    rect.scale(2.0);
    println!("拡大後面積: {}", rect.area()); // 200.0
}
```

---

## 5. トレイト

### 5.1 トレイト定義と実装

### 例5: トレイトの定義と実装

```rust
trait Summary {
    fn summarize_author(&self) -> String;

    // デフォルト実装
    fn summarize(&self) -> String {
        format!("({}による記事をもっと読む...)", self.summarize_author())
    }
}

struct Article {
    title: String,
    author: String,
    content: String,
}

impl Summary for Article {
    fn summarize_author(&self) -> String {
        self.author.clone()
    }

    fn summarize(&self) -> String {
        format!("{} -- {} (by {})", self.title, &self.content[..20], self.author)
    }
}

struct Tweet {
    username: String,
    text: String,
}

impl Summary for Tweet {
    fn summarize_author(&self) -> String {
        format!("@{}", self.username)
    }
    // summarize() はデフォルト実装を使用
}
```

### 例6: トレイト境界付きジェネリクス

```rust
use std::fmt::Display;

// 方法1: トレイト境界構文
fn notify<T: Summary + Display>(item: &T) {
    println!("速報: {}", item.summarize());
}

// 方法2: where句 (複雑な場合に推奨)
fn complex_function<T, U>(t: &T, u: &U) -> String
where
    T: Summary + Clone,
    U: Display + Debug,
{
    format!("{}: {:?}", t.summarize(), u)
}

// 方法3: impl Trait 構文 (引数の場合)
fn notify_simple(item: &impl Summary) {
    println!("速報: {}", item.summarize());
}

// impl Trait は戻り値にも使える
fn create_summarizable() -> impl Summary {
    Tweet {
        username: String::from("rustlang"),
        text: String::from("Rust is great!"),
    }
}
```

### 5.2 よく使う標準トレイト

```
┌───────────────┬───────────────────────────────────────────┐
│ トレイト       │ 用途                                      │
├───────────────┼───────────────────────────────────────────┤
│ Display       │ {} フォーマット表示                        │
│ Debug         │ {:?} デバッグ表示                         │
│ Clone         │ 明示的な深いコピー (.clone())              │
│ Copy          │ 暗黙のビットコピー                        │
│ PartialEq/Eq │ == / != 比較                              │
│ PartialOrd/Ord│ < > <= >= 比較 / ソート                   │
│ Hash          │ ハッシュ値計算(HashMapのキーに必要)       │
│ Default       │ デフォルト値生成                           │
│ From/Into     │ 型変換                                    │
│ Iterator      │ イテレータプロトコル                       │
│ Drop          │ デストラクタ(スコープ終了時の処理)        │
│ Deref         │ 自動参照解決                              │
│ Send/Sync     │ スレッド安全性マーカー                    │
└───────────────┴───────────────────────────────────────────┘
```

### 例7: derive マクロで自動実装

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
struct Config {
    host: String,
    port: u16,
    debug: bool,
}

fn main() {
    let config = Config {
        host: "localhost".to_string(),
        port: 8080,
        debug: true,
    };
    let config2 = config.clone();
    println!("{:?}", config);
    println!("同じ? {}", config == config2);

    let default_config = Config::default();
    println!("デフォルト: {:?}", default_config);
}
```

---

## 6. ジェネリクス

### 例8: ジェネリック構造体と関数

```rust
struct Pair<T> {
    first: T,
    second: T,
}

impl<T: PartialOrd + Display> Pair<T> {
    fn new(first: T, second: T) -> Self {
        Self { first, second }
    }

    fn larger(&self) -> &T {
        if self.first >= self.second {
            &self.first
        } else {
            &self.second
        }
    }
}

fn main() {
    let pair = Pair::new(10, 20);
    println!("大きい方: {}", pair.larger()); // 20
}
```

---

## 7. 比較表

### 7.1 struct vs enum

| 特性 | struct | enum |
|------|--------|------|
| 代数型 | 直積型 (product type) | 直和型 (sum type) |
| フィールド | 全フィールド同時に持つ | バリアントの1つだけ |
| パターンマッチ | 分解代入 | match で網羅的に分岐 |
| メモリ | 全フィールドの合計 | 最大バリアント + タグ |
| 用途 | データのまとまり | 状態・選択肢の表現 |

### 7.2 静的ディスパッチ vs 動的ディスパッチ

| 特性 | 静的 (impl Trait / ジェネリクス) | 動的 (dyn Trait) |
|------|-------------------------------|------------------|
| 仕組み | 単態化 (monomorphization) | vtable (仮想関数テーブル) |
| 実行速度 | 高速(インライン化可能) | オーバーヘッドあり |
| バイナリサイズ | 大きくなりやすい | 小さい |
| 型消去 | なし(コンパイル時に具体型決定) | あり |
| 使い方 | `fn f(x: impl Trait)` | `fn f(x: &dyn Trait)` |
| オブジェクト安全 | 不要 | 必要 |

---

## 8. アンチパターン

### アンチパターン1: String フィールドに &str を入れようとする

```rust
// BAD: ライフタイムが必要で複雑になる
// struct User<'a> {
//     name: &'a str,  // ライフタイム注釈が伝播して大変
// }

// GOOD: 所有型を使う (ほとんどのケースで推奨)
struct User {
    name: String,  // 構造体は自身のデータを所有する
}
```

### アンチパターン2: トレイトオブジェクトの不必要な使用

```rust
// BAD: ジェネリクスで十分なのに動的ディスパッチ
fn process(items: &[Box<dyn Summary>]) {
    for item in items {
        println!("{}", item.summarize());
    }
}

// GOOD: 同一型なら静的ディスパッチ
fn process_good<T: Summary>(items: &[T]) {
    for item in items {
        println!("{}", item.summarize());
    }
}
// 注: 異なる型を混在させるならdyn Traitが正解
```

---

## 9. FAQ

### Q1: struct のフィールドを String にするか &str にするかの基準は？

**A:** 原則として構造体には所有型(String)を使います。構造体が独立してデータを管理でき、ライフタイムの伝播を避けられます。パフォーマンスが重要で短命な構造体(パーサーの中間結果など)の場合のみ `&str` + ライフタイムを検討してください。

### Q2: `impl Trait` と `dyn Trait` はどう使い分けますか？

**A:**
- **`impl Trait`**: コンパイル時に具体型が決まる場合。高速で型安全。
- **`dyn Trait`**: 実行時に異なる型を扱う場合。`Vec<Box<dyn Trait>>` のようにコレクションに異なる型を入れたいとき。

### Q3: trait を実装するとき、孤児ルールとは何ですか？

**A:** 他のクレートが定義したトレイトを、他のクレートが定義した型に実装することはできません。少なくとも型またはトレイトのどちらかが自分のクレートで定義されている必要があります。これにより実装の衝突を防ぎます。

```rust
// OK: 自分のトレイトを外部の型に実装
impl MyTrait for Vec<i32> { ... }

// OK: 外部のトレイトを自分の型に実装
impl Display for MyStruct { ... }

// NG: 外部のトレイトを外部の型に実装
// impl Display for Vec<i32> { ... }  // コンパイルエラー
```

---

## 10. まとめ

| 概念 | 要点 |
|------|------|
| struct | 名前付き/タプル/ユニットの3種類。直積型 |
| enum | バリアントを持つ直和型。パターンマッチで分岐 |
| impl | メソッドと関連関数を定義するブロック |
| trait | インターフェース定義。デフォルト実装も可能 |
| ジェネリクス | 型パラメータで汎用コードを記述 |
| トレイト境界 | ジェネリクスに制約を付ける (`T: Clone + Debug`) |
| derive | 標準トレイトの自動実装 |
| 静的/動的ディスパッチ | 単態化 vs vtable。用途に応じて選択 |

---

## 次に読むべきガイド

- [03-error-handling.md](03-error-handling.md) -- Result/Option を活用したエラー処理
- [04-collections-iterators.md](04-collections-iterators.md) -- コレクションとイテレータ
- [../01-advanced/00-lifetimes.md](../01-advanced/00-lifetimes.md) -- ライフタイム詳解

---

## 参考文献

1. **The Rust Programming Language - Ch.5 Structs, Ch.6 Enums, Ch.10 Generics/Traits** -- https://doc.rust-lang.org/book/
2. **Rust by Example - Custom Types** -- https://doc.rust-lang.org/rust-by-example/custom_types.html
3. **Rust API Guidelines - Type Safety** -- https://rust-lang.github.io/api-guidelines/type-safety.html
