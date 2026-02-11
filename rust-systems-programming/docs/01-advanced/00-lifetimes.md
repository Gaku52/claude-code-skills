# ライフタイム詳解 -- 参照の有効期間をコンパイル時に証明する仕組み

> ライフタイムはRustコンパイラが参照の有効期間を追跡する仕組みであり、ダングリング参照やuse-after-freeをコンパイル時に排除する。

---

## この章で学ぶこと

1. **ライフタイム注釈 'a** -- 関数シグネチャにおけるライフタイムパラメータの意味と書き方を理解する
2. **ライフタイム省略規則** -- コンパイラが暗黙に注釈を推論する3つの規則を習得する
3. **高度なライフタイム** -- 構造体のライフタイム、HRTB、'static を学ぶ

---

## 1. ライフタイムの基本概念

### 1.1 ダングリング参照の防止

```rust
// このコードはコンパイルエラーになる
// fn dangle() -> &String {
//     let s = String::from("hello");
//     &s  // s はこの関数終了時に drop → ダングリング参照！
// }

// 正しい方法: 所有権を返す
fn no_dangle() -> String {
    String::from("hello")
}
```

### 1.2 ライフタイムの可視化

```
fn main() {
    let r;                // ---------+-- 'a
                          //          |
    {                     //          |
        let x = 5;       // -+-- 'b  |
        r = &x;          //  |       |
    }                     // -+       |  ← x が drop される
                          //          |
    // println!("{}", r); //          |  ← r は無効な参照 → エラー
}                         // ---------+

'b は 'a より短い → r = &x は不正
```

---

## 2. ライフタイム注釈

### 例1: 基本的なライフタイム注釈

```rust
// 2つの文字列スライスのうち長い方を返す
// 戻り値のライフタイムは引数のライフタイムの短い方に制約される
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

fn main() {
    let string1 = String::from("long string");
    let result;
    {
        let string2 = String::from("xyz");
        result = longest(string1.as_str(), string2.as_str());
        println!("長い方: {}", result); // OK: string2 はまだ有効
    }
    // println!("{}", result); // エラー: string2 が drop 済み
}
```

### 例2: 異なるライフタイムを持つ引数

```rust
// x と y が異なるライフタイムでも良い場合
fn first<'a, 'b>(x: &'a str, _y: &'b str) -> &'a str {
    x // 戻り値は x のライフタイムにのみ依存
}

fn main() {
    let s1 = String::from("hello");
    let result;
    {
        let s2 = String::from("world");
        result = first(&s1, &s2);
    }
    println!("{}", result); // OK: result は s1 のライフタイム
}
```

---

## 3. ライフタイム省略規則

```
┌──────────────────────────────────────────────────────────┐
│            ライフタイム省略規則 (Elision Rules)           │
├──────────────────────────────────────────────────────────┤
│                                                          │
│ 規則1 (入力): 各参照パラメータに個別のライフタイムを割当 │
│   fn f(x: &str, y: &str)                                │
│   → fn f<'a, 'b>(x: &'a str, y: &'b str)               │
│                                                          │
│ 規則2 (出力): 入力ライフタイムが1つなら出力にも適用      │
│   fn f(x: &str) -> &str                                 │
│   → fn f<'a>(x: &'a str) -> &'a str                     │
│                                                          │
│ 規則3 (メソッド): &self のライフタイムを出力に適用       │
│   fn f(&self, x: &str) -> &str                          │
│   → fn f<'a, 'b>(&'a self, x: &'b str) -> &'a str      │
│                                                          │
│ 3つの規則で決定できない場合 → 明示的な注釈が必要        │
└──────────────────────────────────────────────────────────┘
```

### 例3: 省略規則の適用例

```rust
// 省略可能なケース
fn first_word(s: &str) -> &str {          // 規則1 + 規則2 で推論
    s.split_whitespace().next().unwrap_or("")
}

// 省略不可能なケース → 明示的注釈が必要
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {  // 入力が2つ
    if x.len() > y.len() { x } else { y }
}
```

---

## 4. 構造体のライフタイム

### 例4: 参照を持つ構造体

```rust
#[derive(Debug)]
struct Excerpt<'a> {
    part: &'a str,
}

impl<'a> Excerpt<'a> {
    fn level(&self) -> i32 {
        3 // 省略規則3: &self のライフタイムが適用
    }

    fn announce_and_return(&self, announcement: &str) -> &str {
        println!("お知らせ: {}", announcement);
        self.part // 省略規則3: &self → 戻り値のライフタイム
    }
}

fn main() {
    let novel = String::from("むかしむかし。ある所に...");
    let first_sentence;
    {
        let excerpt = Excerpt {
            part: novel.split('。').next().unwrap(),
        };
        first_sentence = excerpt.announce_and_return("注目！");
        println!("{:?}", excerpt);
    }
    // first_sentence は novel のスライスなので、novel が有効な限りOK
    println!("{}", first_sentence);
}
```

---

## 5. 'static ライフタイム

### 例5: 'static の正しい使い方

```rust
// 文字列リテラルは 'static
let s: &'static str = "この文字列はバイナリに埋め込まれる";

// 'static 境界: 型が参照を含まない or 'static 参照のみ
fn spawn_task<T: Send + 'static>(value: T) {
    std::thread::spawn(move || {
        println!("スレッドで処理中");
        drop(value);
    });
}

// 'static は「永遠に生きる」ではなく「永遠に生きられる」という意味
// 所有型(String, Vec<T>)は全て 'static 境界を満たす
fn accepts_static<T: 'static>(val: T) {
    // T が参照を含むなら 'static 参照のみ
    // T が所有型なら常に満たす
}

fn main() {
    let owned = String::from("hello");
    accepts_static(owned); // OK: String は所有型

    let s: &'static str = "hello";
    accepts_static(s); // OK: 'static 参照

    // let local = String::from("hello");
    // accepts_static(&local); // エラー: &local は 'static ではない
}
```

---

## 6. 高ランクトレイト境界 (HRTB)

```
┌──────────────────────────────────────────────────────┐
│  HRTB (Higher-Rank Trait Bounds)                     │
│                                                      │
│  for<'a> は「任意のライフタイム 'a に対して」の意味   │
│                                                      │
│  fn apply<F>(f: F)                                   │
│  where                                               │
│      F: for<'a> Fn(&'a str) -> &'a str               │
│                                                      │
│  → F は「どんなライフタイムの参照を渡されても        │
│     動作する関数」でなければならない                  │
└──────────────────────────────────────────────────────┘
```

### 例6: HRTB の実用例

```rust
fn apply_to_both<F>(f: F, a: &str, b: &str)
where
    F: for<'a> Fn(&'a str) -> &'a str,
{
    println!("{}", f(a));
    println!("{}", f(b));
}

fn identity(s: &str) -> &str {
    s
}

fn main() {
    let s1 = String::from("hello");
    let s2 = String::from("world");
    apply_to_both(identity, &s1, &s2);
}
```

---

## 7. ライフタイムのサブタイピング

```
ライフタイムの包含関係:
  'a: 'b は「'a は 'b より長く生きる」(outlives)

  ┌─────────────────────────────────┐
  │ 'static                        │
  │  ┌───────────────────────────┐  │
  │  │ 'a                       │  │
  │  │  ┌─────────────────────┐ │  │
  │  │  │ 'b                  │ │  │
  │  │  └─────────────────────┘ │  │
  │  └───────────────────────────┘  │
  └─────────────────────────────────┘

  'static: 'a: 'b
  'static は全てのライフタイムより長い
  より長いライフタイムの参照は、
  短いライフタイムが期待される場所で使える
```

### 例7: ライフタイム境界

```rust
// 'a: 'b は「'a は少なくとも 'b と同じ長さ」
fn select<'a, 'b: 'a>(first: &'a str, second: &'b str) -> &'a str {
    if first.len() > second.len() {
        first
    } else {
        second // 'b: 'a なので 'b の参照を 'a として返せる
    }
}
```

---

## 8. 比較表

### 8.1 ライフタイムの種類

| 種類 | 記法 | 意味 | 例 |
|------|------|------|-----|
| 名前付き | `'a` | 明示的なライフタイムパラメータ | `fn f<'a>(x: &'a str)` |
| 省略 | なし | コンパイラが推論 | `fn f(x: &str) -> &str` |
| 'static | `'static` | プログラム全期間 | `&'static str` |
| 匿名 | `'_` | 推論を明示的に要求 | `impl Iterator<Item = &'_ str>` |

### 8.2 'static の誤解と実際

| 誤解 | 実際 |
|------|------|
| 永遠にメモリに残る | 永遠に有効な「資格がある」 |
| 文字列リテラルだけ | 所有型は全て `'static` を満たす |
| ヒープにある | バイナリの静的領域にある(リテラルの場合) |
| 使うべきでない | スレッドに渡す値には必要 |

---

## 9. アンチパターン

### アンチパターン1: 不要な 'static 制約

```rust
// BAD: 'static を要求しすぎ
fn process(data: &'static str) {
    println!("{}", data);
}

// GOOD: 任意のライフタイムを受け入れる
fn process_good(data: &str) {
    println!("{}", data);
}
```

### アンチパターン2: ライフタイムで戦うより所有型を使う

```rust
// BAD: ライフタイムが複雑になりすぎ
// struct Parser<'input, 'config, 'db> {
//     input: &'input str,
//     config: &'config Config,
//     db: &'db Database,
// }

// GOOD: 所有型で簡潔にする
struct Parser {
    input: String,
    config: Config,
    db: Database,
}
// パフォーマンスが問題になったら後でライフタイムを導入
```

---

## 10. FAQ

### Q1: ライフタイム注釈は実行時に何か影響しますか？

**A:** いいえ。ライフタイム注釈は完全にコンパイル時の情報です。実行時のコードやパフォーマンスには一切影響しません。バイナリにライフタイムの情報は含まれません。

### Q2: NLL (Non-Lexical Lifetimes) とは何ですか？

**A:** Rust 2018 Editionで導入された改善です。従来、参照のライフタイムはレキシカルスコープ(ブロック終端)まで続きましたが、NLLでは「最後に使用された地点」で終了します。これにより、以前はコンパイルエラーだった正当なコードが通るようになりました。

### Q3: `'_` (匿名ライフタイム)はいつ使いますか？

**A:** ライフタイムの存在を明示しつつ、具体的な名前は不要な場合に使います:
```rust
// impl ブロックでライフタイムの存在だけ示す
impl fmt::Display for ImportantExcerpt<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.part)
    }
}
```

---

## 11. まとめ

| 概念 | 要点 |
|------|------|
| ライフタイム | 参照の有効期間をコンパイラが追跡する仕組み |
| 'a 注釈 | 参照間の関係をコンパイラに伝える |
| 省略規則 | 3つの規則で多くの場合は注釈不要 |
| 構造体のLT | 参照フィールドを持つ構造体にはLT注釈が必要 |
| 'static | プログラム全体の期間。所有型は全て満たす |
| HRTB | `for<'a>` で任意のライフタイムに対する制約 |
| NLL | 最後の使用地点でライフタイム終了 |

---

## 次に読むべきガイド

- [01-smart-pointers.md](01-smart-pointers.md) -- Box/Rc/Arc でライフタイムの制約を緩和する
- [02-closures-fn-traits.md](02-closures-fn-traits.md) -- クロージャとライフタイムの関係
- [03-unsafe-rust.md](03-unsafe-rust.md) -- unsafe でライフタイムチェックを回避する危険性

---

## 参考文献

1. **The Rust Programming Language - Ch.10.3 Validating References with Lifetimes** -- https://doc.rust-lang.org/book/ch10-03-lifetime-syntax.html
2. **The Rustonomicon - Lifetimes** -- https://doc.rust-lang.org/nomicon/lifetimes.html
3. **Common Rust Lifetime Misconceptions (pretzelhammer)** -- https://github.com/pretzelhammer/rust-blog/blob/master/posts/common-rust-lifetime-misconceptions.md
