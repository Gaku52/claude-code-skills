# ライフタイム詳解 -- 参照の有効期間をコンパイル時に証明する仕組み

> ライフタイムはRustコンパイラが参照の有効期間を追跡する仕組みであり、ダングリング参照やuse-after-freeをコンパイル時に排除する。

---

## この章で学ぶこと

1. **ライフタイム注釈 'a** -- 関数シグネチャにおけるライフタイムパラメータの意味と書き方を理解する
2. **ライフタイム省略規則** -- コンパイラが暗黙に注釈を推論する3つの規則を習得する
3. **高度なライフタイム** -- 構造体のライフタイム、HRTB、'static を学ぶ
4. **NLL (Non-Lexical Lifetimes)** -- Rust 2018以降の改善されたライフタイム解析を理解する
5. **実践パターン** -- 実務で遭遇する複雑なライフタイムシナリオへの対処法を習得する

---

## 1. ライフタイムの基本概念

### 1.1 なぜライフタイムが必要なのか

Rustはガベージコレクタ (GC) を持たない言語であり、メモリ安全性をコンパイル時に保証する。ライフタイムはその核心的な仕組みの1つであり、以下の問題を防ぐ:

- **ダングリング参照 (dangling reference)**: 既に解放されたメモリへの参照
- **Use-after-free**: 解放後のメモリアクセス
- **データ競合 (data race)**: 参照の有効期間管理による排他制御の基盤

C/C++ ではこれらは実行時にクラッシュや未定義動作を引き起こすが、Rustではコンパイル時に検出・排除される。

```rust
// C言語で起きる典型的なダングリング参照
// int* create_int() {
//     int x = 42;
//     return &x;  // スタックフレームが消える → ダングリング!
// }

// Rustでは同等のコードがコンパイルエラーになる
// fn create_ref() -> &i32 {
//     let x = 42;
//     &x  // コンパイルエラー: `x` does not live long enough
// }

// 正しい方法: 所有権を返す
fn create_value() -> i32 {
    42
}

// またはヒープに確保して所有権を返す
fn create_string() -> String {
    String::from("hello")
}
```

### 1.2 ダングリング参照の防止

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

fn main() {
    let s = no_dangle();
    println!("{}", s); // OK: s が所有権を持っている
}
```

### 1.3 ライフタイムの可視化

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

### 1.4 借用チェッカーの動作原理

借用チェッカー (borrow checker) は以下のステップでライフタイムを検証する:

1. **ライフタイムの割り当て**: 各参照にライフタイムリージョンを割り当てる
2. **制約の収集**: 関数シグネチャ、変数の使用箇所から制約を収集する
3. **制約の解決**: 全ての制約を同時に満たすライフタイムの割り当てが存在するか検証する
4. **エラー報告**: 制約を満たせない場合、具体的なエラーメッセージを生成する

```rust
fn example() {
    let x = String::from("hello");  // x のライフタイム開始
    let r = &x;                      // r は x への参照。'r <= 'x が制約
    println!("{}", r);               // r の最終使用地点
    // r のライフタイム終了 (NLL)
    drop(x);                         // x のライフタイム終了 → OK
}

fn failing_example() {
    let r;
    {
        let x = String::from("hello");
        r = &x;                      // 'r は外側のスコープまで続く
    }                                // x の drop → 'x 終了
    // println!("{}", r);            // 'r > 'x → 制約違反 → エラー
}
```

---

## 2. ライフタイム注釈

### 2.1 ライフタイム注釈の構文

ライフタイム注釈はアポストロフィ `'` に続く小文字のアルファベットで記述する。慣例では `'a`, `'b`, `'c` のように短い名前を使う。

```rust
// 基本構文
&'a T        // ライフタイム 'a を持つ不変参照
&'a mut T    // ライフタイム 'a を持つ可変参照

// ジェネリックライフタイムパラメータ
fn function<'a>(x: &'a str) -> &'a str { x }

// 複数のライフタイムパラメータ
fn function2<'a, 'b>(x: &'a str, y: &'b str) -> &'a str { x }

// ライフタイム境界付き
fn function3<'a, 'b: 'a>(x: &'a str, y: &'b str) -> &'a str {
    if x.len() > 0 { x } else { y }
}
```

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

### 例3: 戻り値が新しい値の場合

```rust
// 戻り値が引数の参照ではなく新しい値の場合、ライフタイム注釈は不要
fn combine(x: &str, y: &str) -> String {
    format!("{}{}", x, y) // 新しい String を返す → ライフタイム不要
}

// 以下はコンパイルエラーになる
// fn bad_return<'a>(x: &'a str) -> &'a str {
//     let s = String::from("created inside");
//     &s  // ローカル変数への参照は返せない
// }

fn main() {
    let result = combine("hello", " world");
    println!("{}", result);
}
```

### 例4: 複数の戻り値候補

```rust
// 条件によって異なる引数を返す場合、ライフタイムの共通化が必要
fn select<'a>(condition: bool, x: &'a str, y: &'a str) -> &'a str {
    if condition { x } else { y }
}

// より精密なライフタイム指定
fn select_first<'a, 'b>(condition: bool, x: &'a str, _y: &'b str) -> &'a str {
    if condition {
        x
    } else {
        // y は返せない: 'b != 'a
        // 代わりにデフォルト値を返す
        "default"  // &'static str は任意のライフタイムに変換可能
    }
}

fn main() {
    let s1 = String::from("first");
    let result;
    {
        let s2 = String::from("second");
        result = select(true, &s1, &s2);
        println!("{}", result);
    }

    let s3 = String::from("third");
    let result2;
    {
        let s4 = String::from("fourth");
        result2 = select_first(false, &s3, &s4);
    }
    println!("{}", result2); // OK: "default" は 'static
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

### 例5: 省略規則の適用例

```rust
// === 規則1 + 規則2 で省略可能 ===

// 省略形
fn first_word(s: &str) -> &str {
    s.split_whitespace().next().unwrap_or("")
}

// 展開するとこうなる
fn first_word_explicit<'a>(s: &'a str) -> &'a str {
    s.split_whitespace().next().unwrap_or("")
}

// === 規則1のみ → 出力ライフタイムが決まらない → 明示必要 ===

// 省略不可能なケース → 明示的注釈が必要
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {  // 入力が2つ
    if x.len() > y.len() { x } else { y }
}

// === 規則3 の適用 ===

struct MyString {
    data: String,
}

impl MyString {
    // 規則3: &self のライフタイムが戻り値に適用
    // 省略形
    fn as_str(&self) -> &str {
        &self.data
    }

    // 展開するとこうなる
    fn as_str_explicit<'a>(&'a self) -> &'a str {
        &self.data
    }

    // 規則3: 他の引数のライフタイムは無視される
    fn with_prefix(&self, prefix: &str) -> &str {
        // &self のライフタイムが戻り値に適用される
        // prefix のライフタイムではない
        &self.data
    }
}
```

### 例6: 省略規則が適用されない複雑なケース

```rust
// ケース1: 2つの入力参照、戻り値がどちらに依存するか不明
// fn ambiguous(x: &str, y: &str) -> &str { ... }  // コンパイルエラー
fn not_ambiguous<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() { x } else { y }
}

// ケース2: トレイトオブジェクトのライフタイム
// Box<dyn Trait> のライフタイムはデフォルトで 'static
fn create_trait_obj() -> Box<dyn std::fmt::Display> {
    Box::new(42)  // i32 は 'static
}

// 非 'static のトレイトオブジェクト
fn create_trait_obj_with_ref<'a>(s: &'a str) -> Box<dyn std::fmt::Display + 'a> {
    Box::new(s)
}

// ケース3: impl Trait のライフタイム
fn create_iter<'a>(s: &'a str) -> impl Iterator<Item = char> + 'a {
    s.chars()
}

fn main() {
    let s = String::from("hello world");
    let result = not_ambiguous(&s, "default");
    println!("{}", result);

    let obj = create_trait_obj();
    println!("{}", obj);

    let chars: Vec<char> = create_iter(&s).collect();
    println!("{:?}", chars);
}
```

---

## 4. 構造体のライフタイム

### 例7: 参照を持つ構造体

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

### 例8: 複数のライフタイムを持つ構造体

```rust
#[derive(Debug)]
struct Pair<'a, 'b> {
    first: &'a str,
    second: &'b str,
}

impl<'a, 'b> Pair<'a, 'b> {
    fn new(first: &'a str, second: &'b str) -> Self {
        Pair { first, second }
    }

    fn first(&self) -> &'a str {
        self.first
    }

    fn second(&self) -> &'b str {
        self.second
    }

    // 両方のライフタイムに依存する戻り値
    fn longer(&self) -> &str
    where
        'a: 'b,  // 'a が 'b より長生きする制約
    {
        if self.first.len() > self.second.len() {
            self.first
        } else {
            self.second
        }
    }
}

fn main() {
    let s1 = String::from("hello");
    let result;
    {
        let s2 = String::from("world!!!");
        let pair = Pair::new(&s1, &s2);
        println!("first: {}, second: {}", pair.first(), pair.second());
        // pair.longer() は 'b の制約内でのみ使用可能
        let longer = pair.longer();
        println!("longer: {}", longer);
    }
    // result = pair.longer(); // pair が drop 済みなのでエラー
}
```

### 例9: 構造体のライフタイムとジェネリクスの組み合わせ

```rust
use std::fmt::Display;

#[derive(Debug)]
struct Annotated<'a, T> {
    label: &'a str,
    value: T,
}

impl<'a, T: Display> Annotated<'a, T> {
    fn new(label: &'a str, value: T) -> Self {
        Annotated { label, value }
    }

    fn display(&self) {
        println!("{}: {}", self.label, self.value);
    }

    fn label(&self) -> &'a str {
        self.label
    }
}

impl<'a, T: Display> std::fmt::Display for Annotated<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.label, self.value)
    }
}

fn main() {
    let label = String::from("温度");
    let annotated = Annotated::new(&label, 36.5_f64);
    annotated.display();
    println!("{}", annotated);
    println!("ラベル: {}", annotated.label());
}
```

### 例10: 自己参照構造体の問題と解決策

```rust
// 自己参照構造体は直接作れない
// struct SelfRef {
//     data: String,
//     reference: &str,  // data を参照したいが、ライフタイムを指定できない
// }

// 解決策1: インデックスで間接参照
#[derive(Debug)]
struct TextWithHighlight {
    text: String,
    highlight_start: usize,
    highlight_end: usize,
}

impl TextWithHighlight {
    fn new(text: String, start: usize, end: usize) -> Self {
        assert!(end <= text.len());
        assert!(start <= end);
        TextWithHighlight {
            text,
            highlight_start: start,
            highlight_end: end,
        }
    }

    fn highlighted(&self) -> &str {
        &self.text[self.highlight_start..self.highlight_end]
    }

    fn full_text(&self) -> &str {
        &self.text
    }
}

// 解決策2: 分離した構造体
#[derive(Debug)]
struct TextOwner {
    text: String,
}

#[derive(Debug)]
struct TextRef<'a> {
    owner: &'a TextOwner,
    start: usize,
    end: usize,
}

impl<'a> TextRef<'a> {
    fn new(owner: &'a TextOwner, start: usize, end: usize) -> Self {
        assert!(end <= owner.text.len());
        TextRef { owner, start, end }
    }

    fn get(&self) -> &str {
        &self.owner.text[self.start..self.end]
    }
}

fn main() {
    // 解決策1
    let tw = TextWithHighlight::new("Hello, Rust World!".to_string(), 7, 11);
    println!("全体: {}", tw.full_text());
    println!("強調: {}", tw.highlighted());

    // 解決策2
    let owner = TextOwner {
        text: "Hello, Rust World!".to_string(),
    };
    let text_ref = TextRef::new(&owner, 7, 11);
    println!("参照: {}", text_ref.get());
}
```

---

## 5. 'static ライフタイム

### 5.1 'static の2つの意味

`'static` には2つの異なる意味があり、混同しやすい:

1. **`&'static T`**: プログラム全期間にわたって有効な参照
2. **`T: 'static`**: 型 T が 'static ライフタイム境界を満たす (所有型は全て満たす)

```
┌──────────────────────────────────────────────────────┐
│  'static の2つの意味                                  │
├──────────────────────────────────────────────────────┤
│                                                      │
│  &'static T = プログラム全期間有効な参照              │
│    例: 文字列リテラル &'static str                    │
│    例: static 変数への参照                            │
│    例: Box::leak() で作った参照                       │
│                                                      │
│  T: 'static = T が参照を含まないか、'static参照のみ  │
│    例: String, Vec<i32>, i32 (所有型は全て満たす)     │
│    例: &'static str (参照なら 'static であること)     │
│                                                      │
│  重要: T: 'static は「永遠に生きる」ではなく          │
│        「永遠に生きることが"可能"」という意味          │
└──────────────────────────────────────────────────────┘
```

### 例11: 'static の正しい使い方

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

### 例12: Box::leak と 'static 参照の作成

```rust
fn create_static_str(s: String) -> &'static str {
    // Box::leak でヒープメモリを意図的にリークし、'static 参照を得る
    // 注意: メモリが解放されないので、限定的な場面でのみ使用すること
    Box::leak(s.into_boxed_str())
}

// lazy_static! やOnceCellの内部実装でも使われるパターン
use std::sync::OnceLock;

static CONFIG: OnceLock<String> = OnceLock::new();

fn get_config() -> &'static str {
    CONFIG.get_or_init(|| {
        // 実際にはファイルや環境変数から読み込む
        String::from("production")
    })
}

fn main() {
    let dynamic_string = String::from("動的に作った文字列");
    let static_ref = create_static_str(dynamic_string);
    println!("{}", static_ref);

    let config = get_config();
    println!("Config: {}", config);
}
```

### 例13: 'static の誤用と修正

```rust
// 誤用1: 不必要な 'static 制約
// BAD: 'static を要求しすぎ
fn process_bad(data: &'static str) {
    println!("{}", data);
}

// GOOD: 任意のライフタイムを受け入れる
fn process_good(data: &str) {
    println!("{}", data);
}

// 誤用2: トレイトオブジェクトのデフォルト 'static
// BAD: 意図せず 'static を要求
fn take_display_bad(item: Box<dyn std::fmt::Display>) {
    // Box<dyn Display> は Box<dyn Display + 'static> と同じ
    println!("{}", item);
}

// GOOD: 明示的にライフタイムを指定
fn take_display_good<'a>(item: Box<dyn std::fmt::Display + 'a>) {
    println!("{}", item);
}

fn main() {
    // process_bad には String の参照を渡せない
    // let s = String::from("hello");
    // process_bad(&s); // エラー

    process_bad("リテラルはOK");
    process_good("リテラルもOK");
    let s = String::from("変数もOK");
    process_good(&s);
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
│                                                      │
│  HRTB が必要な典型的な場面:                           │
│  - クロージャ引数が参照を受け取り参照を返す場合      │
│  - トレイトオブジェクトがジェネリックなライフタイムの │
│    メソッドを持つ場合                                 │
│  - Iterator::for_each のようなコールバック系API       │
└──────────────────────────────────────────────────────┘
```

### 例14: HRTB の実用例

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

fn first_word(s: &str) -> &str {
    s.split_whitespace().next().unwrap_or(s)
}

fn main() {
    let s1 = String::from("hello world");
    let s2 = String::from("rust programming");
    apply_to_both(identity, &s1, &s2);
    apply_to_both(first_word, &s1, &s2);
}
```

### 例15: HRTB とクロージャの組み合わせ

```rust
// HRTB を使ったパーサーコンビネータのような設計
trait Parser {
    fn parse<'input>(&self, input: &'input str) -> Option<(&'input str, &'input str)>;
}

struct Literal {
    expected: String,
}

impl Parser for Literal {
    fn parse<'input>(&self, input: &'input str) -> Option<(&'input str, &'input str)> {
        if input.starts_with(&self.expected) {
            Some((&input[..self.expected.len()], &input[self.expected.len()..]))
        } else {
            None
        }
    }
}

struct Sequence {
    parsers: Vec<Box<dyn Parser>>,
}

impl Parser for Sequence {
    fn parse<'input>(&self, input: &'input str) -> Option<(&'input str, &'input str)> {
        let mut remaining = input;
        let mut matched_end = 0;

        for parser in &self.parsers {
            match parser.parse(remaining) {
                Some((_, rest)) => {
                    matched_end = input.len() - rest.len();
                    remaining = rest;
                }
                None => return None,
            }
        }

        Some((&input[..matched_end], remaining))
    }
}

fn apply_parser<P>(parser: &P, inputs: &[&str])
where
    P: for<'a> Fn(&'a str) -> Option<(&'a str, &'a str)>,
{
    for input in inputs {
        match parser(input) {
            Some((matched, rest)) => {
                println!("マッチ: '{}', 残り: '{}'", matched, rest);
            }
            None => {
                println!("マッチなし: '{}'", input);
            }
        }
    }
}

fn main() {
    let literal = Literal {
        expected: "hello".to_string(),
    };

    let inputs = ["hello world", "hello", "goodbye", "hello!"];
    for input in &inputs {
        match literal.parse(input) {
            Some((matched, rest)) => println!("'{}' → matched='{}', rest='{}'", input, matched, rest),
            None => println!("'{}' → no match", input),
        }
    }

    // クロージャ版
    let prefix_parser = |input: &str| -> Option<(&str, &str)> {
        if input.starts_with("rust") {
            Some((&input[..4], &input[4..]))
        } else {
            None
        }
    };

    let test_inputs = ["rust is great", "rust", "python"];
    apply_parser(&prefix_parser, &test_inputs);
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
  (共変性: covariance)
```

### 例16: ライフタイム境界

```rust
// 'a: 'b は「'a は少なくとも 'b と同じ長さ」
fn select<'a, 'b: 'a>(first: &'a str, second: &'b str) -> &'a str {
    if first.len() > second.len() {
        first
    } else {
        second // 'b: 'a なので 'b の参照を 'a として返せる
    }
}

fn main() {
    let s1 = String::from("hello");
    let result;
    {
        let s2 = String::from("world!!");
        result = select(&s1, &s2);
        println!("{}", result);
    }
    // result は s1 のライフタイム ('a) に制約される
    // s2 のライフタイム ('b) は 'a 以上なので OK
}
```

### 例17: 共変性と反変性

```rust
// ライフタイムの共変性 (covariance)
// &'long T を &'short T として使える (サブタイプ)
fn demonstrate_covariance() {
    let long_lived = String::from("long");

    // 'long を 'short として使う
    fn take_short<'short>(s: &'short str) -> &'short str {
        s
    }

    // 'static は全てのライフタイムのサブタイプ
    let static_str: &'static str = "static";
    let result = take_short(static_str); // 'static → 'short は OK
    println!("{}", result);

    let result2 = take_short(&long_lived); // 通常のライフタイムも OK
    println!("{}", result2);
}

// ライフタイム境界の実践例
struct Container<'a> {
    data: Vec<&'a str>,
}

impl<'a> Container<'a> {
    fn new() -> Self {
        Container { data: Vec::new() }
    }

    // 'b: 'a → 'b は 'a より長生きする
    // つまり、'a より長いライフタイムの参照を追加できる
    fn add<'b: 'a>(&mut self, item: &'b str) {
        self.data.push(item);
    }

    fn get_all(&self) -> &[&'a str] {
        &self.data
    }
}

fn main() {
    demonstrate_covariance();

    let s1 = String::from("hello");
    let s2 = String::from("world");

    let mut container = Container::new();
    container.add(&s1);
    container.add(&s2);
    container.add("static string"); // &'static str も追加可能

    for item in container.get_all() {
        println!("{}", item);
    }
}
```

---

## 8. NLL (Non-Lexical Lifetimes)

### 8.1 NLL の概要

Rust 2018 Edition で導入された NLL (Non-Lexical Lifetimes) は、ライフタイムの終了地点をレキシカルスコープ (ブロック終端) ではなく「最後に使用された地点」に基づいて判断する仕組みである。

```
┌──────────────────────────────────────────────────────┐
│  NLL 以前 (Rust 2015)                                │
│                                                      │
│  let mut data = vec![1, 2, 3];                       │
│  let r = &data[0];         // 'r がスコープ終端まで  │
│  println!("{}", r);                                  │
│  // r はもう使わないのに...                           │
│  data.push(4);             // エラー！'r がまだ有効   │
│                                                      │
├──────────────────────────────────────────────────────┤
│  NLL 以後 (Rust 2018+)                               │
│                                                      │
│  let mut data = vec![1, 2, 3];                       │
│  let r = &data[0];         // 'r 開始                │
│  println!("{}", r);        // 'r の最後の使用 → 終了 │
│  data.push(4);             // OK！'r は終了済み       │
└──────────────────────────────────────────────────────┘
```

### 例18: NLL による改善

```rust
fn main() {
    // NLL がなければコンパイルエラーになるコード

    // ケース1: 条件分岐での借用
    let mut data = vec![1, 2, 3, 4, 5];
    let first = &data[0];
    println!("first: {}", first);
    // NLL: first の最後の使用はここ → ライフタイム終了
    data.push(6); // OK
    println!("data: {:?}", data);

    // ケース2: HashMap の entry パターン
    use std::collections::HashMap;
    let mut map = HashMap::new();
    map.insert("key", vec![1]);

    // NLL がなければ、get と insert を同時に使えなかった
    match map.get("key") {
        Some(v) => println!("found: {:?}", v),
        None => {
            map.insert("key", vec![2]);
        }
    }

    // ケース3: 条件付き可変借用
    let mut v = vec![1, 2, 3];
    let r = &v;
    println!("不変借用: {:?}", r);
    // r はもう使わない → NLL でライフタイム終了
    v.push(4);
    println!("変更後: {:?}", v);
}
```

### 例19: NLL でも解決できないケース

```rust
fn main() {
    // ケース1: 不変と可変の同時借用が本当に重なる場合
    let mut data = vec![1, 2, 3];
    let r = &data[0];
    // data.push(4); // エラー: r はまだこの後で使用される
    println!("{}", r);
    data.push(4); // OK: r はもう使わない

    // ケース2: 構造体のフィールドごとの借用
    struct Pair {
        first: String,
        second: String,
    }

    let mut pair = Pair {
        first: String::from("hello"),
        second: String::from("world"),
    };

    // 別々のフィールドへの同時可変借用は OK
    let r1 = &mut pair.first;
    let r2 = &mut pair.second;
    r1.push_str("!");
    r2.push_str("!");
    println!("{}, {}", r1, r2);

    // ケース3: メソッド経由だと借用が分離できない
    // let r3 = &pair.first;
    // pair.second.push_str("!"); // OK: 別フィールド
    // println!("{}", r3);

    // しかしメソッド経由は全体への借用
    // let r4 = pair.get_first(); // &self → 全体を不変借用
    // pair.set_second("!"); // エラー: &mut self が必要だが全体が借用中
}
```

---

## 9. 高度なライフタイムパターン

### 例20: ライフタイムとトレイトの組み合わせ

```rust
trait Processor<'a> {
    fn process(&self, input: &'a str) -> &'a str;
}

struct TrimProcessor;

impl<'a> Processor<'a> for TrimProcessor {
    fn process(&self, input: &'a str) -> &'a str {
        input.trim()
    }
}

struct PrefixProcessor {
    len: usize,
}

impl<'a> Processor<'a> for PrefixProcessor {
    fn process(&self, input: &'a str) -> &'a str {
        if input.len() > self.len {
            &input[..self.len]
        } else {
            input
        }
    }
}

fn apply_processors<'a>(input: &'a str, processors: &[&dyn Processor<'a>]) -> &'a str {
    let mut result = input;
    for processor in processors {
        result = processor.process(result);
    }
    result
}

fn main() {
    let input = String::from("  Hello, World!  ");
    let trim = TrimProcessor;
    let prefix = PrefixProcessor { len: 5 };

    let processors: Vec<&dyn Processor> = vec![&trim, &prefix];
    let result = apply_processors(&input, &processors);
    println!("結果: '{}'", result); // "Hello"
}
```

### 例21: ライフタイムとイテレータ

```rust
struct WordIterator<'a> {
    text: &'a str,
    position: usize,
}

impl<'a> WordIterator<'a> {
    fn new(text: &'a str) -> Self {
        WordIterator { text, position: 0 }
    }
}

impl<'a> Iterator for WordIterator<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        // 先頭の空白をスキップ
        while self.position < self.text.len()
            && self.text.as_bytes()[self.position] == b' '
        {
            self.position += 1;
        }

        if self.position >= self.text.len() {
            return None;
        }

        let start = self.position;

        // 単語の終端を探す
        while self.position < self.text.len()
            && self.text.as_bytes()[self.position] != b' '
        {
            self.position += 1;
        }

        Some(&self.text[start..self.position])
    }
}

fn main() {
    let text = String::from("Rust is a systems programming language");
    let words: Vec<&str> = WordIterator::new(&text).collect();
    println!("{:?}", words);
    // ["Rust", "is", "a", "systems", "programming", "language"]

    // イテレータアダプタとの組み合わせ
    let long_words: Vec<&str> = WordIterator::new(&text)
        .filter(|w| w.len() > 3)
        .collect();
    println!("長い単語: {:?}", long_words);
    // ["Rust", "systems", "programming", "language"]
}
```

### 例22: GAT (Generic Associated Types) とライフタイム

```rust
// GAT を使ったストリーミング処理パターン
trait StreamingIterator {
    type Item<'a> where Self: 'a;

    fn next<'a>(&'a mut self) -> Option<Self::Item<'a>>;
}

struct WindowIterator {
    data: Vec<i32>,
    position: usize,
    window_size: usize,
}

impl StreamingIterator for WindowIterator {
    type Item<'a> = &'a [i32];

    fn next<'a>(&'a mut self) -> Option<Self::Item<'a>> {
        if self.position + self.window_size > self.data.len() {
            None
        } else {
            let window = &self.data[self.position..self.position + self.window_size];
            self.position += 1;
            Some(window)
        }
    }
}

fn main() {
    let mut iter = WindowIterator {
        data: vec![1, 2, 3, 4, 5],
        position: 0,
        window_size: 3,
    };

    while let Some(window) = iter.next() {
        println!("ウィンドウ: {:?}", window);
    }
    // [1, 2, 3]
    // [2, 3, 4]
    // [3, 4, 5]
}
```

### 例23: ライフタイムと非同期プログラミング

```rust
use std::future::Future;

// 非同期関数のライフタイム
// async fn は戻り値が impl Future + 'lifetime であり、
// 引数のライフタイムに依存する

async fn process_data(data: &str) -> usize {
    // data の参照は Future が完了するまで有効でなければならない
    data.len()
}

// 明示的なライフタイム注釈付き async
fn process_data_explicit<'a>(data: &'a str) -> impl Future<Output = usize> + 'a {
    async move {
        data.len()
    }
}

// トレイトオブジェクトとしての async 関数
fn create_async_processor<'a>(
    data: &'a str,
) -> Box<dyn Future<Output = String> + 'a> {
    Box::new(async move {
        format!("処理結果: {}", data.to_uppercase())
    })
}

// async ブロックとライフタイムの注意点
fn example_async_lifetime() {
    let data = String::from("hello");

    // async ブロックは内部の参照のライフタイムに依存
    let _future = async {
        println!("{}", &data);
    };

    // move async ブロックは所有権を取得
    let _future_move = async move {
        println!("{}", data);
    };
    // data はムーブされたのでここでは使えない
}
```

---

## 10. 比較表

### 10.1 ライフタイムの種類

| 種類 | 記法 | 意味 | 例 |
|------|------|------|-----|
| 名前付き | `'a` | 明示的なライフタイムパラメータ | `fn f<'a>(x: &'a str)` |
| 省略 | なし | コンパイラが推論 | `fn f(x: &str) -> &str` |
| 'static | `'static` | プログラム全期間 | `&'static str` |
| 匿名 | `'_` | 推論を明示的に要求 | `impl Iterator<Item = &'_ str>` |
| HRTB | `for<'a>` | 任意のライフタイムに対して | `F: for<'a> Fn(&'a str)` |

### 10.2 'static の誤解と実際

| 誤解 | 実際 |
|------|------|
| 永遠にメモリに残る | 永遠に有効な「資格がある」 |
| 文字列リテラルだけ | 所有型は全て `'static` を満たす |
| ヒープにある | バイナリの静的領域にある(リテラルの場合) |
| 使うべきでない | スレッドに渡す値には必要 |
| メモリリークする | 所有型なら通常通り drop される |

### 10.3 ライフタイム省略規則の適用パターン

| パターン | 省略前 | 省略後 | 適用規則 |
|----------|--------|--------|----------|
| 単一入力 | `fn f<'a>(x: &'a str) -> &'a str` | `fn f(x: &str) -> &str` | 規則1+2 |
| メソッド | `fn f<'a>(&'a self) -> &'a str` | `fn f(&self) -> &str` | 規則1+3 |
| 複数入力 | 省略不可 | `fn f<'a>(x: &'a str, y: &'a str) -> &'a str` | なし |
| メソッド+引数 | `fn f<'a,'b>(&'a self, x: &'b str) -> &'a str` | `fn f(&self, x: &str) -> &str` | 規則1+3 |
| 参照なし | `fn f(x: i32) -> i32` | `fn f(x: i32) -> i32` | 不要 |

---

## 11. アンチパターン

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

// BAD: トレイトオブジェクトに不要な 'static
fn take_processor(p: Box<dyn Fn(&str) -> String + 'static>) {
    println!("{}", p("hello"));
}

// GOOD: 必要な場合のみ 'static を指定
fn take_processor_good(p: Box<dyn Fn(&str) -> String>) {
    // Box<dyn Fn + 'static> と同じだが、意図が明確
    println!("{}", p("hello"));
}
// もしスレッドに渡すなど 'static が本当に必要な場合は明示する
fn take_processor_thread(p: Box<dyn Fn(&str) -> String + Send + 'static>) {
    std::thread::spawn(move || {
        println!("{}", p("hello"));
    });
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
struct Config {
    max_depth: usize,
}

struct Database {
    connection: String,
}

struct Parser {
    input: String,
    config: Config,
    db: Database,
}

impl Parser {
    fn parse(&self) -> Result<Vec<String>, String> {
        // 処理
        Ok(vec![self.input.clone()])
    }
}
// パフォーマンスが問題になったら後でライフタイムを導入
```

### アンチパターン3: ライフタイムの過剰な伝播

```rust
// BAD: ライフタイムが構造体の利用者すべてに伝播する
struct BadTokenizer<'a> {
    source: &'a str,
    tokens: Vec<&'a str>,
}

// すべての使用箇所でライフタイムを指定する必要がある
// fn process_tokens<'a>(tokenizer: &BadTokenizer<'a>) { ... }
// fn analyze<'a>(tokens: &[&'a str]) { ... }

// GOOD: インデックスベースで所有権の問題を回避
struct GoodTokenizer {
    source: String,
    token_ranges: Vec<(usize, usize)>,
}

impl GoodTokenizer {
    fn new(source: String) -> Self {
        let mut ranges = Vec::new();
        let mut start = 0;
        for (i, ch) in source.char_indices() {
            if ch.is_whitespace() {
                if start < i {
                    ranges.push((start, i));
                }
                start = i + ch.len_utf8();
            }
        }
        if start < source.len() {
            ranges.push((start, source.len()));
        }
        GoodTokenizer {
            source,
            token_ranges: ranges,
        }
    }

    fn tokens(&self) -> Vec<&str> {
        self.token_ranges
            .iter()
            .map(|&(start, end)| &self.source[start..end])
            .collect()
    }
}

fn main() {
    let tokenizer = GoodTokenizer::new("hello world rust".to_string());
    println!("{:?}", tokenizer.tokens());
}
```

### アンチパターン4: コレクションに参照を詰め込む

```rust
// BAD: Vec に参照を詰め込もうとしてライフタイムで苦労する
fn collect_references_bad<'a>() -> Vec<&'a str> {
    let mut results = Vec::new();
    // let s = String::from("hello");
    // results.push(&s);  // エラー: s のライフタイムが足りない
    results
}

// GOOD: 所有型のコレクションを使う
fn collect_owned() -> Vec<String> {
    let mut results = Vec::new();
    let s = String::from("hello");
    results.push(s);
    results
}

// GOOD: 入力のスライスから参照を集める
fn collect_from_input<'a>(input: &'a str) -> Vec<&'a str> {
    input.split_whitespace().collect()
}

fn main() {
    let owned = collect_owned();
    println!("{:?}", owned);

    let input = String::from("hello world rust");
    let refs = collect_from_input(&input);
    println!("{:?}", refs);
}
```

---

## 12. 実践的なライフタイムのデバッグ

### 12.1 よくあるコンパイルエラーと対処法

```rust
// エラー1: "lifetime may not live long enough"
// fn bad1<'a>(x: &str) -> &'a str { x }
// 修正: 入力と出力のライフタイムを一致させる
fn good1<'a>(x: &'a str) -> &'a str { x }

// エラー2: "cannot return reference to local variable"
// fn bad2() -> &str {
//     let s = String::from("hello");
//     &s
// }
// 修正: 所有型を返す
fn good2() -> String {
    String::from("hello")
}

// エラー3: "borrowed value does not live long enough"
fn good3() {
    let result;
    let s = String::from("hello");
    result = &s; // s と同じスコープなので OK
    println!("{}", result);
} // s と result が同時に drop

// エラー4: "cannot borrow as mutable because it is also borrowed as immutable"
fn good4() {
    let mut v = vec![1, 2, 3];
    let first = v[0]; // コピー（i32 は Copy）
    v.push(4);
    println!("{}, {:?}", first, v);

    // 参照の場合は NLL で解決
    let r = &v[0];
    println!("{}", r); // r の最後の使用
    v.push(5); // OK: r はもう使わない
}

fn main() {
    let s = good1("hello");
    println!("{}", s);
    println!("{}", good2());
    good3();
    good4();
}
```

### 12.2 ライフタイムエラーの読み方

```
error[E0597]: `x` does not live long enough
  --> src/main.rs:4:13
   |
3  |     let r;
   |         - borrow later stored here     ← r が参照を保持
4  |     let x = 5;
5  |     r = &x;
   |         ^^ borrowed value does not live long enough  ← x の寿命が足りない
6  | }
   | - `x` dropped here while still borrowed  ← x がドロップされる地点

対処法:
1. r と x のスコープを合わせる
2. r に x の値をコピー/クローンする
3. x を外側のスコープに移動する
```

---

## 13. FAQ

### Q1: ライフタイム注釈は実行時に何か影響しますか？

**A:** いいえ。ライフタイム注釈は完全にコンパイル時の情報です。実行時のコードやパフォーマンスには一切影響しません。バイナリにライフタイムの情報は含まれません。これはRustの「ゼロコスト抽象化」の一例です。

### Q2: NLL (Non-Lexical Lifetimes) とは何ですか？

**A:** Rust 2018 Editionで導入された改善です。従来、参照のライフタイムはレキシカルスコープ(ブロック終端)まで続きましたが、NLLでは「最後に使用された地点」で終了します。これにより、以前はコンパイルエラーだった正当なコードが通るようになりました。NLL は現在のRustではデフォルトで有効です。

### Q3: `'_` (匿名ライフタイム)はいつ使いますか？

**A:** ライフタイムの存在を明示しつつ、具体的な名前は不要な場合に使います:
```rust
// impl ブロックでライフタイムの存在だけ示す
impl fmt::Display for ImportantExcerpt<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.part)
    }
}

// 関数のシグネチャで明示的に省略を示す
fn takes_ref(s: &'_ str) -> &'_ str {
    s
}
```

### Q4: ライフタイムとジェネリクスの相互作用は？

**A:** ライフタイムパラメータはジェネリクスパラメータの一種です。慣例として型パラメータの前にライフタイムパラメータを書きます:
```rust
fn example<'a, 'b, T, U>(x: &'a T, y: &'b U) -> &'a T
where
    T: Clone,
    U: std::fmt::Debug,
    'b: 'a,
{
    println!("{:?}", y);
    x
}
```

### Q5: ライフタイムの分散 (variance) とは何ですか？

**A:** ライフタイムの分散とは、ライフタイムパラメータの代入互換性に関する規則です:
- **共変 (covariant)**: `&'long T` を `&'short T` として使える。ほとんどの参照型がこれ
- **反変 (contravariant)**: `fn(&'short T)` を `fn(&'long T)` として使える。関数引数の位置
- **不変 (invariant)**: `&mut T` の `T` に対するライフタイムは不変。`Cell<&'a T>` も不変

```rust
// 共変の例
fn covariant_example<'short>(s: &'short str) {
    let static_str: &'static str = "hello";
    let _: &'short str = static_str; // 'static → 'short OK (共変)
}

// 不変の例
fn invariant_example() {
    let mut x: &str = "hello";
    let y: &str = "world";
    x = y; // OK: 両方とも &str

    // Cell は不変なので以下は制約が厳しい
    use std::cell::Cell;
    let cell: Cell<&str> = Cell::new("hello");
    // Cell<&'a str> は 'a に対して不変
}
```

### Q6: Polonius とは何ですか？

**A:** Polonius はRustの次世代借用チェッカーで、現在のNLLベースのチェッカーよりも正確にライフタイムを解析します。NLLでもまだ「安全なのにコンパイルエラーになる」ケースがいくつかあり、Poloniusはこれらを解決します。2024年時点ではnightly版で `-Z polonius` フラグにより試験的に使用可能です。

```rust
// 現在のNLLでは通らないがPoloniusでは通るコード例
// fn get_or_insert(map: &mut HashMap<String, String>, key: &str) -> &String {
//     if let Some(value) = map.get(key) {
//         return value;
//     }
//     map.insert(key.to_string(), "default".to_string());
//     map.get(key).unwrap()
// }
```

---

## 14. まとめ

| 概念 | 要点 |
|------|------|
| ライフタイム | 参照の有効期間をコンパイラが追跡する仕組み |
| 'a 注釈 | 参照間の関係をコンパイラに伝える |
| 省略規則 | 3つの規則で多くの場合は注釈不要 |
| 構造体のLT | 参照フィールドを持つ構造体にはLT注釈が必要 |
| 'static | プログラム全体の期間。所有型は全て満たす |
| HRTB | `for<'a>` で任意のライフタイムに対する制約 |
| NLL | 最後の使用地点でライフタイム終了 |
| サブタイピング | 'a: 'b で outlives 関係を表現 |
| 共変性 | &'long T を &'short T として使える |
| GAT | ジェネリック関連型でライフタイム依存の型を表現 |
| Polonius | 次世代借用チェッカー (開発中) |

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
4. **Rust Reference - Lifetime Elision** -- https://doc.rust-lang.org/reference/lifetime-elision.html
5. **Rust RFC 2094 - Non-Lexical Lifetimes** -- https://rust-lang.github.io/rfcs/2094-nll.html
6. **Polonius - Next Generation Borrow Checker** -- https://github.com/rust-lang/polonius
