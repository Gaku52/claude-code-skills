# 型とトレイト -- Rustの型システムとポリモーフィズムの基盤

> Rustの型システムは struct/enum による代数的データ型と trait によるアドホックポリモーフィズムを基盤とし、ジェネリクスと組み合わせてゼロコスト抽象化を実現する。

---

## この章で学ぶこと

1. **struct と enum** -- 代数的データ型(直積型・直和型)の定義と活用を理解する
2. **トレイト** -- インターフェースの定義、実装、デフォルトメソッドを習得する
3. **ジェネリクスとトレイト境界** -- 型パラメータと制約による汎用コードの書き方を学ぶ
4. **動的ディスパッチと静的ディスパッチ** -- トレイトオブジェクトと単態化の使い分けを理解する
5. **高度なトレイトパターン** -- 関連型、スーパートレイト、ブランケット実装を学ぶ

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
│ スライス │ [T] -- 動的サイズ型(DST)、常に参照で使用  │
├──────────┼──────────────────────────────────────────┤
│ 参照     │ &T, &mut T                               │
├──────────┼──────────────────────────────────────────┤
│ 文字列   │ str -- 文字列スライス(DST)                │
├──────────┼──────────────────────────────────────────┤
│ unit     │ () -- 値なし(C の void に相当)           │
├──────────┼──────────────────────────────────────────┤
│ never    │ ! -- 返らない関数の戻り値型              │
└──────────┴──────────────────────────────────────────┘
```

### 1.2 数値型の詳細

```rust
fn main() {
    // 整数リテラルの記法
    let decimal = 98_222;           // 10進数（_ で区切り可能）
    let hex = 0xff;                 // 16進数
    let octal = 0o77;              // 8進数
    let binary = 0b1111_0000;       // 2進数
    let byte = b'A';               // バイトリテラル (u8)

    // 型サフィックス
    let x = 42u32;                  // u32 型
    let y = 3.14f64;               // f64 型

    // 型変換（as キャスト）
    let a: i32 = 42;
    let b: f64 = a as f64;         // 拡大変換
    let c: u8 = 300u16 as u8;      // 縮小変換（切り捨て: 44）

    // 安全な型変換
    let d: u16 = 300;
    let e: u8 = u8::try_from(d).unwrap_or(u8::MAX);  // Result を返す

    println!("decimal={}, hex={}, binary={}", decimal, hex, binary);
    println!("b={}, c={}, e={}", b, c, e);
}
```

### 1.3 文字列型の体系

```
┌──────────────────────────────────────────────────────────┐
│                 Rust の文字列型体系                        │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  所有型              参照型（スライス）                    │
│  ┌──────────┐       ┌──────────┐                        │
│  │  String   │ ────> │  &str    │  Deref coercion       │
│  │ (ヒープ)  │       │ (参照)   │                        │
│  └──────────┘       └──────────┘                        │
│                                                          │
│  ┌──────────┐       ┌──────────┐                        │
│  │ OsString │ ────> │  &OsStr  │  OS固有の文字列        │
│  └──────────┘       └──────────┘                        │
│                                                          │
│  ┌──────────┐       ┌──────────┐                        │
│  │  CString │ ────> │  &CStr   │  C互換(NUL終端)       │
│  └──────────┘       └──────────┘                        │
│                                                          │
│  ┌──────────┐       ┌──────────┐                        │
│  │ PathBuf  │ ────> │  &Path   │  ファイルパス          │
│  └──────────┘       └──────────┘                        │
└──────────────────────────────────────────────────────────┘
```

```rust
fn main() {
    // String (所有型) と &str (借用型)
    let owned: String = String::from("hello");
    let borrowed: &str = "hello";  // 文字列リテラルは &'static str

    // String → &str
    let slice: &str = &owned;
    let slice2: &str = owned.as_str();

    // &str → String
    let owned2: String = borrowed.to_string();
    let owned3: String = String::from(borrowed);

    // 文字列の連結
    let s1 = String::from("hello");
    let s2 = String::from(" world");
    let s3 = s1 + &s2;  // s1 はムーブされる、s2 は借用
    // println!("{}", s1);  // エラー: s1 はムーブ済み
    println!("{}", s3);

    // format! マクロ（どの変数もムーブしない）
    let s4 = String::from("hello");
    let s5 = format!("{} {}", s4, "world");
    println!("{}, {}", s4, s5); // 両方有効
}
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

impl User {
    // 関連関数（コンストラクタ）
    fn new(name: &str, email: &str, age: u32) -> Self {
        Self {
            name: name.to_string(),
            email: email.to_string(),
            age,
            active: true,
        }
    }

    // フィールド更新構文を使ったビルダー風メソッド
    fn with_active(self, active: bool) -> Self {
        Self { active, ..self }
    }

    fn display_name(&self) -> &str {
        &self.name
    }

    fn is_adult(&self) -> bool {
        self.age >= 18
    }
}

fn main() {
    let user = User::new("田中", "tanaka@example.com", 30);
    println!("{} ({}歳)", user.name, user.age);

    // 構造体更新構文
    let user2 = User {
        name: String::from("鈴木"),
        email: String::from("suzuki@example.com"),
        ..user  // 残りのフィールドは user から取得（ムーブ注意）
    };
    println!("{} ({}歳)", user2.name, user2.age); // age=30, active=true

    // メソッドチェーン
    let user3 = User::new("佐藤", "sato@example.com", 15)
        .with_active(false);
    println!("{}は成人? {}", user3.display_name(), user3.is_adult());
}
```

### 例2: タプル構造体とユニット構造体

```rust
// タプル構造体: フィールド名なし（ニュータイプパターンに有用）
struct Color(u8, u8, u8);
struct Meters(f64);
struct Celsius(f64);
struct Fahrenheit(f64);

// ニュータイプパターンで型安全性を確保
impl Celsius {
    fn to_fahrenheit(&self) -> Fahrenheit {
        Fahrenheit(self.0 * 9.0 / 5.0 + 32.0)
    }
}

impl Fahrenheit {
    fn to_celsius(&self) -> Celsius {
        Celsius((self.0 - 32.0) * 5.0 / 9.0)
    }
}

// ユニット構造体: フィールドなし(マーカー型に使用)
struct Marker;
struct Production;
struct Development;

fn main() {
    let red = Color(255, 0, 0);
    let distance = Meters(42.0);
    let temp_c = Celsius(100.0);
    let temp_f = temp_c.to_fahrenheit();

    println!("R={}", red.0);
    println!("距離={}m", distance.0);
    println!("{}°C = {}°F", temp_c.0, temp_f.0);

    // Meters と f64 を混同するミスを型システムが防止
    // let wrong: Meters = Celsius(30.0);  // コンパイルエラー！
}
```

### 2.1 構造体のメモリレイアウト

```
┌──────────────────────────────────────────────────────┐
│ struct User のメモリレイアウト                         │
├──────────────────────────────────────────────────────┤
│                                                      │
│  スタック                       ヒープ                │
│  ┌──────────────────────┐                            │
│  │ name: String         │                            │
│  │   ptr ─────────────────────> "田中"               │
│  │   len: 6             │                            │
│  │   cap: 6             │                            │
│  ├──────────────────────┤                            │
│  │ email: String        │                            │
│  │   ptr ─────────────────────> "tanaka@example.com" │
│  │   len: 18            │                            │
│  │   cap: 18            │                            │
│  ├──────────────────────┤                            │
│  │ age: u32 = 30        │   (4バイト)                │
│  ├──────────────────────┤                            │
│  │ active: bool = true  │   (1バイト + パディング)   │
│  └──────────────────────┘                            │
│                                                      │
│  コンパイラはフィールドの順序を最適化して             │
│  パディングを最小化する（repr(C) でない限り）         │
└──────────────────────────────────────────────────────┘
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

    fn perimeter(&self) -> f64 {
        match self {
            Shape::Circle { radius } => 2.0 * std::f64::consts::PI * radius,
            Shape::Rectangle { width, height } => 2.0 * (width + height),
            Shape::Triangle { base, height } => {
                // 二等辺三角形と仮定
                let side = ((*base / 2.0).powi(2) + height.powi(2)).sqrt();
                base + 2.0 * side
            }
        }
    }

    fn describe(&self) -> String {
        match self {
            Shape::Circle { radius } => format!("半径{}の円", radius),
            Shape::Rectangle { width, height } => format!("{}x{}の長方形", width, height),
            Shape::Triangle { base, height } => format!("底辺{}, 高さ{}の三角形", base, height),
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
        println!("{}: 面積={:.2}, 周囲={:.2}", s.describe(), s.area(), s.perimeter());
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

### 3.2 高度な enum パターン

```rust
// enum にデータを持たせて状態を表現
#[derive(Debug)]
enum HttpResponse {
    Ok { body: String, content_type: String },
    NotFound { path: String },
    Redirect { url: String, permanent: bool },
    ServerError { message: String, code: u16 },
}

impl HttpResponse {
    fn status_code(&self) -> u16 {
        match self {
            HttpResponse::Ok { .. } => 200,
            HttpResponse::NotFound { .. } => 404,
            HttpResponse::Redirect { permanent: true, .. } => 301,
            HttpResponse::Redirect { permanent: false, .. } => 302,
            HttpResponse::ServerError { code, .. } => *code,
        }
    }

    fn is_success(&self) -> bool {
        matches!(self, HttpResponse::Ok { .. })
    }
}

// C互換の enum
#[repr(u8)]
enum Color {
    Red = 1,
    Green = 2,
    Blue = 3,
}

// enum のサイズ最適化（Null Pointer Optimization）
fn size_demo() {
    use std::mem::size_of;

    // Option<Box<T>> は Box<T> と同じサイズ！
    // None は内部的に null ポインタで表現される
    assert_eq!(size_of::<Box<i32>>(), size_of::<Option<Box<i32>>>());
    assert_eq!(size_of::<&i32>(), size_of::<Option<&i32>>());

    println!("Box<i32>: {} bytes", size_of::<Box<i32>>());
    println!("Option<Box<i32>>: {} bytes", size_of::<Option<Box<i32>>>());
}

fn main() {
    let response = HttpResponse::Ok {
        body: "<h1>Hello</h1>".to_string(),
        content_type: "text/html".to_string(),
    };
    println!("ステータス: {}", response.status_code());
    println!("成功?: {}", response.is_success());

    size_demo();
}
```

### 3.3 enum のメモリレイアウト

```
┌──────────────────────────────────────────────────────┐
│ enum Shape のメモリレイアウト                          │
├──────────────────────────────────────────────────────┤
│                                                      │
│  全バリアントが同じサイズのメモリを占有する            │
│  (最大バリアントのサイズ + タグのサイズ)              │
│                                                      │
│  Circle:                                             │
│  ┌─────────┬──────────────┬──────────────┐          │
│  │ tag = 0 │ radius: f64  │  (未使用)    │          │
│  └─────────┴──────────────┴──────────────┘          │
│                                                      │
│  Rectangle:                                          │
│  ┌─────────┬──────────────┬──────────────┐          │
│  │ tag = 1 │ width: f64   │ height: f64  │          │
│  └─────────┴──────────────┴──────────────┘          │
│                                                      │
│  Triangle:                                           │
│  ┌─────────┬──────────────┬──────────────┐          │
│  │ tag = 2 │ base: f64    │ height: f64  │          │
│  └─────────┴──────────────┴──────────────┘          │
│                                                      │
│  サイズ = tag(1-8bytes) + max(バリアント) + padding   │
└──────────────────────────────────────────────────────┘
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

    fn is_square(&self) -> bool {
        (self.width - self.height).abs() < f64::EPSILON
    }

    fn can_hold(&self, other: &Rectangle) -> bool {
        self.width >= other.width && self.height >= other.height
    }

    // 可変メソッド
    fn scale(&mut self, factor: f64) {
        self.width *= factor;
        self.height *= factor;
    }

    // self を消費するメソッド（Builder パターン等で使用）
    fn into_square(self) -> Rectangle {
        let side = self.width.max(self.height);
        Rectangle { width: side, height: side }
    }
}

// 複数の impl ブロックを持てる（トレイト実装との分離に有用）
impl std::fmt::Display for Rectangle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Rectangle({}x{})", self.width, self.height)
    }
}

fn main() {
    let mut rect = Rectangle::new(10.0, 5.0);
    println!("{}", rect);                    // Rectangle(10x5)
    println!("面積: {}", rect.area());       // 50.0
    println!("周囲: {}", rect.perimeter());  // 30.0
    println!("正方形? {}", rect.is_square()); // false

    let small = Rectangle::new(3.0, 2.0);
    println!("包含可能? {}", rect.can_hold(&small)); // true

    rect.scale(2.0);
    println!("拡大後: {}", rect);            // Rectangle(20x10)

    let sq = Rectangle::square(5.0);
    println!("{} は正方形? {}", sq, sq.is_square()); // true
}
```

---

## 5. トレイト

### 5.1 トレイト定義と実装

### 例5: トレイトの定義と実装

```rust
trait Summary {
    // 必須メソッド
    fn summarize_author(&self) -> String;

    // デフォルト実装
    fn summarize(&self) -> String {
        format!("({}による記事をもっと読む...)", self.summarize_author())
    }

    // デフォルト実装が他のメソッドを呼ぶこともできる
    fn preview(&self) -> String {
        let summary = self.summarize();
        if summary.len() > 50 {
            format!("{}...", &summary[..50])
        } else {
            summary
        }
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
use std::fmt::{Display, Debug};

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

// impl Trait は戻り値にも使える（ただし単一の具体型のみ）
fn create_summarizable() -> impl Summary {
    Tweet {
        username: String::from("rustlang"),
        text: String::from("Rust is great!"),
    }
}

// 複数のトレイト境界の組み合わせ
fn process_and_display<T>(item: T)
where
    T: Summary + Display + Clone + Debug,
{
    let cloned = item.clone();
    println!("Summary: {}", item.summarize());
    println!("Display: {}", item);
    println!("Debug: {:?}", cloned);
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
│ TryFrom/TryInto│ 失敗しうる型変換                         │
│ Iterator      │ イテレータプロトコル                       │
│ IntoIterator  │ for ループで使えるようにする               │
│ Drop          │ デストラクタ(スコープ終了時の処理)        │
│ Deref/DerefMut│ 自動参照解決 / スマートポインタ           │
│ AsRef/AsMut   │ 参照としての変換                          │
│ Borrow        │ 借用としての変換（Hash/Eq の一貫性保証）  │
│ Send/Sync     │ スレッド安全性マーカー                    │
│ Sized         │ コンパイル時にサイズが既知                 │
│ Fn/FnMut/FnOnce│ クロージャ/関数呼び出し                  │
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
    // Config { host: "", port: 0, debug: false }
}
```

### 例8: Display と From/Into の実装

```rust
use std::fmt;

#[derive(Debug)]
struct Temperature {
    celsius: f64,
}

impl Temperature {
    fn new(celsius: f64) -> Self {
        Temperature { celsius }
    }

    fn fahrenheit(&self) -> f64 {
        self.celsius * 9.0 / 5.0 + 32.0
    }
}

// Display トレイトの手動実装
impl fmt::Display for Temperature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1}°C ({:.1}°F)", self.celsius, self.fahrenheit())
    }
}

// From トレイトの実装（型変換）
impl From<f64> for Temperature {
    fn from(celsius: f64) -> Self {
        Temperature { celsius }
    }
}

impl From<i32> for Temperature {
    fn from(celsius: i32) -> Self {
        Temperature { celsius: celsius as f64 }
    }
}

// From<T> for U を実装すると、Into<U> for T が自動的に使える
fn display_temp(temp: impl Into<Temperature>) {
    let t: Temperature = temp.into();
    println!("{}", t);
}

fn main() {
    let temp = Temperature::new(100.0);
    println!("{}", temp);       // Display: "100.0°C (212.0°F)"
    println!("{:?}", temp);     // Debug: "Temperature { celsius: 100.0 }"

    // From/Into による変換
    let t1: Temperature = 36.5f64.into();
    let t2: Temperature = Temperature::from(100);
    display_temp(25.0f64);
    display_temp(0);
}
```

### 例9: PartialEq と Ord の実装

```rust
#[derive(Debug, Clone)]
struct Student {
    name: String,
    score: u32,
}

// PartialEq: 名前が同じなら同じ学生とみなす
impl PartialEq for Student {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}
impl Eq for Student {}

// Ord: スコアでソート（降順）
impl PartialOrd for Student {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Student {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.score.cmp(&self.score) // 降順
    }
}

fn main() {
    let mut students = vec![
        Student { name: "田中".to_string(), score: 85 },
        Student { name: "鈴木".to_string(), score: 92 },
        Student { name: "佐藤".to_string(), score: 78 },
    ];

    students.sort(); // Ord に基づいてソート（スコア降順）
    for s in &students {
        println!("{}: {}点", s.name, s.score);
    }
    // 鈴木: 92点, 田中: 85点, 佐藤: 78点
}
```

---

## 6. ジェネリクス

### 例10: ジェネリック構造体と関数

```rust
use std::fmt::Display;

// ジェネリック構造体
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

// 異なる型のペア
struct MixedPair<T, U> {
    first: T,
    second: U,
}

impl<T: Display, U: Display> MixedPair<T, U> {
    fn display(&self) {
        println!("({}, {})", self.first, self.second);
    }
}

// 特定の型にのみ追加メソッドを提供
impl Pair<f64> {
    fn average(&self) -> f64 {
        (self.first + self.second) / 2.0
    }
}

// ジェネリック関数
fn find_max<T: PartialOrd>(list: &[T]) -> Option<&T> {
    if list.is_empty() {
        return None;
    }
    let mut max = &list[0];
    for item in &list[1..] {
        if item > max {
            max = item;
        }
    }
    Some(max)
}

fn main() {
    let pair = Pair::new(10, 20);
    println!("大きい方: {}", pair.larger()); // 20

    let float_pair = Pair::new(3.14, 2.71);
    println!("平均: {}", float_pair.average()); // 2.925

    let mixed = MixedPair { first: "hello", second: 42 };
    mixed.display(); // (hello, 42)

    let numbers = vec![34, 50, 25, 100, 65];
    println!("最大値: {}", find_max(&numbers).unwrap());
}
```

### 6.1 単態化（Monomorphization）

```
ジェネリクスのコンパイル時の処理:

ソースコード:
  fn max<T: PartialOrd>(a: T, b: T) -> T { ... }

  max(1i32, 2i32);
  max(3.14f64, 2.71f64);
  max("hello", "world");

コンパイル後（単態化）:
  fn max_i32(a: i32, b: i32) -> i32 { ... }
  fn max_f64(a: f64, b: f64) -> f64 { ... }
  fn max_str(a: &str, b: &str) -> &str { ... }

  max_i32(1, 2);
  max_f64(3.14, 2.71);
  max_str("hello", "world");

→ 実行時のオーバーヘッドなし（ゼロコスト抽象化）
→ ただしバイナリサイズは増加する可能性あり
```

---

## 7. 動的ディスパッチとトレイトオブジェクト

### 例11: dyn Trait（動的ディスパッチ）

```rust
trait Animal {
    fn name(&self) -> &str;
    fn sound(&self) -> &str;
    fn info(&self) -> String {
        format!("{} は「{}」と鳴く", self.name(), self.sound())
    }
}

struct Dog { name: String }
struct Cat { name: String }
struct Bird { name: String }

impl Animal for Dog {
    fn name(&self) -> &str { &self.name }
    fn sound(&self) -> &str { "ワン" }
}

impl Animal for Cat {
    fn name(&self) -> &str { &self.name }
    fn sound(&self) -> &str { "ニャー" }
}

impl Animal for Bird {
    fn name(&self) -> &str { &self.name }
    fn sound(&self) -> &str { "チュン" }
}

fn main() {
    // 異なる型を同じコレクションに入れる → dyn Trait が必要
    let animals: Vec<Box<dyn Animal>> = vec![
        Box::new(Dog { name: "ポチ".to_string() }),
        Box::new(Cat { name: "タマ".to_string() }),
        Box::new(Bird { name: "ピー太".to_string() }),
    ];

    for animal in &animals {
        println!("{}", animal.info());
    }

    // 関数の引数としても使える
    fn describe_animal(animal: &dyn Animal) {
        println!("動物: {} - {}", animal.name(), animal.sound());
    }

    describe_animal(&Dog { name: "シロ".to_string() });
    describe_animal(&Cat { name: "クロ".to_string() });
}
```

### 7.1 vtable（仮想関数テーブル）の仕組み

```
  トレイトオブジェクト &dyn Animal のメモリレイアウト:

  ファットポインタ（2ワード）
  ┌──────────────┐
  │ data ptr ────────────> 実際のデータ (Dog, Cat, etc.)
  │ vtable ptr ──────────> vtable
  └──────────────┘

  Dog の vtable:
  ┌──────────────────────┐
  │ drop()               │  → Dog::drop
  │ size                 │  → sizeof(Dog)
  │ align                │  → alignof(Dog)
  │ name()               │  → Dog::name
  │ sound()              │  → Dog::sound
  │ info()               │  → Animal::info (デフォルト実装)
  └──────────────────────┘

  Cat の vtable:
  ┌──────────────────────┐
  │ drop()               │  → Cat::drop
  │ size                 │  → sizeof(Cat)
  │ align                │  → alignof(Cat)
  │ name()               │  → Cat::name
  │ sound()              │  → Cat::sound
  │ info()               │  → Animal::info (デフォルト実装)
  └──────────────────────┘
```

### 7.2 オブジェクト安全性

```rust
// オブジェクト安全なトレイト（dyn Trait として使用可能）
trait Drawable {
    fn draw(&self);
    fn bounding_box(&self) -> (f64, f64, f64, f64);
}

// オブジェクト安全でないトレイト（dyn Trait として使用不可）
trait NotObjectSafe {
    fn create() -> Self;           // Self を返す関連関数
    fn compare(&self, other: &Self);  // Self を引数に取る
    fn generic_method<T>(&self, t: T);  // ジェネリックメソッド
}

// オブジェクト安全の条件:
// 1. Self: Sized を要求しない
// 2. メソッドの戻り値に Self を使わない（where Self: Sized ガード付きは除く）
// 3. ジェネリックな型パラメータを持たない
// 4. 関連定数を持たない

// 部分的にオブジェクト安全にする技法
trait Clonable: Clone {
    fn clone_box(&self) -> Box<dyn Clonable>;
}

impl<T: Clone + Clonable + 'static> Clonable for T {
    fn clone_box(&self) -> Box<dyn Clonable> {
        Box::new(self.clone())
    }
}
```

---

## 8. 高度なトレイトパターン

### 8.1 関連型 (Associated Types)

```rust
// 関連型を使ったイテレータの定義
trait MyIterator {
    type Item;  // 関連型

    fn next(&mut self) -> Option<Self::Item>;
}

struct Counter {
    count: u32,
    max: u32,
}

impl MyIterator for Counter {
    type Item = u32;  // 関連型を具体化

    fn next(&mut self) -> Option<Self::Item> {
        if self.count < self.max {
            self.count += 1;
            Some(self.count)
        } else {
            None
        }
    }
}

// 関連型 vs ジェネリクスの比較
// 関連型: 1つの型に対して1つの実装のみ
// ジェネリクス: 1つの型に対して複数の実装が可能

// ジェネリクス版（複数の変換先を定義可能）
trait ConvertTo<T> {
    fn convert(&self) -> T;
}

struct Celsius(f64);

impl ConvertTo<f64> for Celsius {
    fn convert(&self) -> f64 { self.0 }
}

impl ConvertTo<String> for Celsius {
    fn convert(&self) -> String { format!("{}°C", self.0) }
}
```

### 8.2 スーパートレイト

```rust
use std::fmt;

// Display を要求するトレイト（スーパートレイト）
trait Printable: fmt::Display + fmt::Debug {
    fn print(&self) {
        println!("Display: {}", self);
    }

    fn debug_print(&self) {
        println!("Debug: {:?}", self);
    }

    fn pretty_print(&self) {
        println!("===== {} =====", self);
    }
}

#[derive(Debug)]
struct Report {
    title: String,
    content: String,
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.title, self.content)
    }
}

// Display + Debug を実装しているので Printable を実装可能
impl Printable for Report {}

fn main() {
    let report = Report {
        title: "月次報告".to_string(),
        content: "売上は前月比10%増".to_string(),
    };
    report.print();
    report.debug_print();
    report.pretty_print();
}
```

### 8.3 ブランケット実装

```rust
// ブランケット実装: 条件を満たす全ての型に対して一括実装
trait Greet {
    fn greet(&self) -> String;
}

// Display を実装している全ての型に Greet を実装
impl<T: std::fmt::Display> Greet for T {
    fn greet(&self) -> String {
        format!("こんにちは、{}さん！", self)
    }
}

fn main() {
    println!("{}", "太郎".greet());     // こんにちは、太郎さん！
    println!("{}", 42.greet());          // こんにちは、42さん！
    println!("{}", 3.14f64.greet());     // こんにちは、3.14さん！
}
```

### 8.4 演算子オーバーロード

```rust
use std::ops::{Add, Mul, Neg};

#[derive(Debug, Clone, Copy, PartialEq)]
struct Vector2D {
    x: f64,
    y: f64,
}

impl Vector2D {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn magnitude(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }

    fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y
    }
}

// + 演算子
impl Add for Vector2D {
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

// * 演算子（スカラー倍）
impl Mul<f64> for Vector2D {
    type Output = Self;

    fn mul(self, scalar: f64) -> Self::Output {
        Self {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
}

// - 演算子（符号反転）
impl Neg for Vector2D {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}

// Display
impl std::fmt::Display for Vector2D {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

fn main() {
    let v1 = Vector2D::new(3.0, 4.0);
    let v2 = Vector2D::new(1.0, 2.0);

    println!("v1 + v2 = {}", v1 + v2);       // (4, 6)
    println!("v1 * 2 = {}", v1 * 2.0);        // (6, 8)
    println!("-v1 = {}", -v1);                 // (-3, -4)
    println!("|v1| = {}", v1.magnitude());     // 5
    println!("v1 . v2 = {}", v1.dot(&v2));     // 11
}
```

---

## 9. 比較表

### 9.1 struct vs enum

| 特性 | struct | enum |
|------|--------|------|
| 代数型 | 直積型 (product type) | 直和型 (sum type) |
| フィールド | 全フィールド同時に持つ | バリアントの1つだけ |
| パターンマッチ | 分解代入 | match で網羅的に分岐 |
| メモリ | 全フィールドの合計 | 最大バリアント + タグ |
| 用途 | データのまとまり | 状態・選択肢の表現 |
| 型安全性 | フィールドの型で保証 | バリアントの網羅性で保証 |

### 9.2 静的ディスパッチ vs 動的ディスパッチ

| 特性 | 静的 (impl Trait / ジェネリクス) | 動的 (dyn Trait) |
|------|-------------------------------|------------------|
| 仕組み | 単態化 (monomorphization) | vtable (仮想関数テーブル) |
| 実行速度 | 高速(インライン化可能) | オーバーヘッドあり |
| バイナリサイズ | 大きくなりやすい | 小さい |
| 型消去 | なし(コンパイル時に具体型決定) | あり |
| 使い方 | `fn f(x: impl Trait)` | `fn f(x: &dyn Trait)` |
| オブジェクト安全 | 不要 | 必要 |
| コレクション | 同一型のみ | 異なる型を混在可能 |
| コンパイル時間 | 型ごとにコード生成（遅くなりうる） | コード共有（速い） |

### 9.3 トレイト関連の比較

| パターン | 記法 | 用途 |
|----------|------|------|
| トレイト境界 | `T: Clone + Debug` | ジェネリック関数の制約 |
| where句 | `where T: Clone` | 複雑な境界を読みやすく |
| impl Trait (引数) | `item: &impl Summary` | 簡潔なトレイト境界 |
| impl Trait (戻り値) | `-> impl Summary` | 具体型を隠す |
| dyn Trait | `&dyn Summary` | 動的ディスパッチ |
| Box<dyn Trait> | `Box<dyn Summary>` | ヒープ上のトレイトオブジェクト |
| 関連型 | `type Item = u32;` | 型ごとに1つの関連型 |
| derive | `#[derive(Debug)]` | 標準トレイトの自動実装 |

---

## 10. アンチパターン

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

### アンチパターン3: 不必要に複雑なトレイト境界

```rust
// BAD: 使わないトレイト境界を付ける
fn print_item<T: Display + Debug + Clone + Send + Sync>(item: &T) {
    println!("{}", item);  // Display しか使っていない
}

// GOOD: 必要最小限のトレイト境界
fn print_item_good<T: Display>(item: &T) {
    println!("{}", item);
}
```

### アンチパターン4: enum の過度な使用

```rust
// BAD: 状態が追加されるたびに全てのmatch式を修正する必要がある
enum Shape {
    Circle(f64),
    Rectangle(f64, f64),
    Triangle(f64, f64),
    // Pentagon, Hexagon, ... と増えていく
}

fn area(shape: &Shape) -> f64 {
    match shape {
        Shape::Circle(r) => std::f64::consts::PI * r * r,
        Shape::Rectangle(w, h) => w * h,
        Shape::Triangle(b, h) => 0.5 * b * h,
        // 新しいバリアントを追加するたびにここも修正
    }
}

// GOOD: トレイトを使えば拡張に開かれた設計になる
trait ShapeTrait {
    fn area(&self) -> f64;
}

// 新しい形状は新しい struct + impl で追加するだけ
struct Pentagon { side: f64 }
impl ShapeTrait for Pentagon {
    fn area(&self) -> f64 {
        // 正五角形の面積
        0.25 * (5.0f64).sqrt() * (5.0 + 2.0 * (5.0f64).sqrt()) * self.side.powi(2)
    }
}
```

---

## 11. FAQ

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

### Q4: 関連型とジェネリクスはどう違いますか？

**A:**
- **関連型**: 1つの型に対して1つの実装のみ可能。Iterator の Item が代表例
- **ジェネリクス**: 1つの型に対して複数の実装が可能。From<T> が代表例

```rust
// 関連型: Vec<i32> の Iterator は Item=&i32 のみ
impl Iterator for MyIter {
    type Item = u32;  // 固定
    fn next(&mut self) -> Option<u32> { ... }
}

// ジェネリクス: 同じ型に複数の From を実装可能
impl From<String> for MyType { ... }
impl From<i32> for MyType { ... }
```

### Q5: derive できるトレイトの一覧は？

**A:** 標準ライブラリでは以下のトレイトが derive 可能です:
- `Debug`, `Clone`, `Copy`
- `PartialEq`, `Eq`
- `PartialOrd`, `Ord`
- `Hash`, `Default`

外部クレートでは `serde::Serialize`, `serde::Deserialize`, `thiserror::Error` なども derive 可能です。

---

## 12. まとめ

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
| 関連型 | トレイト内で定義する型パラメータ |
| 演算子オーバーロード | std::ops のトレイトを実装 |
| ブランケット実装 | 条件を満たす全型への一括実装 |

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
4. **The Rustonomicon - Trait Objects** -- https://doc.rust-lang.org/nomicon/exotic-sizes.html
5. **Rust Design Patterns** -- https://rust-unofficial.github.io/patterns/
