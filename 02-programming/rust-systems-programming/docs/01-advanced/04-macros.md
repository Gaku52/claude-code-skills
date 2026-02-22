# Rustマクロ — メタプログラミングの力

> コンパイル時にコードを生成する宣言的マクロ・手続き的マクロの仕組みと実践パターンを体系的に学ぶ

## この章で学ぶこと

1. **宣言的マクロ (`macro_rules!`)** — パターンマッチでコードを展開する仕組みと再帰展開
2. **手続き的マクロ (proc-macro)** — derive / attribute / function-like マクロの実装手法
3. **マクロ設計原則** — 衛生性(Hygiene)、デバッグ手法、実用的なベストプラクティス

---

## 1. マクロの分類と全体像

```
┌─────────────────────────────────────────────────────┐
│                Rust マクロ体系                        │
├──────────────────────┬──────────────────────────────┤
│   宣言的マクロ        │     手続き的マクロ             │
│   (Declarative)      │     (Procedural)             │
│                      │                              │
│  macro_rules! で定義  │  別クレートで定義              │
│  パターンマッチベース  │  TokenStream → TokenStream    │
│                      │                              │
│  例: vec!, println!  │  ┌────────────────────────┐  │
│                      │  │ #[derive(..)]          │  │
│                      │  │ #[属性マクロ]           │  │
│                      │  │ 関数風マクロ!()         │  │
│                      │  └────────────────────────┘  │
└──────────────────────┴──────────────────────────────┘
```

### マクロが必要になる場面

Rust のマクロはメタプログラミングの中核であり、以下のような場面で特に威力を発揮する。

1. **ボイラープレート削減** — 同一パターンの繰り返し実装（複数型への `impl`、テストケース生成）
2. **DSL（ドメイン固有言語）** — 構造化されたデータリテラル、宣言的 API 定義
3. **コンパイル時検証** — SQL文・正規表現・環境変数の静的チェック
4. **条件付きコード生成** — `cfg!` に基づくプラットフォーム固有コード
5. **derive 自動実装** — `Serialize`、`Debug`、`Clone` 等のトレイト自動導出

マクロは**コンパイル時**に展開されるため、実行時オーバーヘッドはゼロである。

### マクロの展開フェーズ

```
ソースコード
    │
    ▼
┌──────────────┐
│  字句解析     │  ソース → トークンストリーム
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  マクロ展開   │  macro_rules! / proc-macro がここで実行される
│              │  ※ 複数パスで再帰的に展開
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  名前解決     │  展開後のコードに対して名前を解決
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  型チェック   │  完全に展開されたコードの型チェック
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  借用チェック  │  所有権・ライフタイム検証
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  コード生成   │  LLVM IR → 機械語
└──────────────┘
```

重要な点として、マクロ展開はコンパイルの**初期段階**で行われる。展開後のコードは通常の Rust コードとして型チェック・借用チェックを受けるため、マクロが生成したコードにバグがあればコンパイルエラーとして検出される。

---

## 2. 宣言的マクロ (`macro_rules!`)

### コード例1: 基本的な宣言的マクロ

```rust
/// カスタム HashMap リテラルマクロ
macro_rules! map {
    // 空のマップ
    () => { std::collections::HashMap::new() };

    // key => value ペアを受け取る
    ( $( $key:expr => $value:expr ),+ $(,)? ) => {{
        let mut m = std::collections::HashMap::new();
        $( m.insert($key, $value); )+
        m
    }};
}

fn main() {
    let scores = map! {
        "Alice" => 95,
        "Bob"   => 82,
        "Carol" => 91,
    };
    println!("{:?}", scores);
    // {"Carol": 91, "Alice": 95, "Bob": 82}
}
```

### パターンマッチの動作原理

`macro_rules!` は上から順にアーム（パターン）を試行し、最初にマッチしたものを展開する。各アームは `(パターン) => { 展開テンプレート }` の形式を取る。

```rust
macro_rules! explain {
    // アーム1: 空の入力
    () => {
        println!("引数なし");
    };

    // アーム2: 単一の式
    ($single:expr) => {
        println!("単一の式: {}", $single);
    };

    // アーム3: カンマ区切りの複数式
    ($first:expr, $($rest:expr),+) => {
        println!("最初: {}", $first);
        // 残りを再帰的に処理
        explain!($($rest),+);
    };
}

fn main() {
    explain!();             // "引数なし"
    explain!(42);           // "単一の式: 42"
    explain!(1, 2, 3);      // "最初: 1" → "最初: 2" → "単一の式: 3"
}
```

### フラグメント指定子一覧

```
┌──────────────┬──────────────────────────────┬──────────────┐
│ 指定子        │ マッチ対象                    │ 例            │
├──────────────┼──────────────────────────────┼──────────────┤
│ $x:expr      │ 式                           │ 1 + 2        │
│ $x:ty        │ 型                           │ Vec<i32>     │
│ $x:ident     │ 識別子                        │ my_var       │
│ $x:pat       │ パターン                      │ Some(x)      │
│ $x:path      │ パス                          │ std::io::Read│
│ $x:stmt      │ 文                           │ let x = 1;   │
│ $x:block     │ ブロック                      │ { ... }      │
│ $x:item      │ アイテム                      │ fn foo() {}  │
│ $x:meta      │ メタ(属性の中身)              │ derive(Debug)│
│ $x:tt        │ トークンツリー(何でも1つ)     │ +, foo, {}   │
│ $x:literal   │ リテラル                      │ 42, "hello"  │
│ $x:lifetime  │ ライフタイム                   │ 'a           │
│ $x:vis       │ 可視性修飾子                   │ pub          │
└──────────────┴──────────────────────────────┴──────────────┘
```

### フラグメント指定子の後続制限

フラグメント指定子には「後に続けられるトークン」に制限がある。これを理解していないとコンパイルエラーに悩まされる。

```rust
// $x:expr の後に続けられるのは => , ; のみ
macro_rules! ok_after_expr {
    ($e:expr, $f:expr) => { $e + $f };      // OK: カンマ
    // ($e:expr $f:expr) => { $e + $f };     // NG: 区切りなし
}

// $x:ty の後に続けられるのは => , = | ; : > >> [ { as where のみ
macro_rules! ok_after_ty {
    ($t:ty : $e:expr) => { let _: $t = $e; };  // OK: コロン
}

// $x:pat の後に続けられるのは => , = | if in のみ
macro_rules! ok_after_pat {
    ($p:pat => $e:expr) => {
        match some_val {
            $p => $e,
            _ => panic!(),
        }
    };
}

// $x:tt は何でもマッチするので後続制限なし
macro_rules! flexible {
    ($($t:tt)*) => { $($t)* };  // 何でもそのまま展開
}
```

### コード例2: 再帰マクロ — コンパイル時カウント

```rust
/// コンパイル時にトークン数を数える
macro_rules! count {
    ()                          => { 0usize };
    ($head:tt $($tail:tt)*)     => { 1usize + count!($($tail)*) };
}

/// 固定サイズ配列を型安全に生成
macro_rules! fixed_vec {
    ( $( $elem:expr ),* $(,)? ) => {{
        const LEN: usize = count!($( { $elem } )*);
        let arr: [_; LEN] = [ $( $elem ),* ];
        arr
    }};
}

fn main() {
    let arr = fixed_vec![10, 20, 30];
    // arr は [i32; 3] 型
    assert_eq!(arr.len(), 3);
}
```

### コード例: 複数型への一括 impl（反復マクロ）

実務で最も頻繁に使われるパターンの一つが、複数の型に同じトレイト実装を適用するマクロである。

```rust
/// 数値型に対する共通トレイトを一括実装
trait Numeric {
    fn zero() -> Self;
    fn one() -> Self;
    fn is_positive(&self) -> bool;
}

macro_rules! impl_numeric {
    // 整数型用
    (int: $($t:ty),+ $(,)?) => {
        $(
            impl Numeric for $t {
                fn zero() -> Self { 0 }
                fn one() -> Self { 1 }
                fn is_positive(&self) -> bool { *self > 0 }
            }
        )+
    };
    // 浮動小数点型用
    (float: $($t:ty),+ $(,)?) => {
        $(
            impl Numeric for $t {
                fn zero() -> Self { 0.0 }
                fn one() -> Self { 1.0 }
                fn is_positive(&self) -> bool { *self > 0.0 }
            }
        )+
    };
}

impl_numeric!(int: i8, i16, i32, i64, i128, isize);
impl_numeric!(int: u8, u16, u32, u64, u128, usize);
impl_numeric!(float: f32, f64);

fn double<T: Numeric + std::ops::Add<Output = T> + Copy>(x: T) -> T {
    x + x
}

fn main() {
    assert_eq!(double(21i32), 42);
    assert_eq!(double(1.5f64), 3.0);
    assert!(42i32.is_positive());
    assert!(!(-1i32).is_positive());
}
```

### コード例: TT Muncher パターン（トークン消費パターン）

TT Muncher は宣言的マクロで複雑な構文を解析するための高度なパターンで、トークンを先頭から1つずつ「食べていく」方式で再帰処理する。

```rust
/// HTML 風の構造を構築する DSL マクロ
macro_rules! html {
    // 終了条件: 空
    () => { String::new() };

    // テキストノード
    (text($text:expr) $($rest:tt)*) => {{
        let mut s = $text.to_string();
        s.push_str(&html!($($rest)*));
        s
    }};

    // 自閉じタグ <br/>
    ($tag:ident / $($rest:tt)*) => {{
        let mut s = format!("<{}/>", stringify!($tag));
        s.push_str(&html!($($rest)*));
        s
    }};

    // 開始タグ + 子要素 + 閉じタグ
    ($tag:ident { $($children:tt)* } $($rest:tt)*) => {{
        let mut s = format!("<{}>", stringify!($tag));
        s.push_str(&html!($($children)*));
        s.push_str(&format!("</{}>", stringify!($tag)));
        s.push_str(&html!($($rest)*));
        s
    }};
}

fn main() {
    let page = html! {
        div {
            h1 { text("Hello, World!") }
            p { text("Rust マクロで HTML を生成") }
            br /
            p { text("TT Muncher パターンの例") }
        }
    };
    println!("{}", page);
    // <div><h1>Hello, World!</h1><p>Rust マクロで HTML を生成</p><br/><p>TT Muncher パターンの例</p></div>
}
```

### コード例: Push-down Accumulation パターン

再帰中に中間結果を蓄積していくパターン。JSON ビルダーの例を示す。

```rust
/// 簡易 JSON ビルダーマクロ
macro_rules! json {
    // null
    (null) => { JsonValue::Null };

    // boolean
    (true) => { JsonValue::Bool(true) };
    (false) => { JsonValue::Bool(false) };

    // 数値
    ($num:literal) => { JsonValue::Number($num as f64) };

    // 文字列
    ($s:literal) => { JsonValue::Str($s.to_string()) };

    // 配列
    ([ $($elem:tt),* $(,)? ]) => {
        JsonValue::Array(vec![ $(json!($elem)),* ])
    };

    // オブジェクト
    ({ $($key:literal : $value:tt),* $(,)? }) => {{
        let mut map = std::collections::BTreeMap::new();
        $(
            map.insert($key.to_string(), json!($value));
        )*
        JsonValue::Object(map)
    }};
}

#[derive(Debug, Clone, PartialEq)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    Str(String),
    Array(Vec<JsonValue>),
    Object(std::collections::BTreeMap<String, JsonValue>),
}

impl std::fmt::Display for JsonValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JsonValue::Null => write!(f, "null"),
            JsonValue::Bool(b) => write!(f, "{}", b),
            JsonValue::Number(n) => write!(f, "{}", n),
            JsonValue::Str(s) => write!(f, "\"{}\"", s),
            JsonValue::Array(arr) => {
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            JsonValue::Object(map) => {
                write!(f, "{{")?;
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "\"{}\": {}", k, v)?;
                }
                write!(f, "}}")
            }
        }
    }
}

fn main() {
    let data = json!({
        "name": "Alice",
        "age": 30,
        "active": true,
        "scores": [95, 87, 92],
        "address": {
            "city": "Tokyo",
            "zip": "100-0001"
        }
    });
    println!("{}", data);
}
```

### コード例: テストケース自動生成マクロ

```rust
/// パラメタライズドテストを生成するマクロ
macro_rules! test_cases {
    (
        fn $test_name:ident($input:ident : $in_ty:ty) -> $out_ty:ty $body:block
        cases {
            $( $case_name:ident : $case_input:expr => $expected:expr ),+ $(,)?
        }
    ) => {
        // テスト対象の関数を生成
        fn $test_name($input: $in_ty) -> $out_ty $body

        // 各テストケースを個別のテスト関数として生成
        $(
            #[test]
            fn $case_name() {
                let result = $test_name($case_input);
                assert_eq!(result, $expected,
                    "テストケース '{}' が失敗: 入力={:?}, 期待={:?}, 実際={:?}",
                    stringify!($case_name), $case_input, $expected, result
                );
            }
        )+
    };
}

test_cases! {
    fn fizzbuzz(n: u32) -> String {
        match (n % 3, n % 5) {
            (0, 0) => "FizzBuzz".to_string(),
            (0, _) => "Fizz".to_string(),
            (_, 0) => "Buzz".to_string(),
            _ => n.to_string(),
        }
    }
    cases {
        test_one:       1  => "1".to_string(),
        test_three:     3  => "Fizz".to_string(),
        test_five:      5  => "Buzz".to_string(),
        test_fifteen:   15 => "FizzBuzz".to_string(),
        test_seven:     7  => "7".to_string(),
    }
}
```

### macro_rules! のスコープとエクスポート

```rust
// マクロのスコープルール:
// 1. macro_rules! は定義位置より後でのみ使用可能（テキスト順序に依存）
// 2. モジュール内で定義したマクロは #[macro_export] で公開可能
// 3. #[macro_export] されたマクロはクレートルートに配置される

// モジュール内でのマクロ定義と公開
mod utils {
    // このマクロはクレートルートに配置される
    #[macro_export]
    macro_rules! public_macro {
        () => { println!("公開マクロ") };
    }

    // macro_export なしだと同一モジュール内のみ
    macro_rules! private_macro {
        () => { println!("非公開マクロ") };
    }

    pub fn use_private() {
        private_macro!();
    }
}

// #[macro_use] による一括インポート（2015 edition スタイル）
// #[macro_use] extern crate some_crate;

// 2018+ edition では use でインポート可能
// use some_crate::public_macro;

// $crate を使った正しいパス参照
#[macro_export]
macro_rules! create_vec {
    ($($elem:expr),* $(,)?) => {
        {
            // $crate は定義元クレートを指す
            // 外部クレートから使われても正しく解決される
            let mut v = $crate::__internal::new_vec();
            $(v.push($elem);)*
            v
        }
    };
}

// マクロから参照される内部モジュール
#[doc(hidden)]
pub mod __internal {
    pub fn new_vec<T>() -> Vec<T> {
        Vec::new()
    }
}
```

---

## 3. 手続き的マクロ

### プロジェクト構成

```
┌─ my-project/
│  ├─ Cargo.toml          (ワークスペース)
│  ├─ my-derive/
│  │  ├─ Cargo.toml       ← proc-macro = true
│  │  └─ src/lib.rs       ← マクロ実装
│  └─ my-app/
│     ├─ Cargo.toml       ← my-derive を依存に追加
│     └─ src/main.rs
```

### syn / quote / proc-macro2 の役割分担

```
┌─────────────────────────────────────────────────────────────┐
│                 手続き的マクロの三種の神器                      │
├────────────────┬────────────────────────────────────────────┤
│  proc-macro2   │ TokenStream の抽象化レイヤー                │
│                │ テスト時にも使えるトークンストリーム型を提供    │
├────────────────┼────────────────────────────────────────────┤
│  syn           │ TokenStream → AST (構文木) のパーサー        │
│                │ DeriveInput, ItemFn, Expr 等の型を提供       │
│                │ パース失敗時にスパン付きエラーを返せる          │
├────────────────┼────────────────────────────────────────────┤
│  quote         │ AST → TokenStream のコード生成               │
│                │ quote! マクロでRustコードテンプレートを記述     │
│                │ #var で変数補間、#(#var)* で繰り返し           │
└────────────────┴────────────────────────────────────────────┘
```

### Cargo.toml の設定例

```toml
# my-derive/Cargo.toml
[package]
name = "my-derive"
version = "0.1.0"
edition = "2021"

[lib]
proc-macro = true

[dependencies]
syn = { version = "2", features = ["full", "extra-traits"] }
quote = "1"
proc-macro2 = "1"

# テスト用（オプション）
[dev-dependencies]
trybuild = "1"   # コンパイルエラーのテスト
```

### コード例3: derive マクロの実装

```rust
// my-derive/Cargo.toml
// [lib]
// proc-macro = true
//
// [dependencies]
// syn = { version = "2", features = ["full"] }
// quote = "1"
// proc-macro2 = "1"

// my-derive/src/lib.rs
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

/// フィールド名とその値を表示する Describe トレイトを自動実装
#[proc_macro_derive(Describe)]
pub fn derive_describe(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;

    let fields = if let syn::Data::Struct(data) = &ast.data {
        data.fields.iter().map(|f| {
            let fname = f.ident.as_ref().unwrap();
            quote! {
                write!(f, "  {}: {:?}\n", stringify!(#fname), &self.#fname)?;
            }
        }).collect::<Vec<_>>()
    } else {
        panic!("Describe は構造体のみ対応");
    };

    let expanded = quote! {
        impl std::fmt::Display for #name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                writeln!(f, "{}:", stringify!(#name))?;
                #( #fields )*
                Ok(())
            }
        }
    };

    TokenStream::from(expanded)
}

// my-app/src/main.rs
// use my_derive::Describe;
//
// #[derive(Debug, Describe)]
// struct Config {
//     host: String,
//     port: u16,
//     debug: bool,
// }
//
// fn main() {
//     let cfg = Config {
//         host: "localhost".into(),
//         port: 8080,
//         debug: true,
//     };
//     println!("{}", cfg);
//     // Config:
//     //   host: "localhost"
//     //   port: 8080
//     //   debug: true
// }
```

### コード例: derive マクロで列挙体に対応する

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields};

/// 列挙体のバリアント名を文字列として返す EnumName トレイトを自動実装
#[proc_macro_derive(EnumName)]
pub fn derive_enum_name(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;

    let variants = match &ast.data {
        Data::Enum(data) => &data.variants,
        _ => {
            return syn::Error::new_spanned(
                &ast.ident,
                "EnumName は列挙体にのみ適用できます",
            )
            .to_compile_error()
            .into();
        }
    };

    // 各バリアントに対するマッチアームを生成
    let match_arms = variants.iter().map(|v| {
        let variant_name = &v.ident;
        let name_str = variant_name.to_string();
        match &v.fields {
            Fields::Unit => {
                quote! { #name::#variant_name => #name_str }
            }
            Fields::Unnamed(_) => {
                quote! { #name::#variant_name(..) => #name_str }
            }
            Fields::Named(_) => {
                quote! { #name::#variant_name { .. } => #name_str }
            }
        }
    });

    // all_names メソッド用のバリアント名リスト
    let all_names = variants.iter().map(|v| {
        let name_str = v.ident.to_string();
        quote! { #name_str }
    });

    let expanded = quote! {
        impl #name {
            /// このバリアントの名前を文字列として返す
            pub fn variant_name(&self) -> &'static str {
                match self {
                    #(#match_arms,)*
                }
            }

            /// 全バリアント名のスライスを返す
            pub fn all_variant_names() -> &'static [&'static str] {
                &[#(#all_names,)*]
            }
        }
    };

    TokenStream::from(expanded)
}

// 使用例:
// #[derive(EnumName)]
// enum Color {
//     Red,
//     Green,
//     Blue,
//     Custom(u8, u8, u8),
//     Named { name: String },
// }
//
// fn main() {
//     let c = Color::Custom(255, 0, 128);
//     println!("{}", c.variant_name());  // "Custom"
//     println!("{:?}", Color::all_variant_names());
//     // ["Red", "Green", "Blue", "Custom", "Named"]
// }
```

### コード例: derive マクロにヘルパー属性を追加する

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Lit, Meta, NestedMeta};

/// フィールドのバリデーションを自動生成する Validate derive
/// ヘルパー属性 #[validate(...)] をサポート
#[proc_macro_derive(Validate, attributes(validate))]
pub fn derive_validate(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;

    let fields = match &ast.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(f) => &f.named,
            _ => panic!("名前付きフィールドのみ対応"),
        },
        _ => panic!("構造体のみ対応"),
    };

    let validations = fields.iter().filter_map(|field| {
        let field_name = field.ident.as_ref().unwrap();

        // #[validate(...)] 属性を探す
        let validate_attrs: Vec<_> = field.attrs.iter()
            .filter(|attr| attr.path().is_ident("validate"))
            .collect();

        if validate_attrs.is_empty() {
            return None;
        }

        let mut checks = Vec::new();

        for attr in &validate_attrs {
            // 属性の内容を解析（簡易版）
            // 実際の実装では syn のパース機能を使う
            let meta = attr.parse_args::<Meta>().ok()?;

            match &meta {
                Meta::NameValue(nv) if nv.path.is_ident("min_len") => {
                    if let Lit::Int(lit) = &nv.lit {
                        let min: usize = lit.base10_parse().unwrap();
                        checks.push(quote! {
                            if self.#field_name.len() < #min {
                                errors.push(format!(
                                    "'{}' の長さが最小値 {} 未満です（実際: {}）",
                                    stringify!(#field_name), #min, self.#field_name.len()
                                ));
                            }
                        });
                    }
                }
                Meta::Path(path) if path.is_ident("non_empty") => {
                    checks.push(quote! {
                        if self.#field_name.is_empty() {
                            errors.push(format!(
                                "'{}' は空であってはなりません",
                                stringify!(#field_name)
                            ));
                        }
                    });
                }
                _ => {}
            }
        }

        Some(quote! { #(#checks)* })
    });

    let expanded = quote! {
        impl #name {
            pub fn validate(&self) -> Result<(), Vec<String>> {
                let mut errors = Vec::new();
                #(#validations)*
                if errors.is_empty() {
                    Ok(())
                } else {
                    Err(errors)
                }
            }
        }
    };

    TokenStream::from(expanded)
}

// 使用例:
// #[derive(Validate)]
// struct User {
//     #[validate(non_empty)]
//     #[validate(min_len = 3)]
//     name: String,
//
//     #[validate(non_empty)]
//     email: String,
// }
```

### コード例4: attribute マクロ

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

/// 関数の実行時間を計測する属性マクロ
#[proc_macro_attribute]
pub fn measure_time(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let name = &func.sig.ident;
    let block = &func.block;
    let sig = &func.sig;
    let vis = &func.vis;
    let attrs = &func.attrs;

    let expanded = quote! {
        #(#attrs)*
        #vis #sig {
            let __start = std::time::Instant::now();
            let __result = (|| #block)();
            let __elapsed = __start.elapsed();
            eprintln!("[measure] {} took {:?}", stringify!(#name), __elapsed);
            __result
        }
    };

    TokenStream::from(expanded)
}

// 使用例:
// #[measure_time]
// fn heavy_computation() -> u64 {
//     (0..1_000_000).sum()
// }
```

### コード例: attribute マクロで引数を受け取る

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn, LitInt};

/// リトライ回数を指定する属性マクロ
/// #[retry(3)] のように使用
#[proc_macro_attribute]
pub fn retry(attr: TokenStream, item: TokenStream) -> TokenStream {
    let max_retries = parse_macro_input!(attr as LitInt);
    let max_retries_val: usize = max_retries.base10_parse().unwrap();
    let func = parse_macro_input!(item as ItemFn);

    let name = &func.sig.ident;
    let vis = &func.vis;
    let inputs = &func.sig.inputs;
    let output = &func.sig.output;
    let block = &func.block;
    let attrs = &func.attrs;

    // 戻り値が Result 型であることを前提とする
    let expanded = quote! {
        #(#attrs)*
        #vis fn #name(#inputs) #output {
            let mut __attempts = 0usize;
            loop {
                __attempts += 1;
                let __result = (|| #block)();
                match __result {
                    Ok(val) => return Ok(val),
                    Err(e) => {
                        if __attempts >= #max_retries_val {
                            eprintln!(
                                "[retry] {} が {} 回のリトライ後も失敗: {:?}",
                                stringify!(#name), #max_retries_val, e
                            );
                            return Err(e);
                        }
                        eprintln!(
                            "[retry] {} 試行 {}/{} が失敗、リトライします...",
                            stringify!(#name), __attempts, #max_retries_val
                        );
                        std::thread::sleep(std::time::Duration::from_millis(
                            100 * __attempts as u64
                        ));
                    }
                }
            }
        }
    };

    TokenStream::from(expanded)
}

// 使用例:
// #[retry(3)]
// fn fetch_data(url: &str) -> Result<String, Box<dyn std::error::Error>> {
//     let response = reqwest::blocking::get(url)?;
//     Ok(response.text()?)
// }
```

### コード例5: function-like マクロ

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, LitStr};

/// SQL文をコンパイル時に検証する（簡易版）
#[proc_macro]
pub fn checked_sql(input: TokenStream) -> TokenStream {
    let sql_lit = parse_macro_input!(input as LitStr);
    let sql = sql_lit.value();

    // 簡易バリデーション
    let upper = sql.to_uppercase();
    if !upper.starts_with("SELECT")
        && !upper.starts_with("INSERT")
        && !upper.starts_with("UPDATE")
        && !upper.starts_with("DELETE")
    {
        return syn::Error::new(
            sql_lit.span(),
            "SQL は SELECT/INSERT/UPDATE/DELETE で始まる必要があります",
        )
        .to_compile_error()
        .into();
    }

    let expanded = quote! { #sql };
    TokenStream::from(expanded)
}

// 使用例:
// let query = checked_sql!("SELECT * FROM users WHERE id = $1");
// let bad   = checked_sql!("DROP TABLE users"); // コンパイルエラー!
```

### コード例: function-like マクロで構造化入力を解析する

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse::{Parse, ParseStream}, Ident, LitStr, Token, punctuated::Punctuated};

/// ルーティングテーブルを定義する function-like マクロ
/// routes! {
///     GET "/users"        => list_users,
///     POST "/users"       => create_user,
///     GET "/users/{id}"   => get_user,
/// }

struct Route {
    method: Ident,
    path: LitStr,
    handler: Ident,
}

impl Parse for Route {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let method: Ident = input.parse()?;
        let path: LitStr = input.parse()?;
        input.parse::<Token![=>]>()?;
        let handler: Ident = input.parse()?;
        Ok(Route { method, path, handler })
    }
}

struct Routes {
    routes: Punctuated<Route, Token![,]>,
}

impl Parse for Routes {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let routes = Punctuated::parse_terminated(input)?;
        Ok(Routes { routes })
    }
}

#[proc_macro]
pub fn routes(input: TokenStream) -> TokenStream {
    let routes = syn::parse_macro_input!(input as Routes);

    let route_entries = routes.routes.iter().map(|r| {
        let method_str = r.method.to_string();
        let path = &r.path;
        let handler = &r.handler;
        quote! {
            Router::route(#method_str, #path, #handler)
        }
    });

    let expanded = quote! {
        {
            let mut router = Router::new();
            #(router.add(#route_entries);)*
            router
        }
    };

    TokenStream::from(expanded)
}
```

---

## 4. マクロのデバッグ

```
┌─────────────────────────────────────────────────────────┐
│              マクロデバッグフロー                          │
│                                                         │
│  1. cargo expand        ← マクロ展開結果をRustとして表示  │
│     └─ cargo install cargo-expand                       │
│                                                         │
│  2. trace_macros!(true) ← 宣言的マクロの展開過程をトレース│
│     └─ #![feature(trace_macros)] (nightly)              │
│                                                         │
│  3. eprintln! in proc   ← 手続き的マクロ内で標準エラー出力│
│     └─ コンパイル時に出力される                           │
│                                                         │
│  4. syn::Error          ← スパン付きエラーでIDEに通知     │
│     └─ .to_compile_error().into()                       │
└─────────────────────────────────────────────────────────┘
```

### コード例6: cargo expand の活用

```bash
# インストール
cargo install cargo-expand

# 特定の関数だけ展開
cargo expand --lib -- some_module::some_fn

# テストコードも含めて展開
cargo expand --tests

# 特定 derive だけ確認
cargo expand --lib | grep -A 50 "impl Display"

# 特定のアイテムだけ展開
cargo expand --lib ::my_module::MyStruct
```

### 手続き的マクロのデバッグテクニック

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(MyDerive)]
pub fn my_derive(input: TokenStream) -> TokenStream {
    // デバッグ1: 入力トークンストリームを標準エラーに出力
    eprintln!("=== Input TokenStream ===");
    eprintln!("{}", input.to_string());

    let ast = parse_macro_input!(input as DeriveInput);

    // デバッグ2: パースされた AST を確認
    // syn の "extra-traits" feature が必要
    eprintln!("=== Parsed AST ===");
    eprintln!("{:#?}", ast);

    let name = &ast.ident;
    let expanded = quote! {
        impl #name {
            pub fn hello() {
                println!("Hello from {}", stringify!(#name));
            }
        }
    };

    // デバッグ3: 生成されるコードを確認
    eprintln!("=== Generated Code ===");
    eprintln!("{}", expanded.to_string());

    TokenStream::from(expanded)
}
```

### trybuild によるコンパイルエラーテスト

手続き的マクロが正しいエラーメッセージを出すことを検証するには `trybuild` クレートが有効。

```rust
// tests/ui.rs
#[test]
fn ui_tests() {
    let t = trybuild::TestCases::new();
    // コンパイルが成功するべきテスト
    t.pass("tests/ui/pass_*.rs");
    // コンパイルが失敗するべきテスト（エラーメッセージも検証）
    t.compile_fail("tests/ui/fail_*.rs");
}

// tests/ui/pass_basic.rs
// use my_derive::EnumName;
//
// #[derive(EnumName)]
// enum Color {
//     Red,
//     Green,
//     Blue,
// }
//
// fn main() {
//     let c = Color::Red;
//     assert_eq!(c.variant_name(), "Red");
// }

// tests/ui/fail_struct.rs
// use my_derive::EnumName;
//
// #[derive(EnumName)]
// struct NotAnEnum {  // エラー: EnumName は列挙体にのみ適用できます
//     field: i32,
// }
//
// fn main() {}

// tests/ui/fail_struct.stderr（期待されるエラーメッセージ）
// error: EnumName は列挙体にのみ適用できます
//   --> tests/ui/fail_struct.rs:4:8
//    |
//  4 | struct NotAnEnum {
//    |        ^^^^^^^^^^
```

### trace_macros! による展開トレース

```rust
// nightly コンパイラが必要
#![feature(trace_macros)]

macro_rules! factorial {
    (1) => { 1u64 };
    ($n:literal) => { $n as u64 * factorial!($n - 1) };
}

fn main() {
    trace_macros!(true);
    let result = factorial!(5);
    trace_macros!(false);
    println!("5! = {}", result);
}

// コンパイル時の出力:
// note: trace_macro
//  --> src/main.rs:9:18
//   |
// 9 |     let result = factorial!(5);
//   |                  ^^^^^^^^^^^^
//   |
//   = note: expanding `factorial! { 5 }`
//   = note: to `5 as u64 * factorial!(5 - 1)`
//   = note: expanding `factorial! { 5 - 1 }`
//   ...
```

---

## 5. マクロの比較

### 宣言的 vs 手続き的マクロ

| 観点 | 宣言的 (`macro_rules!`) | 手続き的 (proc-macro) |
|---|---|---|
| 定義場所 | 同一クレート内 | 別クレート (`proc-macro = true`) |
| 入力 | パターンマッチ | `TokenStream` |
| 依存関係 | なし | `syn`, `quote`, `proc-macro2` |
| 衛生性 | 部分的に保証 | 手動管理 |
| 適用範囲 | 式・文・アイテム | derive / 属性 / 関数風 |
| デバッグ | `trace_macros!` | `eprintln!` + `cargo expand` |
| 学習コスト | 低〜中 | 中〜高 |
| ユースケース | DSL、ユーティリティ | 自動実装、コード生成 |
| コンパイル時間 | 影響小 | 依存クレート分の追加コスト |
| IDE サポート | 展開結果が見えにくい | rust-analyzer が部分対応 |

### syn の主要型

| 型 | 用途 | 例 |
|---|---|---|
| `DeriveInput` | derive マクロの入力全体 | 構造体/列挙体の定義 |
| `ItemFn` | 関数定義 | attribute マクロで関数をラップ |
| `ItemStruct` | 構造体定義 | 構造体の変換 |
| `ItemEnum` | 列挙体定義 | 列挙体の分析 |
| `Fields` | フィールド集合 | Named / Unnamed / Unit |
| `Expr` | 任意の式 | `parse_quote!(x + 1)` |
| `Type` | 型表現 | `Vec<String>` |
| `LitStr` | 文字列リテラル | `"hello"` |
| `LitInt` | 整数リテラル | `42` |
| `Ident` | 識別子 | 変数名、関数名 |
| `Path` | パス | `std::io::Result` |
| `Attribute` | 属性 | `#[derive(Debug)]` |
| `Generics` | ジェネリクスパラメータ | `<T: Clone>` |
| `WhereClause` | where 句 | `where T: Debug` |

### quote! の補間構文

```rust
use quote::quote;
use syn::Ident;
use proc_macro2::Span;

// 単一変数の補間
let name = Ident::new("MyStruct", Span::call_site());
let tokens = quote! { struct #name {} };
// → struct MyStruct {}

// 繰り返しの補間
let fields = vec!["x", "y", "z"];
let field_idents: Vec<_> = fields.iter()
    .map(|f| Ident::new(f, Span::call_site()))
    .collect();
let tokens = quote! {
    struct Point {
        #(pub #field_idents: f64,)*
    }
};
// → struct Point { pub x: f64, pub y: f64, pub z: f64, }

// 区切り文字付き繰り返し
let values = vec![1i32, 2, 3];
let tokens = quote! {
    let sum = #(#values)+*;
};
// → let sum = 1 + 2 + 3;

// ネストされた繰り返し
let methods = vec![("get_x", "x"), ("get_y", "y")];
let method_tokens: Vec<_> = methods.iter().map(|(method, field)| {
    let method_ident = Ident::new(method, Span::call_site());
    let field_ident = Ident::new(field, Span::call_site());
    quote! {
        pub fn #method_ident(&self) -> f64 {
            self.#field_ident
        }
    }
}).collect();

let tokens = quote! {
    impl Point {
        #(#method_tokens)*
    }
};

// format_ident! で識別子を動的に生成
let prefix = "get";
let field = "name";
let getter = quote::format_ident!("{}_{}", prefix, field);
// → get_name
```

---

## 6. アンチパターン

### アンチパターン1: マクロの過剰使用

```rust
// NG: 関数やジェネリクスで十分な処理をマクロにする
macro_rules! add_numbers {
    ($a:expr, $b:expr) => { $a + $b };
}

// OK: 通常の関数で十分
fn add<T: std::ops::Add<Output = T>>(a: T, b: T) -> T {
    a + b
}

// マクロが適切な場面:
// - 型に跨る繰り返しパターン (impl Trait for A, B, C...)
// - コンパイル時計算/検証
// - DSL (ドメイン固有言語)
// - ボイラープレート削減 (derive 対応できない場合)
```

### アンチパターン2: 衛生性を無視した変数キャプチャ

```rust
// NG: 外側の変数名に依存するマクロ
macro_rules! bad_log {
    ($msg:expr) => {
        // logger が呼び出し元スコープに存在する前提 — 危険!
        logger.log($msg);
    };
}

// OK: 必要な依存は引数で明示
macro_rules! good_log {
    ($logger:expr, $msg:expr) => {
        $logger.log($msg);
    };
}

// OK: 手続き的マクロで衛生的な変数名を生成
// quote! {
//     let __internal_logger = get_logger();
//     __internal_logger.log(#msg);
// }
```

### アンチパターン3: 式の多重評価

```rust
// NG: $val が複数回評価される
macro_rules! bad_max {
    ($a:expr, $b:expr) => {
        if $a > $b { $a } else { $b }
    };
}

// 問題: 副作用のある式が2回実行される
// bad_max!(expensive_call(), another_call());
// → if expensive_call() > another_call() { expensive_call() } else { another_call() }
// expensive_call() が2回呼ばれる可能性がある!

// OK: 一度変数に束縛する
macro_rules! good_max {
    ($a:expr, $b:expr) => {{
        let __a = $a;
        let __b = $b;
        if __a > __b { __a } else { __b }
    }};
}
```

### アンチパターン4: 型の不一致を隠蔽するマクロ

```rust
// NG: 暗黙の型変換を行うマクロ
macro_rules! bad_convert {
    ($val:expr) => {
        $val as i64  // 情報の損失が見えない
    };
}

// OK: 明示的な型変換関数
fn safe_convert(val: u32) -> i64 {
    i64::from(val)  // From トレイトで安全に変換
}

// OK: TryFrom で失敗を明示
fn try_convert(val: i64) -> Result<u32, std::num::TryFromIntError> {
    u32::try_from(val)
}
```

### アンチパターン5: 巨大マクロ

```rust
// NG: 数百行のマクロ — 読めない、デバッグできない、IDEサポートなし

// OK: マクロは薄いラッパーにとどめ、実装はヘルパー関数に委譲
macro_rules! register_handler {
    ($name:ident, $method:expr, $path:expr) => {
        // マクロはグルーコードだけ
        fn $name(registry: &mut Registry) {
            // 実際のロジックは通常の関数に委譲
            __register_handler_impl(registry, $method, $path, $name::handler);
        }
    };
}

// ヘルパー関数は通常のRustコードなのでIDEサポートも効く
fn __register_handler_impl(
    registry: &mut Registry,
    method: &str,
    path: &str,
    handler: fn(&Request) -> Response,
) {
    registry.add_route(method, path, handler);
}
```

---

## 7. 実践: Builder パターンの自動生成

### コード例7: derive(Builder) の簡易実装

```rust
// builder-derive/src/lib.rs
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields};

#[proc_macro_derive(Builder)]
pub fn derive_builder(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let builder_name = syn::Ident::new(
        &format!("{}Builder", name),
        name.span(),
    );

    let fields = match &ast.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(f) => &f.named,
            _ => panic!("名前付きフィールドのみ対応"),
        },
        _ => panic!("構造体のみ対応"),
    };

    // Builder のフィールド (全て Option<T>)
    let builder_fields = fields.iter().map(|f| {
        let name = &f.ident;
        let ty = &f.ty;
        quote! { #name: Option<#ty> }
    });

    // setter メソッド
    let setters = fields.iter().map(|f| {
        let name = &f.ident;
        let ty = &f.ty;
        quote! {
            pub fn #name(mut self, val: #ty) -> Self {
                self.#name = Some(val);
                self
            }
        }
    });

    // build メソッドのフィールド取り出し
    let build_fields = fields.iter().map(|f| {
        let name = &f.ident;
        quote! {
            #name: self.#name.ok_or(
                format!("フィールド '{}' が未設定", stringify!(#name))
            )?
        }
    });

    // Builder のデフォルト値 (全て None)
    let none_fields = fields.iter().map(|f| {
        let name = &f.ident;
        quote! { #name: None }
    });

    let expanded = quote! {
        pub struct #builder_name {
            #( #builder_fields, )*
        }

        impl #name {
            pub fn builder() -> #builder_name {
                #builder_name {
                    #( #none_fields, )*
                }
            }
        }

        impl #builder_name {
            #( #setters )*

            pub fn build(self) -> Result<#name, String> {
                Ok(#name {
                    #( #build_fields, )*
                })
            }
        }
    };

    TokenStream::from(expanded)
}
```

### コード例: Builder パターンにデフォルト値とバリデーションを追加

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput, Data, Fields, Lit, Meta};

/// 拡張版 Builder — デフォルト値とバリデーションをサポート
/// #[builder(default = "value")] でデフォルト値を指定
/// #[builder(required)] で必須フィールドを明示
#[proc_macro_derive(ExtBuilder, attributes(builder))]
pub fn derive_ext_builder(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let builder_name = quote::format_ident!("{}Builder", name);

    let fields = match &ast.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(f) => &f.named,
            _ => return err(name, "名前付きフィールドのみ対応"),
        },
        _ => return err(name, "構造体のみ対応"),
    };

    let mut builder_field_defs = Vec::new();
    let mut setter_defs = Vec::new();
    let mut build_assigns = Vec::new();
    let mut default_assigns = Vec::new();

    for field in fields.iter() {
        let field_name = field.ident.as_ref().unwrap();
        let field_ty = &field.ty;

        // #[builder(...)] 属性を解析
        let mut has_default = false;
        let mut default_expr = None;

        for attr in &field.attrs {
            if !attr.path().is_ident("builder") {
                continue;
            }
            // 属性の中身を解析（簡易版）
            // 実際には syn::parse を使ってもっと堅牢にする
            if let Ok(Meta::NameValue(nv)) = attr.parse_args::<Meta>() {
                if nv.path.is_ident("default") {
                    has_default = true;
                    if let Lit::Str(s) = &nv.lit {
                        default_expr = Some(s.value());
                    }
                }
            }
        }

        builder_field_defs.push(quote! {
            #field_name: Option<#field_ty>
        });

        setter_defs.push(quote! {
            pub fn #field_name(mut self, val: #field_ty) -> Self {
                self.#field_name = Some(val);
                self
            }
        });

        if has_default {
            if let Some(expr_str) = default_expr {
                let expr: proc_macro2::TokenStream = expr_str.parse().unwrap();
                build_assigns.push(quote! {
                    #field_name: self.#field_name.unwrap_or_else(|| #expr)
                });
            } else {
                build_assigns.push(quote! {
                    #field_name: self.#field_name.unwrap_or_default()
                });
            }
        } else {
            build_assigns.push(quote! {
                #field_name: self.#field_name.ok_or_else(||
                    format!("必須フィールド '{}' が未設定です", stringify!(#field_name))
                )?
            });
        }

        default_assigns.push(quote! { #field_name: None });
    }

    let expanded = quote! {
        pub struct #builder_name {
            #(#builder_field_defs,)*
        }

        impl #name {
            pub fn builder() -> #builder_name {
                #builder_name {
                    #(#default_assigns,)*
                }
            }
        }

        impl #builder_name {
            #(#setter_defs)*

            pub fn build(self) -> Result<#name, String> {
                Ok(#name {
                    #(#build_assigns,)*
                })
            }
        }
    };

    TokenStream::from(expanded)
}

fn err(ident: &syn::Ident, msg: &str) -> TokenStream {
    syn::Error::new_spanned(ident, msg)
        .to_compile_error()
        .into()
}

// 使用例:
// #[derive(ExtBuilder)]
// struct ServerConfig {
//     host: String,                           // 必須
//     port: u16,                              // 必須
//     #[builder(default = "30")]
//     timeout_secs: u64,                      // デフォルト: 30
//     #[builder(default = "true")]
//     tls_enabled: bool,                      // デフォルト: true
//     #[builder(default = "String::from(\"INFO\")")]
//     log_level: String,                      // デフォルト: "INFO"
// }
//
// fn main() {
//     let config = ServerConfig::builder()
//         .host("0.0.0.0".to_string())
//         .port(8080)
//         // timeout_secs, tls_enabled, log_level はデフォルト値が使われる
//         .build()
//         .unwrap();
// }
```

---

## 8. 実践: 高度なマクロパターン

### コード例: enum_dispatch パターン（静的ディスパッチの自動生成）

```rust
/// トレイトオブジェクトの動的ディスパッチを静的ディスパッチに変換する
/// enum_dispatch! パターンの簡易実装
macro_rules! enum_dispatch {
    (
        trait $trait_name:ident {
            $(fn $method:ident(&self $(, $arg:ident : $arg_ty:ty)*) -> $ret:ty;)*
        }

        enum $enum_name:ident {
            $($variant:ident($inner:ty)),+ $(,)?
        }
    ) => {
        // トレイト定義
        trait $trait_name {
            $(fn $method(&self $(, $arg: $arg_ty)*) -> $ret;)*
        }

        // 列挙体定義
        enum $enum_name {
            $($variant($inner),)+
        }

        // From 実装
        $(
            impl From<$inner> for $enum_name {
                fn from(v: $inner) -> Self {
                    $enum_name::$variant(v)
                }
            }
        )+

        // トレイト実装（match で各バリアントに委譲）
        impl $trait_name for $enum_name {
            $(
                fn $method(&self $(, $arg: $arg_ty)*) -> $ret {
                    match self {
                        $($enum_name::$variant(inner) => inner.$method($($arg),*),)+
                    }
                }
            )*
        }
    };
}

// 使用例
enum_dispatch! {
    trait Shape {
        fn area(&self) -> f64;
        fn name(&self) -> &'static str;
    }

    enum AnyShape {
        Circle(Circle),
        Rectangle(Rectangle),
        Triangle(Triangle),
    }
}

struct Circle { radius: f64 }
impl Shape for Circle {
    fn area(&self) -> f64 { std::f64::consts::PI * self.radius * self.radius }
    fn name(&self) -> &'static str { "Circle" }
}

struct Rectangle { width: f64, height: f64 }
impl Shape for Rectangle {
    fn area(&self) -> f64 { self.width * self.height }
    fn name(&self) -> &'static str { "Rectangle" }
}

struct Triangle { base: f64, height: f64 }
impl Shape for Triangle {
    fn area(&self) -> f64 { 0.5 * self.base * self.height }
    fn name(&self) -> &'static str { "Triangle" }
}

fn print_area(shape: &AnyShape) {
    // dyn Shape ではなく enum match で静的ディスパッチ
    println!("{}: area = {:.2}", shape.name(), shape.area());
}

fn main() {
    let shapes: Vec<AnyShape> = vec![
        Circle { radius: 5.0 }.into(),
        Rectangle { width: 4.0, height: 6.0 }.into(),
        Triangle { base: 3.0, height: 8.0 }.into(),
    ];

    for shape in &shapes {
        print_area(shape);
    }
    // Circle: area = 78.54
    // Rectangle: area = 24.00
    // Triangle: area = 12.00
}
```

### コード例: type-state パターンのマクロ化

```rust
/// 型状態パターンを自動生成するマクロ
/// コンパイル時に不正な状態遷移を防止する
macro_rules! state_machine {
    (
        machine $machine:ident {
            states: [ $($state:ident),+ $(,)? ],
            transitions: [
                $($from:ident -> $to:ident : $event:ident ($($arg:ident : $arg_ty:ty),*) ),+ $(,)?
            ]
        }
    ) => {
        // 状態型（ゼロサイズ型）
        $(
            #[derive(Debug)]
            pub struct $state;
        )+

        // 状態マシン本体
        #[derive(Debug)]
        pub struct $machine<State> {
            _state: std::marker::PhantomData<State>,
            // 実際にはここにデータフィールドが入る
            data: String,
        }

        // 初期状態の生成
        impl $machine<$($state)?> {
            // ここは最初の状態に対してのみ
        }

        // 各遷移メソッドを生成
        $(
            impl $machine<$from> {
                pub fn $event(self $(, $arg: $arg_ty)*) -> $machine<$to> {
                    println!("[state] {} -> {} (event: {})",
                        stringify!($from),
                        stringify!($to),
                        stringify!($event)
                    );
                    $machine {
                        _state: std::marker::PhantomData,
                        data: self.data,
                    }
                }
            }
        )+
    };
}

// 使用例: HTTP リクエストの状態マシン
state_machine! {
    machine HttpRequest {
        states: [Created, HeadersSent, BodySent, Complete],
        transitions: [
            Created -> HeadersSent : send_headers(headers: Vec<(String, String)>),
            HeadersSent -> BodySent : send_body(body: Vec<u8>),
            BodySent -> Complete : finish(),
        ]
    }
}

impl HttpRequest<Created> {
    pub fn new(url: &str) -> Self {
        HttpRequest {
            _state: std::marker::PhantomData,
            data: url.to_string(),
        }
    }
}

fn main() {
    let req = HttpRequest::new("https://example.com");
    let req = req.send_headers(vec![
        ("Content-Type".into(), "application/json".into()),
    ]);
    let req = req.send_body(b"{}".to_vec());
    let _req = req.finish();

    // コンパイルエラー: Created 状態から直接 send_body は呼べない
    // let req = HttpRequest::new("https://example.com");
    // let req = req.send_body(b"{}".to_vec()); // ERROR!
}
```

### コード例: 型安全なビットフラグマクロ

```rust
/// ビットフラグ型を自動生成するマクロ
macro_rules! bitflags {
    (
        $(#[$outer:meta])*
        $vis:vis struct $name:ident : $repr:ty {
            $(
                $(#[$inner:meta])*
                const $flag:ident = $value:expr;
            )*
        }
    ) => {
        $(#[$outer])*
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        $vis struct $name {
            bits: $repr,
        }

        impl $name {
            $(
                $(#[$inner])*
                pub const $flag: $name = $name { bits: $value };
            )*

            /// 空のフラグセット
            pub const fn empty() -> Self {
                $name { bits: 0 }
            }

            /// 全フラグが立ったセット
            pub const fn all() -> Self {
                $name { bits: $( $value )|* }
            }

            /// 指定フラグが含まれているか
            pub const fn contains(self, other: Self) -> bool {
                (self.bits & other.bits) == other.bits
            }

            /// フラグが空かどうか
            pub const fn is_empty(self) -> bool {
                self.bits == 0
            }

            /// 生のビット値を取得
            pub const fn bits(self) -> $repr {
                self.bits
            }

            /// フラグを追加
            pub fn insert(&mut self, other: Self) {
                self.bits |= other.bits;
            }

            /// フラグを除去
            pub fn remove(&mut self, other: Self) {
                self.bits &= !other.bits;
            }

            /// フラグをトグル
            pub fn toggle(&mut self, other: Self) {
                self.bits ^= other.bits;
            }
        }

        impl std::ops::BitOr for $name {
            type Output = Self;
            fn bitor(self, rhs: Self) -> Self {
                $name { bits: self.bits | rhs.bits }
            }
        }

        impl std::ops::BitAnd for $name {
            type Output = Self;
            fn bitand(self, rhs: Self) -> Self {
                $name { bits: self.bits & rhs.bits }
            }
        }

        impl std::ops::BitOrAssign for $name {
            fn bitor_assign(&mut self, rhs: Self) {
                self.bits |= rhs.bits;
            }
        }

        impl std::ops::Not for $name {
            type Output = Self;
            fn not(self) -> Self {
                $name { bits: !self.bits & Self::all().bits }
            }
        }

        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let mut first = true;
                write!(f, "{}(", stringify!($name))?;
                $(
                    if self.contains($name::$flag) {
                        if !first { write!(f, " | ")?; }
                        write!(f, stringify!($flag))?;
                        first = false;
                    }
                )*
                if first {
                    write!(f, "empty")?;
                }
                write!(f, ")")
            }
        }
    };
}

// 使用例
bitflags! {
    /// ファイルのアクセス権限
    pub struct Permissions: u32 {
        /// 読み取り権限
        const READ    = 0b001;
        /// 書き込み権限
        const WRITE   = 0b010;
        /// 実行権限
        const EXECUTE = 0b100;
    }
}

fn main() {
    let mut perms = Permissions::READ | Permissions::WRITE;
    println!("{:?}", perms);
    // Permissions(READ | WRITE)

    assert!(perms.contains(Permissions::READ));
    assert!(!perms.contains(Permissions::EXECUTE));

    perms.insert(Permissions::EXECUTE);
    println!("{:?}", perms);
    // Permissions(READ | WRITE | EXECUTE)

    perms.remove(Permissions::WRITE);
    println!("{:?}", perms);
    // Permissions(READ | EXECUTE)

    let readonly = !Permissions::WRITE & Permissions::all();
    println!("{:?}", readonly);
    // Permissions(READ | EXECUTE)
}
```

---

## 9. 衛生性（Hygiene）の深掘り

### 宣言的マクロの衛生性

Rust の `macro_rules!` マクロは「部分的に衛生的」である。マクロ内で導入された識別子は呼び出し元のスコープと衝突しないが、型名やトレイト名は衛生的でない場合がある。

```rust
macro_rules! hygiene_demo {
    () => {
        // この `x` はマクロ内で導入された変数
        // 呼び出し元の `x` とは別のスコープ
        let x = 42;
        println!("マクロ内の x: {}", x);
    };
}

fn main() {
    let x = 100;
    hygiene_demo!();
    // マクロ内の x: 42
    println!("main の x: {}", x);
    // main の x: 100
    // → 衝突しない（衛生的）
}
```

### 手続き的マクロの衛生性管理

```rust
use quote::quote;
use proc_macro2::Span;

// 衛生的な変数名の生成テクニック

// 方法1: __ プレフィックスで衝突を避ける（慣習的）
let var = quote! { __internal_counter };

// 方法2: Span::call_site() — 呼び出し元のスコープで解決
let name = syn::Ident::new("result", Span::call_site());
// → 呼び出し元のスコープの `result` と同じ名前空間

// 方法3: Span::mixed_site() — 宣言的マクロに近い衛生性
let name = syn::Ident::new("result", Span::mixed_site());
// → ローカル変数は衛生的、パス解決は呼び出し元

// ベストプラクティス: 内部変数は衝突しにくい名前を使う
let expanded = quote! {
    {
        let __macro_internal_value = compute();
        let __macro_internal_guard = lock.lock().unwrap();
        use_value(__macro_internal_value, &__macro_internal_guard)
    }
};
```

---

## 10. パフォーマンスとコンパイル時間

### マクロがコンパイル時間に与える影響

```
┌────────────────────────────────────────────────────────┐
│          マクロとコンパイル時間の関係                     │
├──────────────────┬─────────────────────────────────────┤
│ macro_rules!     │ ・展開は高速（パターンマッチのみ）     │
│                  │ ・深い再帰は遅くなる                   │
│                  │ ・再帰制限: #![recursion_limit="256"]  │
├──────────────────┼─────────────────────────────────────┤
│ proc-macro       │ ・別クレートのコンパイルが必要          │
│                  │ ・syn の features が多いほど遅い       │
│                  │ ・初回ビルドが特に遅い                 │
├──────────────────┼─────────────────────────────────────┤
│ 展開後コード      │ ・展開結果が大量だと型チェックが遅い   │
│                  │ ・単相化(monomorphization)のコストに注意│
└──────────────────┴─────────────────────────────────────┘
```

### コンパイル時間の最適化

```toml
# Cargo.toml — syn の feature を最小限に
[dependencies]
syn = { version = "2", features = ["derive", "parsing"] }
# "full" は使わない（パース対象を限定する）

# proc-macro のキャッシュを活用
# proc-macro クレートを分離することで、
# アプリケーションコード変更時に proc-macro の再コンパイルを回避
```

```rust
// 再帰マクロの深度を制御
// 深い再帰が必要な場合は制限を引き上げる
#![recursion_limit = "512"]

// ただし、再帰の深さは O(n) から O(log n) に最適化可能な場合がある
// 例: カウントマクロの最適化

// O(n) 版 — 要素数に比例して再帰が深くなる
macro_rules! count_linear {
    () => { 0usize };
    ($x:tt $($xs:tt)*) => { 1 + count_linear!($($xs)*) };
}

// O(log n) 版 — バイナリ分割で再帰の深さを半減
macro_rules! count_log {
    () => { 0usize };
    ($x:tt) => { 1usize };
    ($($a:tt)+ ; $($b:tt)+) => {
        count_log!($($a)+) + count_log!($($b)+)
    };
    ($a:tt $($rest:tt)*) => {
        count_log!($a) + count_log!($($rest)*)
    };
}
```

---

## FAQ

### Q1: `macro_rules!` と `proc-macro` のどちらを使うべき?

**A:** まず `macro_rules!` で実現できるか検討してください。パターンマッチで十分なケース（繰り返し展開、簡易DSL）は宣言的マクロが適切です。型情報の解析やコード構造の変換が必要な場合は手続き的マクロを使います。

判断フローチャート:

```
単純な繰り返し展開？ → Yes → macro_rules!
         ↓ No
DSL（独自構文）？ → Yes → macro_rules!（シンプルなら）
         ↓           → proc-macro（複雑なら）
型情報の解析が必要？ → Yes → proc-macro (derive)
         ↓ No
関数/構造体の変換？ → Yes → proc-macro (attribute)
         ↓ No
コンパイル時検証？ → Yes → proc-macro (function-like)
```

### Q2: マクロのエラーメッセージを改善するには?

**A:** 手続き的マクロでは `syn::Error::new_spanned(tokens, message)` を使い、問題のあるトークンにスパンを紐付けます。宣言的マクロでは `compile_error!("メッセージ")` を活用します。

```rust
// 手続き的マクロ内
if !is_valid {
    return syn::Error::new_spanned(
        &field.ident,
        "このフィールドは String 型である必要があります"
    ).to_compile_error().into();
}

// 宣言的マクロ内で条件付きエラー
macro_rules! assert_type {
    ($val:expr, String) => { /* OK */ };
    ($val:expr, $other:ty) => {
        compile_error!(concat!(
            "期待する型は String ですが ",
            stringify!($other),
            " が指定されました"
        ));
    };
}

// 複数エラーを返す（手続き的マクロ）
fn validate_fields(fields: &[Field]) -> Result<(), TokenStream> {
    let mut errors = Vec::new();
    for field in fields {
        if !is_supported_type(&field.ty) {
            errors.push(syn::Error::new_spanned(
                &field.ty,
                format!("サポートされていない型です: {:?}", field.ty),
            ));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        // 複数のエラーを結合
        let mut combined = errors[0].clone();
        for error in &errors[1..] {
            combined.combine(error.clone());
        }
        Err(combined.to_compile_error().into())
    }
}
```

### Q3: `quote!` 内で変数を補間する方法は?

**A:** `#var` で単一の変数を、`#( #var )*` で繰り返しを補間します。

```rust
let name = quote::format_ident!("my_func");
let types = vec![quote!(i32), quote!(String)];

let expanded = quote! {
    fn #name(#( arg: #types ),*) {}
    // → fn my_func(arg: i32, arg: String) {}
};
```

### Q4: proc-macro クレートでテストを書く方法は?

**A:** proc-macro クレートでは通常のユニットテストが書けないため、以下の戦略を取る。

```rust
// 戦略1: ロジックを非 proc-macro クレートに分離
// my-derive-core/src/lib.rs (普通のライブラリクレート)
pub fn generate_impl(ast: &syn::DeriveInput) -> proc_macro2::TokenStream {
    // ここにロジックを書く（テスト可能）
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation() {
        let input: syn::DeriveInput = syn::parse_quote! {
            struct Foo {
                bar: i32,
                baz: String,
            }
        };
        let output = generate_impl(&input);
        let expected = quote::quote! {
            // 期待されるコード
        };
        assert_eq!(output.to_string(), expected.to_string());
    }
}

// my-derive/src/lib.rs (proc-macro クレート — 薄いラッパー)
#[proc_macro_derive(MyDerive)]
pub fn my_derive(input: TokenStream) -> TokenStream {
    let ast = syn::parse_macro_input!(input as syn::DeriveInput);
    my_derive_core::generate_impl(&ast).into()
}

// 戦略2: trybuild でコンパイルテスト（前述）
// 戦略3: integration test で実際に derive を使うコードを書く
```

### Q5: マクロで async fn を扱うには?

**A:** attribute マクロで async fn をラップする場合、特別な対応が必要。

```rust
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

#[proc_macro_attribute]
pub fn async_measure(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let name = &func.sig.ident;
    let block = &func.block;
    let vis = &func.vis;
    let attrs = &func.attrs;
    let sig = &func.sig;

    // async かどうかで生成コードを分岐
    let expanded = if sig.asyncness.is_some() {
        quote! {
            #(#attrs)*
            #vis #sig {
                let __start = std::time::Instant::now();
                let __result = async move #block .await;
                let __elapsed = __start.elapsed();
                eprintln!("[async_measure] {} took {:?}", stringify!(#name), __elapsed);
                __result
            }
        }
    } else {
        quote! {
            #(#attrs)*
            #vis #sig {
                let __start = std::time::Instant::now();
                let __result = (|| #block)();
                let __elapsed = __start.elapsed();
                eprintln!("[measure] {} took {:?}", stringify!(#name), __elapsed);
                __result
            }
        }
    };

    TokenStream::from(expanded)
}
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| 宣言的マクロ | `macro_rules!` でパターンマッチ。同一クレートで定義可能 |
| 手続き的マクロ | 別クレート必須。`syn` + `quote` が標準ツール |
| derive マクロ | `#[derive(Foo)]` で自動実装。最も使用頻度が高い |
| attribute マクロ | `#[foo]` で関数/構造体を変換 |
| function-like | `foo!(...)` で自由形式の入力を受け取る |
| デバッグ | `cargo expand` が最重要ツール |
| 衛生性 | 宣言的マクロは部分保証。手続き的は手動管理 |
| 設計原則 | 関数で済むならマクロにしない。エラーメッセージを親切に |
| パフォーマンス | syn の features を最小限に。再帰深度に注意 |
| テスト | trybuild + ロジック分離が定石 |

## 次に読むべきガイド

- [async/await基礎](../02-async/00-async-basics.md) — Rustの非同期プログラミングモデル
- [テスト](../04-ecosystem/01-testing.md) — マクロのテスト手法を含むテスト戦略
- [ベストプラクティス](../04-ecosystem/04-best-practices.md) — API設計とマクロの適切な使い所

## 参考文献

1. **The Rust Reference — Macros**: https://doc.rust-lang.org/reference/macros.html
2. **The Little Book of Rust Macros**: https://veykril.github.io/tlborm/
3. **syn crate documentation**: https://docs.rs/syn/latest/syn/
4. **Proc-Macro Workshop (dtolnay)**: https://github.com/dtolnay/proc-macro-workshop
5. **quote crate documentation**: https://docs.rs/quote/latest/quote/
6. **Rust API Guidelines — Macros**: https://rust-lang.github.io/api-guidelines/macros.html
