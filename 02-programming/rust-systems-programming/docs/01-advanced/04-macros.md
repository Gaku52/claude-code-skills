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

### syn の主要型

| 型 | 用途 | 例 |
|---|---|---|
| `DeriveInput` | derive マクロの入力全体 | 構造体/列挙体の定義 |
| `ItemFn` | 関数定義 | attribute マクロで関数をラップ |
| `Fields` | フィールド集合 | Named / Unnamed / Unit |
| `Expr` | 任意の式 | `parse_quote!(x + 1)` |
| `Type` | 型表現 | `Vec<String>` |
| `LitStr` | 文字列リテラル | `"hello"` |
| `Ident` | 識別子 | 変数名、関数名 |

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

---

## FAQ

### Q1: `macro_rules!` と `proc-macro` のどちらを使うべき?

**A:** まず `macro_rules!` で実現できるか検討してください。パターンマッチで十分なケース（繰り返し展開、簡易DSL）は宣言的マクロが適切です。型情報の解析やコード構造の変換が必要な場合は手続き的マクロを使います。

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

## 次に読むべきガイド

- [async/await基礎](../02-async/00-async-basics.md) — Rustの非同期プログラミングモデル
- [テスト](../04-ecosystem/01-testing.md) — マクロのテスト手法を含むテスト戦略
- [ベストプラクティス](../04-ecosystem/04-best-practices.md) — API設計とマクロの適切な使い所

## 参考文献

1. **The Rust Reference — Macros**: https://doc.rust-lang.org/reference/macros.html
2. **The Little Book of Rust Macros**: https://veykril.github.io/tlborm/
3. **syn crate documentation**: https://docs.rs/syn/latest/syn/
4. **Proc-Macro Workshop (dtolnay)**: https://github.com/dtolnay/proc-macro-workshop
