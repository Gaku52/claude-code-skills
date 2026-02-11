# エラーハンドリング -- Rustの型安全なエラー処理パターン

> Rustは例外機構を持たず、Result<T, E> と Option<T> を用いた明示的なエラー処理により、全てのエラーパスをコンパイル時に検証する。

---

## この章で学ぶこと

1. **Result と Option** -- 失敗可能性を型で表現し、パターンマッチで安全に処理する方法を理解する
2. **? 演算子とエラー伝播** -- ボイラープレートを減らす構文糖衣と変換の仕組みを習得する
3. **thiserror / anyhow** -- 実務で使われるエラー処理クレートの使い分けを学ぶ

---

## 1. Rust のエラー処理哲学

```
┌─────────────────────────────────────────────────────┐
│           Rust のエラー分類                           │
├──────────────────┬──────────────────────────────────┤
│ 回復不能エラー   │ panic!() -- プログラム中断        │
│                  │ 配列の範囲外アクセス等             │
├──────────────────┼──────────────────────────────────┤
│ 回復可能エラー   │ Result<T, E> -- 呼び出し元が処理  │
│                  │ ファイル未発見、パースエラー等     │
├──────────────────┼──────────────────────────────────┤
│ 値の不在         │ Option<T> -- None は正常な状態     │
│                  │ 検索結果なし、設定項目なし等       │
└──────────────────┴──────────────────────────────────┘
```

---

## 2. Option<T>

### 例1: Option の基本

```rust
fn find_user(id: u64) -> Option<String> {
    match id {
        1 => Some(String::from("田中")),
        2 => Some(String::from("鈴木")),
        _ => None,
    }
}

fn main() {
    // パターンマッチ
    match find_user(1) {
        Some(name) => println!("ユーザー: {}", name),
        None => println!("見つかりません"),
    }

    // if let
    if let Some(name) = find_user(2) {
        println!("ユーザー: {}", name);
    }

    // unwrap_or でデフォルト値
    let name = find_user(99).unwrap_or(String::from("不明"));
    println!("ユーザー: {}", name);
}
```

### 例2: Option のコンビネータ

```rust
fn main() {
    let numbers = vec![1, 2, 3, 4, 5];

    // map: Some の中の値を変換
    let first_doubled: Option<i32> = numbers.first().map(|x| x * 2);
    println!("{:?}", first_doubled); // Some(2)

    // and_then: ネストした Option をフラットに
    let result = Some("42")
        .and_then(|s| s.parse::<i32>().ok())
        .map(|n| n * 2);
    println!("{:?}", result); // Some(84)

    // filter: 条件を満たさなければ None
    let even = Some(4).filter(|x| x % 2 == 0);
    let odd = Some(3).filter(|x| x % 2 == 0);
    println!("{:?}, {:?}", even, odd); // Some(4), None

    // unwrap_or_else: デフォルト値を遅延評価
    let value = None::<i32>.unwrap_or_else(|| {
        println!("デフォルト値を計算中...");
        0
    });
    println!("{}", value);
}
```

---

## 3. Result<T, E>

### 例3: Result の基本

```rust
use std::fs;
use std::io;

fn read_username() -> Result<String, io::Error> {
    let content = fs::read_to_string("username.txt")?;
    Ok(content.trim().to_string())
}

fn main() {
    match read_username() {
        Ok(name) => println!("ユーザー名: {}", name),
        Err(e) => println!("エラー: {}", e),
    }
}
```

### 例4: ? 演算子によるエラー伝播

```rust
use std::fs::File;
use std::io::{self, Read};

// ? 演算子なし (冗長)
fn read_file_verbose(path: &str) -> Result<String, io::Error> {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => return Err(e),
    };
    let mut buf = String::new();
    match file.read_to_string(&mut buf) {
        Ok(_) => Ok(buf),
        Err(e) => Err(e),
    }
}

// ? 演算子あり (簡潔)
fn read_file_concise(path: &str) -> Result<String, io::Error> {
    let mut file = File::open(path)?;
    let mut buf = String::new();
    file.read_to_string(&mut buf)?;
    Ok(buf)
}

// さらに簡潔
fn read_file_short(path: &str) -> Result<String, io::Error> {
    std::fs::read_to_string(path)
}
```

### ? 演算子の動作フロー

```
         read_file_concise()
              │
    File::open(path)?
              │
         ┌────┴────┐
         │         │
      Ok(file)  Err(e)
         │         │
         │    return Err(e)  ← 早期リターン
         │
    file.read_to_string(&mut buf)?
              │
         ┌────┴────┐
         │         │
      Ok(n)    Err(e)
         │         │
    Ok(buf)   return Err(e)
```

---

## 4. カスタムエラー型

### 例5: 手動でカスタムエラーを定義

```rust
use std::fmt;
use std::num::ParseIntError;

#[derive(Debug)]
enum AppError {
    IoError(std::io::Error),
    ParseError(ParseIntError),
    ValidationError(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::IoError(e) => write!(f, "IOエラー: {}", e),
            AppError::ParseError(e) => write!(f, "パースエラー: {}", e),
            AppError::ValidationError(msg) => write!(f, "検証エラー: {}", msg),
        }
    }
}

impl std::error::Error for AppError {}

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        AppError::IoError(e)
    }
}

impl From<ParseIntError> for AppError {
    fn from(e: ParseIntError) -> Self {
        AppError::ParseError(e)
    }
}

fn load_config(path: &str) -> Result<u32, AppError> {
    let content = std::fs::read_to_string(path)?; // IoError に自動変換
    let port: u32 = content.trim().parse()?;       // ParseError に自動変換
    if port < 1024 {
        return Err(AppError::ValidationError(
            format!("ポート {} は予約済み", port),
        ));
    }
    Ok(port)
}
```

---

## 5. thiserror と anyhow

### 例6: thiserror(ライブラリ向け)

```rust
use thiserror::Error;

#[derive(Debug, Error)]
enum DatabaseError {
    #[error("接続エラー: {0}")]
    ConnectionFailed(String),

    #[error("クエリエラー: {query}")]
    QueryFailed { query: String, source: sqlx::Error },

    #[error("レコードが見つかりません: ID={id}")]
    NotFound { id: u64 },

    #[error(transparent)]
    Io(#[from] std::io::Error),
}
```

### 例7: anyhow(アプリケーション向け)

```rust
use anyhow::{Context, Result, bail, ensure};

fn load_config(path: &str) -> Result<Config> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("設定ファイル '{}' を読み込めません", path))?;

    let config: Config = toml::from_str(&content)
        .context("設定ファイルのパースに失敗")?;

    ensure!(config.port > 0, "ポートは正の整数でなければなりません");

    if config.host.is_empty() {
        bail!("ホストが指定されていません");
    }

    Ok(config)
}

fn main() {
    match load_config("config.toml") {
        Ok(config) => println!("設定を読み込みました"),
        Err(e) => {
            // エラーチェーンを全て表示
            eprintln!("エラー: {:#}", e);
            std::process::exit(1);
        }
    }
}
```

---

## 6. エラーチェーンの図解

```
┌────────────────────────────────────────┐
│ anyhow::Error                          │
│ "設定ファイル 'config.toml' を         │
│  読み込めません"                        │
│                                        │
│  Caused by:                            │
│  ┌──────────────────────────────────┐  │
│  │ std::io::Error                   │  │
│  │ kind: NotFound                   │  │
│  │ "No such file or directory"      │  │
│  └──────────────────────────────────┘  │
└────────────────────────────────────────┘

.with_context() で文脈を追加すると、
エラーチェーンとして階層的にたどれる
```

```
┌────────────────────────────────────────────┐
│  使い分けフローチャート                      │
│                                            │
│  ライブラリ開発？                           │
│    ├── Yes → thiserror で具体的なエラー型   │
│    │         (利用者が match できる)         │
│    └── No                                  │
│         アプリケーション開発？              │
│           ├── Yes → anyhow                 │
│           │         (エラーチェーン重視)     │
│           └── プロトタイプ → anyhow         │
└────────────────────────────────────────────┘
```

---

## 7. 比較表

### 7.1 エラー処理手法の比較

| 手法 | 用途 | 利点 | 欠点 |
|------|------|------|------|
| `match` | 個別パターン処理 | 網羅的、安全 | 冗長 |
| `?` | エラー伝播 | 簡潔 | エラー変換が必要 |
| `unwrap()` | テスト/プロト | 短い | 本番で危険 |
| `expect("msg")` | テスト/不変条件 | メッセージ付き | 本番で危険 |
| `unwrap_or(v)` | デフォルト値 | 安全、簡潔 | 常にデフォルト計算 |
| `unwrap_or_else(f)` | 遅延デフォルト | 安全、効率的 | やや冗長 |

### 7.2 thiserror vs anyhow

| 特性 | thiserror | anyhow |
|------|-----------|--------|
| 目的 | ライブラリのエラー型定義 | アプリのエラー処理 |
| エラー型 | 具体的な enum | anyhow::Error (型消去) |
| パターンマッチ | 可能 | downcast が必要 |
| エラーチェーン | 手動で source 実装 | 自動 (context) |
| From 実装 | #[from] で自動 | 暗黙の型変換 |
| 推奨場面 | 公開API、ライブラリ | バイナリ、CLI、サーバー |

---

## 8. アンチパターン

### アンチパターン1: unwrap の乱用

```rust
// BAD: 本番コードで unwrap
fn get_port() -> u16 {
    std::env::var("PORT").unwrap().parse().unwrap()
}

// GOOD: 適切なエラー処理
fn get_port_good() -> Result<u16, anyhow::Error> {
    let port = std::env::var("PORT")
        .context("PORT 環境変数が設定されていません")?
        .parse()
        .context("PORT の値が数値ではありません")?;
    Ok(port)
}
```

### アンチパターン2: エラーの握りつぶし

```rust
// BAD: エラーを無視
fn save_data(data: &str) {
    let _ = std::fs::write("data.txt", data);  // エラーを捨てている！
}

// GOOD: エラーを適切に伝播
fn save_data_good(data: &str) -> Result<(), std::io::Error> {
    std::fs::write("data.txt", data)?;
    Ok(())
}
```

### アンチパターン3: 過度に広いエラー型

```rust
// BAD: Box<dyn Error> を安易に使う（型情報が失われる）
fn do_something() -> Result<(), Box<dyn std::error::Error>> {
    // 何のエラーが返るか呼び出し元にわからない
    Ok(())
}

// GOOD: 具体的なエラー型を定義
#[derive(Debug, thiserror::Error)]
enum MyError {
    #[error("IO error")]
    Io(#[from] std::io::Error),
    #[error("parse error")]
    Parse(#[from] std::num::ParseIntError),
}

fn do_something_good() -> Result<(), MyError> {
    Ok(())
}
```

---

## 9. FAQ

### Q1: panic! はいつ使うべきですか？

**A:** 以下のケースのみです:
- **プログラムの不変条件が破れた場合** (バグ)
- **テストコード** (assert!, unwrap)
- **プロトタイプ段階** (後で適切なエラー処理に置き換え)
- **回復不可能な初期化エラー** (main の最初期のみ)

本番コードのビジネスロジック内では原則として Result を使用してください。

### Q2: `?` 演算子は main 関数で使えますか？

**A:** はい。main の戻り値型を `Result` にすれば使えます:

```rust
fn main() -> Result<(), anyhow::Error> {
    let config = load_config("config.toml")?;
    run_server(config)?;
    Ok(())
}
```

### Q3: Option と Result を相互変換するには？

**A:**
```rust
// Option → Result
let opt: Option<i32> = Some(42);
let res: Result<i32, &str> = opt.ok_or("値がありません");

// Result → Option
let res: Result<i32, String> = Ok(42);
let opt: Option<i32> = res.ok();  // Err は None になる
```

---

## 10. まとめ

| 概念 | 要点 |
|------|------|
| Option<T> | 値の有無を型で表現。None は正常な不在 |
| Result<T, E> | 成功/失敗を型で表現。全エラーパスを明示 |
| ? 演算子 | Err/None を早期リターンする構文糖衣 |
| From トレイト | ? でのエラー型自動変換の仕組み |
| thiserror | ライブラリ向け。カスタムエラー型の derive |
| anyhow | アプリ向け。エラーチェーンと context |
| panic! | 回復不能なバグのみ。本番ロジックでは使わない |

---

## 次に読むべきガイド

- [04-collections-iterators.md](04-collections-iterators.md) -- コレクションとイテレータ
- [../01-advanced/02-closures-fn-traits.md](../01-advanced/02-closures-fn-traits.md) -- クロージャと Fn トレイト
- [../04-ecosystem/04-best-practices.md](../04-ecosystem/04-best-practices.md) -- エラー設計のベストプラクティス

---

## 参考文献

1. **The Rust Programming Language - Ch.9 Error Handling** -- https://doc.rust-lang.org/book/ch09-00-error-handling.html
2. **thiserror ドキュメント** -- https://docs.rs/thiserror/
3. **anyhow ドキュメント** -- https://docs.rs/anyhow/
4. **Rust Error Handling Best Practices (Andrew Gallant)** -- https://blog.burntsushi.net/rust-error-handling/
