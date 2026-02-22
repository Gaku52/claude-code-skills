# エラーハンドリング -- Rustの型安全なエラー処理パターン

> Rustは例外機構を持たず、Result<T, E> と Option<T> を用いた明示的なエラー処理により、全てのエラーパスをコンパイル時に検証する。

---

## この章で学ぶこと

1. **Result と Option** -- 失敗可能性を型で表現し、パターンマッチで安全に処理する方法を理解する
2. **? 演算子とエラー伝播** -- ボイラープレートを減らす構文糖衣と変換の仕組みを習得する
3. **カスタムエラー型** -- 独自のエラー型を定義し、From トレイトで変換を自動化する方法を学ぶ
4. **thiserror / anyhow** -- 実務で使われるエラー処理クレートの使い分けを学ぶ
5. **実践的なエラー設計** -- ライブラリとアプリケーションでのエラー設計パターンを身につける

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

Rustのエラー処理哲学は「全てのエラーを型で表現する」というものである。多くの言語が例外機構（try/catch）を採用しているのに対し、Rustは意図的に例外を排除し、戻り値としてエラーを返す方式を採用した。この設計の利点は:

1. **明示性**: 関数のシグネチャを見るだけでエラーが発生しうるかがわかる
2. **網羅性**: コンパイラがエラー処理の漏れを検出する
3. **パフォーマンス**: 例外のスタック巻き戻しコストがない
4. **合成可能性**: `?` 演算子やコンビネータでエラー処理を簡潔に連鎖できる

### 1.1 panic! と回復不能エラー

```rust
fn main() {
    // 明示的なパニック
    // panic!("致命的なエラー！");

    // 暗黙的なパニック（境界外アクセス）
    let v = vec![1, 2, 3];
    // let x = v[10]; // パニック: index out of bounds

    // パニック時のバックトレースを有効にするには:
    // RUST_BACKTRACE=1 cargo run

    // unwrap / expect もパニックを引き起こす
    let result: Result<i32, &str> = Err("エラー");
    // result.unwrap();  // パニック
    // result.expect("カスタムメッセージ");  // パニック（メッセージ付き）
}
```

パニックはプログラムの不変条件が破られた場合（バグ）に使うものであり、通常のエラーハンドリングには `Result` を使用すべきである。

### 1.2 パニックの伝播と catch_unwind

```rust
use std::panic;

fn risky_operation() {
    panic!("何かがおかしい！");
}

fn main() {
    // catch_unwind でパニックをキャッチ（FFI境界などで使用）
    let result = panic::catch_unwind(|| {
        risky_operation();
    });

    match result {
        Ok(()) => println!("正常終了"),
        Err(_) => println!("パニックが発生しましたが、回復しました"),
    }

    println!("プログラムは続行中...");

    // 注意: catch_unwind は一般的なエラーハンドリングには使わない
    // FFI 境界やスレッドプール内での使用が主な用途
}
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

    // let-else（Rust 2021+）
    let Some(name) = find_user(1) else {
        println!("見つかりません");
        return;
    };
    println!("見つかった: {}", name);

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

    // and_then: ネストした Option をフラットに（flatMap に相当）
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

    // or / or_else: 最初の Some を返す
    let a: Option<i32> = None;
    let b: Option<i32> = Some(42);
    let c: Option<i32> = Some(100);
    println!("{:?}", a.or(b));        // Some(42)
    println!("{:?}", b.or(c));        // Some(42) -- 最初の Some

    // zip: 2つの Option を結合
    let x = Some(1);
    let y = Some("hello");
    let z: Option<i32> = None;
    println!("{:?}", x.zip(y));       // Some((1, "hello"))
    println!("{:?}", x.zip(z));       // None

    // flatten: Option<Option<T>> → Option<T>
    let nested: Option<Option<i32>> = Some(Some(42));
    println!("{:?}", nested.flatten()); // Some(42)

    // transpose: Option<Result<T, E>> ↔ Result<Option<T>, E>
    let opt_result: Option<Result<i32, String>> = Some(Ok(42));
    let result_opt: Result<Option<i32>, String> = opt_result.transpose();
    println!("{:?}", result_opt); // Ok(Some(42))
}
```

### Option の連鎖パターン

```rust
#[derive(Debug)]
struct Config {
    database: Option<DatabaseConfig>,
}

#[derive(Debug)]
struct DatabaseConfig {
    host: Option<String>,
    port: Option<u16>,
}

fn get_db_url(config: &Config) -> Option<String> {
    // Option のチェーン: 各段階で None なら早期に None を返す
    let db = config.database.as_ref()?;
    let host = db.host.as_ref()?;
    let port = db.port?;
    Some(format!("postgres://{}:{}/mydb", host, port))
}

fn main() {
    let config = Config {
        database: Some(DatabaseConfig {
            host: Some("localhost".to_string()),
            port: Some(5432),
        }),
    };

    match get_db_url(&config) {
        Some(url) => println!("DB URL: {}", url),
        None => println!("データベース設定が不完全です"),
    }

    // host が None の場合
    let incomplete_config = Config {
        database: Some(DatabaseConfig {
            host: None,
            port: Some(5432),
        }),
    };
    println!("不完全: {:?}", get_db_url(&incomplete_config)); // None
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

### 例5: Result のコンビネータ

```rust
use std::num::ParseIntError;

fn parse_and_double(s: &str) -> Result<i32, ParseIntError> {
    s.parse::<i32>().map(|n| n * 2)
}

fn main() {
    // map: Ok の中の値を変換
    let result = "21".parse::<i32>().map(|n| n * 2);
    println!("{:?}", result); // Ok(42)

    // map_err: Err の中の値を変換
    let result = "abc".parse::<i32>()
        .map_err(|e| format!("パースエラー: {}", e));
    println!("{:?}", result); // Err("パースエラー: ...")

    // and_then: Result の連鎖
    let result = "42".parse::<i32>()
        .and_then(|n| {
            if n > 0 {
                Ok(n)
            } else {
                Err("0以下".parse::<i32>().unwrap_err())
            }
        });
    println!("{:?}", result);

    // unwrap_or / unwrap_or_else
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8080);
    println!("ポート: {}", port);

    // 複数の Result を collect
    let strings = vec!["1", "2", "3", "4", "5"];
    let numbers: Result<Vec<i32>, _> = strings
        .iter()
        .map(|s| s.parse::<i32>())
        .collect();
    println!("{:?}", numbers); // Ok([1, 2, 3, 4, 5])

    // エラーが含まれる場合
    let mixed = vec!["1", "abc", "3"];
    let result: Result<Vec<i32>, _> = mixed
        .iter()
        .map(|s| s.parse::<i32>())
        .collect();
    println!("{:?}", result); // Err(ParseIntError)
}
```

### 例6: 複数のエラー型を扱う

```rust
use std::io;
use std::num::ParseIntError;

// 方法1: Box<dyn Error>
fn process_file_boxed(path: &str) -> Result<i32, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;  // io::Error
    let number: i32 = content.trim().parse()?;       // ParseIntError
    Ok(number * 2)
}

// 方法2: カスタムエラー enum
#[derive(Debug)]
enum ProcessError {
    Io(io::Error),
    Parse(ParseIntError),
}

impl std::fmt::Display for ProcessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessError::Io(e) => write!(f, "IOエラー: {}", e),
            ProcessError::Parse(e) => write!(f, "パースエラー: {}", e),
        }
    }
}

impl std::error::Error for ProcessError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ProcessError::Io(e) => Some(e),
            ProcessError::Parse(e) => Some(e),
        }
    }
}

impl From<io::Error> for ProcessError {
    fn from(e: io::Error) -> Self {
        ProcessError::Io(e)
    }
}

impl From<ParseIntError> for ProcessError {
    fn from(e: ParseIntError) -> Self {
        ProcessError::Parse(e)
    }
}

fn process_file(path: &str) -> Result<i32, ProcessError> {
    let content = std::fs::read_to_string(path)?; // io::Error → ProcessError
    let number: i32 = content.trim().parse()?;      // ParseIntError → ProcessError
    Ok(number * 2)
}

fn main() {
    match process_file("number.txt") {
        Ok(n) => println!("結果: {}", n),
        Err(ProcessError::Io(e)) => eprintln!("ファイルエラー: {}", e),
        Err(ProcessError::Parse(e)) => eprintln!("数値変換エラー: {}", e),
    }
}
```

---

## 4. カスタムエラー型

### 例7: 手動でカスタムエラーを定義

```rust
use std::fmt;
use std::num::ParseIntError;

#[derive(Debug)]
enum AppError {
    IoError(std::io::Error),
    ParseError(ParseIntError),
    ValidationError(String),
    NotFoundError { resource: String, id: u64 },
    AuthError { user: String, reason: String },
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::IoError(e) => write!(f, "IOエラー: {}", e),
            AppError::ParseError(e) => write!(f, "パースエラー: {}", e),
            AppError::ValidationError(msg) => write!(f, "検証エラー: {}", msg),
            AppError::NotFoundError { resource, id } => {
                write!(f, "{} (ID={}) が見つかりません", resource, id)
            }
            AppError::AuthError { user, reason } => {
                write!(f, "認証エラー (ユーザー: {}): {}", user, reason)
            }
        }
    }
}

impl std::error::Error for AppError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            AppError::IoError(e) => Some(e),
            AppError::ParseError(e) => Some(e),
            _ => None,
        }
    }
}

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

fn find_user(id: u64) -> Result<String, AppError> {
    if id == 0 {
        return Err(AppError::NotFoundError {
            resource: "ユーザー".to_string(),
            id,
        });
    }
    Ok(format!("user_{}", id))
}
```

---

## 5. thiserror と anyhow

### 例8: thiserror(ライブラリ向け)

```rust
use thiserror::Error;

#[derive(Debug, Error)]
enum DatabaseError {
    #[error("接続エラー: {0}")]
    ConnectionFailed(String),

    #[error("クエリエラー: {query}")]
    QueryFailed {
        query: String,
        #[source]
        source: std::io::Error,
    },

    #[error("レコードが見つかりません: ID={id}")]
    NotFound { id: u64 },

    #[error("認証エラー: ユーザー '{user}' のアクセスが拒否されました")]
    AuthFailed { user: String },

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Parse(#[from] std::num::ParseIntError),
}

// thiserror の利点:
// 1. Display の自動実装（#[error("...")] アトリビュート）
// 2. From の自動実装（#[from] アトリビュート）
// 3. source() の自動実装（#[source] アトリビュート）
// 4. ボイラープレートの大幅削減

fn connect_db(url: &str) -> Result<(), DatabaseError> {
    if url.is_empty() {
        return Err(DatabaseError::ConnectionFailed(
            "URLが空です".to_string(),
        ));
    }
    Ok(())
}

fn find_record(id: u64) -> Result<String, DatabaseError> {
    if id == 0 {
        return Err(DatabaseError::NotFound { id });
    }
    Ok(format!("record_{}", id))
}
```

### 例9: anyhow(アプリケーション向け)

```rust
use anyhow::{Context, Result, bail, ensure, anyhow};

#[derive(Debug)]
struct Config {
    host: String,
    port: u16,
    database: String,
}

fn load_config(path: &str) -> Result<Config> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("設定ファイル '{}' を読み込めません", path))?;

    let lines: Vec<&str> = content.lines().collect();

    ensure!(lines.len() >= 3, "設定ファイルには少なくとも3行必要です");

    let host = lines[0].trim().to_string();
    let port: u16 = lines[1].trim().parse()
        .context("ポート番号のパースに失敗")?;
    let database = lines[2].trim().to_string();

    if host.is_empty() {
        bail!("ホストが指定されていません");
    }

    if port == 0 {
        return Err(anyhow!("ポート0は無効です"));
    }

    Ok(Config { host, port, database })
}

fn run_server(config: &Config) -> Result<()> {
    println!("サーバー起動: {}:{}/{}", config.host, config.port, config.database);
    Ok(())
}

fn main() {
    match load_config("config.txt") {
        Ok(config) => {
            if let Err(e) = run_server(&config) {
                // エラーチェーンを全て表示
                eprintln!("エラー: {:#}", e);
                // エラーチェーンを個別に表示
                for cause in e.chain() {
                    eprintln!("  原因: {}", cause);
                }
                std::process::exit(1);
            }
        }
        Err(e) => {
            eprintln!("設定読み込みエラー: {:#}", e);
            std::process::exit(1);
        }
    }
}
```

### 例10: anyhow と thiserror の組み合わせ

```rust
// ライブラリ層: thiserror で具体的なエラー型を定義
mod db {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum DbError {
        #[error("接続エラー: {0}")]
        Connection(String),
        #[error("クエリエラー: {0}")]
        Query(String),
        #[error("レコード未発見: {0}")]
        NotFound(u64),
    }

    pub fn find_user(id: u64) -> Result<String, DbError> {
        match id {
            0 => Err(DbError::NotFound(id)),
            _ => Ok(format!("user_{}", id)),
        }
    }
}

// アプリケーション層: anyhow でエラーを集約
mod app {
    use anyhow::{Context, Result};
    use super::db;

    pub fn get_user_name(id: u64) -> Result<String> {
        let user = db::find_user(id)
            .with_context(|| format!("ユーザーID {} の取得に失敗", id))?;
        Ok(user)
    }
}

fn main() {
    match app::get_user_name(0) {
        Ok(name) => println!("ユーザー: {}", name),
        Err(e) => {
            eprintln!("エラー: {:#}", e);
            // anyhow のエラーチェーンが表示される:
            // エラー: ユーザーID 0 の取得に失敗: レコード未発見: 0

            // ダウンキャストで具体的なエラー型を取得
            if let Some(db_err) = e.downcast_ref::<db::DbError>() {
                match db_err {
                    db::DbError::NotFound(id) => {
                        eprintln!("ヒント: ID {} は存在しません", id);
                    }
                    _ => {}
                }
            }
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
│                                            │
│  ハイブリッドアプローチ:                    │
│    ライブラリ層 → thiserror                 │
│    アプリ層 → anyhow (thiserror を包む)     │
└────────────────────────────────────────────┘
```

---

## 7. 実践的なエラーハンドリングパターン

### 7.1 関数内でのエラー変換

```rust
use std::io;
use std::num::ParseIntError;

// エラー変換の様々なパターン
fn demo_error_conversion() -> Result<(), Box<dyn std::error::Error>> {
    // map_err: エラー型を変換
    let _port: u16 = "8080".parse()
        .map_err(|e: ParseIntError| io::Error::new(io::ErrorKind::InvalidData, e))?;

    // From トレイトによる自動変換（? 演算子が使う）
    // ? は Err(e) を Err(From::from(e)) に変換する

    // ok_or: Option → Result 変換
    let env_var = std::env::var("HOME").ok();
    let home = env_var.ok_or_else(|| io::Error::new(
        io::ErrorKind::NotFound,
        "HOME環境変数が設定されていません"
    ))?;
    println!("HOME: {}", home);

    Ok(())
}
```

### 7.2 エラーのログとリカバリ

```rust
fn process_items(items: &[&str]) -> Vec<i32> {
    items.iter()
        .filter_map(|item| {
            match item.parse::<i32>() {
                Ok(n) => Some(n),
                Err(e) => {
                    eprintln!("警告: '{}' をパースできません: {}", item, e);
                    None  // エラーをスキップして続行
                }
            }
        })
        .collect()
}

fn process_with_defaults(items: &[&str]) -> Vec<i32> {
    items.iter()
        .map(|item| {
            item.parse::<i32>().unwrap_or_else(|_| {
                eprintln!("'{}' をデフォルト値 0 に置換", item);
                0
            })
        })
        .collect()
}

fn main() {
    let items = vec!["1", "abc", "3", "def", "5"];

    let filtered = process_items(&items);
    println!("フィルタ結果: {:?}", filtered); // [1, 3, 5]

    let defaulted = process_with_defaults(&items);
    println!("デフォルト結果: {:?}", defaulted); // [1, 0, 3, 0, 5]
}
```

### 7.3 リトライパターン

```rust
use std::time::Duration;
use std::thread;

fn unreliable_operation() -> Result<String, String> {
    use std::time::SystemTime;
    let secs = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();

    if secs % 3 == 0 {
        Ok("成功！".to_string())
    } else {
        Err("一時的なエラー".to_string())
    }
}

fn retry<F, T, E>(mut operation: F, max_retries: u32, delay: Duration) -> Result<T, E>
where
    F: FnMut() -> Result<T, E>,
    E: std::fmt::Display,
{
    let mut last_err = None;
    for attempt in 1..=max_retries {
        match operation() {
            Ok(value) => return Ok(value),
            Err(e) => {
                eprintln!("試行 {}/{}: {}", attempt, max_retries, e);
                last_err = Some(e);
                if attempt < max_retries {
                    thread::sleep(delay);
                }
            }
        }
    }
    Err(last_err.unwrap())
}

fn main() {
    match retry(unreliable_operation, 5, Duration::from_millis(100)) {
        Ok(result) => println!("結果: {}", result),
        Err(e) => eprintln!("全ての試行が失敗: {}", e),
    }
}
```

### 7.4 エラーの集約

```rust
fn validate_user_input(
    name: &str,
    email: &str,
    age: &str,
) -> Result<(String, String, u32), Vec<String>> {
    let mut errors = Vec::new();

    if name.is_empty() {
        errors.push("名前は必須です".to_string());
    }

    if !email.contains('@') {
        errors.push("メールアドレスの形式が不正です".to_string());
    }

    let age_result = age.parse::<u32>();
    if age_result.is_err() {
        errors.push("年齢は数値で入力してください".to_string());
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    Ok((
        name.to_string(),
        email.to_string(),
        age_result.unwrap(),
    ))
}

fn main() {
    match validate_user_input("", "invalid-email", "abc") {
        Ok((name, email, age)) => {
            println!("有効: {} / {} / {}歳", name, email, age);
        }
        Err(errors) => {
            eprintln!("入力エラー:");
            for error in &errors {
                eprintln!("  - {}", error);
            }
        }
    }
}
```

---

## 8. 比較表

### 8.1 エラー処理手法の比較

| 手法 | 用途 | 利点 | 欠点 |
|------|------|------|------|
| `match` | 個別パターン処理 | 網羅的、安全 | 冗長 |
| `?` | エラー伝播 | 簡潔 | エラー変換が必要 |
| `unwrap()` | テスト/プロト | 短い | 本番で危険 |
| `expect("msg")` | テスト/不変条件 | メッセージ付き | 本番で危険 |
| `unwrap_or(v)` | デフォルト値 | 安全、簡潔 | 常にデフォルト計算 |
| `unwrap_or_else(f)` | 遅延デフォルト | 安全、効率的 | やや冗長 |
| `unwrap_or_default()` | Default実装型 | 非常に簡潔 | Default必要 |
| `if let` | 特定パターンのみ | 簡潔 | Err/None 処理なし |
| `let-else` | 早期リターン | 読みやすい | Rust 2021+ |
| `map` / `and_then` | 変換チェーン | 関数型スタイル | 慣れが必要 |

### 8.2 thiserror vs anyhow

| 特性 | thiserror | anyhow |
|------|-----------|--------|
| 目的 | ライブラリのエラー型定義 | アプリのエラー処理 |
| エラー型 | 具体的な enum | anyhow::Error (型消去) |
| パターンマッチ | 可能 | downcast が必要 |
| エラーチェーン | 手動で source 実装 | 自動 (context) |
| From 実装 | #[from] で自動 | 暗黙の型変換 |
| 推奨場面 | 公開API、ライブラリ | バイナリ、CLI、サーバー |
| コードサイズ | やや多い | 少ない |
| 依存クレート数 | 少ない（proc-macro） | 少ない |

### 8.3 エラー型の選択ガイド

| 場面 | 推奨エラー型 | 理由 |
|------|-------------|------|
| ライブラリの公開API | thiserror enum | 利用者がパターンマッチ可能 |
| CLI アプリ | anyhow::Error | エラーメッセージが重要 |
| Web サーバー | thiserror + anyhow | レスポンスコードのマッピング |
| プロトタイプ | anyhow / Box<dyn Error> | 素早い開発 |
| 内部モジュール | thiserror enum | 型安全なエラー処理 |
| テストコード | unwrap / expect | 失敗時のスタックトレース |

---

## 9. アンチパターン

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

// GOOD: エラーを明示的にログして続行
fn save_data_with_logging(data: &str) {
    if let Err(e) = std::fs::write("data.txt", data) {
        eprintln!("警告: データの保存に失敗しました: {}", e);
        // 重要でない場合は続行
    }
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
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse error: {0}")]
    Parse(#[from] std::num::ParseIntError),
}

fn do_something_good() -> Result<(), MyError> {
    Ok(())
}
```

### アンチパターン4: panic! をエラーハンドリングに使う

```rust
// BAD: パニックで「エラーハンドリング」
fn parse_config(s: &str) -> Config {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        panic!("不正な設定形式");  // パニックはバグ検出用！
    }
    Config {
        key: parts[0].to_string(),
        value: parts[1].to_string(),
    }
}

// GOOD: Result で返す
fn parse_config_good(s: &str) -> Result<Config, String> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        return Err(format!("不正な設定形式: '{}'", s));
    }
    Ok(Config {
        key: parts[0].to_string(),
        value: parts[1].to_string(),
    })
}

struct Config {
    key: String,
    value: String,
}
```

### アンチパターン5: エラーメッセージの情報不足

```rust
// BAD: 曖昧なエラーメッセージ
fn load_user(id: u64) -> Result<String, String> {
    Err("エラー".to_string())  // 何のエラー？どこで？
}

// GOOD: 文脈付きのエラーメッセージ
fn load_user_good(id: u64) -> Result<String, anyhow::Error> {
    let path = format!("/data/users/{}.json", id);
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("ユーザーID {} のファイル '{}' を読み込めません", id, path))?;
    let user: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("ユーザーID {} のJSONパースに失敗", id))?;
    Ok(user["name"].as_str().unwrap_or("不明").to_string())
}
```

---

## 10. FAQ

### Q1: panic! はいつ使うべきですか？

**A:** 以下のケースのみです:
- **プログラムの不変条件が破れた場合** (バグ)
- **テストコード** (assert!, unwrap)
- **プロトタイプ段階** (後で適切なエラー処理に置き換え)
- **回復不可能な初期化エラー** (main の最初期のみ)
- **assert! / debug_assert!** による契約プログラミング

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

`std::process::ExitCode` を使えばより細かい終了コード制御も可能です:

```rust
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("エラー: {:#}", e);
            std::process::ExitCode::FAILURE
        }
    }
}

fn run() -> anyhow::Result<()> {
    // ここで ? が使える
    Ok(())
}
```

### Q3: Option と Result を相互変換するには？

**A:**
```rust
// Option → Result
let opt: Option<i32> = Some(42);
let res: Result<i32, &str> = opt.ok_or("値がありません");

// Option → Result (遅延評価)
let res2: Result<i32, String> = opt.ok_or_else(|| format!("値が見つかりません"));

// Result → Option (Ok → Some, Err → None)
let res: Result<i32, String> = Ok(42);
let opt: Option<i32> = res.ok();  // Err は None になる

// Result → Option (Ok → None, Err → Some)
let res: Result<i32, String> = Err("error".to_string());
let opt: Option<String> = res.err();  // Ok は None になる
```

### Q4: `expect` と `unwrap` はどう使い分けますか？

**A:** `expect` は `unwrap` の上位互換です。パニック時にカスタムメッセージを表示できるため、デバッグが容易になります。プロトタイプ段階でも `expect` を使うことを推奨します。

```rust
// unwrap: "called `Result::unwrap()` on an `Err` value: ..." というメッセージ
let file = File::open("config.toml").unwrap();

// expect: カスタムメッセージでなぜこの操作が成功すべきかを説明
let file = File::open("config.toml")
    .expect("config.toml はプロジェクトルートに存在するはず");
```

### Q5: `Box<dyn Error>` と `anyhow::Error` の違いは？

**A:**
- `Box<dyn Error>`: 標準ライブラリのみで使える型消去されたエラー型。最小限の機能
- `anyhow::Error`: `context()` によるエラーチェーン、`downcast()` による元の型の復元、`{:#}` による詳細表示など、豊富な機能を提供

実務では `anyhow::Error` の方が圧倒的に便利です。ただし、ライブラリの公開APIには使わないでください。

### Q6: エラー型を設計する際のベストプラクティスは？

**A:**
1. **ライブラリ**: `thiserror` で enum を定義。利用者が `match` で分岐できるようにする
2. **アプリケーション**: `anyhow` で `context` を活用。エラーメッセージの質を重視
3. **エラーメッセージ**: 「何が起こったか」「何をしようとしていたか」「どう対処すべきか」を含める
4. **エラーの粒度**: 呼び出し元が異なる処理をする必要がある場合のみバリアントを分ける

---

## 11. std::error::Error トレイトの詳細

### 11.1 Error トレイトの定義

```rust
// std::error::Error の定義（簡略版）
pub trait Error: Debug + Display {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}
```

`Error` トレイトは `Debug` と `Display` の両方をスーパートレイトとして要求する。これにより、エラー型は常に人間が読める形式（`Display`）と開発者向けの詳細形式（`Debug`）の両方で表示できる。

### 11.2 エラーチェーンの走査

```rust
use std::error::Error;
use std::fmt;

// エラーチェーンを全て表示するヘルパー関数
fn print_error_chain(err: &dyn Error) {
    eprintln!("エラー: {}", err);
    let mut current = err.source();
    let mut depth = 1;
    while let Some(cause) = current {
        eprintln!("  {}. 原因: {}", depth, cause);
        current = cause.source();
        depth += 1;
    }
}

// カスタムエラー型でチェーンを構成
#[derive(Debug)]
struct ServiceError {
    message: String,
    source: Option<Box<dyn Error + Send + Sync>>,
}

impl fmt::Display for ServiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "サービスエラー: {}", self.message)
    }
}

impl Error for ServiceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_ref().map(|e| e.as_ref() as &(dyn Error + 'static))
    }
}

impl ServiceError {
    fn new(message: impl Into<String>) -> Self {
        Self { message: message.into(), source: None }
    }

    fn with_source(message: impl Into<String>, source: impl Error + Send + Sync + 'static) -> Self {
        Self {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }
}

fn connect_database() -> Result<(), ServiceError> {
    let io_err = std::io::Error::new(
        std::io::ErrorKind::ConnectionRefused,
        "ポート5432への接続が拒否されました"
    );
    Err(ServiceError::with_source("データベース接続に失敗", io_err))
}

fn main() {
    if let Err(e) = connect_database() {
        print_error_chain(&e);
        // 出力:
        // エラー: サービスエラー: データベース接続に失敗
        //   1. 原因: ポート5432への接続が拒否されました
    }
}
```

### 11.3 Send + Sync とエラー型

マルチスレッド環境でエラーを安全に扱うためには `Send + Sync` バウンドが重要である。

```rust
use std::error::Error;

// スレッド安全なエラー型
type BoxError = Box<dyn Error + Send + Sync + 'static>;

// スレッド間でエラーを送信する例
fn spawn_worker() -> Result<String, BoxError> {
    let handle = std::thread::spawn(|| -> Result<String, BoxError> {
        let content = std::fs::read_to_string("data.txt")?;
        let number: i32 = content.trim().parse()?;
        Ok(format!("結果: {}", number * 2))
    });

    match handle.join() {
        Ok(result) => result,
        Err(_) => Err("ワーカースレッドがパニックしました".into()),
    }
}

fn main() {
    match spawn_worker() {
        Ok(value) => println!("{}", value),
        Err(e) => eprintln!("ワーカーエラー: {}", e),
    }
}
```

### 11.4 ダウンキャストによるエラー型の復元

```rust
use std::error::Error;

fn might_fail() -> Result<(), Box<dyn Error>> {
    let result: Result<i32, _> = "abc".parse();
    result?;
    Ok(())
}

fn main() {
    if let Err(e) = might_fail() {
        // ダウンキャスト: Box<dyn Error> → 具体的な型
        if let Some(parse_err) = e.downcast_ref::<std::num::ParseIntError>() {
            eprintln!("パースエラーを検出: {}", parse_err);
        } else if let Some(io_err) = e.downcast_ref::<std::io::Error>() {
            eprintln!("IOエラーを検出: {}", io_err);
        } else {
            eprintln!("不明なエラー: {}", e);
        }

        // downcast で所有権を取得することも可能
        // let concrete: Box<std::num::ParseIntError> = e.downcast().unwrap();
    }
}
```

---

## 12. 実務で使われるエラーハンドリング設計パターン

### 12.1 レイヤードアーキテクチャでのエラー設計

```rust
// === インフラ層 ===
mod infra {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum InfraError {
        #[error("DB接続エラー: {0}")]
        Database(String),
        #[error("ネットワークエラー: {0}")]
        Network(String),
        #[error("ファイルシステムエラー: {0}")]
        FileSystem(#[from] std::io::Error),
    }
}

// === ドメイン層 ===
mod domain {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum DomainError {
        #[error("ユーザーが見つかりません: ID={0}")]
        UserNotFound(u64),
        #[error("残高不足: 必要額={required}, 現在額={current}")]
        InsufficientBalance { required: u64, current: u64 },
        #[error("不正な操作: {0}")]
        InvalidOperation(String),
        #[error("インフラエラー")]
        Infrastructure(#[from] super::infra::InfraError),
    }
}

// === アプリケーション層 ===
mod application {
    use anyhow::{Context, Result};
    use super::domain::DomainError;

    pub fn transfer_money(from: u64, to: u64, amount: u64) -> Result<()> {
        // ドメインエラーは anyhow でラップされ、コンテキストが付与される
        let _from_user = find_user(from)
            .with_context(|| format!("送金元ユーザー {} の取得に失敗", from))?;
        let _to_user = find_user(to)
            .with_context(|| format!("送金先ユーザー {} の取得に失敗", to))?;

        // ドメインバリデーション
        validate_transfer(amount)
            .context("送金バリデーションに失敗")?;

        println!("送金成功: {} → {} ({}円)", from, to, amount);
        Ok(())
    }

    fn find_user(id: u64) -> Result<String, DomainError> {
        if id == 0 {
            Err(DomainError::UserNotFound(id))
        } else {
            Ok(format!("user_{}", id))
        }
    }

    fn validate_transfer(amount: u64) -> Result<(), DomainError> {
        if amount == 0 {
            Err(DomainError::InvalidOperation("送金額は0より大きくなければなりません".into()))
        } else {
            Ok(())
        }
    }
}
```

### 12.2 HTTP API でのエラーマッピング

```rust
use thiserror::Error;

#[derive(Debug, Error)]
enum ApiError {
    #[error("リソースが見つかりません: {0}")]
    NotFound(String),
    #[error("認証エラー: {0}")]
    Unauthorized(String),
    #[error("バリデーションエラー: {0}")]
    BadRequest(String),
    #[error("内部エラー")]
    Internal(#[source] anyhow::Error),
}

impl ApiError {
    fn status_code(&self) -> u16 {
        match self {
            ApiError::NotFound(_) => 404,
            ApiError::Unauthorized(_) => 401,
            ApiError::BadRequest(_) => 400,
            ApiError::Internal(_) => 500,
        }
    }

    fn to_json(&self) -> String {
        format!(
            r#"{{"error": {{"code": {}, "message": "{}"}}}}"#,
            self.status_code(),
            self
        )
    }
}

fn handle_request(path: &str) -> Result<String, ApiError> {
    match path {
        "/users/1" => Ok(r#"{"id": 1, "name": "田中"}"#.to_string()),
        "/users/0" => Err(ApiError::NotFound("ユーザーID 0".to_string())),
        "/admin" => Err(ApiError::Unauthorized("管理者権限が必要です".to_string())),
        _ => Err(ApiError::NotFound(format!("パス '{}'", path))),
    }
}

fn main() {
    let paths = vec!["/users/1", "/users/0", "/admin", "/unknown"];
    for path in paths {
        match handle_request(path) {
            Ok(body) => println!("200 OK: {}", body),
            Err(e) => println!("{} Error: {}", e.status_code(), e.to_json()),
        }
    }
}
```

---

## 13. まとめ

| 概念 | 要点 |
|------|------|
| Option<T> | 値の有無を型で表現。None は正常な不在 |
| Result<T, E> | 成功/失敗を型で表現。全エラーパスを明示 |
| ? 演算子 | Err/None を早期リターンする構文糖衣 |
| From トレイト | ? でのエラー型自動変換の仕組み |
| thiserror | ライブラリ向け。カスタムエラー型の derive |
| anyhow | アプリ向け。エラーチェーンと context |
| panic! | 回復不能なバグのみ。本番ロジックでは使わない |
| コンビネータ | map/and_then/or_else で関数型エラー処理 |
| エラー集約 | Vec<Error> で複数エラーを一括報告 |
| リトライ | Result を返す操作の再試行パターン |
| Error トレイト | Debug + Display。source() でチェーン走査 |
| Send + Sync | マルチスレッドでのエラー転送に必須 |
| ダウンキャスト | Box<dyn Error> から具体型を復元 |
| レイヤード設計 | インフラ→ドメイン→アプリで段階的にエラー変換 |

---

## 次に読むべきガイド

- [04-collections-iterators.md](04-collections-iterators.md) -- コレクションとイテレータ
- [../01-advanced/02-closures-fn-traits.md](../01-advanced/02-closures-fn-traits.md) -- クロージャと Fn トレイト
- [../04-ecosystem/04-best-practices.md](../04-ecosystem/04-best-practices.md) -- エラー設計のベストプラクティス

---

## 14. 参考文献

1. **The Rust Programming Language - Ch.9 Error Handling** -- https://doc.rust-lang.org/book/ch09-00-error-handling.html
2. **thiserror ドキュメント** -- https://docs.rs/thiserror/
3. **anyhow ドキュメント** -- https://docs.rs/anyhow/
4. **Rust Error Handling Best Practices (Andrew Gallant)** -- https://blog.burntsushi.net/rust-error-handling/
5. **The Rust API Guidelines - Error Handling** -- https://rust-lang.github.io/api-guidelines/interoperability.html
6. **Error Handling in Rust (Nick Cameron)** -- https://www.ncameron.org/blog/error-handling-in-rust/
