# CLIツール — clap、クロスコンパイル

> clap による引数解析、カラー出力、プログレスバー、クロスコンパイル/配布まで、実用的な CLI ツール開発の全工程を習得する

## この章で学ぶこと

1. **引数解析** — clap derive API による型安全なCLI定義、サブコマンド、バリデーション
2. **ユーザー体験** — カラー出力、プログレスバー、対話的プロンプト
3. **クロスコンパイルと配布** — cross、GitHub Actions、cargo-dist

---

## 1. CLIツールの構成要素

```
┌────────────────── CLI ツール構成 ──────────────────┐
│                                                     │
│  ┌─ 引数解析 ─────────────────────────────────────┐│
│  │  clap (derive / builder)                       ││
│  │  → コマンド、フラグ、オプション、サブコマンド    ││
│  └────────────────────────────────────────────────┘│
│                                                     │
│  ┌─ 出力・UX ─────────────────────────────────────┐│
│  │  colored / owo-colors  → カラー出力             ││
│  │  indicatif             → プログレスバー          ││
│  │  dialoguer             → 対話的プロンプト        ││
│  │  tabled                → テーブル表示           ││
│  └────────────────────────────────────────────────┘│
│                                                     │
│  ┌─ I/O・設定 ────────────────────────────────────┐│
│  │  serde + toml/json     → 設定ファイル           ││
│  │  directories           → XDG パス               ││
│  │  tracing               → ログ出力               ││
│  └────────────────────────────────────────────────┘│
│                                                     │
│  ┌─ ビルド・配布 ──────────────────────────────────┐│
│  │  cross                 → クロスコンパイル        ││
│  │  cargo-dist            → バイナリ配布           ││
│  │  GitHub Actions        → CI/CD                  ││
│  └────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

---

## 2. clap による引数解析

### コード例1: derive API の基本

```rust
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// ファイル管理ツール — ファイルの検索・変換・分析を行う
#[derive(Parser, Debug)]
#[command(name = "filetool")]
#[command(version, about, long_about = None)]
struct Cli {
    /// 詳細出力を有効にする
    #[arg(short, long, global = true)]
    verbose: bool,

    /// 設定ファイルのパス
    #[arg(short, long, default_value = "~/.config/filetool/config.toml")]
    config: PathBuf,

    /// 出力フォーマット
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Text)]
    format: OutputFormat,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// ファイルを検索する
    Search {
        /// 検索パターン (正規表現)
        pattern: String,

        /// 検索対象ディレクトリ
        #[arg(short, long, default_value = ".")]
        dir: PathBuf,

        /// 最大結果数
        #[arg(short = 'n', long, default_value_t = 100)]
        max_results: usize,

        /// ファイル拡張子フィルタ
        #[arg(short, long)]
        extension: Option<String>,
    },

    /// ファイルを変換する
    Convert {
        /// 入力ファイル
        input: PathBuf,

        /// 出力ファイル
        output: PathBuf,

        /// 変換先フォーマット
        #[arg(short, long)]
        to: String,
    },

    /// ディレクトリを分析する
    Analyze {
        /// 対象パス
        #[arg(default_value = ".")]
        path: PathBuf,

        /// 深度制限
        #[arg(short, long)]
        depth: Option<usize>,
    },
}

#[derive(ValueEnum, Clone, Debug)]
enum OutputFormat {
    Text,
    Json,
    Csv,
}

fn main() {
    let cli = Cli::parse();

    if cli.verbose {
        eprintln!("設定ファイル: {:?}", cli.config);
        eprintln!("出力形式: {:?}", cli.format);
    }

    match &cli.command {
        Commands::Search { pattern, dir, max_results, extension } => {
            println!("検索: '{}' in {:?} (最大{}件)", pattern, dir, max_results);
            if let Some(ext) = extension {
                println!("拡張子フィルタ: .{}", ext);
            }
        }
        Commands::Convert { input, output, to } => {
            println!("変換: {:?} -> {:?} ({}形式)", input, output, to);
        }
        Commands::Analyze { path, depth } => {
            println!("分析: {:?} (深度: {:?})", path, depth);
        }
    }
}

// 使用例:
// $ filetool search "TODO" --dir src/ -n 50 --extension rs
// $ filetool convert input.csv output.json --to json
// $ filetool analyze /var/log --depth 3 --verbose --format json
```

### コード例2: バリデーション

```rust
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
struct Cli {
    /// ポート番号 (1024-65535)
    #[arg(short, long, default_value_t = 8080)]
    #[arg(value_parser = clap::value_parser!(u16).range(1024..=65535))]
    port: u16,

    /// 入力ファイル (存在確認)
    #[arg(value_parser = validate_file_exists)]
    input: PathBuf,

    /// 並行度 (1-256)
    #[arg(short = 'j', long, default_value_t = num_cpus::get())]
    #[arg(value_parser = clap::value_parser!(usize).range(1..=256))]
    jobs: usize,
}

fn validate_file_exists(s: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(s);
    if path.exists() {
        Ok(path)
    } else {
        Err(format!("ファイルが見つかりません: {}", s))
    }
}
```

---

## 3. ユーザー体験の向上

### コード例3: カラー出力とプログレスバー

```rust
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};

fn main() {
    // カラー出力
    println!("{}", "成功!".green().bold());
    println!("{}", "警告: ディスク容量低下".yellow());
    println!("{}", "エラー: ファイル破損".red().bold());
    println!("{}", "情報: 処理開始".blue());

    // 単一プログレスバー
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})"
        ).unwrap()
        .progress_chars("=>-")
    );

    for i in 0..100 {
        pb.set_position(i);
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    pb.finish_with_message("完了!");

    // マルチプログレスバー
    let multi = MultiProgress::new();
    let style = ProgressStyle::with_template(
        "{prefix:.bold} [{bar:30}] {pos}/{len}"
    ).unwrap();

    let pb1 = multi.add(ProgressBar::new(50));
    let pb2 = multi.add(ProgressBar::new(80));
    pb1.set_style(style.clone());
    pb1.set_prefix("ダウンロード");
    pb2.set_style(style);
    pb2.set_prefix("変換      ");

    // 並行処理でバーを更新
    let h1 = std::thread::spawn(move || {
        for i in 0..=50 { pb1.set_position(i); std::thread::sleep(std::time::Duration::from_millis(30)); }
        pb1.finish();
    });
    let h2 = std::thread::spawn(move || {
        for i in 0..=80 { pb2.set_position(i); std::thread::sleep(std::time::Duration::from_millis(20)); }
        pb2.finish();
    });
    h1.join().unwrap();
    h2.join().unwrap();
}
```

### コード例4: 対話的プロンプト

```rust
use dialoguer::{Confirm, Input, Select, MultiSelect, Password};
use console::style;

fn main() -> anyhow::Result<()> {
    // テキスト入力
    let name: String = Input::new()
        .with_prompt("プロジェクト名")
        .default("my-project".into())
        .interact_text()?;

    // 選択
    let frameworks = &["Axum", "Actix-web", "Rocket", "Warp"];
    let selection = Select::new()
        .with_prompt("フレームワークを選択")
        .items(frameworks)
        .default(0)
        .interact()?;
    println!("選択: {}", frameworks[selection]);

    // 複数選択
    let features = &["Database", "Auth", "WebSocket", "GraphQL", "Metrics"];
    let selections = MultiSelect::new()
        .with_prompt("機能を選択 (スペースキーで選択)")
        .items(features)
        .interact()?;
    for &i in &selections {
        println!("  + {}", features[i]);
    }

    // 確認
    if Confirm::new()
        .with_prompt("プロジェクトを作成しますか?")
        .default(true)
        .interact()?
    {
        println!("{} プロジェクト '{}' を作成しました!", style("✓").green(), name);
    }

    Ok(())
}
```

---

## 4. クロスコンパイルと配布

### コード例5: クロスコンパイル設定

```toml
# .cargo/config.toml

# macOS (Apple Silicon)
[target.aarch64-apple-darwin]
# ネイティブの場合は設定不要

# Linux (x86_64, musl で静的リンク)
[target.x86_64-unknown-linux-musl]
linker = "x86_64-linux-musl-gcc"

# Windows (MinGW クロスコンパイル)
[target.x86_64-pc-windows-gnu]
linker = "x86_64-w64-mingw32-gcc"
```

```bash
# cross を使ったクロスコンパイル (Docker ベース)
cargo install cross

# Linux (静的リンク)
cross build --release --target x86_64-unknown-linux-musl

# Windows
cross build --release --target x86_64-pc-windows-gnu

# ARM Linux (Raspberry Pi)
cross build --release --target aarch64-unknown-linux-gnu
```

### CI/CD パイプライン

```
┌───────── GitHub Actions リリースフロー ─────────┐
│                                                   │
│  git push --tags v1.0.0                          │
│    │                                              │
│    ▼                                              │
│  ┌──────────────────────────────────┐            │
│  │  Matrix Build                    │            │
│  │  ┌─ ubuntu   → x86_64-linux    │            │
│  │  ├─ ubuntu   → aarch64-linux   │            │
│  │  ├─ macos    → x86_64-darwin   │            │
│  │  ├─ macos    → aarch64-darwin  │            │
│  │  └─ windows  → x86_64-windows │            │
│  └──────────────────────────────────┘            │
│    │                                              │
│    ▼                                              │
│  ┌──────────────────────────────────┐            │
│  │  GitHub Release                   │            │
│  │  - filetool-linux-x86_64.tar.gz  │            │
│  │  - filetool-linux-aarch64.tar.gz │            │
│  │  - filetool-darwin-x86_64.tar.gz │            │
│  │  - filetool-darwin-aarch64.tar.gz│            │
│  │  - filetool-windows-x86_64.zip   │            │
│  └──────────────────────────────────┘            │
└───────────────────────────────────────────────────┘
```

### コード例6: GitHub Actions ワークフロー

```yaml
# .github/workflows/release.yml
name: Release
on:
  push:
    tags: ['v*']

jobs:
  build:
    strategy:
      matrix:
        include:
          - target: x86_64-unknown-linux-musl
            os: ubuntu-latest
            archive: tar.gz
          - target: aarch64-unknown-linux-musl
            os: ubuntu-latest
            archive: tar.gz
          - target: x86_64-apple-darwin
            os: macos-latest
            archive: tar.gz
          - target: aarch64-apple-darwin
            os: macos-latest
            archive: tar.gz
          - target: x86_64-pc-windows-msvc
            os: windows-latest
            archive: zip
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.target }}
      - name: Build
        run: cargo build --release --target ${{ matrix.target }}
      - name: Upload
        uses: actions/upload-artifact@v4
        with:
          name: filetool-${{ matrix.target }}
          path: target/${{ matrix.target }}/release/filetool*
```

---

## 5. 比較表

### CLI 引数解析ライブラリ比較

| ライブラリ | API スタイル | コンパイル速度 | 機能 | 特徴 |
|---|---|---|---|---|
| clap (derive) | 構造体マクロ | 遅め | 最も豊富 | 業界標準 |
| clap (builder) | ビルダーパターン | 遅め | 最も豊富 | 動的定義 |
| argh | derive | 高速 | 基本的 | Google 製、軽量 |
| pico-args | 手動 | 最速 | 最小限 | 依存ゼロ |
| bpaf | derive + builder | 中程度 | 豊富 | 合成可能 |

### 配布方式比較

| 方式 | 対象 | 利点 | 欠点 |
|---|---|---|---|
| GitHub Release | 全OS | 広い到達性 | 手動ダウンロード |
| cargo install | Rust 開発者 | 最も簡単 | rustc 必要 |
| Homebrew (tap) | macOS/Linux | パッケージ管理 | tap 管理が必要 |
| cargo-dist | 全OS | 自動化 | 設定が必要 |
| Docker イメージ | サーバー | 環境独立 | Docker 必要 |

---

## 6. アンチパターン

### アンチパターン1: エラー出力を stdout に流す

```rust
// NG: エラーメッセージを println! (stdout) で出力
fn bad_main() {
    let result = process_file("input.txt");
    if result.is_err() {
        println!("エラー: ファイルが見つかりません");
        // パイプ時に正常出力と混ざる!
        // $ filetool input.txt | grep pattern  ← エラーも grep される
    }
}

// OK: エラーは stderr、正常出力は stdout
fn good_main() {
    match process_file("input.txt") {
        Ok(output) => print!("{}", output),           // stdout
        Err(e) => {
            eprintln!("エラー: {}", e);                // stderr
            std::process::exit(1);                      // 非ゼロ終了コード
        }
    }
}

fn process_file(_: &str) -> Result<String, String> { Ok("data".into()) }
```

### アンチパターン2: 終了コードの無視

```rust
// NG: 常に exit(0)
fn bad() {
    if let Err(e) = run() {
        eprintln!("{}", e);
        // exit code = 0 (成功扱い) → CI/CD で異常検知できない
    }
}

// OK: 適切な終了コード
fn main() {
    let result = run();
    std::process::exit(match result {
        Ok(_) => 0,     // 成功
        Err(e) => {
            eprintln!("エラー: {:#}", e);
            match e.downcast_ref::<std::io::Error>() {
                Some(io_err) if io_err.kind() == std::io::ErrorKind::NotFound => 2,
                Some(_) => 3,    // I/O エラー
                None => 1,       // その他のエラー
            }
        }
    });
}

fn run() -> anyhow::Result<()> { Ok(()) }
```

---

## FAQ

### Q1: clap の derive API と builder API はどちらを使うべき?

**A:** ほとんどの場合 derive API が推奨です。構造体に `#[derive(Parser)]` を付けるだけで型安全な CLI が定義できます。動的にコマンドを構築する必要がある場合(プラグインシステム等)のみ builder API を検討してください。

### Q2: musl と glibc の違いは?

**A:** musl でビルドすると静的リンクされ、libc に依存しないバイナリになります。どの Linux ディストリビューションでもそのまま動作します。glibc は動的リンクのため、実行環境の glibc バージョンに依存します。配布用バイナリには musl が推奨です。

### Q3: シェル補完を生成するには?

**A:** clap の `clap_complete` クレートで各シェルの補完スクリプトを自動生成できます。

```rust
use clap::CommandFactory;
use clap_complete::{generate, Shell};

// ビルド時に生成
fn main() {
    let mut cmd = Cli::command();
    generate(Shell::Bash, &mut cmd, "filetool", &mut std::io::stdout());
    generate(Shell::Zsh, &mut cmd, "filetool", &mut std::io::stdout());
    generate(Shell::Fish, &mut cmd, "filetool", &mut std::io::stdout());
}
```

---

## まとめ

| 項目 | 要点 |
|---|---|
| clap derive | `#[derive(Parser)]` で型安全な CLI 定義 |
| サブコマンド | `#[derive(Subcommand)]` で enum ベース |
| カラー出力 | colored / owo-colors で見やすい出力 |
| プログレスバー | indicatif で長時間処理の可視化 |
| 対話的UI | dialoguer で入力・選択・確認 |
| クロスコンパイル | cross で Docker ベースの簡単クロスビルド |
| 静的リンク | musl ターゲットで単一バイナリ配布 |
| 終了コード | 0=成功、非0=エラー。stderr にエラー出力 |

## 次に読むべきガイド

- [Cargo/ワークスペース](../04-ecosystem/00-cargo-workspace.md) — パッケージ管理と公開
- [テスト](../04-ecosystem/01-testing.md) — CLI の統合テスト
- [ベストプラクティス](../04-ecosystem/04-best-practices.md) — API設計とエラーハンドリング

## 参考文献

1. **clap documentation**: https://docs.rs/clap/latest/clap/
2. **Command Line Applications in Rust**: https://rust-cli.github.io/book/
3. **cross (cross-compilation)**: https://github.com/cross-rs/cross
