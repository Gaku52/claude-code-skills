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

### コード例2b: 環境変数との連携

```rust
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "myapp")]
struct Cli {
    /// API キー (環境変数 MY_API_KEY でも設定可能)
    #[arg(long, env = "MY_API_KEY")]
    api_key: String,

    /// ログレベル (環境変数 RUST_LOG でも設定可能)
    #[arg(long, env = "RUST_LOG", default_value = "info")]
    log_level: String,

    /// データベース URL
    #[arg(long, env = "DATABASE_URL")]
    database_url: Option<String>,

    /// デバッグモード
    #[arg(long, env = "DEBUG", default_value_t = false)]
    debug: bool,
}

// 優先順位: コマンドライン引数 > 環境変数 > デフォルト値
// $ MY_API_KEY=secret myapp --log-level debug
// $ myapp --api-key secret --database-url postgres://localhost/mydb
```

### コード例2c: グループ化とフラットン

```rust
use clap::{Args, Parser};

#[derive(Parser, Debug)]
struct Cli {
    #[command(flatten)]
    connection: ConnectionArgs,

    #[command(flatten)]
    output: OutputArgs,

    /// 実行するクエリ
    query: String,
}

/// 接続設定
#[derive(Args, Debug)]
struct ConnectionArgs {
    /// ホスト名
    #[arg(long, default_value = "localhost")]
    host: String,

    /// ポート番号
    #[arg(long, default_value_t = 5432)]
    port: u16,

    /// ユーザー名
    #[arg(long, env = "DB_USER")]
    user: String,

    /// パスワード
    #[arg(long, env = "DB_PASSWORD")]
    password: Option<String>,

    /// SSL モード
    #[arg(long, value_enum, default_value_t = SslMode::Prefer)]
    ssl_mode: SslMode,
}

/// 出力設定
#[derive(Args, Debug)]
struct OutputArgs {
    /// 出力フォーマット
    #[arg(short, long, value_enum, default_value_t = Format::Table)]
    format: Format,

    /// 出力ファイル (省略時は stdout)
    #[arg(short, long)]
    output: Option<std::path::PathBuf>,

    /// ヘッダーを非表示
    #[arg(long)]
    no_header: bool,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum SslMode { Disable, Prefer, Require }

#[derive(clap::ValueEnum, Clone, Debug)]
enum Format { Table, Json, Csv, Tsv }

// $ myapp --host db.example.com --user admin --format json "SELECT * FROM users"
```

### コード例2d: カスタム derive とヘルプメッセージ

```rust
use clap::Parser;

/// ログ解析ツール — 構造化ログを検索・集計・可視化する
///
/// 例:
///   logana search "ERROR" --since 1h --format json
///   logana stats --group-by level
///   logana tail --follow /var/log/app.log
#[derive(Parser)]
#[command(
    name = "logana",
    version,
    about,
    long_about = None,
    after_help = "詳細は https://example.com/logana を参照してください。",
    arg_required_else_help = true,
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// ログを検索する
    Search {
        /// 検索パターン (正規表現対応)
        pattern: String,

        /// 期間フィルタ (例: 1h, 30m, 7d)
        #[arg(long, value_parser = parse_duration)]
        since: Option<std::time::Duration>,

        /// ログレベルフィルタ
        #[arg(long, value_delimiter = ',')]
        level: Vec<String>,

        /// コンテキスト行数 (前後)
        #[arg(short = 'C', long, default_value_t = 0)]
        context: usize,
    },
    /// 統計情報を表示する
    Stats {
        /// グループ化キー
        #[arg(long)]
        group_by: Option<String>,

        /// 上位 N 件のみ表示
        #[arg(long, default_value_t = 10)]
        top: usize,
    },
    /// リアルタイムでログを追跡する
    Tail {
        /// 対象ファイル
        file: std::path::PathBuf,

        /// follow モード (-f 相当)
        #[arg(short, long)]
        follow: bool,

        /// フィルタパターン
        #[arg(long)]
        filter: Option<String>,
    },
}

fn parse_duration(s: &str) -> Result<std::time::Duration, String> {
    let len = s.len();
    if len < 2 {
        return Err("期間の形式が不正です (例: 1h, 30m, 7d)".to_string());
    }
    let (num_str, unit) = s.split_at(len - 1);
    let num: u64 = num_str.parse().map_err(|_| "数値の解析に失敗しました".to_string())?;
    match unit {
        "s" => Ok(std::time::Duration::from_secs(num)),
        "m" => Ok(std::time::Duration::from_secs(num * 60)),
        "h" => Ok(std::time::Duration::from_secs(num * 3600)),
        "d" => Ok(std::time::Duration::from_secs(num * 86400)),
        _ => Err(format!("不明な単位: {}. s/m/h/d を使用してください", unit)),
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

### コード例3b: テーブル出力

```rust
use tabled::{Table, Tabled, settings::{Style, Modify, object::Columns, Alignment}};

#[derive(Tabled)]
struct ProcessInfo {
    #[tabled(rename = "PID")]
    pid: u32,
    #[tabled(rename = "名前")]
    name: String,
    #[tabled(rename = "CPU %")]
    cpu: f64,
    #[tabled(rename = "メモリ (MB)")]
    memory_mb: f64,
    #[tabled(rename = "状態")]
    status: String,
}

fn display_processes(processes: &[ProcessInfo]) {
    let table = Table::new(processes)
        .with(Style::rounded())
        .with(Modify::new(Columns::new(2..=3)).with(Alignment::right()))
        .to_string();

    println!("{}", table);
}

fn main() {
    let processes = vec![
        ProcessInfo {
            pid: 1234, name: "nginx".into(),
            cpu: 2.5, memory_mb: 128.4, status: "running".into(),
        },
        ProcessInfo {
            pid: 5678, name: "postgres".into(),
            cpu: 15.3, memory_mb: 512.8, status: "running".into(),
        },
        ProcessInfo {
            pid: 9012, name: "redis".into(),
            cpu: 0.8, memory_mb: 64.2, status: "idle".into(),
        },
    ];

    display_processes(&processes);
    // 出力:
    // ╭──────┬──────────┬───────┬──────────────┬─────────╮
    // │ PID  │ 名前     │ CPU % │ メモリ (MB)  │ 状態    │
    // ├──────┼──────────┼───────┼──────────────┼─────────┤
    // │ 1234 │ nginx    │   2.5 │        128.4 │ running │
    // │ 5678 │ postgres │  15.3 │        512.8 │ running │
    // │ 9012 │ redis    │   0.8 │         64.2 │ idle    │
    // ╰──────┴──────────┴───────┴──────────────┴─────────╯
}
```

### コード例3c: スピナーとステータス表示

```rust
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

/// 不定量の処理にはスピナーを使う
fn process_with_spinner() {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::with_template("{spinner:.green} {msg}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    spinner.set_message("データベースに接続中...");

    // 接続処理のシミュレーション
    std::thread::sleep(Duration::from_secs(2));
    spinner.set_message("スキーマを検証中...");
    std::thread::sleep(Duration::from_secs(1));
    spinner.set_message("マイグレーションを実行中...");
    std::thread::sleep(Duration::from_secs(3));

    spinner.finish_with_message("✓ マイグレーション完了");
}

/// ダウンロードのようなバイトベースの進捗表示
fn download_with_progress(total_bytes: u64) {
    let pb = ProgressBar::new(total_bytes);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})"
        )
        .unwrap()
        .progress_chars("█▓▒░  "),
    );

    let mut downloaded = 0u64;
    while downloaded < total_bytes {
        let chunk = std::cmp::min(8192, total_bytes - downloaded);
        downloaded += chunk;
        pb.set_position(downloaded);
        std::thread::sleep(Duration::from_millis(10));
    }

    pb.finish_with_message("ダウンロード完了");
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

### コード例6b: cargo-dist による自動配布

```toml
# Cargo.toml に以下を追加
[workspace.metadata.dist]
# CI プロバイダ
ci = ["github"]
# インストーラ生成
installers = ["shell", "powershell", "homebrew"]
# ターゲット
targets = [
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-unknown-linux-gnu",
    "x86_64-unknown-linux-musl",
    "x86_64-pc-windows-msvc",
]
# Homebrew tap リポジトリ
tap = "username/homebrew-tap"
# リリース発行
publish-jobs = ["homebrew"]
```

```bash
# cargo-dist の初期化
cargo install cargo-dist
cargo dist init

# ローカルでビルドテスト
cargo dist build

# プラン表示 (何がビルドされるか確認)
cargo dist plan

# タグプッシュでリリース自動実行
git tag v1.0.0
git push --tags
```

---

## 5. 設定ファイルとデータ永続化

### コード例7: XDG 準拠の設定管理

```rust
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AppConfig {
    /// デフォルトの出力フォーマット
    #[serde(default = "default_format")]
    pub format: String,

    /// 並行度
    #[serde(default = "default_jobs")]
    pub jobs: usize,

    /// カラー出力
    #[serde(default = "default_color")]
    pub color: ColorMode,

    /// エディタコマンド
    #[serde(default)]
    pub editor: Option<String>,

    /// カスタムエイリアス
    #[serde(default)]
    pub aliases: std::collections::HashMap<String, Vec<String>>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
pub enum ColorMode {
    Auto,
    Always,
    Never,
}

fn default_format() -> String { "text".into() }
fn default_jobs() -> usize { num_cpus::get() }
fn default_color() -> ColorMode { ColorMode::Auto }

impl AppConfig {
    /// 設定ファイルのパスを取得 (XDG 準拠)
    pub fn config_path() -> Option<PathBuf> {
        ProjectDirs::from("com", "example", "filetool")
            .map(|dirs| dirs.config_dir().join("config.toml"))
    }

    /// データディレクトリのパスを取得
    pub fn data_dir() -> Option<PathBuf> {
        ProjectDirs::from("com", "example", "filetool")
            .map(|dirs| dirs.data_dir().to_path_buf())
    }

    /// キャッシュディレクトリのパスを取得
    pub fn cache_dir() -> Option<PathBuf> {
        ProjectDirs::from("com", "example", "filetool")
            .map(|dirs| dirs.cache_dir().to_path_buf())
    }

    /// 設定を読み込む (なければデフォルト値)
    pub fn load() -> anyhow::Result<Self> {
        let path = Self::config_path()
            .ok_or_else(|| anyhow::anyhow!("設定ディレクトリを特定できません"))?;

        if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            let config: Self = toml::from_str(&content)?;
            Ok(config)
        } else {
            Ok(Self::default())
        }
    }

    /// 設定を保存する
    pub fn save(&self) -> anyhow::Result<()> {
        let path = Self::config_path()
            .ok_or_else(|| anyhow::anyhow!("設定ディレクトリを特定できません"))?;

        // ディレクトリを作成
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = toml::to_string_pretty(self)?;
        std::fs::write(&path, content)?;
        Ok(())
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            format: default_format(),
            jobs: default_jobs(),
            color: default_color(),
            editor: None,
            aliases: std::collections::HashMap::new(),
        }
    }
}

// 設定ファイルの例 (~/.config/filetool/config.toml):
// format = "json"
// jobs = 8
// color = "auto"
// editor = "nvim"
//
// [aliases]
// rs = ["search", "--extension", "rs"]
// todo = ["search", "TODO|FIXME", "--extension", "rs"]
```

### コード例8: 履歴とキャッシュの管理

```rust
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Debug)]
pub struct CommandHistory {
    entries: Vec<HistoryEntry>,
    #[serde(default = "default_max_entries")]
    max_entries: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct HistoryEntry {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub command: String,
    pub args: Vec<String>,
    pub exit_code: i32,
    pub duration_ms: u64,
}

fn default_max_entries() -> usize { 1000 }

impl CommandHistory {
    pub fn load(path: &PathBuf) -> anyhow::Result<Self> {
        if path.exists() {
            let content = std::fs::read_to_string(path)?;
            Ok(serde_json::from_str(&content)?)
        } else {
            Ok(Self {
                entries: Vec::new(),
                max_entries: default_max_entries(),
            })
        }
    }

    pub fn add(&mut self, entry: HistoryEntry) {
        self.entries.push(entry);
        // 古いエントリを削除
        if self.entries.len() > self.max_entries {
            let excess = self.entries.len() - self.max_entries;
            self.entries.drain(..excess);
        }
    }

    pub fn save(&self, path: &PathBuf) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// 直近 N 件を取得
    pub fn recent(&self, n: usize) -> &[HistoryEntry] {
        let start = self.entries.len().saturating_sub(n);
        &self.entries[start..]
    }

    /// 特定コマンドの実行回数
    pub fn command_count(&self, command: &str) -> usize {
        self.entries.iter().filter(|e| e.command == command).count()
    }
}
```

---

## 6. エラーハンドリングとロギング

### コード例9: anyhow + thiserror による構造化エラー

```rust
use thiserror::Error;

/// アプリケーション固有のエラー型
#[derive(Error, Debug)]
pub enum AppError {
    #[error("設定ファイルの読み込みに失敗しました: {path}")]
    ConfigLoad {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("パターン '{pattern}' は無効な正規表現です")]
    InvalidPattern {
        pattern: String,
        #[source]
        source: regex::Error,
    },

    #[error("ファイル '{path}' が見つかりません")]
    FileNotFound { path: std::path::PathBuf },

    #[error("権限が不足しています: {path}")]
    PermissionDenied { path: std::path::PathBuf },

    #[error("操作がタイムアウトしました ({timeout_secs}秒)")]
    Timeout { timeout_secs: u64 },

    #[error("不明なフォーマット: '{format}'. 使用可能: {available}")]
    UnknownFormat {
        format: String,
        available: String,
    },
}

/// main 関数での統合
fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // tracing の初期化
    setup_logging(cli.verbose)?;

    // 実行
    let result = run(&cli);

    // エラー表示のカスタマイズ
    match result {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            // anyhow のエラーチェーンを表示
            if cli.verbose {
                // 詳細: エラーチェーン全体
                eprintln!("{}: {:?}", "エラー".red().bold(), e);
            } else {
                // 簡潔: 最上位のメッセージのみ
                eprintln!("{}: {}", "エラー".red().bold(), e);
            }

            // エラーの種類に応じた終了コード
            let code = match e.downcast_ref::<AppError>() {
                Some(AppError::FileNotFound { .. }) => 2,
                Some(AppError::PermissionDenied { .. }) => 3,
                Some(AppError::InvalidPattern { .. }) => 4,
                Some(AppError::ConfigLoad { .. }) => 5,
                Some(AppError::Timeout { .. }) => 6,
                _ => 1,
            };
            std::process::exit(code);
        }
    }
}

use clap::Parser;
use colored::Colorize;
#[derive(Parser)] struct Cli { #[arg(short, long)] verbose: bool }
fn run(_cli: &Cli) -> anyhow::Result<()> { Ok(()) }
fn setup_logging(_verbose: bool) -> anyhow::Result<()> { Ok(()) }
```

### コード例10: tracing によるログ出力

```rust
use tracing::{debug, error, info, instrument, warn};
use tracing_subscriber::{fmt, EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

/// ロギングのセットアップ
fn setup_logging(verbose: bool, log_file: Option<&std::path::Path>) -> anyhow::Result<()> {
    let env_filter = if verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info"))
    };

    let fmt_layer = fmt::layer()
        .with_target(false)
        .with_thread_ids(false)
        .with_timer(fmt::time::ChronoLocal::new("%H:%M:%S%.3f".to_string()));

    let subscriber = tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer);

    // ファイル出力を追加
    if let Some(log_path) = log_file {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)?;

        let file_layer = fmt::layer()
            .with_writer(file)
            .with_ansi(false)
            .json();

        subscriber.with(file_layer).init();
    } else {
        subscriber.init();
    }

    Ok(())
}

/// instrument マクロで自動トレーシング
#[instrument(skip(content), fields(content_len = content.len()))]
fn process_file(path: &std::path::Path, content: &str) -> anyhow::Result<usize> {
    info!("ファイルを処理中: {:?}", path);

    let lines = content.lines().count();
    debug!(lines, "行数を計算");

    if lines == 0 {
        warn!("空のファイルです: {:?}", path);
    }

    // エラーの場合
    if !path.exists() {
        error!("ファイルが存在しません: {:?}", path);
        anyhow::bail!("ファイルが見つかりません");
    }

    info!(lines, "処理完了");
    Ok(lines)
}

// 使い方:
// RUST_LOG=debug myapp process     ← debug 以上を表示
// RUST_LOG=myapp=trace myapp       ← 自クレートのみ trace
// myapp --verbose process          ← verbose で debug 有効
```

---

## 7. CLI テスト

### コード例11: assert_cmd による統合テスト

```rust
// tests/integration_test.rs
use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::TempDir;
use std::fs;

#[test]
fn test_help_flag() {
    Command::cargo_bin("filetool")
        .unwrap()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("ファイル管理ツール"));
}

#[test]
fn test_version_flag() {
    Command::cargo_bin("filetool")
        .unwrap()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains(env!("CARGO_PKG_VERSION")));
}

#[test]
fn test_search_basic() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("test.rs");
    fs::write(&file, "fn main() {\n    // TODO: implement\n    println!(\"hello\");\n}").unwrap();

    Command::cargo_bin("filetool")
        .unwrap()
        .args(["search", "TODO", "--dir"])
        .arg(dir.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("TODO"));
}

#[test]
fn test_search_no_results() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("empty.txt");
    fs::write(&file, "nothing here").unwrap();

    Command::cargo_bin("filetool")
        .unwrap()
        .args(["search", "NONEXISTENT", "--dir"])
        .arg(dir.path())
        .assert()
        .success()
        .stdout(predicate::str::contains("0 件"));
}

#[test]
fn test_invalid_pattern() {
    Command::cargo_bin("filetool")
        .unwrap()
        .args(["search", "[invalid", "--dir", "."])
        .assert()
        .failure()
        .stderr(predicate::str::contains("無効な正規表現"));
}

#[test]
fn test_nonexistent_directory() {
    Command::cargo_bin("filetool")
        .unwrap()
        .args(["search", "test", "--dir", "/nonexistent/path"])
        .assert()
        .failure()
        .code(2);
}

#[test]
fn test_json_output_format() {
    let dir = TempDir::new().unwrap();
    let file = dir.path().join("data.txt");
    fs::write(&file, "test data").unwrap();

    let output = Command::cargo_bin("filetool")
        .unwrap()
        .args(["search", "test", "--dir"])
        .arg(dir.path())
        .args(["--format", "json"])
        .output()
        .unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    // JSON として妥当かチェック
    let _: serde_json::Value = serde_json::from_str(&stdout)
        .expect("出力が有効な JSON であること");
}
```

### コード例12: trycmd によるスナップショットテスト

```toml
# Cargo.toml
[dev-dependencies]
trycmd = "0.15"
```

```rust
// tests/cli_tests.rs
#[test]
fn cli_tests() {
    trycmd::TestCases::new().case("tests/cmd/*.toml");
}
```

```toml
# tests/cmd/search_help.toml
bin.name = "filetool"
args = ["search", "--help"]
status.code = 0
stdout.is = """
ファイルを検索する

Usage: filetool search [OPTIONS] <PATTERN>

Arguments:
  <PATTERN>  検索パターン (正規表現)

Options:
  -d, --dir <DIR>            検索対象ディレクトリ [default: .]
  -n, --max-results <MAX>    最大結果数 [default: 100]
  -e, --extension <EXT>      ファイル拡張子フィルタ
  -h, --help                 Print help
"""
```

### コード例13: clap のユニットテスト

```rust
use clap::Parser;

#[derive(Parser, Debug, PartialEq)]
struct Cli {
    #[arg(short, long)]
    verbose: bool,

    #[arg(short, long, default_value_t = 8080)]
    port: u16,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(clap::Subcommand, Debug, PartialEq)]
enum Commands {
    Start { #[arg(long)] daemon: bool },
    Stop,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_args() {
        let cli = Cli::parse_from(["myapp"]);
        assert!(!cli.verbose);
        assert_eq!(cli.port, 8080);
        assert!(cli.command.is_none());
    }

    #[test]
    fn test_verbose_flag() {
        let cli = Cli::parse_from(["myapp", "--verbose"]);
        assert!(cli.verbose);
    }

    #[test]
    fn test_custom_port() {
        let cli = Cli::parse_from(["myapp", "--port", "3000"]);
        assert_eq!(cli.port, 3000);
    }

    #[test]
    fn test_subcommand_start() {
        let cli = Cli::parse_from(["myapp", "start", "--daemon"]);
        assert_eq!(cli.command, Some(Commands::Start { daemon: true }));
    }

    #[test]
    fn test_invalid_port_range() {
        let result = Cli::try_parse_from(["myapp", "--port", "99999"]);
        assert!(result.is_err());
    }

    /// ヘルプが正常に生成されることを検証
    #[test]
    fn test_help_generation() {
        // parse_from に --help を渡すとプロセスが終了するため、
        // Command を直接使って検証する
        use clap::CommandFactory;
        let mut cmd = Cli::command();
        let help = cmd.render_help().to_string();
        assert!(help.contains("--verbose"));
        assert!(help.contains("--port"));
    }
}
```

---

## 8. 実践的な CLI ツール設計パターン

### パイプとリダイレクト対応

```rust
use std::io::{self, BufRead, Write, IsTerminal};

/// stdin がパイプされているか TTY かを判定して動作を変える
fn main() -> anyhow::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    if stdin.is_terminal() {
        // 対話モード: ユーザーに入力を求める
        eprintln!("テキストを入力してください (Ctrl+D で終了):");
    }

    // パイプでもTTYでも統一的に処理
    for line in stdin.lock().lines() {
        let line = line?;
        let processed = process_line(&line);
        writeln!(out, "{}", processed)?;
    }

    Ok(())
}

fn process_line(line: &str) -> String {
    line.to_uppercase()
}

// 使い方:
// $ echo "hello world" | myapp          ← パイプ入力
// $ myapp < input.txt > output.txt      ← リダイレクト
// $ myapp                                ← 対話モード
```

### シグナルハンドリング (Graceful Shutdown)

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    // Ctrl+C ハンドラ
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        if !r.load(Ordering::SeqCst) {
            // 2回目の Ctrl+C で強制終了
            eprintln!("\n強制終了します");
            std::process::exit(130);
        }
        eprintln!("\n中断を受信しました。処理を停止しています...");
        r.store(false, Ordering::SeqCst);
    })?;

    // メイン処理ループ
    let total = 1000;
    for i in 0..total {
        if !running.load(Ordering::SeqCst) {
            eprintln!("処理を中断しました ({}/{})", i, total);
            // クリーンアップ処理
            cleanup()?;
            std::process::exit(130); // 128 + SIGINT(2)
        }
        process_item(i)?;
    }

    Ok(())
}

fn process_item(_i: usize) -> anyhow::Result<()> {
    std::thread::sleep(std::time::Duration::from_millis(10));
    Ok(())
}

fn cleanup() -> anyhow::Result<()> {
    eprintln!("一時ファイルを削除中...");
    Ok(())
}
```

### カラー出力の自動判定

```rust
use std::io::IsTerminal;

/// 出力先に応じてカラーを自動制御
struct Output {
    color_enabled: bool,
}

impl Output {
    fn new(color_mode: &str) -> Self {
        let color_enabled = match color_mode {
            "always" => true,
            "never" => false,
            _ => {
                // auto: TTY ならカラー、パイプなら無効
                std::io::stdout().is_terminal()
                    && std::env::var("NO_COLOR").is_err() // NO_COLOR 規約対応
                    && std::env::var("TERM").map_or(true, |t| t != "dumb")
            }
        };
        Self { color_enabled }
    }

    fn success(&self, msg: &str) {
        if self.color_enabled {
            println!("\x1b[32m✓\x1b[0m {}", msg);
        } else {
            println!("[OK] {}", msg);
        }
    }

    fn error(&self, msg: &str) {
        if self.color_enabled {
            eprintln!("\x1b[31m✗\x1b[0m {}", msg);
        } else {
            eprintln!("[ERROR] {}", msg);
        }
    }

    fn warning(&self, msg: &str) {
        if self.color_enabled {
            eprintln!("\x1b[33m⚠\x1b[0m {}", msg);
        } else {
            eprintln!("[WARN] {}", msg);
        }
    }

    fn info(&self, msg: &str) {
        if self.color_enabled {
            println!("\x1b[36mℹ\x1b[0m {}", msg);
        } else {
            println!("[INFO] {}", msg);
        }
    }
}
```

---

## 9. 比較表

### CLI クレート一覧

| カテゴリ | クレート | 用途 | Cargo.toml |
|---|---|---|---|
| 引数解析 | clap | コマンドライン引数 | `clap = { version = "4", features = ["derive", "env"] }` |
| カラー | colored | 色付きターミナル出力 | `colored = "2"` |
| カラー | owo-colors | 軽量カラー出力 | `owo-colors = "4"` |
| 進捗 | indicatif | プログレスバー / スピナー | `indicatif = "0.17"` |
| 対話 | dialoguer | インタラクティブプロンプト | `dialoguer = "0.11"` |
| テーブル | tabled | テーブル表示 | `tabled = "0.16"` |
| コンソール | console | ターミナルユーティリティ | `console = "0.15"` |
| エラー | anyhow | アプリケーションエラー | `anyhow = "1"` |
| エラー | thiserror | ライブラリエラー定義 | `thiserror = "2"` |
| ログ | tracing | 構造化ログ | `tracing = "0.1"` |
| ログ | tracing-subscriber | ログ出力設定 | `tracing-subscriber = { version = "0.3", features = ["env-filter"] }` |
| 設定 | directories | XDG パス | `directories = "5"` |
| 設定 | toml | TOML パーサ | `toml = "0.8"` |
| テスト | assert_cmd | CLI 統合テスト | `assert_cmd = "2"` |
| テスト | predicates | アサーション | `predicates = "3"` |
| テスト | trycmd | スナップショットテスト | `trycmd = "0.15"` |
| シグナル | ctrlc | Ctrl+C ハンドリング | `ctrlc = "3"` |
| ビルド | cross | クロスコンパイル | CLI ツール |
| 配布 | cargo-dist | バイナリ配布 | CLI ツール |

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

## 10. アンチパターン

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

### アンチパターン3: カラーの強制出力

```rust
// NG: パイプ先でもカラーコードが出力されてしまう
fn bad_output() {
    // colored はデフォルトで TTY を検出するが、
    // 環境変数 CLICOLOR_FORCE=1 などで常に有効になる場合がある
    println!("{}", "結果".green());
    // $ myapp | grep pattern → "\x1b[32m結果\x1b[0m" がそのまま出力
}

// OK: NO_COLOR 規約と TTY 判定を尊重
fn good_output() {
    // colored の自動検出を利用
    if std::env::var("NO_COLOR").is_ok() {
        colored::control::set_override(false);
    }

    // もしくは手動制御
    use std::io::IsTerminal;
    if !std::io::stdout().is_terminal() {
        colored::control::set_override(false);
    }

    println!("{}", "結果".green());
}

use colored::Colorize;
```

### アンチパターン4: 大量出力のバッファリング不足

```rust
use std::io::{self, Write, BufWriter};

// NG: 毎行 println! → 毎回 flush でパフォーマンス劣化
fn bad_output(lines: &[String]) {
    for line in lines {
        println!("{}", line); // 暗黙の flush が発生
    }
}

// OK: BufWriter で一括書き出し
fn good_output(lines: &[String]) -> io::Result<()> {
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    for line in lines {
        writeln!(out, "{}", line)?;
    }
    out.flush()?; // 最後に明示的に flush
    Ok(())
}
// 数万行の出力で 10-100 倍の速度差が出る
```

### アンチパターン5: unwrap の乱用

```rust
// NG: CLI ツールで unwrap → ユーザーに不親切なパニックメッセージ
fn bad_main() {
    let content = std::fs::read_to_string("config.toml").unwrap();
    // thread 'main' panicked at 'called `Result::unwrap()` on an `Err` value: ...
    let config: Config = toml::from_str(&content).unwrap();
}

// OK: エラーを人間が読めるメッセージに変換
fn good_main() -> anyhow::Result<()> {
    let content = std::fs::read_to_string("config.toml")
        .context("設定ファイル config.toml を開けませんでした")?;
    let config: Config = toml::from_str(&content)
        .context("設定ファイルの形式が不正です")?;
    Ok(())
}

use anyhow::Context;
use serde::Deserialize;
#[derive(Deserialize)] struct Config {}
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

// ビルド時に生成 (build.rs)
fn main() {
    let mut cmd = Cli::command();
    let out_dir = std::env::var("OUT_DIR").unwrap();

    for shell in [Shell::Bash, Shell::Zsh, Shell::Fish, Shell::PowerShell] {
        let mut file = std::fs::File::create(
            format!("{}/filetool.{}", out_dir, shell)
        ).unwrap();
        generate(shell, &mut cmd, "filetool", &mut file);
    }
}

// サブコマンドで生成するパターン
#[derive(clap::Subcommand)]
enum Commands {
    /// シェル補完スクリプトを生成する
    Completions {
        /// 対象シェル
        #[arg(value_enum)]
        shell: Shell,
    },
    // ... 他のコマンド
}

fn handle_completions(shell: Shell) {
    let mut cmd = Cli::command();
    generate(shell, &mut cmd, "filetool", &mut std::io::stdout());
}

// 使い方:
// $ filetool completions bash > ~/.local/share/bash-completion/completions/filetool
// $ filetool completions zsh > ~/.zfunc/_filetool
// $ filetool completions fish > ~/.config/fish/completions/filetool.fish
```

### Q4: マニュアルページ (man page) を生成するには?

**A:** `clap_mangen` クレートで man ページを自動生成できます。

```rust
// build.rs
use clap::CommandFactory;
use clap_mangen::Man;

fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let cmd = Cli::command();
    let man = Man::new(cmd);
    let mut buffer = Vec::new();
    man.render(&mut buffer).unwrap();
    std::fs::write(format!("{}/filetool.1", out_dir), buffer).unwrap();
}
```

### Q5: バイナリサイズを小さくするには?

**A:** 以下の設定を組み合わせることで、バイナリサイズを大幅に削減できます。

```toml
# Cargo.toml
[profile.release]
opt-level = "z"      # サイズ最適化
lto = true           # Link-Time Optimization
codegen-units = 1    # 単一コード生成ユニット
panic = "abort"      # unwind テーブル除去
strip = true         # デバッグ情報除去
```

| 設定 | サイズ削減効果 | ビルド速度への影響 |
|---|---|---|
| strip = true | 30-50% 削減 | なし |
| lto = true | 10-20% 削減 | 大幅に遅化 |
| opt-level = "z" | 5-15% 削減 | やや遅化 |
| panic = "abort" | 5-10% 削減 | なし |
| codegen-units = 1 | 5-10% 削減 | 遅化 |

さらに `cargo install cargo-bloat` でバイナリ内の関数サイズを分析できます。

### Q6: CLI ツールのバージョニングはどうすべき?

**A:** Cargo.toml の version をシングルソースオブトゥルースにし、clap の `version` マクロで自動反映させます。

```rust
#[derive(clap::Parser)]
#[command(version)] // Cargo.toml の version を使用
struct Cli {
    // ...
}

// git のコミットハッシュも含めたい場合:
// build.rs
fn main() {
    // vergen クレートで環境変数を自動設定
    println!("cargo:rustc-env=GIT_HASH={}",
        std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|_| "unknown".to_string()));
}

// main.rs
#[derive(clap::Parser)]
#[command(version = format!("{} ({})", env!("CARGO_PKG_VERSION"), env!("GIT_HASH")))]
struct Cli {
    // ...
}

// $ myapp --version
// myapp 1.2.3 (a1b2c3d)
```

### Q7: テスト時に stdin をシミュレートするには?

**A:** `assert_cmd` の `write_stdin` メソッドを使います。

```rust
use assert_cmd::Command;

#[test]
fn test_stdin_input() {
    Command::cargo_bin("myapp")
        .unwrap()
        .write_stdin("line1\nline2\nline3\n")
        .assert()
        .success()
        .stdout(predicates::str::contains("3 lines processed"));
}

#[test]
fn test_pipe_simulation() {
    // パイプされた入力のシミュレーション
    Command::cargo_bin("myapp")
        .unwrap()
        .write_stdin("hello world\n")
        .args(["--format", "json"])
        .assert()
        .success();
}
```

### Q8: Rust CLI は Go の CLI と比べてどう?

**A:** 両者にそれぞれ長所があります。

| 項目 | Rust (clap) | Go (cobra) |
|---|---|---|
| コンパイル速度 | 遅い | 速い |
| バイナリサイズ | 小さい (strip 時) | やや大きい |
| 実行速度 | 最速クラス | 十分速い |
| メモリ使用量 | 最小 | GC 分やや多い |
| クロスコンパイル | cross が必要 | ネイティブ対応 |
| 型安全性 | 非常に高い | 高い |
| エコシステム | clap + 多数クレート | cobra が標準 |
| 学習コスト | 高い | 低い |

パフォーマンスが最重要な場合や、メモリ安全性が求められる場合は Rust が適しています。迅速な開発やチーム開発では Go が優位な場合もあります。

---

## まとめ

| 項目 | 要点 |
|---|---|
| clap derive | `#[derive(Parser)]` で型安全な CLI 定義 |
| サブコマンド | `#[derive(Subcommand)]` で enum ベース |
| 環境変数連携 | `#[arg(env = "...")]` で環境変数フォールバック |
| バリデーション | `value_parser` とカスタム関数で入力検証 |
| カラー出力 | colored / owo-colors で見やすい出力 |
| NO_COLOR 対応 | NO_COLOR 環境変数と TTY 判定を尊重 |
| プログレスバー | indicatif で長時間処理の可視化 |
| テーブル表示 | tabled で構造化データの表形式出力 |
| 対話的UI | dialoguer で入力・選択・確認 |
| 設定ファイル | directories + toml で XDG 準拠の設定管理 |
| エラーハンドリング | anyhow + thiserror で構造化エラー |
| ログ出力 | tracing で構造化ログ |
| クロスコンパイル | cross で Docker ベースの簡単クロスビルド |
| 静的リンク | musl ターゲットで単一バイナリ配布 |
| 配布 | cargo-dist で自動バイナリ配布 |
| テスト | assert_cmd + trycmd で CLI 統合テスト |
| シグナル | ctrlc で Ctrl+C の Graceful Shutdown |
| 終了コード | 0=成功、非0=エラー。stderr にエラー出力 |
| パイプ対応 | BufWriter で高速出力、is_terminal() で判定 |

## 次に読むべきガイド

- [Cargo/ワークスペース](../04-ecosystem/00-cargo-workspace.md) — パッケージ管理と公開
- [テスト](../04-ecosystem/01-testing.md) — CLI の統合テスト
- [ベストプラクティス](../04-ecosystem/04-best-practices.md) — API設計とエラーハンドリング

## 参考文献

1. **clap documentation**: https://docs.rs/clap/latest/clap/
2. **Command Line Applications in Rust**: https://rust-cli.github.io/book/
3. **cross (cross-compilation)**: https://github.com/cross-rs/cross
