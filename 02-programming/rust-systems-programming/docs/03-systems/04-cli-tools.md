# CLI Tools — clap, Cross-compilation

> Master the entire workflow of practical CLI tool development, from argument parsing with clap, colored output, and progress bars to cross-compilation and distribution.

## What You'll Learn in This Chapter

1. **Argument parsing** — Type-safe CLI definitions with the clap derive API, subcommands, and validation
2. **User experience** — Colored output, progress bars, and interactive prompts
3. **Cross-compilation and distribution** — cross, GitHub Actions, and cargo-dist


## Prerequisites

Your understanding will deepen if you have the following knowledge before reading this guide:

- Basic programming knowledge
- Understanding of related foundational concepts
- An understanding of the contents of [Embedded/WASM — no_std, wasm-bindgen](./03-embedded-wasm.md)

---

## 1. Components of a CLI Tool

```
┌────────────── CLI Tool Architecture ───────────────┐
│                                                     │
│  ┌─ Argument parsing ─────────────────────────────┐│
│  │  clap (derive / builder)                       ││
│  │  → commands, flags, options, subcommands       ││
│  └────────────────────────────────────────────────┘│
│                                                     │
│  ┌─ Output / UX ──────────────────────────────────┐│
│  │  colored / owo-colors  → colored output         ││
│  │  indicatif             → progress bars          ││
│  │  dialoguer             → interactive prompts    ││
│  │  tabled                → table display          ││
│  └────────────────────────────────────────────────┘│
│                                                     │
│  ┌─ I/O / Configuration ──────────────────────────┐│
│  │  serde + toml/json     → config files           ││
│  │  directories           → XDG paths              ││
│  │  tracing               → log output             ││
│  └────────────────────────────────────────────────┘│
│                                                     │
│  ┌─ Build / Distribution ─────────────────────────┐│
│  │  cross                 → cross-compilation      ││
│  │  cargo-dist            → binary distribution    ││
│  │  GitHub Actions        → CI/CD                  ││
│  └────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

---

## 2. Argument Parsing with clap

### Code Example 1: Basics of the derive API

```rust
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// File management tool — search, convert, and analyze files
#[derive(Parser, Debug)]
#[command(name = "filetool")]
#[command(version, about, long_about = None)]
struct Cli {
    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Path to the configuration file
    #[arg(short, long, default_value = "~/.config/filetool/config.toml")]
    config: PathBuf,

    /// Output format
    #[arg(short, long, value_enum, default_value_t = OutputFormat::Text)]
    format: OutputFormat,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Search files
    Search {
        /// Search pattern (regex)
        pattern: String,

        /// Target directory for search
        #[arg(short, long, default_value = ".")]
        dir: PathBuf,

        /// Maximum number of results
        #[arg(short = 'n', long, default_value_t = 100)]
        max_results: usize,

        /// File extension filter
        #[arg(short, long)]
        extension: Option<String>,
    },

    /// Convert files
    Convert {
        /// Input file
        input: PathBuf,

        /// Output file
        output: PathBuf,

        /// Target format for conversion
        #[arg(short, long)]
        to: String,
    },

    /// Analyze a directory
    Analyze {
        /// Target path
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Depth limit
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
        eprintln!("Config file: {:?}", cli.config);
        eprintln!("Output format: {:?}", cli.format);
    }

    match &cli.command {
        Commands::Search { pattern, dir, max_results, extension } => {
            println!("Search: '{}' in {:?} (max {})", pattern, dir, max_results);
            if let Some(ext) = extension {
                println!("Extension filter: .{}", ext);
            }
        }
        Commands::Convert { input, output, to } => {
            println!("Convert: {:?} -> {:?} (format: {})", input, output, to);
        }
        Commands::Analyze { path, depth } => {
            println!("Analyze: {:?} (depth: {:?})", path, depth);
        }
    }
}

// Usage examples:
// $ filetool search "TODO" --dir src/ -n 50 --extension rs
// $ filetool convert input.csv output.json --to json
// $ filetool analyze /var/log --depth 3 --verbose --format json
```

### Code Example 2: Validation

```rust
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
struct Cli {
    /// Port number (1024-65535)
    #[arg(short, long, default_value_t = 8080)]
    #[arg(value_parser = clap::value_parser!(u16).range(1024..=65535))]
    port: u16,

    /// Input file (existence check)
    #[arg(value_parser = validate_file_exists)]
    input: PathBuf,

    /// Concurrency level (1-256)
    #[arg(short = 'j', long, default_value_t = num_cpus::get())]
    #[arg(value_parser = clap::value_parser!(usize).range(1..=256))]
    jobs: usize,
}

fn validate_file_exists(s: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(s);
    if path.exists() {
        Ok(path)
    } else {
        Err(format!("File not found: {}", s))
    }
}
```

### Code Example 2b: Integration with Environment Variables

```rust
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "myapp")]
struct Cli {
    /// API key (can also be set via the MY_API_KEY environment variable)
    #[arg(long, env = "MY_API_KEY")]
    api_key: String,

    /// Log level (can also be set via the RUST_LOG environment variable)
    #[arg(long, env = "RUST_LOG", default_value = "info")]
    log_level: String,

    /// Database URL
    #[arg(long, env = "DATABASE_URL")]
    database_url: Option<String>,

    /// Debug mode
    #[arg(long, env = "DEBUG", default_value_t = false)]
    debug: bool,
}

// Priority order: command-line arguments > environment variables > default values
// $ MY_API_KEY=secret myapp --log-level debug
// $ myapp --api-key secret --database-url postgres://localhost/mydb
```

### Code Example 2c: Grouping and Flattening

```rust
use clap::{Args, Parser};

#[derive(Parser, Debug)]
struct Cli {
    #[command(flatten)]
    connection: ConnectionArgs,

    #[command(flatten)]
    output: OutputArgs,

    /// Query to execute
    query: String,
}

/// Connection settings
#[derive(Args, Debug)]
struct ConnectionArgs {
    /// Hostname
    #[arg(long, default_value = "localhost")]
    host: String,

    /// Port number
    #[arg(long, default_value_t = 5432)]
    port: u16,

    /// Username
    #[arg(long, env = "DB_USER")]
    user: String,

    /// Password
    #[arg(long, env = "DB_PASSWORD")]
    password: Option<String>,

    /// SSL mode
    #[arg(long, value_enum, default_value_t = SslMode::Prefer)]
    ssl_mode: SslMode,
}

/// Output settings
#[derive(Args, Debug)]
struct OutputArgs {
    /// Output format
    #[arg(short, long, value_enum, default_value_t = Format::Table)]
    format: Format,

    /// Output file (stdout if omitted)
    #[arg(short, long)]
    output: Option<std::path::PathBuf>,

    /// Hide header
    #[arg(long)]
    no_header: bool,
}

#[derive(clap::ValueEnum, Clone, Debug)]
enum SslMode { Disable, Prefer, Require }

#[derive(clap::ValueEnum, Clone, Debug)]
enum Format { Table, Json, Csv, Tsv }

// $ myapp --host db.example.com --user admin --format json "SELECT * FROM users"
```

### Code Example 2d: Custom Derive and Help Messages

```rust
use clap::Parser;

/// Log analysis tool — search, aggregate, and visualize structured logs
///
/// Examples:
///   logana search "ERROR" --since 1h --format json
///   logana stats --group-by level
///   logana tail --follow /var/log/app.log
#[derive(Parser)]
#[command(
    name = "logana",
    version,
    about,
    long_about = None,
    after_help = "For details, see https://example.com/logana.",
    arg_required_else_help = true,
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Search logs
    Search {
        /// Search pattern (regex supported)
        pattern: String,

        /// Time range filter (e.g., 1h, 30m, 7d)
        #[arg(long, value_parser = parse_duration)]
        since: Option<std::time::Duration>,

        /// Log level filter
        #[arg(long, value_delimiter = ',')]
        level: Vec<String>,

        /// Number of context lines (before and after)
        #[arg(short = 'C', long, default_value_t = 0)]
        context: usize,
    },
    /// Display statistics
    Stats {
        /// Grouping key
        #[arg(long)]
        group_by: Option<String>,

        /// Show only top N results
        #[arg(long, default_value_t = 10)]
        top: usize,
    },
    /// Track logs in real time
    Tail {
        /// Target file
        file: std::path::PathBuf,

        /// follow mode (equivalent to -f)
        #[arg(short, long)]
        follow: bool,

        /// Filter pattern
        #[arg(long)]
        filter: Option<String>,
    },
}

fn parse_duration(s: &str) -> Result<std::time::Duration, String> {
    let len = s.len();
    if len < 2 {
        return Err("Invalid duration format (e.g., 1h, 30m, 7d)".to_string());
    }
    let (num_str, unit) = s.split_at(len - 1);
    let num: u64 = num_str.parse().map_err(|_| "Failed to parse number".to_string())?;
    match unit {
        "s" => Ok(std::time::Duration::from_secs(num)),
        "m" => Ok(std::time::Duration::from_secs(num * 60)),
        "h" => Ok(std::time::Duration::from_secs(num * 3600)),
        "d" => Ok(std::time::Duration::from_secs(num * 86400)),
        _ => Err(format!("Unknown unit: {}. Please use s/m/h/d", unit)),
    }
}
```

---

## 3. Improving User Experience

### Code Example 3: Colored Output and Progress Bars

```rust
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle, MultiProgress};

fn main() {
    // Colored output
    println!("{}", "Success!".green().bold());
    println!("{}", "Warning: Low disk space".yellow());
    println!("{}", "Error: File corruption".red().bold());
    println!("{}", "Info: Processing started".blue());

    // Single progress bar
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
    pb.finish_with_message("Done!");

    // Multi progress bar
    let multi = MultiProgress::new();
    let style = ProgressStyle::with_template(
        "{prefix:.bold} [{bar:30}] {pos}/{len}"
    ).unwrap();

    let pb1 = multi.add(ProgressBar::new(50));
    let pb2 = multi.add(ProgressBar::new(80));
    pb1.set_style(style.clone());
    pb1.set_prefix("Download ");
    pb2.set_style(style);
    pb2.set_prefix("Convert  ");

    // Update bars concurrently
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

### Code Example 3b: Table Output

```rust
use tabled::{Table, Tabled, settings::{Style, Modify, object::Columns, Alignment}};

#[derive(Tabled)]
struct ProcessInfo {
    #[tabled(rename = "PID")]
    pid: u32,
    #[tabled(rename = "Name")]
    name: String,
    #[tabled(rename = "CPU %")]
    cpu: f64,
    #[tabled(rename = "Memory (MB)")]
    memory_mb: f64,
    #[tabled(rename = "Status")]
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
    // Output:
    // ╭──────┬──────────┬───────┬──────────────┬─────────╮
    // │ PID  │ Name     │ CPU % │ Memory (MB)  │ Status  │
    // ├──────┼──────────┼───────┼──────────────┼─────────┤
    // │ 1234 │ nginx    │   2.5 │        128.4 │ running │
    // │ 5678 │ postgres │  15.3 │        512.8 │ running │
    // │ 9012 │ redis    │   0.8 │         64.2 │ idle    │
    // ╰──────┴──────────┴───────┴──────────────┴─────────╯
}
```

### Code Example 3c: Spinners and Status Display

```rust
use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

/// Use a spinner for processes with indeterminate length
fn process_with_spinner() {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::with_template("{spinner:.green} {msg}")
            .unwrap()
            .tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
    );
    spinner.set_message("Connecting to the database...");

    // Simulate the connection process
    std::thread::sleep(Duration::from_secs(2));
    spinner.set_message("Validating schema...");
    std::thread::sleep(Duration::from_secs(1));
    spinner.set_message("Running migrations...");
    std::thread::sleep(Duration::from_secs(3));

    spinner.finish_with_message("✓ Migration complete");
}

/// Byte-based progress display, e.g. for downloads
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

    pb.finish_with_message("Download complete");
}
```

### Code Example 4: Interactive Prompts

```rust
use dialoguer::{Confirm, Input, Select, MultiSelect, Password};
use console::style;

fn main() -> anyhow::Result<()> {
    // Text input
    let name: String = Input::new()
        .with_prompt("Project name")
        .default("my-project".into())
        .interact_text()?;

    // Selection
    let frameworks = &["Axum", "Actix-web", "Rocket", "Warp"];
    let selection = Select::new()
        .with_prompt("Choose a framework")
        .items(frameworks)
        .default(0)
        .interact()?;
    println!("Selected: {}", frameworks[selection]);

    // Multi-selection
    let features = &["Database", "Auth", "WebSocket", "GraphQL", "Metrics"];
    let selections = MultiSelect::new()
        .with_prompt("Choose features (space to select)")
        .items(features)
        .interact()?;
    for &i in &selections {
        println!("  + {}", features[i]);
    }

    // Confirmation
    if Confirm::new()
        .with_prompt("Create the project?")
        .default(true)
        .interact()?
    {
        println!("{} Created project '{}'!", style("✓").green(), name);
    }

    Ok(())
}
```

---

## 4. Cross-compilation and Distribution

### Code Example 5: Cross-compilation Configuration

```toml
# .cargo/config.toml

# macOS (Apple Silicon)
[target.aarch64-apple-darwin]
# No configuration needed when building natively

# Linux (x86_64, statically linked with musl)
[target.x86_64-unknown-linux-musl]
linker = "x86_64-linux-musl-gcc"

# Windows (MinGW cross-compilation)
[target.x86_64-pc-windows-gnu]
linker = "x86_64-w64-mingw32-gcc"
```

```bash
# Cross-compile using cross (Docker-based)
cargo install cross

# Linux (static linking)
cross build --release --target x86_64-unknown-linux-musl

# Windows
cross build --release --target x86_64-pc-windows-gnu

# ARM Linux (Raspberry Pi)
cross build --release --target aarch64-unknown-linux-gnu
```

### CI/CD Pipeline

```
┌────────── GitHub Actions Release Flow ──────────┐
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

### Code Example 6: GitHub Actions Workflow

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

### Code Example 6b: Automated Distribution with cargo-dist

```toml
# Add the following to Cargo.toml
[workspace.metadata.dist]
# CI provider
ci = ["github"]
# Generated installers
installers = ["shell", "powershell", "homebrew"]
# Targets
targets = [
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "x86_64-unknown-linux-gnu",
    "x86_64-unknown-linux-musl",
    "x86_64-pc-windows-msvc",
]
# Homebrew tap repository
tap = "username/homebrew-tap"
# Release publishing
publish-jobs = ["homebrew"]
```

```bash
# Initialize cargo-dist
cargo install cargo-dist
cargo dist init

# Test the build locally
cargo dist build

# Show the plan (verify what will be built)
cargo dist plan

# Pushing a tag triggers the release automatically
git tag v1.0.0
git push --tags
```

---

## 5. Configuration Files and Data Persistence

### Code Example 7: XDG-compliant Configuration Management

```rust
use directories::ProjectDirs;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AppConfig {
    /// Default output format
    #[serde(default = "default_format")]
    pub format: String,

    /// Concurrency level
    #[serde(default = "default_jobs")]
    pub jobs: usize,

    /// Colored output
    #[serde(default = "default_color")]
    pub color: ColorMode,

    /// Editor command
    #[serde(default)]
    pub editor: Option<String>,

    /// Custom aliases
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
    /// Get the configuration file path (XDG compliant)
    pub fn config_path() -> Option<PathBuf> {
        ProjectDirs::from("com", "example", "filetool")
            .map(|dirs| dirs.config_dir().join("config.toml"))
    }

    /// Get the data directory path
    pub fn data_dir() -> Option<PathBuf> {
        ProjectDirs::from("com", "example", "filetool")
            .map(|dirs| dirs.data_dir().to_path_buf())
    }

    /// Get the cache directory path
    pub fn cache_dir() -> Option<PathBuf> {
        ProjectDirs::from("com", "example", "filetool")
            .map(|dirs| dirs.cache_dir().to_path_buf())
    }

    /// Load the configuration (use default values if absent)
    pub fn load() -> anyhow::Result<Self> {
        let path = Self::config_path()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine config directory"))?;

        if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            let config: Self = toml::from_str(&content)?;
            Ok(config)
        } else {
            Ok(Self::default())
        }
    }

    /// Save the configuration
    pub fn save(&self) -> anyhow::Result<()> {
        let path = Self::config_path()
            .ok_or_else(|| anyhow::anyhow!("Cannot determine config directory"))?;

        // Create the directory
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

// Example configuration file (~/.config/filetool/config.toml):
// format = "json"
// jobs = 8
// color = "auto"
// editor = "nvim"
//
// [aliases]
// rs = ["search", "--extension", "rs"]
// todo = ["search", "TODO|FIXME", "--extension", "rs"]
```

### Code Example 8: Managing History and Cache

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
        // Remove old entries
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

    /// Get the most recent N entries
    pub fn recent(&self, n: usize) -> &[HistoryEntry] {
        let start = self.entries.len().saturating_sub(n);
        &self.entries[start..]
    }

    /// Number of executions of a specific command
    pub fn command_count(&self, command: &str) -> usize {
        self.entries.iter().filter(|e| e.command == command).count()
    }
}
```

---

## 6. Error Handling and Logging

### Code Example 9: Structured Errors with anyhow + thiserror

```rust
use thiserror::Error;

/// Application-specific error type
#[derive(Error, Debug)]
pub enum AppError {
    #[error("Failed to load configuration file: {path}")]
    ConfigLoad {
        path: std::path::PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("Pattern '{pattern}' is an invalid regular expression")]
    InvalidPattern {
        pattern: String,
        #[source]
        source: regex::Error,
    },

    #[error("File '{path}' not found")]
    FileNotFound { path: std::path::PathBuf },

    #[error("Permission denied: {path}")]
    PermissionDenied { path: std::path::PathBuf },

    #[error("Operation timed out ({timeout_secs}s)")]
    Timeout { timeout_secs: u64 },

    #[error("Unknown format: '{format}'. Available: {available}")]
    UnknownFormat {
        format: String,
        available: String,
    },
}

/// Integration in the main function
fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize tracing
    setup_logging(cli.verbose)?;

    // Run
    let result = run(&cli);

    // Customize error display
    match result {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            // Display the anyhow error chain
            if cli.verbose {
                // Detailed: full error chain
                eprintln!("{}: {:?}", "Error".red().bold(), e);
            } else {
                // Concise: top-level message only
                eprintln!("{}: {}", "Error".red().bold(), e);
            }

            // Exit code based on error kind
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

### Code Example 10: Logging with tracing

```rust
use tracing::{debug, error, info, instrument, warn};
use tracing_subscriber::{fmt, EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

/// Logging setup
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

    // Add file output
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

/// Automatic tracing with the instrument macro
#[instrument(skip(content), fields(content_len = content.len()))]
fn process_file(path: &std::path::Path, content: &str) -> anyhow::Result<usize> {
    info!("Processing file: {:?}", path);

    let lines = content.lines().count();
    debug!(lines, "Counted lines");

    if lines == 0 {
        warn!("Empty file: {:?}", path);
    }

    // Error case
    if !path.exists() {
        error!("File does not exist: {:?}", path);
        anyhow::bail!("File not found");
    }

    info!(lines, "Processing complete");
    Ok(lines)
}

// Usage:
// RUST_LOG=debug myapp process     ← shows debug and above
// RUST_LOG=myapp=trace myapp       ← trace for own crate only
// myapp --verbose process          ← --verbose enables debug
```

---

## 7. CLI Testing

### Code Example 11: Integration Testing with assert_cmd

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
        .stdout(predicate::str::contains("File management tool"));
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
        .stdout(predicate::str::contains("0 results"));
}

#[test]
fn test_invalid_pattern() {
    Command::cargo_bin("filetool")
        .unwrap()
        .args(["search", "[invalid", "--dir", "."])
        .assert()
        .failure()
        .stderr(predicate::str::contains("invalid regular expression"));
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
    // Validate that the output is valid JSON
    let _: serde_json::Value = serde_json::from_str(&stdout)
        .expect("output should be valid JSON");
}
```

### Code Example 12: Snapshot Testing with trycmd

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
Search files

Usage: filetool search [OPTIONS] <PATTERN>

Arguments:
  <PATTERN>  Search pattern (regex)

Options:
  -d, --dir <DIR>            Target directory for search [default: .]
  -n, --max-results <MAX>    Maximum number of results [default: 100]
  -e, --extension <EXT>      File extension filter
  -h, --help                 Print help
"""
```

### Code Example 13: Unit Testing of clap

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

    /// Verify that help is generated correctly
    #[test]
    fn test_help_generation() {
        // Passing --help to parse_from terminates the process,
        // so we use Command directly to verify
        use clap::CommandFactory;
        let mut cmd = Cli::command();
        let help = cmd.render_help().to_string();
        assert!(help.contains("--verbose"));
        assert!(help.contains("--port"));
    }
}
```

---

## 8. Practical CLI Tool Design Patterns

### Pipe and Redirection Support

```rust
use std::io::{self, BufRead, Write, IsTerminal};

/// Detect whether stdin is piped or a TTY and adjust behavior accordingly
fn main() -> anyhow::Result<()> {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = io::BufWriter::new(stdout.lock());

    if stdin.is_terminal() {
        // Interactive mode: prompt the user for input
        eprintln!("Enter text (Ctrl+D to finish):");
    }

    // Process uniformly whether piped or TTY
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

// Usage:
// $ echo "hello world" | myapp          ← piped input
// $ myapp < input.txt > output.txt      ← redirection
// $ myapp                                ← interactive mode
```

### Signal Handling (Graceful Shutdown)

```rust
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn main() -> anyhow::Result<()> {
    // Ctrl+C handler
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        if !r.load(Ordering::SeqCst) {
            // Force exit on second Ctrl+C
            eprintln!("\nForce-quitting");
            std::process::exit(130);
        }
        eprintln!("\nInterrupt received. Stopping processing...");
        r.store(false, Ordering::SeqCst);
    })?;

    // Main processing loop
    let total = 1000;
    for i in 0..total {
        if !running.load(Ordering::SeqCst) {
            eprintln!("Processing aborted ({}/{})", i, total);
            // Cleanup
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
    eprintln!("Removing temporary files...");
    Ok(())
}
```

### Automatic Color Output Detection

```rust
use std::io::IsTerminal;

/// Automatically control color according to the output destination
struct Output {
    color_enabled: bool,
}

impl Output {
    fn new(color_mode: &str) -> Self {
        let color_enabled = match color_mode {
            "always" => true,
            "never" => false,
            _ => {
                // auto: color when TTY, disabled when piped
                std::io::stdout().is_terminal()
                    && std::env::var("NO_COLOR").is_err() // Honor the NO_COLOR convention
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

## 9. Comparison Tables

### CLI Crate Reference

| Category | Crate | Purpose | Cargo.toml |
|---|---|---|---|
| Argument parsing | clap | Command-line arguments | `clap = { version = "4", features = ["derive", "env"] }` |
| Color | colored | Colored terminal output | `colored = "2"` |
| Color | owo-colors | Lightweight colored output | `owo-colors = "4"` |
| Progress | indicatif | Progress bars / spinners | `indicatif = "0.17"` |
| Interaction | dialoguer | Interactive prompts | `dialoguer = "0.11"` |
| Tables | tabled | Table display | `tabled = "0.16"` |
| Console | console | Terminal utilities | `console = "0.15"` |
| Errors | anyhow | Application errors | `anyhow = "1"` |
| Errors | thiserror | Library error definitions | `thiserror = "2"` |
| Logging | tracing | Structured logging | `tracing = "0.1"` |
| Logging | tracing-subscriber | Logging output configuration | `tracing-subscriber = { version = "0.3", features = ["env-filter"] }` |
| Configuration | directories | XDG paths | `directories = "5"` |
| Configuration | toml | TOML parser | `toml = "0.8"` |
| Testing | assert_cmd | CLI integration testing | `assert_cmd = "2"` |
| Testing | predicates | Assertions | `predicates = "3"` |
| Testing | trycmd | Snapshot testing | `trycmd = "0.15"` |
| Signals | ctrlc | Ctrl+C handling | `ctrlc = "3"` |
| Build | cross | Cross-compilation | CLI tool |
| Distribution | cargo-dist | Binary distribution | CLI tool |

### Comparison of CLI Argument Parsing Libraries

| Library | API style | Compile speed | Features | Characteristics |
|---|---|---|---|---|
| clap (derive) | Struct macro | Slow | Most extensive | Industry standard |
| clap (builder) | Builder pattern | Slow | Most extensive | Dynamic definition |
| argh | derive | Fast | Basic | Made by Google, lightweight |
| pico-args | Manual | Fastest | Minimal | Zero dependencies |
| bpaf | derive + builder | Moderate | Rich | Composable |

### Distribution Method Comparison

| Method | Targets | Pros | Cons |
|---|---|---|---|
| GitHub Release | All OS | Wide reach | Manual download |
| cargo install | Rust developers | Simplest | Requires rustc |
| Homebrew (tap) | macOS/Linux | Package management | Tap maintenance required |
| cargo-dist | All OS | Automated | Configuration required |
| Docker image | Servers | Environment independent | Requires Docker |

---

## 10. Anti-patterns

### Anti-pattern 1: Sending Error Output to stdout

```rust
// BAD: Print error message via println! (stdout)
fn bad_main() {
    let result = process_file("input.txt");
    if result.is_err() {
        println!("Error: file not found");
        // Mixed with normal output when piped!
        // $ filetool input.txt | grep pattern  ← errors get grep'd too
    }
}

// GOOD: errors to stderr, normal output to stdout
fn good_main() {
    match process_file("input.txt") {
        Ok(output) => print!("{}", output),           // stdout
        Err(e) => {
            eprintln!("Error: {}", e);                 // stderr
            std::process::exit(1);                      // non-zero exit code
        }
    }
}

fn process_file(_: &str) -> Result<String, String> { Ok("data".into()) }
```

### Anti-pattern 2: Ignoring Exit Codes

```rust
// BAD: always exit(0)
fn bad() {
    if let Err(e) = run() {
        eprintln!("{}", e);
        // exit code = 0 (treated as success) → CI/CD cannot detect anomalies
    }
}

// GOOD: Appropriate exit codes
fn main() {
    let result = run();
    std::process::exit(match result {
        Ok(_) => 0,     // Success
        Err(e) => {
            eprintln!("Error: {:#}", e);
            match e.downcast_ref::<std::io::Error>() {
                Some(io_err) if io_err.kind() == std::io::ErrorKind::NotFound => 2,
                Some(_) => 3,    // I/O error
                None => 1,       // Other errors
            }
        }
    });
}

fn run() -> anyhow::Result<()> { Ok(()) }
```

### Anti-pattern 3: Forced Color Output

```rust
// BAD: color codes are emitted even when piped
fn bad_output() {
    // colored detects TTY by default, but
    // it can be forced on via env vars like CLICOLOR_FORCE=1
    println!("{}", "Result".green());
    // $ myapp | grep pattern → "\x1b[32mResult\x1b[0m" emitted as-is
}

// GOOD: Honor the NO_COLOR convention and TTY detection
fn good_output() {
    // Use colored's automatic detection
    if std::env::var("NO_COLOR").is_ok() {
        colored::control::set_override(false);
    }

    // Or control manually
    use std::io::IsTerminal;
    if !std::io::stdout().is_terminal() {
        colored::control::set_override(false);
    }

    println!("{}", "Result".green());
}

use colored::Colorize;
```

### Anti-pattern 4: Insufficient Buffering for Large Output

```rust
use std::io::{self, Write, BufWriter};

// BAD: println! per line → flush on every call hurts performance
fn bad_output(lines: &[String]) {
    for line in lines {
        println!("{}", line); // Implicit flush occurs
    }
}

// GOOD: Bulk write with BufWriter
fn good_output(lines: &[String]) -> io::Result<()> {
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    for line in lines {
        writeln!(out, "{}", line)?;
    }
    out.flush()?; // Explicit final flush
    Ok(())
}
// For tens of thousands of lines, this can be 10-100x faster
```

### Anti-pattern 5: Overuse of unwrap

```rust
// BAD: unwrap in a CLI tool → unfriendly panic message for users
fn bad_main() {
    let content = std::fs::read_to_string("config.toml").unwrap();
    // thread 'main' panicked at 'called `Result::unwrap()` on an `Err` value: ...
    let config: Config = toml::from_str(&content).unwrap();
}

// GOOD: Convert errors into human-readable messages
fn good_main() -> anyhow::Result<()> {
    let content = std::fs::read_to_string("config.toml")
        .context("Could not open config file config.toml")?;
    let config: Config = toml::from_str(&content)
        .context("Configuration file has an invalid format")?;
    Ok(())
}

use anyhow::Context;
use serde::Deserialize;
#[derive(Deserialize)] struct Config {}
```

---

## FAQ

### Q1: Should I use clap's derive API or builder API?

**A:** In most cases the derive API is recommended. You can define a type-safe CLI just by adding `#[derive(Parser)]` to a struct. Consider the builder API only when you need to construct commands dynamically (such as in plugin systems).

### Q2: What is the difference between musl and glibc?

**A:** Building with musl produces a statically linked binary that does not depend on libc. It runs as-is on any Linux distribution. glibc is dynamically linked, so it depends on the glibc version of the runtime environment. musl is recommended for distribution binaries.

### Q3: How do I generate shell completions?

**A:** clap's `clap_complete` crate can automatically generate completion scripts for various shells.

```rust
use clap::CommandFactory;
use clap_complete::{generate, Shell};

// Generate at build time (build.rs)
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

// Pattern of generating via a subcommand
#[derive(clap::Subcommand)]
enum Commands {
    /// Generate shell completion scripts
    Completions {
        /// Target shell
        #[arg(value_enum)]
        shell: Shell,
    },
    // ... other commands
}

fn handle_completions(shell: Shell) {
    let mut cmd = Cli::command();
    generate(shell, &mut cmd, "filetool", &mut std::io::stdout());
}

// Usage:
// $ filetool completions bash > ~/.local/share/bash-completion/completions/filetool
// $ filetool completions zsh > ~/.zfunc/_filetool
// $ filetool completions fish > ~/.config/fish/completions/filetool.fish
```

### Q4: How do I generate man pages?

**A:** The `clap_mangen` crate can generate man pages automatically.

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

### Q5: How do I reduce binary size?

**A:** Combining the following settings can substantially reduce the binary size.

```toml
# Cargo.toml
[profile.release]
opt-level = "z"      # Optimize for size
lto = true           # Link-Time Optimization
codegen-units = 1    # Single code generation unit
panic = "abort"      # Remove unwind tables
strip = true         # Strip debug info
```

| Setting | Size reduction | Impact on build speed |
|---|---|---|
| strip = true | 30-50% reduction | None |
| lto = true | 10-20% reduction | Significantly slower |
| opt-level = "z" | 5-15% reduction | Slightly slower |
| panic = "abort" | 5-10% reduction | None |
| codegen-units = 1 | 5-10% reduction | Slower |

You can also analyze function sizes inside the binary with `cargo install cargo-bloat`.

### Q6: How should I version a CLI tool?

**A:** Use Cargo.toml's `version` as a single source of truth and have it propagated automatically via clap's `version` macro.

```rust
#[derive(clap::Parser)]
#[command(version)] // Use the version from Cargo.toml
struct Cli {
    // ...
}

// If you want to include the git commit hash as well:
// build.rs
fn main() {
    // The vergen crate sets environment variables automatically
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

### Q7: How do I simulate stdin in tests?

**A:** Use the `write_stdin` method of `assert_cmd`.

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
    // Simulate piped input
    Command::cargo_bin("myapp")
        .unwrap()
        .write_stdin("hello world\n")
        .args(["--format", "json"])
        .assert()
        .success();
}
```

### Q8: How does a Rust CLI compare to a Go CLI?

**A:** Each has its own strengths.

| Item | Rust (clap) | Go (cobra) |
|---|---|---|
| Compile speed | Slow | Fast |
| Binary size | Small (with strip) | Slightly larger |
| Execution speed | Top tier | Sufficiently fast |
| Memory usage | Minimal | Slightly higher due to GC |
| Cross-compilation | Requires cross | Native support |
| Type safety | Very high | High |
| Ecosystem | clap + many crates | cobra is the standard |
| Learning curve | High | Low |

Rust is suitable when performance is paramount or memory safety is required. Go can be more advantageous for rapid development and team-based development.

---

## Summary

| Item | Key points |
|---|---|
| clap derive | Define a type-safe CLI with `#[derive(Parser)]` |
| Subcommands | Enum-based via `#[derive(Subcommand)]` |
| Environment variable integration | Fall back to env vars with `#[arg(env = "...")]` |
| Validation | Validate input using `value_parser` and custom functions |
| Colored output | Readable output with colored / owo-colors |
| NO_COLOR support | Honor the NO_COLOR env var and TTY detection |
| Progress bars | Visualize long-running tasks with indicatif |
| Table display | Tabular display of structured data with tabled |
| Interactive UI | Input, selection, and confirmation with dialoguer |
| Configuration files | XDG-compliant settings management with directories + toml |
| Error handling | Structured errors with anyhow + thiserror |
| Logging | Structured logs via tracing |
| Cross-compilation | Easy Docker-based cross-builds with cross |
| Static linking | Distribute a single binary using the musl target |
| Distribution | Automated binary distribution with cargo-dist |
| Testing | CLI integration tests via assert_cmd + trycmd |
| Signals | Graceful Ctrl+C shutdown with ctrlc |
| Exit codes | 0 = success, non-zero = error. Errors go to stderr |
| Pipe support | Fast output with BufWriter, detect with is_terminal() |

## Recommended Next Reads

- [Cargo / Workspaces](../04-ecosystem/00-cargo-workspace.md) — Package management and publishing
- [Testing](../04-ecosystem/01-testing.md) — CLI integration testing
- [Best Practices](../04-ecosystem/04-best-practices.md) — API design and error handling

## References

1. **clap documentation**: https://docs.rs/clap/latest/clap/
2. **Command Line Applications in Rust**: https://rust-cli.github.io/book/
3. **cross (cross-compilation)**: https://github.com/cross-rs/cross
