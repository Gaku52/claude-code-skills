# Serde — JSON/TOML/YAML

> Rust の事実上の標準シリアライゼーションフレームワーク Serde を、JSON/TOML/YAML の実践的な変換パターンと共に習得する

## この章で学ぶこと

1. **Serde の仕組み** — Serialize/Deserialize trait、derive マクロ、Data Model
2. **フォーマット別実践** — JSON (serde_json), TOML (toml), YAML (serde_yaml) の使い分け
3. **カスタマイズ** — 属性マクロ、カスタムシリアライザ、ゼロコピーデシリアライゼーション

---

## 1. Serde アーキテクチャ

```
┌──────────────── Serde のレイヤー構造 ────────────────┐
│                                                       │
│  ┌─────────────┐   Serialize    ┌─────────────────┐  │
│  │  Rust の型   │ ─────────────→ │  Serde Data     │  │
│  │  struct,enum │               │  Model          │  │
│  │  Vec, HashMap│ ←───────────── │  (29種の型)     │  │
│  └─────────────┘  Deserialize   └────────┬────────┘  │
│                                          │            │
│                                   Serializer /        │
│                                   Deserializer        │
│                                   (フォーマット別)     │
│                                          │            │
│                    ┌─────────────────────┼──────┐    │
│                    ▼                     ▼      ▼    │
│              ┌──────────┐  ┌──────────┐  ┌──────┐   │
│              │   JSON   │  │   TOML   │  │ YAML │   │
│              │serde_json│  │   toml   │  │serde │   │
│              │          │  │          │  │_yaml │   │
│              └──────────┘  └──────────┘  └──────┘   │
│                                                       │
│  この分離により:                                      │
│  - 1回の derive で全フォーマット対応                   │
│  - 新フォーマット追加が容易                            │
│  - 型→型の直接変換も可能 (serde_transcode)            │
└───────────────────────────────────────────────────────┘
```

---

## 2. 基本的な使い方

### コード例1: derive による自動実装

```rust
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
struct Config {
    /// サーバー設定
    server: ServerConfig,
    /// データベース設定
    database: DatabaseConfig,
    /// 機能フラグ
    #[serde(default)]
    features: Features,
}

#[derive(Debug, Serialize, Deserialize)]
struct ServerConfig {
    host: String,
    port: u16,
    #[serde(default = "default_workers")]
    workers: usize,
}

#[derive(Debug, Serialize, Deserialize)]
struct DatabaseConfig {
    url: String,
    #[serde(default = "default_pool_size")]
    pool_size: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    password: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
struct Features {
    auth_enabled: bool,
    rate_limiting: bool,
    cors: bool,
}

impl Default for Features {
    fn default() -> Self {
        Features {
            auth_enabled: true,
            rate_limiting: true,
            cors: false,
        }
    }
}

fn default_workers() -> usize { num_cpus::get() }
fn default_pool_size() -> u32 { 10 }

fn main() -> anyhow::Result<()> {
    // JSON
    let json = r#"{
        "server": { "host": "localhost", "port": 8080 },
        "database": { "url": "postgres://localhost/mydb" }
    }"#;
    let config: Config = serde_json::from_str(json)?;
    println!("JSON: {:?}", config);

    // TOML
    let toml_str = r#"
        [server]
        host = "0.0.0.0"
        port = 3000
        workers = 4

        [database]
        url = "postgres://prod-host/mydb"
        pool_size = 20
    "#;
    let config: Config = toml::from_str(toml_str)?;
    println!("TOML: {:?}", config);

    // 出力
    println!("\n--- JSON出力 ---");
    println!("{}", serde_json::to_string_pretty(&config)?);

    println!("\n--- TOML出力 ---");
    println!("{}", toml::to_string_pretty(&config)?);

    Ok(())
}
```

### コード例2: 列挙体のシリアライゼーション

```rust
use serde::{Serialize, Deserialize};

/// タグ付き列挙体 (デフォルト: 外部タグ)
#[derive(Debug, Serialize, Deserialize)]
enum Message {
    Text(String),
    Image { url: String, width: u32, height: u32 },
    Ping,
}
// JSON: {"Text": "hello"}
// JSON: {"Image": {"url": "...", "width": 800, "height": 600}}
// JSON: "Ping"

/// 内部タグ形式
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
enum Event {
    #[serde(rename = "user.created")]
    UserCreated { id: u64, name: String },
    #[serde(rename = "user.deleted")]
    UserDeleted { id: u64 },
    #[serde(rename = "system.health")]
    HealthCheck { status: String },
}
// JSON: {"type": "user.created", "id": 1, "name": "Alice"}

/// 隣接タグ形式
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data")]
enum ApiResponse {
    Success(serde_json::Value),
    Error { code: u16, message: String },
}
// JSON: {"kind": "Success", "data": {...}}
// JSON: {"kind": "Error", "data": {"code": 404, "message": "Not found"}}

/// タグなし形式 (各バリアントのフィールドで判別)
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum NumberOrString {
    Number(f64),
    Text(String),
}
// JSON: 42.0 or "hello"
```

---

## 3. serde 属性マクロ一覧

```
┌──────────── serde 主要属性一覧 ──────────────┐
│                                                │
│  フィールド属性:                                │
│  #[serde(rename = "fieldName")]  名前変更      │
│  #[serde(alias = "old_name")]    別名          │
│  #[serde(default)]               デフォルト値   │
│  #[serde(default = "fn_name")]   カスタム初期値 │
│  #[serde(skip)]                  無視           │
│  #[serde(skip_serializing)]      出力時のみ無視 │
│  #[serde(skip_deserializing)]    入力時のみ無視 │
│  #[serde(skip_serializing_if)]   条件付き無視   │
│  #[serde(flatten)]               フラットに展開 │
│  #[serde(with = "module")]       カスタム変換   │
│                                                │
│  コンテナ属性:                                  │
│  #[serde(rename_all = "camelCase")]  全名前変更 │
│  #[serde(tag = "type")]          タグ形式      │
│  #[serde(deny_unknown_fields)]   未知フィールド禁止 │
│  #[serde(transparent)]           内部型として扱う │
└────────────────────────────────────────────────┘
```

### コード例3: 実践的な属性マクロ使用

```rust
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]  // 全フィールドを camelCase に
#[serde(deny_unknown_fields)]       // 未知のフィールドを拒否
struct UserProfile {
    user_id: u64,                    // → "userId"
    display_name: String,            // → "displayName"

    #[serde(rename = "email")]       // 明示的リネーム
    email_address: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    bio: Option<String>,             // None なら出力しない

    #[serde(default)]
    is_active: bool,                 // 入力に無ければ false

    #[serde(with = "chrono::serde::ts_seconds")]
    created_at: DateTime<Utc>,       // Unix タイムスタンプとして

    #[serde(flatten)]
    metadata: std::collections::HashMap<String, serde_json::Value>,
    // 追加フィールドをフラットに格納
}

// 入力 JSON:
// {
//   "userId": 1,
//   "displayName": "Alice",
//   "email": "alice@example.com",
//   "isActive": true,
//   "createdAt": 1700000000,
//   "customField": "extra data"
// }
```

### コード例4: カスタムデシリアライザ

```rust
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// 文字列またはu64としてデシリアライズ可能なID
#[derive(Debug, Clone)]
struct FlexibleId(u64);

impl<'de> Deserialize<'de> for FlexibleId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::{self, Visitor};

        struct IdVisitor;

        impl<'de> Visitor<'de> for IdVisitor {
            type Value = FlexibleId;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("数値または文字列のID")
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> Result<FlexibleId, E> {
                Ok(FlexibleId(v))
            }

            fn visit_str<E: de::Error>(self, v: &str) -> Result<FlexibleId, E> {
                v.parse::<u64>()
                    .map(FlexibleId)
                    .map_err(|_| E::custom(format!("無効なID: {}", v)))
            }
        }

        deserializer.deserialize_any(IdVisitor)
    }
}

impl Serialize for FlexibleId {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(self.0)
    }
}

// {"id": 42} も {"id": "42"} も受け付ける
#[derive(Debug, Serialize, Deserialize)]
struct Record {
    id: FlexibleId,
    name: String,
}
```

---

## 4. ゼロコピーデシリアライゼーション

### コード例5: 借用データのデシリアライズ

```rust
use serde::Deserialize;

/// ゼロコピー: 入力文字列から借用してデシリアライズ
#[derive(Debug, Deserialize)]
struct LogEntry<'a> {
    #[serde(borrow)]
    level: &'a str,           // 入力文字列への借用 (コピーなし)
    #[serde(borrow)]
    message: &'a str,
    timestamp: u64,
    #[serde(borrow)]
    tags: Vec<&'a str>,       // 各タグも借用
}

fn parse_logs(json_bytes: &[u8]) -> Vec<LogEntry<'_>> {
    // 入力バッファを直接参照 → メモリコピーなし
    serde_json::from_slice(json_bytes).unwrap()
}

/// 所有版 (比較用)
#[derive(Debug, Deserialize)]
struct OwnedLogEntry {
    level: String,             // String をヒープに確保
    message: String,
    timestamp: u64,
    tags: Vec<String>,
}

fn main() {
    let json = br#"[
        {"level": "INFO", "message": "Server started", "timestamp": 1700000000, "tags": ["boot", "server"]},
        {"level": "ERROR", "message": "Connection lost", "timestamp": 1700000001, "tags": ["network"]}
    ]"#;

    // ゼロコピー版 (高速、メモリ効率的)
    let entries: Vec<LogEntry> = serde_json::from_slice(json).unwrap();
    println!("{:?}", entries);

    // ※ entries のライフタイムは json バッファに依存
    // json が drop されると entries も無効になる
}
```

---

## 5. フォーマット別活用

### コード例6: TOML 設定ファイルの読み書き

```rust
use serde::{Serialize, Deserialize};
use std::path::Path;

#[derive(Debug, Serialize, Deserialize)]
struct AppConfig {
    app: AppSettings,
    #[serde(default)]
    logging: LoggingConfig,
    #[serde(default)]
    plugins: Vec<PluginConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AppSettings {
    name: String,
    version: String,
    debug: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(default)]
struct LoggingConfig {
    level: String,
    file: Option<String>,
    json_format: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        LoggingConfig {
            level: "info".into(),
            file: None,
            json_format: false,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct PluginConfig {
    name: String,
    enabled: bool,
    #[serde(default)]
    settings: toml::Table,
}

fn load_config(path: &Path) -> anyhow::Result<AppConfig> {
    let content = std::fs::read_to_string(path)?;
    let config: AppConfig = toml::from_str(&content)?;
    Ok(config)
}

fn save_config(path: &Path, config: &AppConfig) -> anyhow::Result<()> {
    let content = toml::to_string_pretty(config)?;
    std::fs::write(path, content)?;
    Ok(())
}

// config.toml:
// [app]
// name = "MyApp"
// version = "1.0.0"
// debug = false
//
// [logging]
// level = "debug"
// file = "app.log"
//
// [[plugins]]
// name = "metrics"
// enabled = true
//
// [plugins.settings]
// interval = 30
// endpoint = "http://localhost:9090"
```

---

## 6. 比較表

### フォーマット比較

| 特性 | JSON | TOML | YAML |
|---|---|---|---|
| 用途 | API通信、データ交換 | 設定ファイル | 設定、CI/CD |
| 人間可読性 | 中 | 高い | 高い |
| コメント | 不可 | 可能 (#) | 可能 (#) |
| ネスト | 自由 | 制限あり | 自由 |
| 日付型 | 文字列 | ネイティブ | 文字列 |
| Rust クレート | serde_json | toml | serde_yaml |
| パフォーマンス | 高速 (simd-json) | 中 | 低め |

### serde_json 関数比較

| 関数 | 入力 | 出力 | ゼロコピー |
|---|---|---|---|
| `from_str(&str)` | 文字列 | T | 可能 (`&'a str`) |
| `from_slice(&[u8])` | バイト列 | T | 可能 |
| `from_reader(Read)` | ストリーム | T | 不可 |
| `from_value(Value)` | JSON Value | T | 不可 |
| `to_string(&T)` | T | String | - |
| `to_string_pretty(&T)` | T | String (整形) | - |
| `to_writer(Write, &T)` | T | ストリーム出力 | - |
| `to_value(&T)` | T | JSON Value | - |

---

## 7. アンチパターン

### アンチパターン1: 全フィールドを Option にする

```rust
// NG: 全フィールドを Option にして「何でも受け入れる」
#[derive(Deserialize)]
struct BadConfig {
    host: Option<String>,
    port: Option<u16>,
    database_url: Option<String>,
    // → 実行時に None チェックが散乱
}

// OK: 必須フィールドは型で表現、デフォルト値を活用
#[derive(Deserialize)]
struct GoodConfig {
    host: String,                    // 必須
    #[serde(default = "default_port")]
    port: u16,                       // デフォルト値あり
    database_url: String,            // 必須
    #[serde(default)]
    cache_ttl: Option<u64>,          // 本当にオプションなもののみ
}

fn default_port() -> u16 { 8080 }
```

### アンチパターン2: serde_json::Value の多用

```rust
// NG: 型安全性を放棄して全て Value で扱う
fn bad_process(json: &str) -> String {
    let v: serde_json::Value = serde_json::from_str(json).unwrap();
    let name = v["user"]["name"].as_str().unwrap(); // 実行時パニックの危険
    name.to_string()
}

// OK: 構造体で型安全にデシリアライズ
#[derive(Deserialize)]
struct ApiResponse {
    user: User,
}
#[derive(Deserialize)]
struct User {
    name: String,
}

fn good_process(json: &str) -> Result<String, serde_json::Error> {
    let response: ApiResponse = serde_json::from_str(json)?;
    Ok(response.user.name)
}

// Value が適切な場面:
// - スキーマが不定の JSON (追加フィールドの保持)
// - JSON の部分的な操作 (特定キーの書き換え)
// - 型変換のブリッジ (serde_json::from_value)
```

---

## FAQ

### Q1: `#[serde(flatten)]` のパフォーマンスへの影響は?

**A:** flatten はデシリアライズ時に全フィールドをバッファリングするため、大きな構造体では性能低下が発生します。頻繁にデシリアライズする場合は、flatten を避けて明示的なフィールド定義を検討してください。

### Q2: JSON と TOML の相互変換は可能?

**A:** Serde の Data Model を経由して簡単に変換できます。

```rust
let config: AppConfig = toml::from_str(toml_str)?;
let json_str = serde_json::to_string_pretty(&config)?;
// TOML → Rust 型 → JSON の2段階変換
```

### Q3: バイナリフォーマットはどれがおすすめ?

**A:** 高速性重視なら `bincode` (Rust同士の通信)、サイズ重視なら `MessagePack` (rmp-serde)、スキーマ定義が必要なら `Protocol Buffers` (prost) を選択してください。全て Serde 経由で透過的に使えます。

---

## まとめ

| 項目 | 要点 |
|---|---|
| derive | `#[derive(Serialize, Deserialize)]` で自動実装 |
| JSON | serde_json。API通信の標準。高速 |
| TOML | toml クレート。設定ファイルに最適 |
| YAML | serde_yaml。CI/CD設定でよく使用 |
| 属性マクロ | rename, default, skip で柔軟にカスタマイズ |
| enum 表現 | tag, content, untagged で JSON 形式を制御 |
| ゼロコピー | `#[serde(borrow)]` + `&'a str` で高性能 |
| Value | 型不定 JSON のみに使用。型安全な構造体を優先 |

## 次に読むべきガイド

- [データベース](./03-database.md) — SQLx/SeaORM でのSerde連携
- [Axum](../02-async/04-axum-web.md) — JSON API の構築
- [テスト](./01-testing.md) — テストフィクスチャとしてのSerde活用

## 参考文献

1. **Serde Documentation**: https://serde.rs/
2. **serde_json crate**: https://docs.rs/serde_json/latest/serde_json/
3. **Serde Attributes Reference**: https://serde.rs/attributes.html
