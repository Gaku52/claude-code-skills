# Serde — JSON/TOML/YAML

> Rust の事実上の標準シリアライゼーションフレームワーク Serde を、JSON/TOML/YAML の実践的な変換パターンと共に習得する

## この章で学ぶこと

1. **Serde の仕組み** — Serialize/Deserialize trait、derive マクロ、Data Model
2. **フォーマット別実践** — JSON (serde_json), TOML (toml), YAML (serde_yaml) の使い分け
3. **カスタマイズ** — 属性マクロ、カスタムシリアライザ、ゼロコピーデシリアライゼーション
4. **高度なパターン** — flatten、untagged enum、Visitor パターン、serde_with
5. **パフォーマンス** — ゼロコピー、simd-json、バイナリフォーマット

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

### 1.1 Serde Data Model の29種の型

```
┌──────────── Serde Data Model ──────────────┐
│                                              │
│  プリミティブ:                                │
│    bool, i8, i16, i32, i64, i128            │
│    u8, u16, u32, u64, u128                  │
│    f32, f64                                  │
│    char, string                              │
│    byte_array, bytes                         │
│                                              │
│  複合型:                                     │
│    option       → Option<T>                 │
│    unit         → ()                        │
│    unit_struct  → struct Unit;              │
│    unit_variant → enum E { A }              │
│    newtype_struct → struct N(T);            │
│    newtype_variant → enum E { A(T) }        │
│    seq          → Vec<T>, [T; N]            │
│    tuple        → (T1, T2, ...)             │
│    tuple_struct → struct T(T1, T2)          │
│    tuple_variant → enum E { A(T1, T2) }    │
│    map          → HashMap<K, V>             │
│    struct       → struct S { f: T }         │
│    struct_variant → enum E { A { f: T } }   │
│                                              │
│  derive マクロは Rust の型を上記モデルに       │
│  マッピングするコードを自動生成する             │
└──────────────────────────────────────────────┘
```

### 1.2 Cargo.toml での設定

```toml
[dependencies]
# Serde コア（derive マクロ付き）
serde = { version = "1", features = ["derive"] }

# JSON
serde_json = "1"

# TOML
toml = "0.8"

# YAML
serde_yaml = "0.9"

# バイナリフォーマット
bincode = "1"              # Rust 同士の通信
rmp-serde = "1"            # MessagePack
ciborium = "0.2"           # CBOR
postcard = "1"             # 組み込み向け小型フォーマット

# ユーティリティ
serde_with = "3"           # カスタムシリアライゼーションヘルパー
serde_repr = "0.1"         # 列挙体を整数として表現
serde_ignored = "0.1"      # 未知フィールドの収集

# 高速JSON
simd-json = "0.13"         # SIMD を活用した高速 JSON パーサー
sonic-rs = "0.3"           # 高速 JSON シリアライザ/デシリアライザ
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

### 2.1 serde_json の詳細な使い方

```rust
use serde::{Serialize, Deserialize};
use serde_json::{json, Value, Map};

// --- json! マクロによる動的 JSON 構築 ---
fn build_json_response(user_id: u64, name: &str) -> Value {
    json!({
        "status": "success",
        "data": {
            "user": {
                "id": user_id,
                "name": name,
                "roles": ["admin", "user"],
                "settings": {
                    "theme": "dark",
                    "notifications": true
                }
            }
        },
        "meta": {
            "version": "1.0",
            "timestamp": chrono::Utc::now().to_rfc3339()
        }
    })
}

// --- Value の操作 ---
fn manipulate_json() {
    let mut value = json!({
        "users": [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]
    });

    // ポインタアクセス
    if let Some(name) = value.pointer("/users/0/name") {
        println!("最初のユーザー: {}", name);
    }

    // ミュータブルポインタアクセス
    if let Some(age) = value.pointer_mut("/users/1/age") {
        *age = json!(26);
    }

    // 配列への追加
    if let Some(users) = value["users"].as_array_mut() {
        users.push(json!({"name": "Charlie", "age": 35}));
    }

    // マップの操作
    if let Some(obj) = value.as_object_mut() {
        obj.insert("total".to_string(), json!(3));
    }
}

// --- ストリーミング読み書き ---
use std::io::{BufReader, BufWriter};
use std::fs::File;

fn stream_json() -> anyhow::Result<()> {
    // ファイルからストリーミング読み込み
    let file = File::open("large_data.json")?;
    let reader = BufReader::new(file);
    let data: Vec<Record> = serde_json::from_reader(reader)?;

    // ファイルへストリーミング書き込み
    let file = File::create("output.json")?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &data)?;

    Ok(())
}

// --- 行区切り JSON (NDJSON / JSON Lines) ---
fn process_ndjson(input: &str) -> Vec<Record> {
    input
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

fn write_ndjson(records: &[Record], writer: &mut impl std::io::Write) -> anyhow::Result<()> {
    for record in records {
        serde_json::to_writer(&mut *writer, record)?;
        writeln!(writer)?;
    }
    Ok(())
}

// --- serde_json::Value と型の相互変換 ---
fn value_conversion() -> anyhow::Result<()> {
    // 型 → Value
    let user = User { name: "Alice".into(), age: 30 };
    let value: Value = serde_json::to_value(&user)?;

    // Value → 型
    let user2: User = serde_json::from_value(value)?;

    // Value を使った部分的なデシリアライズ
    let json_str = r#"{"name": "Bob", "age": 25, "extra": "ignored"}"#;
    let value: Value = serde_json::from_str(json_str)?;
    let name: String = serde_json::from_value(value["name"].clone())?;

    Ok(())
}
```

### 2.2 serde_json::RawValue の活用

```rust
use serde::{Serialize, Deserialize};
use serde_json::value::RawValue;

/// RawValue: JSON をパースせずにそのまま保持
/// パフォーマンスが重要な中間処理で有用
#[derive(Serialize, Deserialize)]
struct Envelope<'a> {
    #[serde(rename = "type")]
    msg_type: String,
    /// JSON ペイロードをパースせずに保持
    #[serde(borrow)]
    payload: &'a RawValue,
}

fn route_message(json: &str) -> anyhow::Result<()> {
    // エンベロープだけパースして、ペイロードは型に応じて後でパースする
    let envelope: Envelope = serde_json::from_str(json)?;

    match envelope.msg_type.as_str() {
        "user_event" => {
            let event: UserEvent = serde_json::from_str(envelope.payload.get())?;
            handle_user_event(event);
        }
        "order_event" => {
            let event: OrderEvent = serde_json::from_str(envelope.payload.get())?;
            handle_order_event(event);
        }
        _ => {
            // 未知のイベントもペイロードを保持して転送可能
            println!("未知のイベント: {}", envelope.payload.get());
        }
    }

    Ok(())
}

// RawValue を使った透過的な JSON プロキシ
#[derive(Serialize, Deserialize)]
struct ProxyRequest {
    target: String,
    headers: std::collections::HashMap<String, String>,
    /// リクエストボディをパースせずに転送
    body: Box<RawValue>,
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

### 2.3 列挙体表現の詳細比較

```rust
use serde::{Serialize, Deserialize};

// --- 外部タグ (Externally Tagged) [デフォルト] ---
#[derive(Serialize, Deserialize)]
enum ExternalTag {
    Variant1(String),
    Variant2 { x: i32, y: i32 },
    Variant3,
}
// {"Variant1": "hello"}
// {"Variant2": {"x": 1, "y": 2}}
// "Variant3"
// 利点: 明確。欠点: ネストが深くなる

// --- 内部タグ (Internally Tagged) ---
#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum InternalTag {
    #[serde(rename = "circle")]
    Circle { radius: f64 },
    #[serde(rename = "rectangle")]
    Rectangle { width: f64, height: f64 },
}
// {"type": "circle", "radius": 5.0}
// {"type": "rectangle", "width": 10.0, "height": 20.0}
// 利点: フラット。欠点: タプルバリアントに使えない

// --- 隣接タグ (Adjacently Tagged) ---
#[derive(Serialize, Deserialize)]
#[serde(tag = "t", content = "c")]
enum AdjacentTag {
    Text(String),
    Number(i32),
    Pair(String, i32),
}
// {"t": "Text", "c": "hello"}
// {"t": "Number", "c": 42}
// {"t": "Pair", "c": ["hello", 42]}
// 利点: タプルバリアント可。欠点: 冗長

// --- タグなし (Untagged) ---
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum Untagged {
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<serde_json::Value>),
    Object(std::collections::HashMap<String, serde_json::Value>),
}
// 42, 3.14, "hello", [...], {...}
// 利点: 自然な JSON。欠点: バリアント順序が重要、エラーメッセージが不明瞭

// --- serde_repr: 整数表現 ---
use serde_repr::{Serialize_repr, Deserialize_repr};

#[derive(Debug, Serialize_repr, Deserialize_repr)]
#[repr(u8)]
enum Priority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}
// JSON: 0, 1, 2, 3
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

### 3.1 属性マクロの詳細リファレンス

```rust
use serde::{Serialize, Deserialize};

// ========================================
// コンテナ属性（struct / enum 全体に適用）
// ========================================

// --- rename_all ---
// 全フィールドの名前変換規則を一括指定
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]       // camelCase
// #[serde(rename_all = "snake_case")]    // snake_case
// #[serde(rename_all = "PascalCase")]    // PascalCase
// #[serde(rename_all = "SCREAMING_SNAKE_CASE")]  // SCREAMING_SNAKE_CASE
// #[serde(rename_all = "kebab-case")]    // kebab-case
// #[serde(rename_all = "SCREAMING-KEBAB-CASE")]  // SCREAMING-KEBAB-CASE
// #[serde(rename_all = "lowercase")]     // lowercase
// #[serde(rename_all = "UPPERCASE")]     // UPPERCASE
struct UserProfile {
    user_id: u64,           // → "userId"
    first_name: String,     // → "firstName"
    last_name: String,      // → "lastName"
    is_active: bool,        // → "isActive"
}

// --- シリアライズとデシリアライズで別の規則 ---
#[derive(Serialize, Deserialize)]
#[serde(rename_all(serialize = "camelCase", deserialize = "snake_case"))]
struct MixedNaming {
    field_name: String,     // シリアライズ: "fieldName", デシリアライズ: "field_name"
}

// --- deny_unknown_fields ---
// 未知のフィールドがあるとエラー
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StrictConfig {
    host: String,
    port: u16,
    // {"host": "localhost", "port": 8080, "extra": true} → エラー!
}

// --- transparent ---
// ニュータイプを内部の型として扱う
#[derive(Serialize, Deserialize)]
#[serde(transparent)]
struct UserId(u64);
// JSON: 42 ({"UserId": 42} ではない)

#[derive(Serialize, Deserialize)]
#[serde(transparent)]
struct Email(String);
// JSON: "user@example.com"

// ========================================
// フィールド属性
// ========================================

#[derive(Serialize, Deserialize)]
struct DetailedExample {
    // --- rename ---
    #[serde(rename = "ID")]
    id: u64,

    // --- alias ---
    // デシリアライズ時に複数の名前を受け付ける
    #[serde(alias = "userName", alias = "user_name")]
    name: String,

    // --- default ---
    #[serde(default)]
    count: u32,             // 無ければ 0

    #[serde(default = "default_status")]
    status: String,         // 無ければ "active"

    // --- skip ---
    #[serde(skip)]
    internal_state: u32,    // シリアライズ/デシリアライズの両方で無視

    #[serde(skip_serializing)]
    write_only_field: String, // 書き込みのみ（出力しない）

    #[serde(skip_deserializing)]
    computed_field: String,  // 読み取りのみ（入力から無視）

    // --- skip_serializing_if ---
    #[serde(skip_serializing_if = "Option::is_none")]
    optional_field: Option<String>,

    #[serde(skip_serializing_if = "Vec::is_empty")]
    tags: Vec<String>,

    #[serde(skip_serializing_if = "is_zero")]
    retry_count: u32,

    // --- flatten ---
    // ネストを解消してフラットに展開
    #[serde(flatten)]
    metadata: std::collections::HashMap<String, serde_json::Value>,

    // --- with ---
    // カスタムシリアライゼーションモジュール
    #[serde(with = "chrono::serde::ts_seconds")]
    created_at: chrono::DateTime<chrono::Utc>,

    // --- serialize_with / deserialize_with ---
    #[serde(serialize_with = "serialize_uppercase")]
    #[serde(deserialize_with = "deserialize_trimmed")]
    label: String,

    // --- getter ---
    // シリアライズ時にメソッドを呼ぶ
    #[serde(getter = "DetailedExample::computed_value")]
    computed: String,

    // --- bound ---
    // derive で生成されるトレイト境界をカスタマイズ
    // #[serde(bound(serialize = "T: Serialize + Display"))]
    // #[serde(bound(deserialize = "T: Deserialize<'de> + Default"))]
}

fn default_status() -> String { "active".to_string() }
fn is_zero(v: &u32) -> bool { *v == 0 }

fn serialize_uppercase<S>(value: &str, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(&value.to_uppercase())
}

fn deserialize_trimmed<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    Ok(s.trim().to_string())
}
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

### 3.2 serde_with クレートの活用

```rust
use serde::{Serialize, Deserialize};
use serde_with::{serde_as, DisplayFromStr, DurationSeconds, TimestampSeconds};
use std::collections::HashMap;
use std::time::Duration;

// serde_as は #[serde(with = "...")] をより使いやすくしたマクロ
#[serde_as]
#[derive(Serialize, Deserialize)]
struct AdvancedConfig {
    // --- Duration を秒数として表現 ---
    #[serde_as(as = "DurationSeconds<u64>")]
    timeout: Duration,
    // JSON: {"timeout": 30}

    // --- Display/FromStr による変換 ---
    #[serde_as(as = "DisplayFromStr")]
    ip_address: std::net::IpAddr,
    // JSON: {"ip_address": "192.168.1.1"}

    // --- HashMap のキーを文字列として表現 ---
    #[serde_as(as = "HashMap<DisplayFromStr, _>")]
    port_mapping: HashMap<u16, String>,
    // JSON: {"port_mapping": {"8080": "web", "5432": "db"}}

    // --- Vec をカンマ区切り文字列として ---
    #[serde_as(as = "serde_with::StringWithSeparator::<serde_with::CommaSeparator, String>")]
    tags: Vec<String>,
    // JSON: {"tags": "rust,async,web"}

    // --- DateTime を RFC3339 文字列として ---
    #[serde_as(as = "TimestampSeconds<String>")]
    created_at: chrono::DateTime<chrono::Utc>,

    // --- Option<Vec> で None と空を区別 ---
    #[serde_as(as = "Option<Vec<DisplayFromStr>>")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    allowed_ips: Option<Vec<std::net::IpAddr>>,

    // --- Base64 エンコーディング ---
    #[serde_as(as = "serde_with::base64::Base64")]
    binary_data: Vec<u8>,
    // JSON: {"binary_data": "SGVsbG8gV29ybGQ="}
}
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

### 3.3 より複雑なカスタムデシリアライザ

```rust
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde::de::{self, Visitor, MapAccess, SeqAccess};

/// カンマ区切り文字列または配列として受け入れる
fn deserialize_string_or_array<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    struct StringOrArrayVisitor;

    impl<'de> Visitor<'de> for StringOrArrayVisitor {
        type Value = Vec<String>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("文字列またはstringの配列")
        }

        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            Ok(v.split(',').map(|s| s.trim().to_string()).collect())
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut vec = Vec::new();
            while let Some(value) = seq.next_element::<String>()? {
                vec.push(value);
            }
            Ok(vec)
        }
    }

    deserializer.deserialize_any(StringOrArrayVisitor)
}

#[derive(Deserialize)]
struct FlexibleConfig {
    // "tags": "rust,async,web" も "tags": ["rust", "async", "web"] もOK
    #[serde(deserialize_with = "deserialize_string_or_array")]
    tags: Vec<String>,
}

/// 環境変数からのフォールバック付きデシリアライズ
fn deserialize_with_env_fallback<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    if value.starts_with("${") && value.ends_with("}") {
        let env_var = &value[2..value.len() - 1];
        std::env::var(env_var)
            .map_err(|_| de::Error::custom(format!("環境変数 {} が見つかりません", env_var)))
    } else {
        Ok(value)
    }
}

#[derive(Deserialize)]
struct SecureConfig {
    host: String,
    // "password": "${DB_PASSWORD}" → 環境変数 DB_PASSWORD の値に展開
    #[serde(deserialize_with = "deserialize_with_env_fallback")]
    password: String,
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

### 4.1 ゼロコピーの制約と注意点

```rust
use serde::Deserialize;

// --- ゼロコピーが可能な条件 ---
// 1. 入力が &str や &[u8] である（所有型の String や Vec<u8> ではない）
// 2. エスケープシーケンスを含まない文字列
// 3. from_str() または from_slice() を使用（from_reader() は不可）

// ゼロコピーが可能な場合
#[derive(Deserialize)]
struct ZeroCopy<'a> {
    #[serde(borrow)]
    name: &'a str,       // OK: 入力バッファからの借用
    #[serde(borrow)]
    data: &'a [u8],      // OK: バイト列の借用
    count: u64,           // OK: コピーでも軽量
}

// ゼロコピーが不可能な場合
#[derive(Deserialize)]
struct NeedsCopy {
    // エスケープが含まれる可能性がある → ゼロコピー不可
    // "hello \"world\"" → アンエスケープ後の新しい文字列が必要
    name: String,
}

// --- Cow<str> で条件付きゼロコピー ---
use std::borrow::Cow;

#[derive(Deserialize)]
struct SmartCopy<'a> {
    #[serde(borrow)]
    name: Cow<'a, str>,   // エスケープなし → 借用、エスケープあり → 所有
    // "hello" → Cow::Borrowed("hello")
    // "hello \"world\"" → Cow::Owned("hello \"world\"")
}

// パフォーマンス比較
fn benchmark_zero_copy(json: &[u8]) {
    // ゼロコピー: ~50ns for small JSON
    let _: Vec<LogEntry> = serde_json::from_slice(json).unwrap();

    // 所有コピー: ~200ns for same JSON (String 確保のオーバーヘッド)
    let _: Vec<OwnedLogEntry> = serde_json::from_slice(json).unwrap();
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

### 5.1 YAML の活用

```rust
use serde::{Serialize, Deserialize};

// --- Kubernetes 風の YAML 設定 ---
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Deployment {
    api_version: String,
    kind: String,
    metadata: Metadata,
    spec: DeploymentSpec,
}

#[derive(Debug, Serialize, Deserialize)]
struct Metadata {
    name: String,
    namespace: String,
    #[serde(default)]
    labels: std::collections::HashMap<String, String>,
    #[serde(default)]
    annotations: std::collections::HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct DeploymentSpec {
    replicas: u32,
    selector: Selector,
    template: PodTemplate,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Selector {
    match_labels: std::collections::HashMap<String, String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct PodTemplate {
    metadata: Metadata,
    spec: PodSpec,
}

#[derive(Debug, Serialize, Deserialize)]
struct PodSpec {
    containers: Vec<Container>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct Container {
    name: String,
    image: String,
    ports: Vec<ContainerPort>,
    #[serde(default)]
    env: Vec<EnvVar>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    resources: Option<Resources>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ContainerPort {
    container_port: u16,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct EnvVar {
    name: String,
    value: Option<String>,
    #[serde(rename = "valueFrom", skip_serializing_if = "Option::is_none")]
    value_from: Option<serde_yaml::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Resources {
    limits: ResourceSpec,
    requests: ResourceSpec,
}

#[derive(Debug, Serialize, Deserialize)]
struct ResourceSpec {
    cpu: String,
    memory: String,
}

fn parse_yaml_config() -> anyhow::Result<()> {
    let yaml = r#"
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-app
  namespace: production
  labels:
    app: web-app
    version: "1.0"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web-app
  template:
    metadata:
      name: web-app
      namespace: production
      labels:
        app: web-app
    spec:
      containers:
        - name: app
          image: my-registry/web-app:latest
          ports:
            - containerPort: 8080
              name: http
          env:
            - name: DATABASE_URL
              value: postgres://db:5432/app
            - name: LOG_LEVEL
              value: info
          resources:
            limits:
              cpu: "500m"
              memory: "256Mi"
            requests:
              cpu: "100m"
              memory: "128Mi"
    "#;

    let deployment: Deployment = serde_yaml::from_str(yaml)?;
    println!("{:#?}", deployment);

    // YAML に再出力
    let output = serde_yaml::to_string(&deployment)?;
    println!("{}", output);

    Ok(())
}
```

### 5.2 マルチフォーマット対応の設定ローダー

```rust
use serde::de::DeserializeOwned;
use std::path::Path;

/// ファイル拡張子に基づいて適切なフォーマットでデシリアライズ
fn load_config_auto<T: DeserializeOwned>(path: &Path) -> anyhow::Result<T> {
    let content = std::fs::read_to_string(path)?;

    match path.extension().and_then(|e| e.to_str()) {
        Some("json") => {
            serde_json::from_str(&content).map_err(Into::into)
        }
        Some("toml") => {
            toml::from_str(&content).map_err(Into::into)
        }
        Some("yaml") | Some("yml") => {
            serde_yaml::from_str(&content).map_err(Into::into)
        }
        _ => {
            // 拡張子が不明な場合は順に試行
            serde_json::from_str(&content)
                .or_else(|_| toml::from_str(&content).map_err(Into::into))
                .or_else(|_: anyhow::Error| serde_yaml::from_str(&content).map_err(Into::into))
        }
    }
}

/// 環境変数によるオーバーライド付き設定ローダー
fn load_config_with_env<T: DeserializeOwned + Serialize>(path: &Path) -> anyhow::Result<T> {
    // 1. ファイルから読み込み
    let content = std::fs::read_to_string(path)?;
    let mut value: serde_json::Value = match path.extension().and_then(|e| e.to_str()) {
        Some("toml") => {
            let toml_val: toml::Value = toml::from_str(&content)?;
            serde_json::to_value(&toml_val)?
        }
        Some("yaml") | Some("yml") => {
            let yaml_val: serde_yaml::Value = serde_yaml::from_str(&content)?;
            serde_json::to_value(&yaml_val)?
        }
        _ => serde_json::from_str(&content)?,
    };

    // 2. 環境変数でオーバーライド（APP_ プレフィックス）
    if let Some(obj) = value.as_object_mut() {
        for (key, val) in obj.iter_mut() {
            let env_key = format!("APP_{}", key.to_uppercase().replace('-', "_"));
            if let Ok(env_val) = std::env::var(&env_key) {
                // 型に応じて変換
                if val.is_number() {
                    if let Ok(n) = env_val.parse::<i64>() {
                        *val = serde_json::json!(n);
                    }
                } else if val.is_boolean() {
                    if let Ok(b) = env_val.parse::<bool>() {
                        *val = serde_json::json!(b);
                    }
                } else {
                    *val = serde_json::json!(env_val);
                }
            }
        }
    }

    // 3. 型に変換
    serde_json::from_value(value).map_err(Into::into)
}
```

### 5.3 バイナリフォーマット

```rust
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct SensorData {
    device_id: u32,
    timestamp: u64,
    temperature: f32,
    humidity: f32,
    readings: Vec<f64>,
}

fn binary_formats_comparison() -> anyhow::Result<()> {
    let data = SensorData {
        device_id: 42,
        timestamp: 1700000000,
        temperature: 23.5,
        humidity: 65.2,
        readings: vec![1.0, 2.5, 3.7, 4.2, 5.1],
    };

    // --- JSON (テキスト) ---
    let json = serde_json::to_string(&data)?;
    println!("JSON:     {} bytes", json.len());
    // ~120 bytes

    // --- bincode (Rust 最適化バイナリ) ---
    let bincode_data = bincode::serialize(&data)?;
    println!("bincode:  {} bytes", bincode_data.len());
    // ~60 bytes
    let decoded: SensorData = bincode::deserialize(&bincode_data)?;
    assert_eq!(data, decoded);

    // --- MessagePack ---
    let msgpack = rmp_serde::to_vec(&data)?;
    println!("msgpack:  {} bytes", msgpack.len());
    // ~70 bytes
    let decoded: SensorData = rmp_serde::from_slice(&msgpack)?;
    assert_eq!(data, decoded);

    // --- CBOR ---
    let mut cbor_buf = Vec::new();
    ciborium::into_writer(&data, &mut cbor_buf)?;
    println!("CBOR:     {} bytes", cbor_buf.len());
    let decoded: SensorData = ciborium::from_reader(&cbor_buf[..])?;
    assert_eq!(data, decoded);

    // --- postcard (組み込み向け) ---
    let postcard_data = postcard::to_allocvec(&data)?;
    println!("postcard: {} bytes", postcard_data.len());
    // ~40 bytes (最もコンパクト)

    Ok(())
}

// パフォーマンス目安:
// | フォーマット | サイズ | シリアライズ速度 | デシリアライズ速度 |
// |------------|--------|-----------------|-------------------|
// | JSON       | 大     | 中              | 中                |
// | bincode    | 小     | 非常に高速       | 非常に高速         |
// | MessagePack| 中     | 高速            | 高速              |
// | CBOR       | 中     | 高速            | 高速              |
// | postcard   | 最小   | 高速            | 高速              |
```

---

## 6. 高度なパターン

### 6.1 flatten と untagged の組み合わせ

```rust
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// API レスポンスの汎用エンベロープ
#[derive(Serialize, Deserialize)]
struct ApiResponse<T> {
    success: bool,
    #[serde(flatten)]
    result: ApiResult<T>,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum ApiResult<T> {
    Ok { data: T },
    Err { error: ApiError },
}

#[derive(Serialize, Deserialize)]
struct ApiError {
    code: String,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<HashMap<String, String>>,
}

// 成功時: {"success": true, "data": {...}}
// 失敗時: {"success": false, "error": {"code": "NOT_FOUND", "message": "..."}}

/// Partial Update パターン（PATCH リクエスト用）
#[derive(Serialize, Deserialize)]
struct UserUpdate {
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    email: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    age: Option<u8>,
}

impl UserUpdate {
    fn apply_to(&self, user: &mut User) {
        if let Some(ref name) = self.name {
            user.name = name.clone();
        }
        if let Some(ref email) = self.email {
            user.email = email.clone();
        }
        if let Some(age) = self.age {
            user.age = age;
        }
    }
}
```

### 6.2 型安全なビルダーパターンとの組み合わせ

```rust
use serde::{Serialize, Deserialize};

/// 設定のバリデーション付きデシリアライズ
#[derive(Debug)]
pub struct ValidatedConfig {
    pub host: String,
    pub port: u16,
    pub max_connections: u32,
    pub tls_enabled: bool,
    pub cert_path: Option<String>,
}

// 中間表現（Serde でデシリアライズ）
#[derive(Deserialize)]
struct RawConfig {
    host: String,
    port: u16,
    #[serde(default = "default_max_connections")]
    max_connections: u32,
    #[serde(default)]
    tls_enabled: bool,
    cert_path: Option<String>,
}

fn default_max_connections() -> u32 { 100 }

impl TryFrom<RawConfig> for ValidatedConfig {
    type Error = ConfigError;

    fn try_from(raw: RawConfig) -> Result<Self, Self::Error> {
        // バリデーション
        if raw.port == 0 {
            return Err(ConfigError::InvalidPort);
        }
        if raw.max_connections == 0 || raw.max_connections > 10000 {
            return Err(ConfigError::InvalidMaxConnections(raw.max_connections));
        }
        if raw.tls_enabled && raw.cert_path.is_none() {
            return Err(ConfigError::TlsWithoutCert);
        }

        Ok(ValidatedConfig {
            host: raw.host,
            port: raw.port,
            max_connections: raw.max_connections,
            tls_enabled: raw.tls_enabled,
            cert_path: raw.cert_path,
        })
    }
}

#[derive(Debug, thiserror::Error)]
enum ConfigError {
    #[error("ポート番号が無効です")]
    InvalidPort,
    #[error("最大接続数が無効です: {0}")]
    InvalidMaxConnections(u32),
    #[error("TLS が有効ですが証明書パスが指定されていません")]
    TlsWithoutCert,
}

fn load_validated_config(json: &str) -> Result<ValidatedConfig, Box<dyn std::error::Error>> {
    let raw: RawConfig = serde_json::from_str(json)?;
    let config = ValidatedConfig::try_from(raw)?;
    Ok(config)
}
```

### 6.3 #[serde(remote)] による外部型のシリアライゼーション

```rust
use serde::{Serialize, Deserialize};

// 外部クレートの型に Serialize/Deserialize を実装する
// (孤児ルールにより直接 impl できないため)

// --- 方法1: remote derive ---
mod external_crate {
    pub struct Color {
        pub r: u8,
        pub g: u8,
        pub b: u8,
    }
}

// リモート型の「シャドウ定義」
#[derive(Serialize, Deserialize)]
#[serde(remote = "external_crate::Color")]
struct ColorDef {
    r: u8,
    g: u8,
    b: u8,
}

#[derive(Serialize, Deserialize)]
struct Theme {
    name: String,
    #[serde(with = "ColorDef")]
    primary_color: external_crate::Color,
    #[serde(with = "ColorDef")]
    secondary_color: external_crate::Color,
}

// --- 方法2: ニュータイプラッパー ---
#[derive(Serialize, Deserialize)]
#[serde(transparent)]
struct ColorWrapper(#[serde(with = "ColorDef")] external_crate::Color);

// --- 方法3: From/Into による変換 ---
#[derive(Serialize, Deserialize)]
struct SerializableColor {
    r: u8,
    g: u8,
    b: u8,
}

impl From<&external_crate::Color> for SerializableColor {
    fn from(c: &external_crate::Color) -> Self {
        SerializableColor { r: c.r, g: c.g, b: c.b }
    }
}

impl From<SerializableColor> for external_crate::Color {
    fn from(c: SerializableColor) -> Self {
        external_crate::Color { r: c.r, g: c.g, b: c.b }
    }
}
```

---

## 7. パフォーマンス最適化

### 7.1 simd-json による高速パーシング

```rust
// Cargo.toml:
// [dependencies]
// simd-json = "0.13"
// serde = { version = "1", features = ["derive"] }

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct LargeDataset {
    records: Vec<DataRecord>,
}

#[derive(Debug, Deserialize)]
struct DataRecord {
    id: u64,
    name: String,
    value: f64,
}

fn parse_with_simd_json(json_bytes: &mut [u8]) -> Result<LargeDataset, simd_json::Error> {
    // 注意: simd-json はインプレースで動作するため、入力を mut で受け取る
    simd_json::from_slice(json_bytes)
}

fn parse_with_serde_json(json_str: &str) -> Result<LargeDataset, serde_json::Error> {
    serde_json::from_str(json_str)
}

// パフォーマンス比較（大きい JSON ファイル）:
// serde_json: ~100ms for 100MB
// simd-json:  ~40ms for 100MB (2-3x 高速)
//
// simd-json の制約:
// - 入力が mut &[u8] である必要がある
// - SIMD 対応 CPU が必要（ほとんどのモダン CPU で対応）
// - ストリーミングパースは未対応
```

### 7.2 パフォーマンスのベストプラクティス

```rust
use serde::{Serialize, Deserialize};

// --- 1. 不要なクローンを避ける ---

// NG: 毎回 String を確保
fn parse_many_bad(jsons: &[&str]) -> Vec<String> {
    jsons.iter()
        .filter_map(|j| {
            let v: serde_json::Value = serde_json::from_str(j).ok()?;
            v["name"].as_str().map(String::from)
        })
        .collect()
}

// OK: ゼロコピーで借用
fn parse_many_good<'a>(jsons: &'a [&'a str]) -> Vec<&'a str> {
    #[derive(Deserialize)]
    struct NameOnly<'a> {
        #[serde(borrow)]
        name: &'a str,
    }

    jsons.iter()
        .filter_map(|j| {
            let v: NameOnly = serde_json::from_str(j).ok()?;
            Some(v.name)
        })
        .collect()
}

// --- 2. 部分的なデシリアライズ ---

// NG: 全フィールドをデシリアライズ
#[derive(Deserialize)]
struct FullUser {
    id: u64,
    name: String,
    email: String,
    bio: String,
    avatar_url: String,
    settings: serde_json::Value,
    history: Vec<serde_json::Value>,
}

// OK: 必要なフィールドだけデシリアライズ
#[derive(Deserialize)]
struct UserSummary {
    id: u64,
    name: String,
    // 他のフィールドは無視される（deny_unknown_fields がなければ）
}

// --- 3. バッファの再利用 ---
fn process_stream(reader: impl std::io::BufRead) {
    // serde_json::StreamDeserializer で連続 JSON をパース
    let stream = serde_json::Deserializer::from_reader(reader)
        .into_iter::<DataRecord>();

    for result in stream {
        match result {
            Ok(record) => process_record(&record),
            Err(e) => eprintln!("パースエラー: {}", e),
        }
    }
}

// --- 4. pre-allocated String/Vec ---
fn serialize_with_capacity(records: &[DataRecord]) -> String {
    // 推定サイズを事前確保
    let estimated_size = records.len() * 100;
    let mut buf = Vec::with_capacity(estimated_size);
    serde_json::to_writer(&mut buf, records).unwrap();
    String::from_utf8(buf).unwrap()
}
```

---

## 8. 比較表

### フォーマット比較

| 特性 | JSON | TOML | YAML | bincode | MessagePack |
|---|---|---|---|---|---|
| 用途 | API通信、データ交換 | 設定ファイル | 設定、CI/CD | Rust内部通信 | 言語横断通信 |
| 人間可読性 | 中 | 高い | 高い | 不可 | 不可 |
| コメント | 不可 | 可能 (#) | 可能 (#) | - | - |
| ネスト | 自由 | 制限あり | 自由 | - | 自由 |
| 日付型 | 文字列 | ネイティブ | 文字列 | - | - |
| Rust クレート | serde_json | toml | serde_yaml | bincode | rmp-serde |
| パフォーマンス | 高速 (simd-json) | 中 | 低め | 非常に高速 | 高速 |
| データサイズ | 大 | - | 大 | 小 | 中 |
| スキーマ | なし | なし | なし | なし | なし |

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

### enum 表現形式の比較

| 形式 | 属性 | JSON 例 | 適用場面 |
|---|---|---|---|
| 外部タグ | (デフォルト) | `{"Variant": data}` | Rust 間通信 |
| 内部タグ | `#[serde(tag = "type")]` | `{"type": "variant", ...}` | REST API |
| 隣接タグ | `#[serde(tag = "t", content = "c")]` | `{"t": "variant", "c": data}` | タプルバリアント |
| タグなし | `#[serde(untagged)]` | `data` | 多態的 JSON |
| 整数表現 | `serde_repr` | `0, 1, 2` | DB/Protocol |

---

## 9. アンチパターン

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

### アンチパターン3: デシリアライズエラーの unwrap

```rust
// NG: パースエラーを無視
fn bad_parse(json: &str) -> Config {
    serde_json::from_str(json).unwrap()
    // ユーザー入力ならパニックする可能性大
}

// OK: エラーを適切に処理
fn good_parse(json: &str) -> Result<Config, AppError> {
    serde_json::from_str(json).map_err(|e| {
        AppError::InvalidConfig {
            source: e,
            input: json.to_string(),
        }
    })
}

// OK: 詳細なエラーメッセージ
fn parse_with_context(json: &str, source: &str) -> anyhow::Result<Config> {
    serde_json::from_str(json)
        .with_context(|| format!("設定ファイル '{}' のパースに失敗", source))
}
```

### アンチパターン4: flatten の過剰使用

```rust
// NG: flatten をネストして使用（パフォーマンス低下）
#[derive(Deserialize)]
struct BadNested {
    #[serde(flatten)]
    base: BaseFields,
    #[serde(flatten)]
    extra: ExtraFields,
    #[serde(flatten)]
    more: MoreFields,
    #[serde(flatten)]
    catchall: HashMap<String, Value>,
    // → 各 flatten がバッファリングを要求し、O(n^2) に近くなる
}

// OK: 明示的なフィールド定義
#[derive(Deserialize)]
struct GoodExplicit {
    // base fields
    id: u64,
    name: String,
    // extra fields
    tags: Vec<String>,
    category: String,
    // 残りだけ catchall
    #[serde(flatten)]
    extra: HashMap<String, Value>,
}
```

### アンチパターン5: シリアライゼーションに秘密情報を含める

```rust
// NG: パスワードがシリアライズされる
#[derive(Serialize, Deserialize)]
struct UserAccount {
    email: String,
    password_hash: String,    // API レスポンスに含まれてしまう!
    api_key: String,          // ログに出力されてしまう!
}

// OK: skip_serializing で機密情報を除外
#[derive(Serialize, Deserialize)]
struct SecureUserAccount {
    email: String,
    #[serde(skip_serializing)]   // 出力しない
    password_hash: String,
    #[serde(skip_serializing)]
    api_key: String,
}

// OK: 別の型で API レスポンスを定義
#[derive(Serialize)]
struct UserResponse {
    email: String,
    display_name: String,
    // 機密情報は含めない
}
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

### Q4: derive が使えない場合（手動実装が必要な場合）は?

**A:** 以下のケースでは手動実装が必要です。

```rust
// 1. 既存の型に impl できない場合 → #[serde(remote)] を使用
// 2. 複雑な条件分岐が必要な場合 → Visitor パターン
// 3. 複数の入力形式を受け入れる場合 → deserialize_any
// 4. ストリーミング処理が必要な場合 → SeqAccess/MapAccess

// ほとんどの場合、serde_with クレートで解決可能
```

### Q5: エラーメッセージを改善するには?

**A:** serde_path_to_error を使用すると、エラーが発生したフィールドのパスが分かります。

```rust
// Cargo.toml: serde_path_to_error = "0.1"

fn parse_with_path(json: &str) -> Result<Config, String> {
    let deserializer = &mut serde_json::Deserializer::from_str(json);
    serde_path_to_error::deserialize(deserializer)
        .map_err(|e| format!("パスエラー '{}': {}", e.path(), e.inner()))
    // エラー例: "パスエラー 'server.port': invalid type: string "abc", expected u16"
}
```

### Q6: 大きな JSON ファイルを効率的に処理するには?

**A:** ストリーミングデシリアライズを使用します。

```rust
use std::io::BufReader;
use std::fs::File;

fn process_large_json(path: &str) -> anyhow::Result<()> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    // JSON 配列の各要素を順次処理（全体をメモリに保持しない）
    let stream = serde_json::Deserializer::from_reader(reader)
        .into_iter::<Record>();

    let mut count = 0;
    for record in stream {
        let record = record?;
        process(&record);
        count += 1;
    }
    println!("処理レコード数: {}", count);
    Ok(())
}
```

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
| serde_with | カスタムシリアライゼーションヘルパー集 |
| RawValue | JSON をパースせずに保持。ルーティングに有用 |
| simd-json | SIMD 活用の高速 JSON パーサー |
| バイナリ | bincode, MessagePack, CBOR, postcard |
| flatten | 構造をフラット化。パフォーマンスに注意 |
| remote | 外部型への Serde 実装 |
| serde_path_to_error | エラーの発生箇所を特定 |

## 次に読むべきガイド

- [データベース](./03-database.md) — SQLx/SeaORM でのSerde連携
- [Axum](../02-async/04-axum-web.md) — JSON API の構築
- [テスト](./01-testing.md) — テストフィクスチャとしてのSerde活用

## 参考文献

1. **Serde Documentation**: https://serde.rs/
2. **serde_json crate**: https://docs.rs/serde_json/latest/serde_json/
3. **Serde Attributes Reference**: https://serde.rs/attributes.html
4. **serde_with documentation**: https://docs.rs/serde_with/
5. **simd-json documentation**: https://docs.rs/simd-json/
6. **serde_path_to_error**: https://docs.rs/serde_path_to_error/
7. **serde_repr**: https://docs.rs/serde_repr/
8. **Serde Data Model**: https://serde.rs/data-model.html
