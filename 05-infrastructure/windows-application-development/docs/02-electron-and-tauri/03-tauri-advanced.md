# Tauri 応用

> Tauri v2 のプラグインシステム、カスタムプロトコル、サイドカーバイナリ、マルチウィンドウ管理、そして capabilities によるセキュリティモデルを深く理解し、本格的なデスクトップアプリを構築する。

---

## この章で学ぶこと

1. **プラグインシステム**を活用し、再利用可能な機能モジュールを設計・実装できるようになる
2. **カスタムプロトコルとサイドカー**を使い、高度なネイティブ統合を実現できるようになる
3. **capabilities（権限モデル）**を正しく設定し、セキュアなアプリケーションを構築できるようになる

---

## 1. プラグインシステム

### 1.1 プラグインのアーキテクチャ

```
+----------------------------------------------------------+
|                   Tauri アプリケーション                    |
+----------------------------------------------------------+
|                                                          |
|  tauri::Builder::default()                               |
|    .plugin(tauri_plugin_store::init())     ← 公式        |
|    .plugin(tauri_plugin_sql::init())       ← 公式        |
|    .plugin(my_custom_plugin::init())       ← カスタム    |
|                                                          |
|  +---------------------------------------------------+   |
|  |  プラグインの内部構造                                |   |
|  |                                                   |   |
|  |  ┌──────────┐  ┌──────────┐  ┌──────────────┐   |   |
|  |  │ Rust     │  │ JS API   │  │ Capabilities │   |   |
|  |  │ Backend  │  │ Bindings │  │ (権限定義)    │   |   |
|  |  │          │  │          │  │              │   |   |
|  |  │ commands │  │ invoke() │  │ permissions  │   |   |
|  |  │ state    │  │ listen() │  │ scopes       │   |   |
|  |  │ lifecycle│  │          │  │              │   |   |
|  |  └──────────┘  └──────────┘  └──────────────┘   |   |
|  +---------------------------------------------------+   |
+----------------------------------------------------------+
```

### 1.2 主要公式プラグイン一覧

| プラグイン | 機能 | Cargo パッケージ |
|---|---|---|
| store | Key-Value ストレージ | `tauri-plugin-store` |
| sql | SQLite / MySQL / PostgreSQL | `tauri-plugin-sql` |
| fs | ファイルシステム操作 | `tauri-plugin-fs` |
| dialog | ファイルダイアログ、メッセージボックス | `tauri-plugin-dialog` |
| notification | OS 通知 | `tauri-plugin-notification` |
| clipboard-manager | クリップボード操作 | `tauri-plugin-clipboard-manager` |
| shell | 外部コマンド実行 | `tauri-plugin-shell` |
| http | HTTP クライアント | `tauri-plugin-http` |
| updater | 自動更新 | `tauri-plugin-updater` |
| log | ログ出力 | `tauri-plugin-log` |
| window-state | ウィンドウ位置・サイズの保存 | `tauri-plugin-window-state` |

### コード例 1: 公式プラグインの導入

```bash
# Rust 側のプラグイン追加
cargo add tauri-plugin-store tauri-plugin-sql tauri-plugin-dialog

# フロントエンド側の API パッケージ追加
npm install @tauri-apps/plugin-store @tauri-apps/plugin-sql @tauri-apps/plugin-dialog
```

```rust
// src-tauri/src/main.rs — プラグインの登録
fn main() {
    tauri::Builder::default()
        // Key-Value ストアプラグイン
        .plugin(tauri_plugin_store::Builder::new().build())
        // SQLite データベースプラグイン
        .plugin(
            tauri_plugin_sql::Builder::default()
                .add_migrations(
                    "sqlite:app.db",
                    vec![tauri_plugin_sql::Migration {
                        version: 1,
                        description: "初期テーブル作成",
                        sql: "CREATE TABLE IF NOT EXISTS tasks (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            title TEXT NOT NULL,
                            completed BOOLEAN DEFAULT FALSE
                        );",
                        kind: tauri_plugin_sql::MigrationKind::Up,
                    }],
                )
                .build(),
        )
        // ファイルダイアログプラグイン
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![])
        .run(tauri::generate_context!())
        .expect("Tauri アプリの起動に失敗しました");
}
```

```typescript
// src/lib/store.ts — Key-Value ストアの使用
import { Store } from '@tauri-apps/plugin-store'

// ストアの初期化（ファイルパスはアプリデータディレクトリ相対）
const store = await Store.load('settings.json')

// 値の保存
await store.set('theme', 'dark')
await store.set('language', 'ja')
await store.set('windowSize', { width: 1200, height: 800 })

// 値の取得（型パラメータで型安全に）
const theme = await store.get<string>('theme')
const size = await store.get<{ width: number; height: number }>('windowSize')

// 値の削除
await store.delete('obsoleteKey')

// ディスクに保存（明示的な保存が必要）
await store.save()
```

### コード例 2: カスタムプラグインの作成

```rust
// src-tauri/src/plugins/analytics.rs — カスタム分析プラグイン
use tauri::{
    plugin::{Builder, TauriPlugin},
    AppHandle, Manager, Runtime, Emitter,
};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

/// 分析イベントの型定義
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsEvent {
    pub name: String,
    pub properties: serde_json::Value,
    pub timestamp: u64,
}

/// プラグインの内部状態
pub struct AnalyticsState {
    events: Vec<AnalyticsEvent>,
    enabled: bool,
}

/// イベントを記録するコマンド
#[tauri::command]
async fn track_event<R: Runtime>(
    app: AppHandle<R>,
    name: String,
    properties: serde_json::Value,
) -> Result<(), String> {
    let state = app.state::<Mutex<AnalyticsState>>();
    let mut state = state.lock().map_err(|e| e.to_string())?;

    if !state.enabled {
        return Ok(());
    }

    let event = AnalyticsEvent {
        name: name.clone(),
        properties,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
    };

    state.events.push(event);

    // 100 件溜まったらフラッシュ
    if state.events.len() >= 100 {
        flush_events(&mut state.events)?;
    }

    Ok(())
}

/// 蓄積されたイベントを送信
fn flush_events(events: &mut Vec<AnalyticsEvent>) -> Result<(), String> {
    // バッチ送信のロジック（省略）
    println!("分析イベント {} 件を送信", events.len());
    events.clear();
    Ok(())
}

/// プラグインの初期化関数
pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("analytics")
        // プラグイン内部のコマンドを登録
        .invoke_handler(tauri::generate_handler![track_event])
        // プラグインの状態を管理
        .setup(|app, _api| {
            app.manage(Mutex::new(AnalyticsState {
                events: Vec::new(),
                enabled: true,
            }));
            Ok(())
        })
        .build()
}
```

```typescript
// src/lib/analytics.ts — カスタムプラグインのフロントエンド API
import { invoke } from '@tauri-apps/api/core'

// 分析イベントを記録する関数
export async function trackEvent(
  name: string,
  properties: Record<string, unknown> = {}
): Promise<void> {
  // プラグインコマンドは "plugin:プラグイン名|コマンド名" 形式で呼び出す
  await invoke('plugin:analytics|track_event', { name, properties })
}

// 使用例
trackEvent('page_view', { page: '/dashboard' })
trackEvent('button_click', { button: 'save', section: 'editor' })
```

---

## 2. カスタムプロトコル

### 2.1 プロトコルの動作フロー

```
フロントエンド                    Rust バックエンド
+------------------+             +---------------------------+
|                  |             |                           |
| <img src=        |   HTTP風    | register_assetprotocol    |
|  "asset://       | ──リクエスト→ |   _protocol("asset",     |
|   images/        |             |     |path| {              |
|   photo.jpg">   |             |       ファイルを読み込み    |
|                  |  ←レスポンス── |       バイト列を返す      |
|  [画像が表示]     |   (バイト列) |     })                    |
+------------------+             +---------------------------+
```

### コード例 3: カスタムプロトコルの実装

```rust
// src-tauri/src/main.rs — カスタムプロトコルでローカルファイルを安全に配信
use tauri::http::{Request, Response};
use std::path::PathBuf;

fn main() {
    tauri::Builder::default()
        // "asset://" プロトコルを登録
        .register_assetprotocol_handler("asset", |_ctx, request| {
            handle_asset_request(request)
        })
        .run(tauri::generate_context!())
        .expect("起動に失敗しました");
}

fn handle_asset_request(request: Request<Vec<u8>>) -> Response<Vec<u8>> {
    let uri = request.uri().to_string();
    // "asset://localhost/images/photo.jpg" → "images/photo.jpg"
    let path = uri.replace("asset://localhost/", "");

    // 許可ディレクトリの基準パス
    let base_dir = dirs::document_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("my-app-assets");

    let file_path = base_dir.join(&path);

    // パストラバーサル攻撃の防止
    match file_path.canonicalize() {
        Ok(canonical) if canonical.starts_with(&base_dir) => {
            match std::fs::read(&canonical) {
                Ok(data) => {
                    // MIME タイプの推定
                    let mime = mime_guess::from_path(&canonical)
                        .first_or_octet_stream()
                        .to_string();

                    Response::builder()
                        .status(200)
                        .header("Content-Type", mime)
                        .body(data)
                        .unwrap()
                }
                Err(_) => Response::builder()
                    .status(404)
                    .body(b"ファイルが見つかりません".to_vec())
                    .unwrap(),
            }
        }
        _ => Response::builder()
            .status(403)
            .body(b"アクセス禁止".to_vec())
            .unwrap(),
    }
}
```

---

## 3. サイドカー（外部バイナリ）

### 3.1 サイドカーの仕組み

```
+----------------------------------------------------------+
|                   Tauri アプリ                             |
+----------------------------------------------------------+
|                                                          |
|  Rust バックエンド                                        |
|  ┌────────────────────────┐                              |
|  │  Command::new_sidecar  │                              |
|  │    ("ffmpeg")          │                              |
|  │         │              │                              |
|  └─────────│──────────────┘                              |
|            ↓                                             |
|  +---------------------+                                 |
|  | binaries/           |                                 |
|  |   ffmpeg-x86_64-    |  ← OS/アーキテクチャ別バイナリ  |
|  |   pc-windows-msvc   |                                 |
|  +---------------------+                                 |
|            ↓                                             |
|  [別プロセスとして実行]                                    |
|  stdin/stdout/stderr で通信                               |
+----------------------------------------------------------+
```

### コード例 4: サイドカーの設定と使用

```json
// tauri.conf.json — サイドカーバイナリの登録
{
  "bundle": {
    "externalBin": [
      "binaries/ffmpeg"
    ]
  }
}
```

```
バイナリの配置（OS/アーキテクチャ別の命名規則）:

src-tauri/binaries/
├── ffmpeg-x86_64-pc-windows-msvc.exe    ← Windows (x64)
├── ffmpeg-aarch64-pc-windows-msvc.exe   ← Windows (ARM64)
├── ffmpeg-x86_64-apple-darwin           ← macOS (Intel)
├── ffmpeg-aarch64-apple-darwin          ← macOS (Apple Silicon)
├── ffmpeg-x86_64-unknown-linux-gnu      ← Linux (x64)
└── ffmpeg-aarch64-unknown-linux-gnu     ← Linux (ARM64)
```

```rust
// src-tauri/src/commands/media.rs — サイドカーを使った動画変換
use tauri_plugin_shell::ShellExt;
use tauri::AppHandle;

/// 動画を変換するコマンド
#[tauri::command]
async fn convert_video(
    app: AppHandle,
    input: String,
    output: String,
    format: String,
) -> Result<String, String> {
    // サイドカーバイナリをコマンドとして取得
    let shell = app.shell();

    let output_result = shell
        .sidecar("ffmpeg")
        .map_err(|e| format!("サイドカーの起動に失敗: {}", e))?
        .args([
            "-i", &input,        // 入力ファイル
            "-c:v", "libx264",   // ビデオコーデック
            "-c:a", "aac",       // オーディオコーデック
            "-f", &format,       // 出力フォーマット
            "-y",                // 上書き許可
            &output,             // 出力ファイル
        ])
        .output()
        .await
        .map_err(|e| format!("変換に失敗: {}", e))?;

    if output_result.status.success() {
        Ok(format!("変換完了: {}", output))
    } else {
        let stderr = String::from_utf8_lossy(&output_result.stderr);
        Err(format!("変換エラー: {}", stderr))
    }
}

/// サイドカーの出力をリアルタイムでストリーミング
#[tauri::command]
async fn convert_video_with_progress(
    app: AppHandle,
    input: String,
    output: String,
) -> Result<(), String> {
    use tauri::Emitter;

    let shell = app.shell();

    let (mut rx, _child) = shell
        .sidecar("ffmpeg")
        .map_err(|e| e.to_string())?
        .args(["-i", &input, "-y", &output, "-progress", "pipe:1"])
        .spawn()
        .map_err(|e| e.to_string())?;

    // 子プロセスの出力を非同期で読み取り
    while let Some(event) = rx.recv().await {
        match event {
            tauri_plugin_shell::process::CommandEvent::Stdout(line) => {
                // 進捗情報をフロントエンドにイベントとして送信
                let line_str = String::from_utf8_lossy(&line);
                let _ = app.emit("conversion-progress", line_str.to_string());
            }
            tauri_plugin_shell::process::CommandEvent::Terminated(status) => {
                let _ = app.emit("conversion-complete", status.code);
                break;
            }
            _ => {}
        }
    }

    Ok(())
}
```

---

## 4. マルチウィンドウ

### コード例 5: マルチウィンドウの管理

```rust
// src-tauri/src/commands/window.rs — ウィンドウ管理コマンド
use tauri::{AppHandle, Manager, WebviewUrl, WebviewWindowBuilder};

/// 新しいウィンドウを開くコマンド
#[tauri::command]
async fn open_window(
    app: AppHandle,
    label: String,
    title: String,
    url: String,
    width: f64,
    height: f64,
) -> Result<(), String> {
    // 既存ウィンドウがあればフォーカスして返す
    if let Some(window) = app.get_webview_window(&label) {
        window.set_focus().map_err(|e| e.to_string())?;
        return Ok(());
    }

    // 新しいウィンドウを作成
    WebviewWindowBuilder::new(
        &app,
        &label,
        WebviewUrl::App(url.into()),
    )
    .title(&title)
    .inner_size(width, height)
    .min_inner_size(400.0, 300.0)
    .build()
    .map_err(|e| format!("ウィンドウ作成に失敗: {}", e))?;

    Ok(())
}

/// ウィンドウ間でメッセージを送信するコマンド
#[tauri::command]
async fn send_to_window(
    app: AppHandle,
    target_label: String,
    event: String,
    payload: serde_json::Value,
) -> Result<(), String> {
    use tauri::Emitter;

    if let Some(window) = app.get_webview_window(&target_label) {
        window.emit(&event, payload)
            .map_err(|e| e.to_string())?;
    } else {
        return Err(format!("ウィンドウ '{}' が見つかりません", target_label));
    }

    Ok(())
}

/// 全ウィンドウの一覧を取得するコマンド
#[tauri::command]
fn list_windows(app: AppHandle) -> Vec<String> {
    app.webview_windows()
        .keys()
        .cloned()
        .collect()
}
```

```typescript
// src/lib/windows.ts — フロントエンドからのウィンドウ操作
import { invoke } from '@tauri-apps/api/core'
import { WebviewWindow } from '@tauri-apps/api/webviewWindow'

// 設定ウィンドウを開く
export async function openSettings(): Promise<void> {
  // 既存のウィンドウがあれば取得
  const existing = await WebviewWindow.getByLabel('settings')
  if (existing) {
    await existing.setFocus()
    return
  }

  // 新しいウィンドウを作成
  const settingsWindow = new WebviewWindow('settings', {
    url: '/settings',
    title: '設定',
    width: 600,
    height: 500,
    resizable: true,
    center: true,
  })

  // ウィンドウの作成完了を待つ
  settingsWindow.once('tauri://created', () => {
    console.log('設定ウィンドウを作成しました')
  })

  settingsWindow.once('tauri://error', (e) => {
    console.error('ウィンドウ作成エラー:', e)
  })
}
```

---

## 5. セキュリティ（Capabilities）

### 5.1 Capabilities モデルの概念

```
+----------------------------------------------------------+
|               Tauri v2 セキュリティモデル                   |
+----------------------------------------------------------+
|                                                          |
|  Capability (権限セット)                                  |
|  ┌────────────────────────────────────────────────────┐  |
|  │  "main-capability"                                 │  |
|  │                                                    │  |
|  │  適用対象: windows: ["main"]                       │  |
|  │                                                    │  |
|  │  Permissions (個別権限):                            │  |
|  │  ┌──────────────────┐  ┌──────────────────┐       │  |
|  │  │ fs:read          │  │ dialog:open      │       │  |
|  │  │ scope: ["$HOME"] │  │                  │       │  |
|  │  └──────────────────┘  └──────────────────┘       │  |
|  │  ┌──────────────────┐  ┌──────────────────┐       │  |
|  │  │ store:default    │  │ notification:    │       │  |
|  │  │                  │  │ default          │       │  |
|  │  └──────────────────┘  └──────────────────┘       │  |
|  └────────────────────────────────────────────────────┘  |
|                                                          |
|  原則: 最小権限 — 必要な権限のみ明示的に付与              |
+----------------------------------------------------------+
```

### 5.2 Capabilities の設定

```json
// src-tauri/capabilities/default.json — メインウィンドウの権限定義
{
  "identifier": "main-capability",
  "description": "メインウィンドウに付与する権限",
  "windows": ["main"],
  "permissions": [
    "core:default",
    "dialog:default",
    "notification:default",
    "clipboard-manager:default",
    "store:default",
    {
      "identifier": "fs:read",
      "allow": [
        { "path": "$HOME/Documents/**" },
        { "path": "$APPDATA/**" }
      ]
    },
    {
      "identifier": "fs:write",
      "allow": [
        { "path": "$APPDATA/**" }
      ]
    },
    {
      "identifier": "shell:default",
      "deny": [
        { "name": "rm" },
        { "name": "del" }
      ]
    },
    {
      "identifier": "http:default",
      "allow": [
        { "url": "https://api.example.com/**" }
      ]
    }
  ]
}
```

```json
// src-tauri/capabilities/settings.json — 設定ウィンドウの権限（制限付き）
{
  "identifier": "settings-capability",
  "description": "設定ウィンドウ用の制限された権限",
  "windows": ["settings"],
  "permissions": [
    "core:default",
    "store:default"
  ]
}
```

### 5.3 権限パスの変数一覧

| 変数 | Windows | macOS | 説明 |
|---|---|---|---|
| `$HOME` | `C:\Users\{user}` | `/Users/{user}` | ホームディレクトリ |
| `$APPDATA` | `%APPDATA%\{bundle}` | `~/Library/Application Support/{bundle}` | アプリデータ |
| `$DESKTOP` | `{HOME}\Desktop` | `~/Desktop` | デスクトップ |
| `$DOCUMENT` | `{HOME}\Documents` | `~/Documents` | ドキュメント |
| `$DOWNLOAD` | `{HOME}\Downloads` | `~/Downloads` | ダウンロード |
| `$TEMP` | `%TEMP%` | `/tmp` | 一時ディレクトリ |

---

## 6. アンチパターン

### アンチパターン 1: Capabilities で全権限を許可する

```json
// NG: 全てのプラグインにデフォルト全権限を付与
{
  "identifier": "everything",
  "windows": ["*"],
  "permissions": [
    "fs:default",
    "shell:default",
    "http:default"
  ]
}
```

```json
// OK: ウィンドウごとに必要最小限の権限のみ付与
{
  "identifier": "main-restricted",
  "windows": ["main"],
  "permissions": [
    {
      "identifier": "fs:read",
      "allow": [{ "path": "$DOCUMENT/my-app/**" }]
    },
    {
      "identifier": "http:default",
      "allow": [{ "url": "https://api.example.com/**" }]
    }
  ]
}
```

### アンチパターン 2: プラグインの状態管理で Mutex を長時間ロックする

```rust
// NG: Mutex のロックを保持したまま I/O を実行
#[tauri::command]
async fn sync_data(state: tauri::State<'_, Mutex<AppState>>) -> Result<(), String> {
    let mut state = state.lock().unwrap(); // ロック取得
    // ロックを保持したまま HTTP リクエスト → 他のコマンドがブロックされる
    let data = reqwest::get("https://api.example.com/data")
        .await.map_err(|e| e.to_string())?
        .json::<Data>().await.map_err(|e| e.to_string())?;
    state.data = data;
    Ok(())
} // ここでロック解放
```

```rust
// OK: ロックの保持時間を最小限にする
#[tauri::command]
async fn sync_data(state: tauri::State<'_, Mutex<AppState>>) -> Result<(), String> {
    // 先にデータを取得（ロック不要）
    let data = reqwest::get("https://api.example.com/data")
        .await.map_err(|e| e.to_string())?
        .json::<Data>().await.map_err(|e| e.to_string())?;

    // 状態の更新時のみ短期間ロック
    {
        let mut state = state.lock().unwrap();
        state.data = data;
    } // すぐにロック解放

    Ok(())
}
```

---

## 7. FAQ

### Q1: Tauri のプラグインを自作する場合、crate として公開すべきか？

**A:** アプリ固有のロジックであればプロジェクト内にモジュールとして配置すれば十分である。複数のプロジェクトで再利用する場合は、`tauri-plugin-*` の命名規則で crate.io に公開するか、社内の Git リポジトリで Cargo のパス依存またはGit依存として管理するのが良い。公式のプラグインテンプレート（`cargo tauri plugin init`）を使うとスキャフォールディングが容易である。

### Q2: サイドカーバイナリのサイズが大きい場合、どう対処すべきか？

**A:** 以下の方法がある。(1) 初回起動時にサーバーからダウンロードし、アプリデータディレクトリにキャッシュする、(2) UPX 等の圧縮ツールでバイナリサイズを削減する、(3) サイドカーの代わりに Rust のライブラリクレートとして直接リンクする（可能な場合）。FFmpeg のように巨大なバイナリの場合は (1) のアプローチが現実的である。

### Q3: Tauri v2 の capabilities で、動的に権限を追加できるか？

**A:** capabilities はビルド時に静的に定義されるものであり、実行時に動的に変更することはできない。ただし、ウィンドウ単位で異なる capabilities を割り当てることは可能であるため、権限の異なるウィンドウを必要に応じて開くことで実質的な動的制御が可能である。より細かい実行時制御が必要な場合は、Rust のコマンド側でアクセス制御のロジックを実装する。

---

## 8. まとめ

| トピック | キーポイント |
|---|---|
| プラグインシステム | 公式プラグイン豊富。カスタムプラグインは `Builder::new()` で作成 |
| カスタムプロトコル | `register_assetprotocol_handler` でローカルファイルを安全配信 |
| サイドカー | 外部バイナリを OS/アーキテクチャ別に同梱。stdin/stdout で通信 |
| マルチウィンドウ | `WebviewWindowBuilder` で作成。ラベルで識別・管理 |
| Capabilities | 最小権限原則。ウィンドウ単位で権限を JSON 定義 |
| 状態管理 | `Mutex` + `app.manage()` でスレッドセーフに管理。ロック時間は最小に |

---

## 次に読むべきガイド

- **[00-packaging-and-signing.md](../03-distribution/00-packaging-and-signing.md)** — Tauri アプリのパッケージングと署名
- **[01-auto-update.md](../03-distribution/01-auto-update.md)** — Tauri updater を使った自動更新

---

## 参考文献

1. Tauri, "Plugins", https://v2.tauri.app/develop/plugins/
2. Tauri, "Security — Capabilities", https://v2.tauri.app/security/capabilities/
3. Tauri, "Sidecar", https://v2.tauri.app/develop/sidecar/
4. Tauri, "Multi-Window", https://v2.tauri.app/develop/window-customization/
