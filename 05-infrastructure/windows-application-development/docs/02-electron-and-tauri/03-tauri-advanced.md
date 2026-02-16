# Tauri 応用

> Tauri v2 のプラグインシステム、カスタムプロトコル、サイドカーバイナリ、マルチウィンドウ管理、そして capabilities によるセキュリティモデルを深く理解し、本格的なデスクトップアプリを構築する。

---

## この章で学ぶこと

1. **プラグインシステム**を活用し、再利用可能な機能モジュールを設計・実装できるようになる
2. **カスタムプロトコルとサイドカー**を使い、高度なネイティブ統合を実現できるようになる
3. **capabilities（権限モデル）**を正しく設定し、セキュアなアプリケーションを構築できるようになる
4. **システムトレイとメニュー**を実装し、OS ネイティブな操作体験を提供できるようになる
5. **データベース統合と永続化**パターンを習得し、堅牢なデータ管理を実装できるようになる

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
| global-shortcut | グローバルキーボードショートカット | `tauri-plugin-global-shortcut` |
| process | プロセス管理（終了・再起動） | `tauri-plugin-process` |
| os | OS 情報の取得 | `tauri-plugin-os` |
| deep-link | ディープリンク（カスタム URL スキーム） | `tauri-plugin-deep-link` |
| autostart | OS 起動時の自動起動 | `tauri-plugin-autostart` |

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

### 1.3 公式プラグインの実践的な活用例

#### ファイルダイアログ + ファイルシステムの組み合わせ

```typescript
// src/lib/file-manager.ts — ファイルダイアログとファイル操作の統合
import { open, save, message, confirm } from '@tauri-apps/plugin-dialog'
import { readTextFile, writeTextFile, exists, mkdir } from '@tauri-apps/plugin-fs'
import { appDataDir, join } from '@tauri-apps/api/path'

// ファイルを開くダイアログ + ファイル読み込み
export async function openTextFile(): Promise<{ path: string; content: string } | null> {
  const selected = await open({
    multiple: false,
    directory: false,
    filters: [
      { name: 'テキストファイル', extensions: ['txt', 'md', 'json'] },
      { name: 'すべてのファイル', extensions: ['*'] },
    ],
  })

  if (!selected) return null

  const path = typeof selected === 'string' ? selected : selected[0]
  const content = await readTextFile(path)
  return { path, content }
}

// 名前を付けて保存ダイアログ + ファイル書き込み
export async function saveTextFileAs(content: string): Promise<string | null> {
  const filePath = await save({
    filters: [
      { name: 'テキストファイル', extensions: ['txt'] },
      { name: 'Markdown', extensions: ['md'] },
      { name: 'JSON', extensions: ['json'] },
    ],
    defaultPath: 'untitled.txt',
  })

  if (!filePath) return null

  await writeTextFile(filePath, content)
  return filePath
}

// アプリ固有のデータディレクトリにファイルを保存
export async function saveAppData(filename: string, data: string): Promise<void> {
  const dataDir = await appDataDir()
  const filePath = await join(dataDir, filename)

  // ディレクトリが存在しない場合は作成
  const dirExists = await exists(dataDir)
  if (!dirExists) {
    await mkdir(dataDir, { recursive: true })
  }

  await writeTextFile(filePath, data)
}

// 確認ダイアログ付きの上書き保存
export async function saveWithConfirmation(path: string, content: string): Promise<boolean> {
  const fileExists = await exists(path)
  if (fileExists) {
    const shouldOverwrite = await confirm(
      `${path} は既に存在します。上書きしますか？`,
      { title: '上書き確認', kind: 'warning' }
    )
    if (!shouldOverwrite) return false
  }

  await writeTextFile(path, content)
  await message('ファイルを保存しました', { title: '保存完了', kind: 'info' })
  return true
}
```

#### グローバルショートカットの登録

```typescript
// src/lib/shortcuts.ts — グローバルショートカットの管理
import { register, unregisterAll, isRegistered } from '@tauri-apps/plugin-global-shortcut'

export async function setupGlobalShortcuts(): Promise<void> {
  // 既存のショートカットを全てクリア
  await unregisterAll()

  // Ctrl+Shift+S でクイック保存
  await register('CmdOrCtrl+Shift+S', (event) => {
    if (event.state === 'Pressed') {
      console.log('クイック保存が実行されました')
      // 保存処理を実行
    }
  })

  // Ctrl+Shift+N で新しいウィンドウ
  await register('CmdOrCtrl+Shift+N', (event) => {
    if (event.state === 'Pressed') {
      console.log('新しいウィンドウを開きます')
    }
  })

  // F5 でリフレッシュ
  await register('F5', (event) => {
    if (event.state === 'Pressed') {
      console.log('データをリフレッシュします')
    }
  })
}
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

### コード例 2b: より高度なカスタムプラグイン — 暗号化ストレージ

```rust
// src-tauri/src/plugins/encrypted_store.rs — 暗号化ストレージプラグイン
use tauri::{
    plugin::{Builder, TauriPlugin},
    AppHandle, Manager, Runtime,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use std::path::PathBuf;

/// 暗号化ストレージの内部状態
pub struct EncryptedStoreState {
    data: HashMap<String, serde_json::Value>,
    file_path: PathBuf,
    encryption_key: Vec<u8>,
    dirty: bool,
}

impl EncryptedStoreState {
    fn new(file_path: PathBuf, key: Vec<u8>) -> Self {
        Self {
            data: HashMap::new(),
            file_path,
            encryption_key: key,
            dirty: false,
        }
    }

    /// ディスクからデータを読み込み（復号化）
    fn load(&mut self) -> Result<(), String> {
        if !self.file_path.exists() {
            return Ok(());
        }

        let encrypted = std::fs::read(&self.file_path)
            .map_err(|e| format!("ファイル読み込みエラー: {}", e))?;

        // 簡易的な XOR 暗号化（本番では AES-256-GCM 等を使用すべき）
        let decrypted = self.xor_cipher(&encrypted);
        let json_str = String::from_utf8(decrypted)
            .map_err(|e| format!("UTF-8 デコードエラー: {}", e))?;

        self.data = serde_json::from_str(&json_str)
            .map_err(|e| format!("JSON パースエラー: {}", e))?;

        Ok(())
    }

    /// ディスクにデータを保存（暗号化）
    fn save(&mut self) -> Result<(), String> {
        if !self.dirty {
            return Ok(());
        }

        let json_str = serde_json::to_string(&self.data)
            .map_err(|e| format!("JSON シリアライズエラー: {}", e))?;

        let encrypted = self.xor_cipher(json_str.as_bytes());

        // 親ディレクトリを作成
        if let Some(parent) = self.file_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("ディレクトリ作成エラー: {}", e))?;
        }

        std::fs::write(&self.file_path, &encrypted)
            .map_err(|e| format!("ファイル書き込みエラー: {}", e))?;

        self.dirty = false;
        Ok(())
    }

    fn xor_cipher(&self, data: &[u8]) -> Vec<u8> {
        data.iter()
            .enumerate()
            .map(|(i, byte)| byte ^ self.encryption_key[i % self.encryption_key.len()])
            .collect()
    }
}

/// 値を取得するコマンド
#[tauri::command]
async fn encrypted_get<R: Runtime>(
    app: AppHandle<R>,
    key: String,
) -> Result<Option<serde_json::Value>, String> {
    let state = app.state::<Mutex<EncryptedStoreState>>();
    let state = state.lock().map_err(|e| e.to_string())?;
    Ok(state.data.get(&key).cloned())
}

/// 値を設定するコマンド
#[tauri::command]
async fn encrypted_set<R: Runtime>(
    app: AppHandle<R>,
    key: String,
    value: serde_json::Value,
) -> Result<(), String> {
    let state = app.state::<Mutex<EncryptedStoreState>>();
    let mut state = state.lock().map_err(|e| e.to_string())?;
    state.data.insert(key, value);
    state.dirty = true;
    state.save()?;
    Ok(())
}

/// 値を削除するコマンド
#[tauri::command]
async fn encrypted_delete<R: Runtime>(
    app: AppHandle<R>,
    key: String,
) -> Result<bool, String> {
    let state = app.state::<Mutex<EncryptedStoreState>>();
    let mut state = state.lock().map_err(|e| e.to_string())?;
    let removed = state.data.remove(&key).is_some();
    if removed {
        state.dirty = true;
        state.save()?;
    }
    Ok(removed)
}

/// プラグインの初期化
pub fn init<R: Runtime>(encryption_key: &str) -> TauriPlugin<R> {
    let key = encryption_key.as_bytes().to_vec();

    Builder::new("encrypted-store")
        .invoke_handler(tauri::generate_handler![
            encrypted_get,
            encrypted_set,
            encrypted_delete,
        ])
        .setup(move |app, _api| {
            let app_dir = app.path().app_data_dir()
                .map_err(|e| e.into())?;
            let file_path = app_dir.join("encrypted_store.bin");

            let mut store = EncryptedStoreState::new(file_path, key.clone());
            store.load().map_err(|e| {
                log::error!("暗号化ストアの読み込みに失敗: {}", e);
                Box::new(std::io::Error::new(std::io::ErrorKind::Other, e)) as Box<dyn std::error::Error>
            })?;

            app.manage(Mutex::new(store));
            Ok(())
        })
        .build()
}
```

```typescript
// src/lib/encrypted-store.ts — 暗号化ストレージのフロントエンド API
import { invoke } from '@tauri-apps/api/core'

export class EncryptedStore {
  async get<T>(key: string): Promise<T | null> {
    return invoke<T | null>('plugin:encrypted-store|encrypted_get', { key })
  }

  async set(key: string, value: unknown): Promise<void> {
    await invoke('plugin:encrypted-store|encrypted_set', { key, value })
  }

  async delete(key: string): Promise<boolean> {
    return invoke<boolean>('plugin:encrypted-store|encrypted_delete', { key })
  }
}

// 使用例
const secureStore = new EncryptedStore()
await secureStore.set('api_token', 'sk-xxxxxxxxxxxxx')
const token = await secureStore.get<string>('api_token')
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

### 2.2 カスタムプロトコルの応用例 — ストリーミング配信

```rust
// src-tauri/src/protocols/media.rs — メディアファイルのストリーミング配信
use tauri::http::{Request, Response};
use std::io::Read;

/// Range ヘッダーをパースして部分コンテンツを返す（動画ストリーミング対応）
pub fn handle_media_request(request: Request<Vec<u8>>) -> Response<Vec<u8>> {
    let uri = request.uri().to_string();
    let path = uri.replace("media://localhost/", "");
    let file_path = std::path::PathBuf::from(&path);

    // ファイルのメタデータを取得
    let metadata = match std::fs::metadata(&file_path) {
        Ok(m) => m,
        Err(_) => {
            return Response::builder()
                .status(404)
                .body(b"Not Found".to_vec())
                .unwrap();
        }
    };

    let file_size = metadata.len();
    let mime = mime_guess::from_path(&file_path)
        .first_or_octet_stream()
        .to_string();

    // Range ヘッダーの確認
    let range_header = request.headers().get("Range")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    if range_header.starts_with("bytes=") {
        // Range リクエストの処理
        let range = &range_header[6..];
        let parts: Vec<&str> = range.split('-').collect();
        let start: u64 = parts[0].parse().unwrap_or(0);
        let end: u64 = if parts.len() > 1 && !parts[1].is_empty() {
            parts[1].parse().unwrap_or(file_size - 1)
        } else {
            // チャンクサイズ: 1MB
            std::cmp::min(start + 1_048_576, file_size - 1)
        };

        let length = end - start + 1;

        // ファイルの一部を読み取り
        let mut file = std::fs::File::open(&file_path).unwrap();
        std::io::Seek::seek(&mut file, std::io::SeekFrom::Start(start)).unwrap();
        let mut buffer = vec![0u8; length as usize];
        file.read_exact(&mut buffer).unwrap_or_default();

        Response::builder()
            .status(206) // Partial Content
            .header("Content-Type", &mime)
            .header("Content-Length", length.to_string())
            .header("Content-Range", format!("bytes {}-{}/{}", start, end, file_size))
            .header("Accept-Ranges", "bytes")
            .body(buffer)
            .unwrap()
    } else {
        // 通常のリクエスト（全体を返す）
        let data = std::fs::read(&file_path).unwrap_or_default();

        Response::builder()
            .status(200)
            .header("Content-Type", &mime)
            .header("Content-Length", file_size.to_string())
            .header("Accept-Ranges", "bytes")
            .body(data)
            .unwrap()
    }
}
```

```typescript
// src/components/MediaPlayer.tsx — カスタムプロトコルを使った動画プレイヤー
import { useState } from 'react'

interface MediaPlayerProps {
  filePath: string
}

export function MediaPlayer({ filePath }: MediaPlayerProps) {
  const [isPlaying, setIsPlaying] = useState(false)

  // media:// プロトコルでローカルファイルを参照
  const mediaUrl = `media://localhost/${encodeURIComponent(filePath)}`

  return (
    <div className="media-player">
      <video
        src={mediaUrl}
        controls
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        style={{ maxWidth: '100%', maxHeight: '80vh' }}
      />
      <p>{isPlaying ? '再生中' : '停止中'}</p>
    </div>
  )
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

### 3.2 サイドカーの高度な使用パターン — Python スクリプトの実行

```rust
// src-tauri/src/commands/python_bridge.rs — Python スクリプトをサイドカーとして実行
use tauri::AppHandle;
use tauri_plugin_shell::ShellExt;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct PythonResult {
    pub success: bool,
    pub output: String,
    pub error: String,
}

/// Python スクリプトを実行するコマンド
#[tauri::command]
pub async fn run_python_script(
    app: AppHandle,
    script_name: String,
    args: Vec<String>,
) -> Result<PythonResult, String> {
    let shell = app.shell();

    // Python バイナリをサイドカーとして実行
    let mut command_args = vec![
        format!("scripts/{}", script_name),
    ];
    command_args.extend(args);

    let output = shell
        .sidecar("python")
        .map_err(|e| format!("Python の起動に失敗: {}", e))?
        .args(&command_args)
        .output()
        .await
        .map_err(|e| format!("スクリプト実行エラー: {}", e))?;

    Ok(PythonResult {
        success: output.status.success(),
        output: String::from_utf8_lossy(&output.stdout).to_string(),
        error: String::from_utf8_lossy(&output.stderr).to_string(),
    })
}

/// 長時間実行 Python プロセスの管理
#[tauri::command]
pub async fn start_python_server(
    app: AppHandle,
    port: u16,
) -> Result<u32, String> {
    use tauri::Emitter;

    let shell = app.shell();

    let (mut rx, child) = shell
        .sidecar("python")
        .map_err(|e| e.to_string())?
        .args(["scripts/server.py", "--port", &port.to_string()])
        .spawn()
        .map_err(|e| e.to_string())?;

    let pid = child.pid();

    // バックグラウンドで出力を監視
    let app_clone = app.clone();
    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                tauri_plugin_shell::process::CommandEvent::Stdout(line) => {
                    let line_str = String::from_utf8_lossy(&line);
                    let _ = app_clone.emit("python-stdout", line_str.to_string());
                }
                tauri_plugin_shell::process::CommandEvent::Stderr(line) => {
                    let line_str = String::from_utf8_lossy(&line);
                    let _ = app_clone.emit("python-stderr", line_str.to_string());
                }
                tauri_plugin_shell::process::CommandEvent::Terminated(status) => {
                    let _ = app_clone.emit("python-terminated", status.code);
                    break;
                }
                _ => {}
            }
        }
    });

    Ok(pid)
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

### 4.1 ウィンドウ間の高度な通信パターン

```typescript
// src/lib/window-communication.ts — ウィンドウ間通信マネージャー
import { emit, listen, UnlistenFn } from '@tauri-apps/api/event'
import { getCurrentWebviewWindow } from '@tauri-apps/api/webviewWindow'

interface WindowMessage {
  from: string
  to: string
  type: string
  payload: unknown
  timestamp: number
}

export class WindowCommunicator {
  private windowLabel: string
  private listeners: Map<string, UnlistenFn> = new Map()

  constructor() {
    this.windowLabel = getCurrentWebviewWindow().label
  }

  // 特定のウィンドウにメッセージを送信
  async sendTo(targetLabel: string, type: string, payload: unknown): Promise<void> {
    const message: WindowMessage = {
      from: this.windowLabel,
      to: targetLabel,
      type,
      payload,
      timestamp: Date.now(),
    }
    await emit(`window-msg:${targetLabel}`, message)
  }

  // ブロードキャスト（全ウィンドウに送信）
  async broadcast(type: string, payload: unknown): Promise<void> {
    const message: WindowMessage = {
      from: this.windowLabel,
      to: '*',
      type,
      payload,
      timestamp: Date.now(),
    }
    await emit('window-broadcast', message)
  }

  // メッセージの受信リスナーを登録
  async onMessage(handler: (message: WindowMessage) => void): Promise<void> {
    // 自分宛てのメッセージ
    const directUnlisten = await listen<WindowMessage>(
      `window-msg:${this.windowLabel}`,
      (event) => handler(event.payload)
    )
    this.listeners.set('direct', directUnlisten)

    // ブロードキャストメッセージ
    const broadcastUnlisten = await listen<WindowMessage>(
      'window-broadcast',
      (event) => {
        // 送信元が自分の場合はスキップ
        if (event.payload.from !== this.windowLabel) {
          handler(event.payload)
        }
      }
    )
    this.listeners.set('broadcast', broadcastUnlisten)
  }

  // 全リスナーを解除
  destroy(): void {
    for (const unlisten of this.listeners.values()) {
      unlisten()
    }
    this.listeners.clear()
  }
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

### 5.4 カスタムコマンドの権限定義

プラグインだけでなく、アプリ固有のコマンドにも権限を設定できる。

```json
// src-tauri/capabilities/default.json — カスタムコマンドの権限
{
  "identifier": "main-capability",
  "windows": ["main"],
  "permissions": [
    "core:default",
    {
      "identifier": "core:default",
      "allow": [
        { "cmd": "greet" },
        { "cmd": "get_system_info" },
        { "cmd": "read_file" },
        { "cmd": "write_file" },
        { "cmd": "list_directory" }
      ]
    }
  ]
}
```

---

## 6. システムトレイとメニュー

### 6.1 システムトレイの実装

```rust
// src-tauri/src/tray.rs — システムトレイの設定
use tauri::{
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    menu::{MenuBuilder, MenuItemBuilder, PredefinedMenuItem},
    Manager, Emitter,
};

pub fn setup_tray(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    // メニュー項目の作成
    let show_item = MenuItemBuilder::with_id("show", "ウィンドウを表示")
        .build(app)?;
    let hide_item = MenuItemBuilder::with_id("hide", "ウィンドウを隠す")
        .build(app)?;
    let separator = PredefinedMenuItem::separator(app)?;
    let quit_item = MenuItemBuilder::with_id("quit", "終了")
        .accelerator("CmdOrCtrl+Q")
        .build(app)?;

    // メニューの構築
    let menu = MenuBuilder::new(app)
        .item(&show_item)
        .item(&hide_item)
        .item(&separator)
        .item(&quit_item)
        .build()?;

    // トレイアイコンの構築
    let _tray = TrayIconBuilder::new()
        .icon(app.default_window_icon().unwrap().clone())
        .tooltip("My Tauri App")
        .menu(&menu)
        .on_menu_event(move |app, event| {
            match event.id().as_ref() {
                "show" => {
                    if let Some(window) = app.get_webview_window("main") {
                        let _ = window.show();
                        let _ = window.set_focus();
                    }
                }
                "hide" => {
                    if let Some(window) = app.get_webview_window("main") {
                        let _ = window.hide();
                    }
                }
                "quit" => {
                    app.exit(0);
                }
                _ => {}
            }
        })
        .on_tray_icon_event(|tray, event| {
            match event {
                TrayIconEvent::Click {
                    button: MouseButton::Left,
                    button_state: MouseButtonState::Up,
                    ..
                } => {
                    let app = tray.app_handle();
                    if let Some(window) = app.get_webview_window("main") {
                        let _ = window.show();
                        let _ = window.unminimize();
                        let _ = window.set_focus();
                    }
                }
                TrayIconEvent::DoubleClick {
                    button: MouseButton::Left,
                    ..
                } => {
                    let app = tray.app_handle();
                    if let Some(window) = app.get_webview_window("main") {
                        let visible = window.is_visible().unwrap_or(false);
                        if visible {
                            let _ = window.hide();
                        } else {
                            let _ = window.show();
                            let _ = window.set_focus();
                        }
                    }
                }
                _ => {}
            }
        })
        .build(app)?;

    Ok(())
}
```

### 6.2 アプリケーションメニューの設定

```rust
// src-tauri/src/menu.rs — アプリケーションメニューバーの構築
use tauri::{
    menu::{MenuBuilder, SubmenuBuilder, MenuItemBuilder, PredefinedMenuItem, CheckMenuItemBuilder},
    Manager, Emitter,
};

pub fn setup_menu(app: &tauri::App) -> Result<(), Box<dyn std::error::Error>> {
    // ファイルメニュー
    let new_item = MenuItemBuilder::with_id("file-new", "新規作成")
        .accelerator("CmdOrCtrl+N")
        .build(app)?;
    let open_item = MenuItemBuilder::with_id("file-open", "開く...")
        .accelerator("CmdOrCtrl+O")
        .build(app)?;
    let save_item = MenuItemBuilder::with_id("file-save", "保存")
        .accelerator("CmdOrCtrl+S")
        .build(app)?;
    let save_as_item = MenuItemBuilder::with_id("file-save-as", "名前を付けて保存...")
        .accelerator("CmdOrCtrl+Shift+S")
        .build(app)?;

    let file_menu = SubmenuBuilder::new(app, "ファイル")
        .item(&new_item)
        .item(&open_item)
        .separator()
        .item(&save_item)
        .item(&save_as_item)
        .separator()
        .quit()
        .build()?;

    // 編集メニュー
    let edit_menu = SubmenuBuilder::new(app, "編集")
        .undo()
        .redo()
        .separator()
        .cut()
        .copy()
        .paste()
        .select_all()
        .build()?;

    // 表示メニュー
    let dark_mode = CheckMenuItemBuilder::with_id("view-dark-mode", "ダークモード")
        .checked(false)
        .build(app)?;

    let view_menu = SubmenuBuilder::new(app, "表示")
        .item(&dark_mode)
        .separator()
        .fullscreen()
        .build()?;

    // メニューバーの構築
    let menu = MenuBuilder::new(app)
        .item(&file_menu)
        .item(&edit_menu)
        .item(&view_menu)
        .build()?;

    app.set_menu(menu)?;

    // メニューイベントのハンドリング
    app.on_menu_event(move |app, event| {
        match event.id().as_ref() {
            "file-new" => {
                let _ = app.emit("menu-action", "new");
            }
            "file-open" => {
                let _ = app.emit("menu-action", "open");
            }
            "file-save" => {
                let _ = app.emit("menu-action", "save");
            }
            "file-save-as" => {
                let _ = app.emit("menu-action", "save-as");
            }
            "view-dark-mode" => {
                let _ = app.emit("menu-action", "toggle-dark-mode");
            }
            _ => {}
        }
    });

    Ok(())
}
```

```typescript
// src/hooks/useMenuActions.ts — メニューアクションのリスナー
import { useEffect } from 'react'
import { listen } from '@tauri-apps/api/event'

export function useMenuActions(handlers: {
  onNew?: () => void
  onOpen?: () => void
  onSave?: () => void
  onSaveAs?: () => void
  onToggleDarkMode?: () => void
}) {
  useEffect(() => {
    const unlisten = listen<string>('menu-action', (event) => {
      switch (event.payload) {
        case 'new':
          handlers.onNew?.()
          break
        case 'open':
          handlers.onOpen?.()
          break
        case 'save':
          handlers.onSave?.()
          break
        case 'save-as':
          handlers.onSaveAs?.()
          break
        case 'toggle-dark-mode':
          handlers.onToggleDarkMode?.()
          break
      }
    })

    return () => {
      unlisten.then((fn) => fn())
    }
  }, [handlers])
}
```

---

## 7. データベース統合

### 7.1 SQLite を使った CRUD 操作

```typescript
// src/lib/database.ts — SQLite データベースの操作
import Database from '@tauri-apps/plugin-sql'

// データベース接続（アプリデータディレクトリに保存）
let db: Database | null = null

export async function getDb(): Promise<Database> {
  if (!db) {
    db = await Database.load('sqlite:app.db')
  }
  return db
}

// タスク型の定義
interface Task {
  id: number
  title: string
  description: string
  completed: boolean
  created_at: string
  updated_at: string
}

// CRUD 操作
export const taskRepository = {
  // 全タスクの取得
  async findAll(): Promise<Task[]> {
    const db = await getDb()
    return db.select<Task[]>('SELECT * FROM tasks ORDER BY created_at DESC')
  },

  // ID によるタスク取得
  async findById(id: number): Promise<Task | null> {
    const db = await getDb()
    const results = await db.select<Task[]>('SELECT * FROM tasks WHERE id = $1', [id])
    return results[0] || null
  },

  // タスクの作成
  async create(title: string, description: string): Promise<number> {
    const db = await getDb()
    const result = await db.execute(
      'INSERT INTO tasks (title, description, completed, created_at, updated_at) VALUES ($1, $2, 0, datetime("now"), datetime("now"))',
      [title, description]
    )
    return result.lastInsertId
  },

  // タスクの更新
  async update(id: number, updates: Partial<Task>): Promise<void> {
    const db = await getDb()
    const fields: string[] = []
    const values: unknown[] = []
    let paramIndex = 1

    if (updates.title !== undefined) {
      fields.push(`title = $${paramIndex++}`)
      values.push(updates.title)
    }
    if (updates.description !== undefined) {
      fields.push(`description = $${paramIndex++}`)
      values.push(updates.description)
    }
    if (updates.completed !== undefined) {
      fields.push(`completed = $${paramIndex++}`)
      values.push(updates.completed ? 1 : 0)
    }

    fields.push(`updated_at = datetime("now")`)
    values.push(id)

    await db.execute(
      `UPDATE tasks SET ${fields.join(', ')} WHERE id = $${paramIndex}`,
      values
    )
  },

  // タスクの削除
  async delete(id: number): Promise<void> {
    const db = await getDb()
    await db.execute('DELETE FROM tasks WHERE id = $1', [id])
  },

  // 検索
  async search(keyword: string): Promise<Task[]> {
    const db = await getDb()
    return db.select<Task[]>(
      'SELECT * FROM tasks WHERE title LIKE $1 OR description LIKE $1 ORDER BY created_at DESC',
      [`%${keyword}%`]
    )
  },
}
```

### 7.2 Rust 側での高度なデータベース操作

```rust
// src-tauri/src/database.rs — Rust 側での SQLite 操作（rusqlite 使用）
use rusqlite::{Connection, params};
use serde::Serialize;
use std::sync::Mutex;

pub struct Database {
    conn: Connection,
}

#[derive(Serialize)]
pub struct TaskStats {
    total: i64,
    completed: i64,
    pending: i64,
    completion_rate: f64,
}

impl Database {
    pub fn new(path: &str) -> Result<Self, rusqlite::Error> {
        let conn = Connection::open(path)?;

        // WAL モードを有効化（パフォーマンス向上）
        conn.execute_batch("
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA foreign_keys=ON;
        ")?;

        // テーブル作成
        conn.execute_batch("
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT DEFAULT '',
                completed BOOLEAN DEFAULT 0,
                priority INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_tasks_completed ON tasks(completed);
            CREATE INDEX IF NOT EXISTS idx_tasks_priority ON tasks(priority);
        ")?;

        Ok(Self { conn })
    }

    pub fn get_stats(&self) -> Result<TaskStats, rusqlite::Error> {
        let total: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM tasks",
            [],
            |row| row.get(0),
        )?;

        let completed: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM tasks WHERE completed = 1",
            [],
            |row| row.get(0),
        )?;

        let pending = total - completed;
        let completion_rate = if total > 0 {
            (completed as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        Ok(TaskStats {
            total,
            completed,
            pending,
            completion_rate,
        })
    }
}

/// データベースのコマンド
#[tauri::command]
pub fn get_task_stats(
    db: tauri::State<'_, Mutex<Database>>,
) -> Result<TaskStats, String> {
    let db = db.lock().map_err(|e| e.to_string())?;
    db.get_stats().map_err(|e| e.to_string())
}
```

---

## 8. アンチパターン

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

### アンチパターン 3: サイドカーのプロセスをリークさせる

```rust
// NG: サイドカープロセスを起動して放置（ゾンビプロセス化）
#[tauri::command]
async fn run_background_process(app: AppHandle) -> Result<(), String> {
    let shell = app.shell();
    let (rx, child) = shell
        .sidecar("worker")
        .map_err(|e| e.to_string())?
        .spawn()
        .map_err(|e| e.to_string())?;

    // child を保持せず、rx も読み取らない → プロセスがリーク
    Ok(())
}
```

```rust
// OK: プロセスのライフサイクルを適切に管理
#[tauri::command]
async fn run_background_process(
    app: AppHandle,
    state: tauri::State<'_, Mutex<ProcessManager>>,
) -> Result<u32, String> {
    let shell = app.shell();
    let (mut rx, child) = shell
        .sidecar("worker")
        .map_err(|e| e.to_string())?
        .spawn()
        .map_err(|e| e.to_string())?;

    let pid = child.pid();

    // プロセスを状態管理に登録
    {
        let mut manager = state.lock().map_err(|e| e.to_string())?;
        manager.register(pid, child);
    }

    // 出力を監視し、終了時にクリーンアップ
    let state_clone = state.inner().clone();
    tokio::spawn(async move {
        while let Some(event) = rx.recv().await {
            if let tauri_plugin_shell::process::CommandEvent::Terminated(_) = event {
                if let Ok(mut manager) = state_clone.lock() {
                    manager.unregister(pid);
                }
                break;
            }
        }
    });

    Ok(pid)
}
```

---

## 9. FAQ

### Q1: Tauri のプラグインを自作する場合、crate として公開すべきか？

**A:** アプリ固有のロジックであればプロジェクト内にモジュールとして配置すれば十分である。複数のプロジェクトで再利用する場合は、`tauri-plugin-*` の命名規則で crate.io に公開するか、社内の Git リポジトリで Cargo のパス依存またはGit依存として管理するのが良い。公式のプラグインテンプレート（`cargo tauri plugin init`）を使うとスキャフォールディングが容易である。

### Q2: サイドカーバイナリのサイズが大きい場合、どう対処すべきか？

**A:** 以下の方法がある。(1) 初回起動時にサーバーからダウンロードし、アプリデータディレクトリにキャッシュする、(2) UPX 等の圧縮ツールでバイナリサイズを削減する、(3) サイドカーの代わりに Rust のライブラリクレートとして直接リンクする（可能な場合）。FFmpeg のように巨大なバイナリの場合は (1) のアプローチが現実的である。

### Q3: Tauri v2 の capabilities で、動的に権限を追加できるか？

**A:** capabilities はビルド時に静的に定義されるものであり、実行時に動的に変更することはできない。ただし、ウィンドウ単位で異なる capabilities を割り当てることは可能であるため、権限の異なるウィンドウを必要に応じて開くことで実質的な動的制御が可能である。より細かい実行時制御が必要な場合は、Rust のコマンド側でアクセス制御のロジックを実装する。

### Q4: マルチウィンドウアプリで状態を同期するにはどうすればよいか？

**A:** Tauri のイベントシステムを使うのが最も簡単である。状態が変更されたら全ウィンドウにブロードキャストイベントを送信し、各ウィンドウのフロントエンドが自身の表示を更新する。また、Rust バックエンドの `Mutex<AppState>` を中央の単一ソースとして使い、全ウィンドウからコマンド経由でアクセスする方法も有効である。大規模アプリでは、Redux のような一方向データフローパターンを採用し、バックエンドをストアとして利用することを推奨する。

### Q5: Tauri アプリで OS のネイティブ通知を表示するには？

**A:** `tauri-plugin-notification` を使用する。`npm install @tauri-apps/plugin-notification` でインストールし、capabilities に `notification:default` を追加する。フロントエンドでは `sendNotification({ title: 'タイトル', body: '本文' })` で簡単に通知を送信できる。Windows ではトースト通知、macOS では通知センター、Linux では libnotify を使用する。権限の要求はプラグインが自動的に処理する。

---

## 10. まとめ

| トピック | キーポイント |
|---|---|
| プラグインシステム | 公式プラグイン豊富。カスタムプラグインは `Builder::new()` で作成 |
| カスタムプロトコル | `register_assetprotocol_handler` でローカルファイルを安全配信 |
| サイドカー | 外部バイナリを OS/アーキテクチャ別に同梱。stdin/stdout で通信 |
| マルチウィンドウ | `WebviewWindowBuilder` で作成。ラベルで識別・管理 |
| Capabilities | 最小権限原則。ウィンドウ単位で権限を JSON 定義 |
| 状態管理 | `Mutex` + `app.manage()` でスレッドセーフに管理。ロック時間は最小に |
| システムトレイ | `TrayIconBuilder` でトレイアイコンとメニューを構築 |
| メニュー | `MenuBuilder` でネイティブメニューバーを構築 |
| データベース | `tauri-plugin-sql` で SQLite 統合。Rust 側では rusqlite も選択肢 |

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
5. Tauri, "System Tray", https://v2.tauri.app/develop/system-tray/
6. Tauri, "Menu", https://v2.tauri.app/develop/menu/
7. rusqlite Documentation, https://docs.rs/rusqlite/
