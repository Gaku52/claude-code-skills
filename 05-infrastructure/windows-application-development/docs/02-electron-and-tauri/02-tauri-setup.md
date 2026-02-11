# Tauri セットアップ

> Rust をバックエンドとする軽量デスクトップアプリフレームワーク Tauri v2 の環境構築からプロジェクト作成、フロントエンド統合、コマンド定義、イベントシステムまでを習得する。

---

## この章で学ぶこと

1. **Rust 環境と Tauri CLI** をセットアップし、プロジェクトをゼロから作成できるようになる
2. **Tauri コマンド**を定義し、フロントエンドから Rust 関数を呼び出せるようになる
3. **イベントシステム**を使い、フロントエンドとバックエンド間の双方向通信を実装できるようになる

---

## 1. Tauri とは何か

### 1.1 Electron との比較

```
+---------------------------+    +---------------------------+
|       Electron            |    |         Tauri             |
+---------------------------+    +---------------------------+
|                           |    |                           |
|  +---------------------+ |    |  +---------------------+  |
|  | Chromium (同梱)      | |    |  | OS WebView          |  |
|  | ~150MB              | |    |  | 0MB (OS 組み込み)    |  |
|  +---------------------+ |    |  +---------------------+  |
|  | Node.js (同梱)      | |    |  | Rust バイナリ        |  |
|  | ~40MB               | |    |  | ~3-5MB              |  |
|  +---------------------+ |    |  +---------------------+  |
|                           |    |                           |
|  合計: ~200MB+            |    |  合計: ~3-10MB           |
|  メモリ: ~150MB+          |    |  メモリ: ~30-50MB        |
+---------------------------+    +---------------------------+
```

### 1.2 比較表

| 項目 | Electron | Tauri v2 |
|---|---|---|
| バックエンド言語 | JavaScript (Node.js) | Rust |
| WebView | Chromium (同梱) | OS ネイティブ (WebView2/WebKit) |
| バイナリサイズ | 150-200 MB | 3-10 MB |
| メモリ使用量 | 150-300 MB | 30-80 MB |
| 起動速度 | 1-3 秒 | 0.3-1 秒 |
| 対応 OS | Windows / macOS / Linux | Windows / macOS / Linux / iOS / Android |
| セキュリティモデル | Preload + contextBridge | Capabilities (許可リスト) |
| エコシステム成熟度 | 非常に充実 | 急成長中 |
| 学習コスト | 低（JS/TS のみ） | 中～高（Rust 学習が必要） |

---

## 2. 環境構築

### 2.1 前提条件

```
OS 別の必要ソフトウェア:

Windows:
├── Visual Studio Build Tools 2022
│   └── "C++ によるデスクトップ開発" ワークロード
├── WebView2 Runtime (Windows 11 はプリインストール済み)
└── Rust ツールチェーン

macOS:
├── Xcode Command Line Tools
│   $ xcode-select --install
└── Rust ツールチェーン

Linux:
├── 各種開発ライブラリ
│   $ sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev libayatana-appindicator3-dev
└── Rust ツールチェーン
```

### コード例 1: Rust と Tauri CLI のインストール

```bash
# Rust のインストール（rustup 経由）
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# インストール確認
rustc --version   # rustc 1.77.0 以降
cargo --version   # cargo 1.77.0 以降

# Tauri CLI のインストール（cargo 経由）
cargo install tauri-cli --version "^2.0.0"

# Node.js パッケージとして Tauri CLI を使う場合（代替）
npm install -D @tauri-apps/cli@latest
```

### 2.2 プロジェクト作成

### コード例 2: プロジェクトのスキャフォールディング

```bash
# Tauri プロジェクトをインタラクティブに作成
# フロントエンド: React + TypeScript (Vite) を選択
cargo tauri init

# または npm 経由で作成（フロントエンドテンプレート付き）
npm create tauri-app@latest my-tauri-app -- \
  --template react-ts

# ディレクトリに移動して依存関係をインストール
cd my-tauri-app
npm install

# 開発サーバー起動
cargo tauri dev
# または
npm run tauri dev
```

### 2.3 ディレクトリ構成

```
my-tauri-app/
├── package.json                  ← フロントエンド依存関係
├── vite.config.ts                ← Vite 設定
├── tsconfig.json                 ← TypeScript 設定
│
├── src/                          ← フロントエンド (React)
│   ├── main.tsx                  ← エントリポイント
│   ├── App.tsx                   ← ルートコンポーネント
│   ├── components/               ← UI コンポーネント
│   ├── hooks/                    ← カスタムフック
│   ├── lib/                      ← ユーティリティ
│   │   └── tauri.ts              ← Tauri API ラッパー
│   └── assets/                   ← 静的リソース
│
├── src-tauri/                    ← Tauri バックエンド (Rust)
│   ├── Cargo.toml                ← Rust 依存関係
│   ├── tauri.conf.json           ← Tauri 設定ファイル
│   ├── capabilities/             ← セキュリティ許可定義
│   │   └── default.json
│   ├── src/
│   │   ├── main.rs               ← エントリポイント
│   │   ├── lib.rs                ← ライブラリルート
│   │   └── commands/             ← コマンド定義
│   │       ├── mod.rs
│   │       └── file_ops.rs
│   ├── icons/                    ← アプリアイコン
│   └── target/                   ← ビルド出力
│
└── dist/                         ← フロントエンドビルド出力
```

---

## 3. Tauri 設定ファイル

### コード例 3: tauri.conf.json

```json
{
  "$schema": "https://raw.githubusercontent.com/tauri-apps/tauri/dev/crates/tauri-utils/schema.json",
  "productName": "My Tauri App",
  "version": "0.1.0",
  "identifier": "com.example.my-tauri-app",
  "build": {
    "frontendDist": "../dist",
    "devUrl": "http://localhost:5173",
    "beforeDevCommand": "npm run dev",
    "beforeBuildCommand": "npm run build"
  },
  "app": {
    "windows": [
      {
        "title": "My Tauri App",
        "width": 1200,
        "height": 800,
        "minWidth": 800,
        "minHeight": 600,
        "resizable": true,
        "fullscreen": false
      }
    ],
    "security": {
      "csp": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"
    }
  },
  "bundle": {
    "active": true,
    "targets": "all",
    "icon": [
      "icons/32x32.png",
      "icons/128x128.png",
      "icons/128x128@2x.png",
      "icons/icon.icns",
      "icons/icon.ico"
    ]
  }
}
```

---

## 4. コマンド定義

### 4.1 コマンドの通信フロー

```
フロントエンド (TypeScript)              バックエンド (Rust)
+----------------------------+          +----------------------------+
|                            |          |                            |
|  invoke('greet', {         |  ─IPC──→ | #[tauri::command]          |
|    name: '太郎'            |          | fn greet(name: &str)       |
|  })                        |          |   -> String                |
|                            |          |                            |
|  .then(msg => {            |  ←─IPC── | return format!(            |
|    console.log(msg)        |          |   "こんにちは、{}さん！",    |
|  })                        |          |   name);                   |
+----------------------------+          +----------------------------+

  通信方式: JSON シリアライゼーション (serde)
  エラー処理: Result<T, E> → Promise<T> | catch
```

### コード例 4: コマンドの定義と呼び出し

```rust
// src-tauri/src/main.rs — Tauri アプリのエントリポイント
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;

fn main() {
    tauri::Builder::default()
        // コマンドを登録
        .invoke_handler(tauri::generate_handler![
            commands::greet,
            commands::file_ops::read_file,
            commands::file_ops::write_file,
            commands::file_ops::list_directory,
        ])
        .run(tauri::generate_context!())
        .expect("Tauri アプリの起動に失敗しました");
}
```

```rust
// src-tauri/src/commands/mod.rs — コマンドモジュール
pub mod file_ops;

use serde::{Deserialize, Serialize};

/// あいさつコマンド（シンプルな例）
#[tauri::command]
pub fn greet(name: &str) -> String {
    format!("こんにちは、{}さん！Tauri からのメッセージです。", name)
}

/// 構造化されたデータを返すコマンド
#[derive(Serialize)]
pub struct SystemInfo {
    os: String,
    arch: String,
    hostname: String,
}

#[tauri::command]
pub fn get_system_info() -> SystemInfo {
    SystemInfo {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        hostname: hostname::get()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|_| "不明".to_string()),
    }
}
```

```rust
// src-tauri/src/commands/file_ops.rs — ファイル操作コマンド
use std::fs;
use std::path::PathBuf;
use serde::Serialize;

/// エラー型の定義（フロントエンドに返すエラーメッセージ）
#[derive(Debug, thiserror::Error)]
pub enum FileError {
    #[error("ファイルの読み込みに失敗: {0}")]
    ReadError(#[from] std::io::Error),

    #[error("許可されていないパスです: {0}")]
    ForbiddenPath(String),
}

// Tauri がエラーを JSON に変換できるようにする
impl serde::Serialize for FileError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

/// ファイルを読み込むコマンド
#[tauri::command]
pub fn read_file(path: String) -> Result<String, FileError> {
    let path = PathBuf::from(&path);

    // セキュリティ: 許可されたディレクトリか確認
    validate_path(&path)?;

    let content = fs::read_to_string(&path)?;
    Ok(content)
}

/// ファイルに書き込むコマンド
#[tauri::command]
pub fn write_file(path: String, content: String) -> Result<(), FileError> {
    let path = PathBuf::from(&path);
    validate_path(&path)?;
    fs::write(&path, content)?;
    Ok(())
}

/// ディレクトリ一覧を取得するコマンド
#[derive(Serialize)]
pub struct FileEntry {
    name: String,
    is_dir: bool,
    size: u64,
}

#[tauri::command]
pub fn list_directory(path: String) -> Result<Vec<FileEntry>, FileError> {
    let path = PathBuf::from(&path);
    validate_path(&path)?;

    let entries = fs::read_dir(&path)?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let metadata = entry.metadata().ok()?;
            Some(FileEntry {
                name: entry.file_name().to_string_lossy().to_string(),
                is_dir: metadata.is_dir(),
                size: metadata.len(),
            })
        })
        .collect();

    Ok(entries)
}

/// パスの検証関数
fn validate_path(path: &PathBuf) -> Result<(), FileError> {
    let canonical = path.canonicalize()
        .map_err(|_| FileError::ForbiddenPath(path.display().to_string()))?;

    // ホームディレクトリ配下のみアクセスを許可
    let home = dirs::home_dir()
        .ok_or_else(|| FileError::ForbiddenPath("ホームディレクトリが見つかりません".into()))?;

    if !canonical.starts_with(&home) {
        return Err(FileError::ForbiddenPath(canonical.display().to_string()));
    }

    Ok(())
}
```

### フロントエンドからの呼び出し

```typescript
// src/lib/tauri.ts — Tauri コマンドの型安全なラッパー
import { invoke } from '@tauri-apps/api/core'

// コマンドの戻り値型を定義
interface SystemInfo {
  os: string
  arch: string
  hostname: string
}

interface FileEntry {
  name: string
  is_dir: boolean
  size: number
}

// 型安全な API ラッパー
export const tauriApi = {
  greet: (name: string): Promise<string> =>
    invoke('greet', { name }),

  getSystemInfo: (): Promise<SystemInfo> =>
    invoke('get_system_info'),

  readFile: (path: string): Promise<string> =>
    invoke('read_file', { path }),

  writeFile: (path: string, content: string): Promise<void> =>
    invoke('write_file', { path, content }),

  listDirectory: (path: string): Promise<FileEntry[]> =>
    invoke('list_directory', { path }),
}
```

```tsx
// src/App.tsx — React コンポーネントからコマンドを使用
import { useState } from 'react'
import { tauriApi } from './lib/tauri'

function App() {
  const [greeting, setGreeting] = useState('')
  const [name, setName] = useState('')

  // Tauri コマンドを呼び出す
  const handleGreet = async () => {
    try {
      const message = await tauriApi.greet(name)
      setGreeting(message)
    } catch (error) {
      console.error('コマンド呼び出しエラー:', error)
    }
  }

  return (
    <div className="container">
      <h1>Tauri + React</h1>
      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="名前を入力"
      />
      <button onClick={handleGreet}>あいさつ</button>
      {greeting && <p>{greeting}</p>}
    </div>
  )
}

export default App
```

---

## 5. イベントシステム

### 5.1 イベント通信の全体像

```
+------------------------------------------------------+
|                 Tauri イベントシステム                  |
+------------------------------------------------------+
|                                                      |
|  フロントエンド → バックエンド                          |
|  emit('frontend-event', payload)                     |
|       └──→ app.listen('frontend-event', handler)     |
|                                                      |
|  バックエンド → フロントエンド                          |
|  app.emit('backend-event', payload)                  |
|       └──→ listen('backend-event', callback)         |
|                                                      |
|  バックエンド → 特定ウィンドウ                          |
|  window.emit('window-event', payload)                |
|       └──→ listen('window-event', callback)          |
|                                                      |
|  フロントエンド → フロントエンド（同一ウィンドウ内）     |
|  emit('local-event', payload)                        |
|       └──→ listen('local-event', callback)           |
+------------------------------------------------------+
```

### コード例 5: イベントの送受信

```rust
// src-tauri/src/main.rs — バックエンドからのイベント送信
use tauri::{AppHandle, Manager, Emitter};
use std::time::Duration;

/// 定期的に進捗を通知するコマンド
#[tauri::command]
async fn start_long_task(app: AppHandle) -> Result<String, String> {
    let total_steps = 10;

    for step in 1..=total_steps {
        // 時間のかかる処理をシミュレート
        tokio::time::sleep(Duration::from_millis(500)).await;

        // フロントエンドに進捗イベントを送信
        app.emit("task-progress", serde_json::json!({
            "current": step,
            "total": total_steps,
            "message": format!("ステップ {}/{} を処理中...", step, total_steps),
        })).map_err(|e| e.to_string())?;
    }

    // 完了イベントを送信
    app.emit("task-complete", serde_json::json!({
        "result": "全ステップが完了しました"
    })).map_err(|e| e.to_string())?;

    Ok("タスク完了".to_string())
}
```

```typescript
// src/hooks/useTauriEvent.ts — イベント受信用のカスタムフック
import { useEffect, useState } from 'react'
import { listen, UnlistenFn } from '@tauri-apps/api/event'

// 汎用的なイベントリスナーフック
export function useTauriEvent<T>(
  eventName: string,
  handler: (payload: T) => void
): void {
  useEffect(() => {
    let unlisten: UnlistenFn | undefined

    // イベントリスナーを登録
    listen<T>(eventName, (event) => {
      handler(event.payload)
    }).then((fn) => {
      unlisten = fn
    })

    // クリーンアップ: コンポーネントのアンマウント時にリスナーを解除
    return () => {
      unlisten?.()
    }
  }, [eventName, handler])
}

// 進捗表示用の特化フック
interface Progress {
  current: number
  total: number
  message: string
}

export function useTaskProgress() {
  const [progress, setProgress] = useState<Progress | null>(null)
  const [isComplete, setIsComplete] = useState(false)

  useTauriEvent<Progress>('task-progress', (payload) => {
    setProgress(payload)
  })

  useTauriEvent<{ result: string }>('task-complete', () => {
    setIsComplete(true)
  })

  return { progress, isComplete }
}
```

```tsx
// src/components/TaskRunner.tsx — 進捗表示コンポーネント
import { useState } from 'react'
import { invoke } from '@tauri-apps/api/core'
import { useTaskProgress } from '../hooks/useTauriEvent'

export function TaskRunner() {
  const [isRunning, setIsRunning] = useState(false)
  const { progress, isComplete } = useTaskProgress()

  const handleStart = async () => {
    setIsRunning(true)
    try {
      await invoke('start_long_task')
    } catch (error) {
      console.error('タスクエラー:', error)
    } finally {
      setIsRunning(false)
    }
  }

  // 進捗率の計算
  const percentage = progress
    ? Math.round((progress.current / progress.total) * 100)
    : 0

  return (
    <div className="task-runner">
      <button onClick={handleStart} disabled={isRunning}>
        {isRunning ? '実行中...' : 'タスクを開始'}
      </button>

      {progress && (
        <div className="progress">
          <div className="progress-bar" style={{ width: `${percentage}%` }} />
          <span>{progress.message} ({percentage}%)</span>
        </div>
      )}

      {isComplete && <p>タスクが完了しました</p>}
    </div>
  )
}
```

---

## 6. アンチパターン

### アンチパターン 1: unwrap() を多用してパニックさせる

```rust
// NG: unwrap() でエラー時にプロセスがパニック（クラッシュ）する
#[tauri::command]
fn read_config() -> String {
    let content = std::fs::read_to_string("config.json").unwrap(); // パニックの可能性
    let config: serde_json::Value = serde_json::from_str(&content).unwrap();
    config["name"].as_str().unwrap().to_string()
}
```

```rust
// OK: Result 型で適切にエラーを処理し、フロントエンドに伝える
#[tauri::command]
fn read_config() -> Result<String, String> {
    let content = std::fs::read_to_string("config.json")
        .map_err(|e| format!("設定ファイルの読み込みに失敗: {}", e))?;

    let config: serde_json::Value = serde_json::from_str(&content)
        .map_err(|e| format!("JSON パースエラー: {}", e))?;

    config["name"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "設定に 'name' フィールドがありません".to_string())
}
```

### アンチパターン 2: 非同期コマンドでメインスレッドをブロックする

```rust
// NG: 同期的なファイル I/O でメインスレッドをブロック
#[tauri::command]
fn process_large_file(path: String) -> Result<String, String> {
    // 大きなファイルの読み込みで UI がフリーズ
    let data = std::fs::read_to_string(&path)
        .map_err(|e| e.to_string())?;
    // 重い処理...
    Ok(process(data))
}
```

```rust
// OK: async コマンドで非同期に実行
#[tauri::command]
async fn process_large_file(path: String) -> Result<String, String> {
    // tokio の非同期 I/O を使用（メインスレッドをブロックしない）
    let data = tokio::fs::read_to_string(&path).await
        .map_err(|e| e.to_string())?;

    // CPU バウンドの処理は spawn_blocking で別スレッドに委譲
    let result = tokio::task::spawn_blocking(move || {
        process(&data)
    }).await.map_err(|e| e.to_string())?;

    Ok(result)
}
```

---

## 7. FAQ

### Q1: Rust を知らなくても Tauri は使えるか？

**A:** 基本的なアプリであれば、Tauri が提供する JavaScript API（ファイルダイアログ、通知、クリップボードなど）だけで多くの機能を実装できる。ただし、カスタムコマンドやプラグイン開発では Rust の知識が必要になる。Rust の所有権システムやエラー処理（Result/Option）の基礎を学んでおくことを推奨する。

### Q2: Tauri v1 と v2 の大きな違いは何か？

**A:** Tauri v2 の主な変更点は以下の通り。(1) モバイル対応（iOS/Android）が追加された、(2) セキュリティモデルが `allowlist` から `capabilities` に変更された、(3) プラグインシステムが大幅に刷新された、(4) `@tauri-apps/api` のインポートパスが変更された。v1 からの移行には公式のマイグレーションガイドを参照すべきである。

### Q3: Tauri アプリのデバッグ方法は？

**A:** フロントエンドは通常の Web 開発と同じくブラウザの DevTools（F12 または右クリック→検証）が使える。Rust バックエンドは `println!` マクロでコンソール出力するか、VS Code の LLDB デバッガ拡張を使う。`RUST_LOG=debug cargo tauri dev` で詳細なログを有効化できる。

---

## 8. まとめ

| トピック | キーポイント |
|---|---|
| Tauri の利点 | 小サイズ(3-10MB)、低メモリ、OS WebView 使用 |
| 環境構築 | Rust ツールチェーン + OS 固有の開発ライブラリ |
| プロジェクト構成 | `src/`(フロントエンド) + `src-tauri/`(バックエンド) |
| コマンド | `#[tauri::command]` で Rust 関数を定義、`invoke()` で呼び出し |
| エラー処理 | `Result<T, E>` を使い、フロントエンドにエラーメッセージを返す |
| イベント | `emit()` / `listen()` で双方向の非同期通信 |
| 設定 | `tauri.conf.json` でウィンドウ、セキュリティ、バンドルを設定 |

---

## 次に読むべきガイド

- **[03-tauri-advanced.md](./03-tauri-advanced.md)** — プラグイン、カスタムプロトコル、セキュリティ設定の応用
- **[00-packaging-and-signing.md](../03-distribution/00-packaging-and-signing.md)** — Tauri アプリのパッケージングと署名

---

## 参考文献

1. Tauri, "Getting Started", https://v2.tauri.app/start/
2. Tauri, "Command Guide", https://v2.tauri.app/develop/calling-rust/
3. Tauri, "Event System", https://v2.tauri.app/develop/calling-rust/#event-system
4. The Rust Programming Language, "日本語版", https://doc.rust-jp.rs/book-ja/
