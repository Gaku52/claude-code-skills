# Tauri セットアップ

> Rust をバックエンドとする軽量デスクトップアプリフレームワーク Tauri v2 の環境構築からプロジェクト作成、フロントエンド統合、コマンド定義、イベントシステムまでを習得する。

---

## この章で学ぶこと

1. **Rust 環境と Tauri CLI** をセットアップし、プロジェクトをゼロから作成できるようになる
2. **Tauri コマンド**を定義し、フロントエンドから Rust 関数を呼び出せるようになる
3. **イベントシステム**を使い、フロントエンドとバックエンド間の双方向通信を実装できるようになる
4. **状態管理とライフサイクル**を理解し、アプリの初期化・終了処理を適切に制御できるようになる
5. **開発ワークフロー**を確立し、ホットリロード・デバッグ・テストの効率的な進め方を身につける

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

### 1.3 Tauri v2 の WebView 戦略

Tauri は OS に組み込まれた WebView を利用するため、プラットフォームごとに使用される WebView エンジンが異なる。

```
プラットフォーム別 WebView エンジン:

+------------------+------------------------------+------------------------+
| プラットフォーム   | WebView エンジン              | 注意事項                |
+------------------+------------------------------+------------------------+
| Windows 10/11   | WebView2 (Chromium ベース)    | Evergreen 自動更新      |
| Windows 7/8     | WebView2 (手動インストール)    | ブートストラッパー同梱   |
| macOS            | WKWebView (WebKit)           | OS 標準搭載            |
| Linux            | WebKitGTK                    | パッケージ別途必要       |
| iOS              | WKWebView                    | Tauri v2 で対応         |
| Android          | Android WebView (Chromium)   | Tauri v2 で対応         |
+------------------+------------------------------+------------------------+
```

Windows では WebView2 Runtime が必要だが、Windows 11 にはプリインストールされている。Windows 10 ではアプリのインストーラーにブートストラッパーを同梱して自動インストールする方法が推奨される。

```rust
// Tauri の tauri.conf.json で WebView2 のインストール戦略を指定可能
// "bundle" > "windows" > "webviewInstallMode" で設定する
```

```json
{
  "bundle": {
    "windows": {
      "webviewInstallMode": {
        "type": "downloadBootstrapper"
      }
    }
  }
}
```

`webviewInstallMode` のオプション:

| モード | 説明 | バイナリサイズへの影響 |
|---|---|---|
| `skip` | WebView2 の自動インストールをスキップ | なし |
| `downloadBootstrapper` | インストーラーが自動でダウンロード | 最小 (~1.8MB 追加) |
| `embedBootstrapper` | ブートストラッパーをバイナリに埋め込み | 小 (~1.8MB 追加) |
| `offlineInstaller` | WebView2 のフルインストーラーを同梱 | 大 (~130MB 追加) |
| `fixedVersion` | 特定バージョンを同梱 | 大 (~130MB 追加) |

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

### 2.2 Windows 固有のセットアップ詳細

Windows 環境での開発には Visual Studio Build Tools が必須である。以下の手順で正しくセットアップする。

```powershell
# Visual Studio Build Tools 2022 のダウンロードとインストール
# https://visualstudio.microsoft.com/ja/visual-cpp-build-tools/ からインストーラーを取得

# winget を使ったインストール（推奨）
winget install Microsoft.VisualStudio.2022.BuildTools --silent --override "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"

# WebView2 Runtime のインストール確認
# レジストリで確認する方法
reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" /v pv

# Rust ツールチェーンが MSVC を使用していることを確認
rustup show
# 出力に "stable-x86_64-pc-windows-msvc" が表示されること

# MSVC ツールチェーンを明示的にデフォルトに設定
rustup default stable-msvc
```

```powershell
# よくあるトラブルシューティング

# エラー: "link.exe not found"
# 原因: Visual Studio Build Tools が正しくインストールされていない
# 解決: "C++ によるデスクトップ開発" ワークロードを選択して再インストール

# エラー: "LINK : fatal error LNK1181: cannot open input file 'kernel32.lib'"
# 原因: Windows SDK が不足
# 解決: Visual Studio Installer から "Windows 11 SDK" を追加

# 環境変数の手動設定（通常は不要だが、パスが通らない場合）
$env:PATH += ";C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64"
```

### 2.3 Linux 固有のセットアップ詳細

Linux ディストリビューションごとに必要なパッケージが異なる。

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y \
  libwebkit2gtk-4.1-dev \
  libgtk-3-dev \
  libayatana-appindicator3-dev \
  librsvg2-dev \
  patchelf \
  libssl-dev \
  libsoup-3.0-dev \
  javascriptcoregtk-4.1-dev

# Fedora / RHEL
sudo dnf install -y \
  webkit2gtk4.1-devel \
  gtk3-devel \
  libappindicator-gtk3-devel \
  librsvg2-devel \
  patchelf \
  openssl-devel \
  libsoup3-devel \
  javascriptcoregtk4.1-devel

# Arch Linux
sudo pacman -S --needed \
  webkit2gtk-4.1 \
  gtk3 \
  libappindicator-gtk3 \
  librsvg \
  patchelf \
  openssl \
  libsoup3

# openSUSE
sudo zypper install -y \
  webkit2gtk3-devel \
  gtk3-devel \
  libappindicator-devel \
  librsvg-devel \
  patchelf \
  libopenssl-devel
```

### 2.4 プロジェクト作成

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

### フロントエンドフレームワーク別テンプレート

Tauri は様々なフロントエンドフレームワークと組み合わせ可能である。

```bash
# React + TypeScript (Vite)
npm create tauri-app@latest my-app -- --template react-ts

# Vue + TypeScript (Vite)
npm create tauri-app@latest my-app -- --template vue-ts

# Svelte + TypeScript (Vite)
npm create tauri-app@latest my-app -- --template svelte-ts

# SolidJS + TypeScript (Vite)
npm create tauri-app@latest my-app -- --template solid-ts

# Angular (独自ビルドシステム)
npm create tauri-app@latest my-app -- --template angular

# Vanilla JavaScript (Vite)
npm create tauri-app@latest my-app -- --template vanilla

# Next.js との統合（SSG モード）
npx create-next-app@latest my-next-tauri --typescript
cd my-next-tauri
npm install @tauri-apps/cli @tauri-apps/api
npx tauri init
```

### Next.js + Tauri の統合設定

Next.js と Tauri を組み合わせる場合、SSG (Static Site Generation) モードを使用する。

```typescript
// next.config.ts — Next.js の設定を SSG モードに変更
import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  output: 'export', // 静的エクスポート（SSG モード）
  // Tauri はファイルプロトコルで配信するため、相対パスを使用
  assetPrefix: process.env.TAURI_ENV_PLATFORM ? '/' : undefined,
  images: {
    unoptimized: true, // SSG では画像最適化を無効化
  },
}

export default nextConfig
```

```json
// src-tauri/tauri.conf.json — Next.js 用の設定
{
  "build": {
    "frontendDist": "../out",
    "devUrl": "http://localhost:3000",
    "beforeDevCommand": "npm run dev",
    "beforeBuildCommand": "npm run build"
  }
}
```

### 2.5 ディレクトリ構成

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

### 3.1 設定ファイルの詳細解説

`tauri.conf.json` はアプリケーションの中核設定を管理するファイルである。各セクションの意味と使い分けを理解することが重要である。

```json
{
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
        "label": "main",
        "title": "My Tauri App",
        "width": 1200,
        "height": 800,
        "minWidth": 800,
        "minHeight": 600,
        "maxWidth": 1920,
        "maxHeight": 1080,
        "resizable": true,
        "fullscreen": false,
        "center": true,
        "decorations": true,
        "transparent": false,
        "alwaysOnTop": false,
        "visible": true,
        "skipTaskbar": false,
        "fileDropEnabled": true,
        "url": "index.html"
      }
    ],
    "security": {
      "csp": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' asset: https: data:; connect-src 'self' https://api.example.com",
      "dangerousDisableAssetCspModification": false,
      "freezePrototype": true
    },
    "trayIcon": {
      "iconPath": "icons/tray-icon.png",
      "tooltip": "My Tauri App"
    },
    "withGlobalTauri": false
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
    ],
    "resources": [
      "assets/**/*"
    ],
    "windows": {
      "certificateThumbprint": null,
      "digestAlgorithm": "sha256",
      "timestampUrl": "http://timestamp.comodoca.com",
      "wix": null,
      "nsis": {
        "installerIcon": "icons/icon.ico",
        "headerImage": "icons/nsis-header.bmp",
        "sidebarImage": "icons/nsis-sidebar.bmp",
        "installMode": "currentUser",
        "languages": ["Japanese", "English"],
        "displayLanguageSelector": true
      },
      "webviewInstallMode": {
        "type": "downloadBootstrapper"
      }
    },
    "macOS": {
      "frameworks": [],
      "minimumSystemVersion": "10.15",
      "signingIdentity": null,
      "entitlements": null
    },
    "linux": {
      "deb": {
        "depends": ["libwebkit2gtk-4.1-0", "libgtk-3-0"],
        "section": "utils"
      },
      "appimage": {
        "bundleMediaFramework": true
      }
    }
  }
}
```

### 3.2 Cargo.toml の設定

```toml
# src-tauri/Cargo.toml — Rust プロジェクトの依存関係設定
[package]
name = "my-tauri-app"
version = "0.1.0"
description = "A Tauri desktop application"
authors = ["Your Name <your@email.com>"]
license = "MIT"
repository = "https://github.com/your-org/my-tauri-app"
edition = "2021"

[build-dependencies]
tauri-build = { version = "2", features = [] }

[dependencies]
tauri = { version = "2", features = ["tray-icon"] }
tauri-plugin-shell = "2"
tauri-plugin-store = "2"
tauri-plugin-dialog = "2"
tauri-plugin-fs = "2"
tauri-plugin-notification = "2"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["full"] }
thiserror = "1"
dirs = "5"
hostname = "0.3"
log = "0.4"
env_logger = "0.10"

[features]
default = ["custom-protocol"]
custom-protocol = ["tauri/custom-protocol"]

# リリースビルドの最適化設定
[profile.release]
panic = "abort"       # パニック時にスタックアンワインドを省略（バイナリサイズ削減）
codegen-units = 1     # コンパイル単位を1に（最適化向上、ビルド時間増加）
lto = true            # Link Time Optimization を有効化
opt-level = "s"       # サイズ最適化（"z" だとさらに小さい）
strip = true          # デバッグ情報を除去
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

### 4.2 高度なコマンドパターン

#### 非同期コマンドと AppHandle の利用

```rust
// src-tauri/src/commands/advanced.rs — 高度なコマンドパターン
use tauri::{AppHandle, Manager, State};
use serde::{Deserialize, Serialize};
use std::sync::Mutex;

/// アプリケーション状態の定義
#[derive(Default)]
pub struct AppState {
    pub counter: i32,
    pub history: Vec<String>,
}

/// 状態を変更するコマンド（State を注入）
#[tauri::command]
pub fn increment_counter(state: State<'_, Mutex<AppState>>) -> Result<i32, String> {
    let mut state = state.lock().map_err(|e| e.to_string())?;
    state.counter += 1;
    state.history.push(format!("カウンター更新: {}", state.counter));
    Ok(state.counter)
}

/// Window ハンドルを使ったコマンド
#[tauri::command]
pub async fn get_window_info(app: AppHandle) -> Result<WindowInfo, String> {
    let window = app.get_webview_window("main")
        .ok_or("メインウィンドウが見つかりません")?;

    let position = window.outer_position().map_err(|e| e.to_string())?;
    let size = window.outer_size().map_err(|e| e.to_string())?;
    let is_focused = window.is_focused().map_err(|e| e.to_string())?;

    Ok(WindowInfo {
        x: position.x,
        y: position.y,
        width: size.width,
        height: size.height,
        is_focused,
    })
}

#[derive(Serialize)]
pub struct WindowInfo {
    x: i32,
    y: i32,
    width: u32,
    height: u32,
    is_focused: bool,
}

/// 複数の引数と複雑な戻り値の型を持つコマンド
#[derive(Deserialize)]
pub struct SearchQuery {
    keyword: String,
    directory: String,
    extensions: Vec<String>,
    max_results: usize,
    case_sensitive: bool,
}

#[derive(Serialize)]
pub struct SearchResult {
    path: String,
    line_number: usize,
    content: String,
    score: f64,
}

#[tauri::command]
pub async fn search_files(query: SearchQuery) -> Result<Vec<SearchResult>, String> {
    use std::fs;

    let mut results = Vec::new();
    let dir = std::path::PathBuf::from(&query.directory);

    fn walk_dir(
        dir: &std::path::Path,
        query: &SearchQuery,
        results: &mut Vec<SearchResult>,
    ) -> Result<(), String> {
        let entries = fs::read_dir(dir).map_err(|e| e.to_string())?;

        for entry in entries.flatten() {
            let path = entry.path();

            if path.is_dir() {
                walk_dir(&path, query, results)?;
            } else if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_string();
                if query.extensions.contains(&ext_str) || query.extensions.is_empty() {
                    if let Ok(content) = fs::read_to_string(&path) {
                        for (i, line) in content.lines().enumerate() {
                            let matches = if query.case_sensitive {
                                line.contains(&query.keyword)
                            } else {
                                line.to_lowercase().contains(&query.keyword.to_lowercase())
                            };

                            if matches {
                                results.push(SearchResult {
                                    path: path.display().to_string(),
                                    line_number: i + 1,
                                    content: line.to_string(),
                                    score: 1.0,
                                });

                                if results.len() >= query.max_results {
                                    return Ok(());
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    walk_dir(&dir, &query, &mut results)?;
    Ok(results)
}
```

#### main.rs での状態管理の登録

```rust
// src-tauri/src/main.rs — 状態管理を含むエントリポイント
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;

use std::sync::Mutex;
use commands::advanced::AppState;

fn main() {
    tauri::Builder::default()
        // アプリケーション状態を管理
        .manage(Mutex::new(AppState::default()))
        // セットアップ時の初期化処理
        .setup(|app| {
            // アプリデータディレクトリの作成
            let app_dir = app.path().app_data_dir()
                .expect("アプリデータディレクトリの取得に失敗");
            std::fs::create_dir_all(&app_dir)
                .expect("ディレクトリの作成に失敗");

            log::info!("アプリデータディレクトリ: {:?}", app_dir);

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::greet,
            commands::advanced::increment_counter,
            commands::advanced::get_window_info,
            commands::advanced::search_files,
            commands::file_ops::read_file,
            commands::file_ops::write_file,
            commands::file_ops::list_directory,
        ])
        .run(tauri::generate_context!())
        .expect("Tauri アプリの起動に失敗しました");
}
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

### 5.2 フロントエンドからバックエンドへのイベント送信

```typescript
// src/lib/events.ts — フロントエンドからのイベント送信
import { emit } from '@tauri-apps/api/event'

// フロントエンドからバックエンドにイベントを送信
export async function notifyBackend(eventName: string, data: unknown): Promise<void> {
  await emit(eventName, data)
}

// ユーザーアクションの通知例
export async function notifyUserAction(action: string, details: Record<string, unknown>): Promise<void> {
  await emit('user-action', {
    action,
    details,
    timestamp: Date.now(),
  })
}

// ウィンドウのライフサイクルイベントをリスン
import { listen } from '@tauri-apps/api/event'
import { getCurrentWebviewWindow } from '@tauri-apps/api/webviewWindow'

export async function setupWindowListeners(): Promise<void> {
  const appWindow = getCurrentWebviewWindow()

  // ウィンドウのフォーカス変更
  await appWindow.onFocusChanged(({ payload: focused }) => {
    console.log(`ウィンドウフォーカス: ${focused ? '取得' : '喪失'}`)
  })

  // ウィンドウの閉じるイベント（キャンセル可能）
  await appWindow.onCloseRequested(async (event) => {
    // 未保存のデータがある場合は確認ダイアログを表示
    const hasUnsavedChanges = checkUnsavedChanges()
    if (hasUnsavedChanges) {
      const confirmed = await confirm('未保存の変更があります。終了しますか？')
      if (!confirmed) {
        event.preventDefault() // 閉じるのをキャンセル
      }
    }
  })

  // ウィンドウの移動イベント
  await appWindow.onMoved(({ payload: position }) => {
    console.log(`ウィンドウ移動: (${position.x}, ${position.y})`)
  })

  // ウィンドウのリサイズイベント
  await appWindow.onResized(({ payload: size }) => {
    console.log(`ウィンドウリサイズ: ${size.width}x${size.height}`)
  })

  // ファイルドロップイベント
  await appWindow.onDragDropEvent((event) => {
    if (event.payload.type === 'drop') {
      console.log('ドロップされたファイル:', event.payload.paths)
    } else if (event.payload.type === 'hover') {
      console.log('ファイルがホバー中')
    } else if (event.payload.type === 'cancel') {
      console.log('ドラッグがキャンセルされました')
    }
  })
}

function checkUnsavedChanges(): boolean {
  // 実際のアプリでは状態管理から判定する
  return false
}
```

```rust
// src-tauri/src/events.rs — バックエンドでのイベント受信
use tauri::{AppHandle, Listener};

/// バックエンドでフロントエンドからのイベントをリスンする
pub fn setup_event_listeners(app: &AppHandle) {
    // ユーザーアクションのリスン
    let app_handle = app.clone();
    app.listen("user-action", move |event| {
        if let Some(payload) = event.payload().as_ref() {
            log::info!("ユーザーアクション受信: {}", payload);
            // 分析ログへの記録など
        }
    });

    // 設定変更のリスン
    let app_handle2 = app.clone();
    app.listen("settings-changed", move |event| {
        if let Some(payload) = event.payload().as_ref() {
            log::info!("設定変更: {}", payload);
            // 設定の反映処理
        }
    });
}
```

---

## 6. 開発ワークフロー

### 6.1 ホットリロードの動作

Tauri の開発モード (`cargo tauri dev`) では、フロントエンドとバックエンドで異なるホットリロード機構が働く。

```
開発モードの動作フロー:

+-------------------------------------------------------------------+
|  $ cargo tauri dev                                                 |
|     |                                                              |
|     +---> [beforeDevCommand] npm run dev                           |
|     |       → Vite dev server が localhost:5173 で起動              |
|     |       → HMR (Hot Module Replacement) でフロントエンド即座反映  |
|     |                                                              |
|     +---> [Rust ビルド & 起動]                                      |
|             → src-tauri/ のソースコードをコンパイル                   |
|             → Rust ファイル変更時は自動再コンパイル & 再起動          |
|             → フロントエンドは WebView が devUrl を読み込み          |
+-------------------------------------------------------------------+
```

```json
// vite.config.ts — HMR の設定（Tauri 環境向け）
{
  "server": {
    "port": 5173,
    "strictPort": true,
    "host": "localhost",
    "hmr": {
      "protocol": "ws",
      "host": "localhost",
      "port": 5173
    },
    "watch": {
      "ignored": ["**/src-tauri/**"]
    }
  }
}
```

### 6.2 デバッグ手法

```bash
# バックエンド (Rust) のログ出力を有効化
RUST_LOG=debug cargo tauri dev

# 特定のモジュールのみデバッグ
RUST_LOG=my_tauri_app::commands=debug cargo tauri dev

# WebView の DevTools を開く (開発モードでは F12 / 右クリック→検証)
# リリースビルドでも DevTools を有効にする場合:
# tauri.conf.json の "app" > "windows" で "devtools": true を設定
```

```rust
// src-tauri/src/main.rs — ログの初期化
fn main() {
    // env_logger を初期化（RUST_LOG 環境変数でレベル制御）
    env_logger::init();

    log::info!("アプリケーション開始");
    log::debug!("デバッグモード有効");

    tauri::Builder::default()
        .setup(|app| {
            log::info!("セットアップ開始");

            // デバッグビルドでは DevTools を自動で開く
            #[cfg(debug_assertions)]
            {
                if let Some(window) = app.get_webview_window("main") {
                    window.open_devtools();
                }
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("Tauri アプリの起動に失敗しました");
}
```

### 6.3 テストの書き方

```rust
// src-tauri/src/commands/file_ops.rs — ユニットテスト
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_list_directory() {
        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path();

        // テスト用ファイルを作成
        fs::write(dir_path.join("test.txt"), "hello").unwrap();
        fs::create_dir(dir_path.join("subdir")).unwrap();

        let result = list_directory(dir_path.display().to_string());
        // 注: validate_path をモック化するか、テスト用にスキップする必要がある
        // この例ではホームディレクトリ配下であることを前提とする

        match result {
            Ok(entries) => {
                assert!(entries.len() >= 2);
                assert!(entries.iter().any(|e| e.name == "test.txt" && !e.is_dir));
                assert!(entries.iter().any(|e| e.name == "subdir" && e.is_dir));
            }
            Err(_) => {
                // ホームディレクトリ外の tempdir ではエラーになる場合がある
            }
        }
    }

    #[test]
    fn test_greet() {
        let result = super::super::greet("テスト");
        assert!(result.contains("テスト"));
        assert!(result.contains("Tauri"));
    }
}
```

```typescript
// src/__tests__/tauri-mock.test.ts — フロントエンドのモックテスト
import { vi, describe, it, expect, beforeEach } from 'vitest'

// Tauri の invoke をモック
vi.mock('@tauri-apps/api/core', () => ({
  invoke: vi.fn(),
}))

import { invoke } from '@tauri-apps/api/core'
import { tauriApi } from '../lib/tauri'

describe('tauriApi', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('greet が正しい引数で invoke を呼ぶ', async () => {
    const mockInvoke = invoke as ReturnType<typeof vi.fn>
    mockInvoke.mockResolvedValue('こんにちは、太郎さん！')

    const result = await tauriApi.greet('太郎')

    expect(mockInvoke).toHaveBeenCalledWith('greet', { name: '太郎' })
    expect(result).toBe('こんにちは、太郎さん！')
  })

  it('readFile がエラーを正しくハンドリングする', async () => {
    const mockInvoke = invoke as ReturnType<typeof vi.fn>
    mockInvoke.mockRejectedValue('許可されていないパスです')

    await expect(tauriApi.readFile('/etc/passwd')).rejects.toBe(
      '許可されていないパスです'
    )
  })

  it('getSystemInfo が構造化データを返す', async () => {
    const mockInvoke = invoke as ReturnType<typeof vi.fn>
    mockInvoke.mockResolvedValue({
      os: 'windows',
      arch: 'x86_64',
      hostname: 'my-pc',
    })

    const info = await tauriApi.getSystemInfo()
    expect(info.os).toBe('windows')
    expect(info.arch).toBe('x86_64')
  })
})
```

---

## 7. ビルドと最適化

### 7.1 リリースビルド

```bash
# リリースビルド（全プラットフォーム向け）
cargo tauri build

# Windows 向けのみ（NSIS インストーラー）
cargo tauri build --target x86_64-pc-windows-msvc

# macOS Universal Binary（Intel + Apple Silicon）
cargo tauri build --target universal-apple-darwin

# Linux 向け（AppImage + deb）
cargo tauri build --target x86_64-unknown-linux-gnu

# デバッグ情報付きリリースビルド
cargo tauri build --debug

# バンドルタイプを指定
cargo tauri build --bundles nsis,msi
cargo tauri build --bundles deb,appimage
cargo tauri build --bundles dmg,app
```

### 7.2 バイナリサイズの最適化

```toml
# Cargo.toml — サイズ最適化の設定
[profile.release]
panic = "abort"
codegen-units = 1
lto = true
opt-level = "z"    # "s" よりさらに小さく
strip = true

# UPX による追加圧縮（Tauri の afterBuild フック）
```

```json
// tauri.conf.json — バンドルサイズを最小化する設定
{
  "bundle": {
    "resources": [],
    "windows": {
      "nsis": {
        "compression": "lzma"
      }
    }
  }
}
```

```bash
# ビルド後のバイナリサイズ確認
ls -lh src-tauri/target/release/my-tauri-app
ls -lh src-tauri/target/release/bundle/

# UPX による圧縮（オプション）
upx --best --lzma src-tauri/target/release/my-tauri-app.exe
```

---

## 8. アンチパターン

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

### アンチパターン 3: フロントエンドでイベントリスナーをリークする

```typescript
// NG: useEffect のクリーンアップなしでリスナーを登録
function BadComponent() {
  useEffect(() => {
    // リスナーが登録されるが、解除されない → メモリリーク
    listen('some-event', (event) => {
      console.log(event.payload)
    })
  }, [])
  return <div>Bad</div>
}
```

```typescript
// OK: クリーンアップ関数でリスナーを確実に解除
function GoodComponent() {
  useEffect(() => {
    let unlisten: (() => void) | undefined

    listen('some-event', (event) => {
      console.log(event.payload)
    }).then((fn) => {
      unlisten = fn
    })

    return () => {
      unlisten?.()
    }
  }, [])
  return <div>Good</div>
}
```

### アンチパターン 4: CSP を無効化する

```json
// NG: Content Security Policy を完全に無効化
{
  "app": {
    "security": {
      "csp": null
    }
  }
}
```

```json
// OK: 必要最小限の CSP を設定
{
  "app": {
    "security": {
      "csp": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' asset: https: data:; connect-src 'self' https://api.example.com"
    }
  }
}
```

---

## 9. FAQ

### Q1: Rust を知らなくても Tauri は使えるか？

**A:** 基本的なアプリであれば、Tauri が提供する JavaScript API（ファイルダイアログ、通知、クリップボードなど）だけで多くの機能を実装できる。ただし、カスタムコマンドやプラグイン開発では Rust の知識が必要になる。Rust の所有権システムやエラー処理（Result/Option）の基礎を学んでおくことを推奨する。

### Q2: Tauri v1 と v2 の大きな違いは何か？

**A:** Tauri v2 の主な変更点は以下の通り。(1) モバイル対応（iOS/Android）が追加された、(2) セキュリティモデルが `allowlist` から `capabilities` に変更された、(3) プラグインシステムが大幅に刷新された、(4) `@tauri-apps/api` のインポートパスが変更された。v1 からの移行には公式のマイグレーションガイドを参照すべきである。

### Q3: Tauri アプリのデバッグ方法は？

**A:** フロントエンドは通常の Web 開発と同じくブラウザの DevTools（F12 または右クリック→検証）が使える。Rust バックエンドは `println!` マクロでコンソール出力するか、VS Code の LLDB デバッガ拡張を使う。`RUST_LOG=debug cargo tauri dev` で詳細なログを有効化できる。

### Q4: Tauri は Electron の完全な代替になるか？

**A:** 多くのユースケースでは代替になりうるが、注意点もある。(1) WebView がプラットフォーム依存のため、ブラウザ間の互換性問題が発生する可能性がある（特に Windows の WebView2 と macOS の WebKit の差異）、(2) Node.js のエコシステム（例: native addon）に依存するアプリは移行コストが高い、(3) Rust の学習曲線がチームのスキルセットに合わない場合がある。パフォーマンスとバイナリサイズが重要な場合は Tauri が有利で、Node.js エコシステムの活用やブラウザ一貫性が重要な場合は Electron が有利である。

### Q5: Tauri でクロスコンパイルはできるか？

**A:** Tauri は原則としてネイティブコンパイル（ターゲット OS 上でビルド）を推奨している。これは WebView の依存関係が OS 固有であるためである。CI/CD では GitHub Actions のマトリックスビルドを使い、各 OS のランナーでそれぞれビルドするのが標準的なアプローチである。ただし、Rust 自体のクロスコンパイル（`cargo build --target`）は可能で、Tauri 部分を除いたライブラリのテストなどには利用できる。

### Q6: Tauri アプリのメモリ使用量を削減するにはどうすればよいか？

**A:** (1) フロントエンドの JavaScript バンドルサイズを削減する（Tree Shaking、コード分割）、(2) 大量のデータは Rust 側で処理し、フロントエンドには表示に必要な最小限のデータのみ渡す、(3) イベントリスナーを適切にクリーンアップし、メモリリークを防ぐ、(4) 画像やメディアは遅延読み込みする、(5) Rust 側で `Box`、`Arc` を活用してヒープ使用量を最適化する。

---

## 10. まとめ

| トピック | キーポイント |
|---|---|
| Tauri の利点 | 小サイズ(3-10MB)、低メモリ、OS WebView 使用 |
| 環境構築 | Rust ツールチェーン + OS 固有の開発ライブラリ |
| プロジェクト構成 | `src/`(フロントエンド) + `src-tauri/`(バックエンド) |
| コマンド | `#[tauri::command]` で Rust 関数を定義、`invoke()` で呼び出し |
| エラー処理 | `Result<T, E>` を使い、フロントエンドにエラーメッセージを返す |
| イベント | `emit()` / `listen()` で双方向の非同期通信 |
| 設定 | `tauri.conf.json` でウィンドウ、セキュリティ、バンドルを設定 |
| 状態管理 | `Mutex<T>` + `.manage()` でスレッドセーフに状態を共有 |
| 開発ワークフロー | `cargo tauri dev` でホットリロード、DevTools でデバッグ |
| テスト | Rust 側は `#[cfg(test)]`、フロントエンドは Vitest + mock |
| ビルド最適化 | `lto`、`strip`、`opt-level="z"` でバイナリサイズを削減 |

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
5. Tauri, "Configuration Reference", https://v2.tauri.app/reference/config/
6. Tauri, "Window Customization", https://v2.tauri.app/develop/window-customization/
7. Vite, "HMR Guide", https://vitejs.dev/guide/api-hmr.html
