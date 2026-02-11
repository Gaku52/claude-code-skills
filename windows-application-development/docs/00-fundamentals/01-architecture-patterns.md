# アーキテクチャパターン

> デスクトップアプリのアーキテクチャはプロセス分離とIPC通信が核心。メインプロセス/レンダラーモデル、セキュアなIPC設計、preloadスクリプト、コンテキスト分離まで、安全で堅牢なアプリ設計を解説する。

## この章で学ぶこと

- [ ] メインプロセス/レンダラープロセスモデルを理解する
- [ ] IPC通信パターン（invoke/handle、send/on）を実装できる
- [ ] preloadスクリプトでセキュアなブリッジを構築できる

---

## 1. プロセスモデル

```
デスクトップアプリのプロセス構造:

  ┌─────────────────────────────────────────────┐
  │              メインプロセス                    │
  │  (Node.js / Rust バックエンド)                │
  │                                              │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
  │  │ファイルI/O │  │ネイティブ │  │ OS API   │   │
  │  │          │  │メニュー  │  │ 通知/トレイ│   │
  │  └──────────┘  └──────────┘  └──────────┘   │
  │              ▲  IPC  ▼                       │
  ├──────────────┼──────┼───────────────────────┤
  │              ▼      ▲                        │
  │         ┌───────────────┐                    │
  │         │ preload.js    │ ← ブリッジ層       │
  │         │ contextBridge │                    │
  │         └───────┬───────┘                    │
  │                 │                            │
  │  ┌──────────────▼──────────────┐             │
  │  │     レンダラープロセス        │             │
  │  │  (Chromium / WebView)       │             │
  │  │  React / Vue / Svelte       │             │
  │  │  HTML / CSS / JavaScript    │             │
  │  └─────────────────────────────┘             │
  └─────────────────────────────────────────────┘

  Electron のプロセスモデル:
    メインプロセス:   1つ（Node.js ランタイム）
    レンダラープロセス: ウィンドウごとに1つ（Chromium）
    preload:         レンダラーごとに1つ（隔離されたコンテキスト）

  Tauri のプロセスモデル:
    コアプロセス:     1つ（Rust バックエンド）
    WebView プロセス: ウィンドウごとに1つ（OS WebView）
    → Chromium を同梱しないため軽量
```

### 1.1 Electron のメインプロセス

```typescript
// main.ts — Electron メインプロセス
import { app, BrowserWindow, ipcMain } from 'electron';
import path from 'path';

let mainWindow: BrowserWindow | null = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      // セキュリティ設定
      nodeIntegration: false,      // レンダラーで Node.js 無効
      contextIsolation: true,      // コンテキスト分離有効
      sandbox: true,               // サンドボックス有効
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  // 開発時: Vite dev server
  if (process.env.NODE_ENV === 'development') {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'));
  }
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
```

### 1.2 Tauri のコアプロセス

```rust
// src-tauri/src/main.rs — Tauri コアプロセス
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! From Rust backend.", name)
}

#[tauri::command]
async fn read_file(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path)
        .map_err(|e| e.to_string())
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![greet, read_file])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

---

## 2. IPC 通信パターン

```
IPC（Inter-Process Communication）パターン:

  パターン1: Request-Response（invoke/handle）
  ┌──────────┐  invoke('get-data', args)  ┌──────────┐
  │レンダラー  │ ───────────────────────→  │メイン     │
  │          │                            │          │
  │          │  ←──────────────────────── │          │
  │          │  Promise<result>           │          │
  └──────────┘                            └──────────┘

  パターン2: Fire-and-Forget（send/on）
  ┌──────────┐  send('log', data)         ┌──────────┐
  │レンダラー  │ ───────────────────────→  │メイン     │
  │          │  （応答なし）               │          │
  └──────────┘                            └──────────┘

  パターン3: Push（メイン→レンダラー）
  ┌──────────┐                            ┌──────────┐
  │レンダラー  │  ←──────────────────────  │メイン     │
  │          │  webContents.send('event') │          │
  └──────────┘                            └──────────┘
```

### 2.1 Electron IPC 実装

```typescript
// preload.ts — セキュアなブリッジ
import { contextBridge, ipcRenderer } from 'electron';

// レンダラーに公開するAPI（ホワイトリスト方式）
contextBridge.exposeInMainWorld('electronAPI', {
  // パターン1: Request-Response
  getAppVersion: () => ipcRenderer.invoke('get-app-version'),
  readFile: (path: string) => ipcRenderer.invoke('read-file', path),
  saveFile: (path: string, data: string) =>
    ipcRenderer.invoke('save-file', path, data),

  // パターン2: Fire-and-Forget
  logEvent: (event: string) => ipcRenderer.send('log-event', event),

  // パターン3: メインからの通知を受信
  onUpdateAvailable: (callback: (version: string) => void) => {
    const handler = (_event: any, version: string) => callback(version);
    ipcRenderer.on('update-available', handler);
    // クリーンアップ関数を返す
    return () => ipcRenderer.removeListener('update-available', handler);
  },
});

// main.ts — ハンドラー登録
ipcMain.handle('get-app-version', () => {
  return app.getVersion();
});

ipcMain.handle('read-file', async (_event, filePath: string) => {
  // パス検証（セキュリティ）
  const safePath = path.resolve(filePath);
  if (!safePath.startsWith(app.getPath('documents'))) {
    throw new Error('Access denied: path outside documents');
  }
  return fs.promises.readFile(safePath, 'utf-8');
});

ipcMain.handle('save-file', async (_event, filePath: string, data: string) => {
  const safePath = path.resolve(filePath);
  if (!safePath.startsWith(app.getPath('documents'))) {
    throw new Error('Access denied');
  }
  await fs.promises.writeFile(safePath, data, 'utf-8');
  return { success: true };
});

// メイン→レンダラー通知
function notifyUpdate(version: string) {
  mainWindow?.webContents.send('update-available', version);
}
```

### 2.2 Tauri コマンド通信

```typescript
// フロントエンド（TypeScript）
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

// コマンド呼び出し（Request-Response）
const greeting = await invoke<string>('greet', { name: 'Gaku' });

// イベント受信（メイン→フロントエンド）
const unlisten = await listen<string>('file-changed', (event) => {
  console.log('File changed:', event.payload);
});

// クリーンアップ
unlisten();
```

---

## 3. セキュリティモデル

```
セキュリティ多層防御:

  レイヤー1: プロセス分離
    → レンダラーは Node.js API にアクセス不可
    → nodeIntegration: false（必須）

  レイヤー2: コンテキスト分離
    → contextIsolation: true（必須）
    → preload と Web ページは別コンテキスト

  レイヤー3: サンドボックス
    → sandbox: true
    → ファイルシステム・プロセスへの直接アクセス不可

  レイヤー4: CSP（Content Security Policy）
    → インラインスクリプト禁止
    → 外部リソース読み込み制限

  レイヤー5: API ホワイトリスト
    → contextBridge で必要な API のみ公開
    → 入力検証をメインプロセス側で実施

  Tauri のセキュリティモデル:
    → Capabilities（権限宣言）でAPI単位の許可
    → デフォルトで全API無効
    → ウィンドウ単位で権限を設定可能
```

```typescript
// Electron — CSP 設定
// main.ts
mainWindow.webContents.session.webRequest.onHeadersReceived(
  (details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [
          "default-src 'self';" +
          "script-src 'self';" +
          "style-src 'self' 'unsafe-inline';" +
          "img-src 'self' data: https:;" +
          "connect-src 'self' https://api.example.com;"
        ],
      },
    });
  }
);
```

```json
// Tauri — capabilities 設定
// src-tauri/capabilities/default.json
{
  "identifier": "default",
  "description": "Default capabilities",
  "windows": ["main"],
  "permissions": [
    "core:default",
    "dialog:allow-open",
    "dialog:allow-save",
    "fs:allow-read",
    "notification:default"
  ]
}
```

---

## 4. preload スクリプト設計

```
preload 設計原則:

  ✓ ホワイトリスト方式（必要な API のみ公開）
  ✓ 入力のサニタイズ（レンダラーからの入力を信頼しない）
  ✓ 型安全な API 定義
  ✗ ipcRenderer を直接公開しない
  ✗ require/import を公開しない
  ✗ Node.js API を直接公開しない
```

```typescript
// preload.ts — 型安全な API 設計
import { contextBridge, ipcRenderer } from 'electron';

// API の型定義
export interface ElectronAPI {
  // ファイル操作
  file: {
    open: () => Promise<{ path: string; content: string } | null>;
    save: (content: string) => Promise<boolean>;
    saveAs: (content: string) => Promise<string | null>;
  };
  // アプリ情報
  app: {
    getVersion: () => Promise<string>;
    getPlatform: () => string;
  };
  // イベント
  events: {
    onMenuAction: (callback: (action: string) => void) => () => void;
  };
}

contextBridge.exposeInMainWorld('electronAPI', {
  file: {
    open: () => ipcRenderer.invoke('file:open'),
    save: (content: string) => ipcRenderer.invoke('file:save', content),
    saveAs: (content: string) => ipcRenderer.invoke('file:saveAs', content),
  },
  app: {
    getVersion: () => ipcRenderer.invoke('app:version'),
    getPlatform: () => process.platform,
  },
  events: {
    onMenuAction: (callback: (action: string) => void) => {
      const handler = (_: any, action: string) => callback(action);
      ipcRenderer.on('menu:action', handler);
      return () => ipcRenderer.removeListener('menu:action', handler);
    },
  },
} satisfies ElectronAPI);

// renderer.d.ts — レンダラー側の型定義
declare global {
  interface Window {
    electronAPI: import('./preload').ElectronAPI;
  }
}
```

---

## 5. アンチパターン

```
よくある間違い:

  ✗ nodeIntegration: true にする
    → レンダラーから Node.js API に直接アクセス可能
    → XSS 攻撃で任意コード実行のリスク

  ✗ contextIsolation: false にする
    → preload のコンテキストが Web ページと共有
    → prototype pollution 攻撃のリスク

  ✗ ipcRenderer を丸ごと公開する
    → 任意のチャンネルにメッセージ送信可能
    → ホワイトリスト方式で制限すべき

  ✗ メインプロセスで入力検証しない
    → レンダラーからの入力は常に信頼できない
    → パストラバーサル、インジェクション攻撃のリスク
```

---

## FAQ

### Q1: Electron と Tauri のセキュリティモデルの違いは？
Electron はデフォルトで緩い設定（手動で厳格化が必要）。Tauri はデフォルトで全て無効（必要なAPIのみ capabilities で許可）。Tauri の方がセキュア・バイ・デフォルト。

### Q2: preload スクリプトは複数使えるか？
Electron では BrowserWindow ごとに1つの preload を指定。複数の機能は1つの preload 内でモジュール化して管理する。

### Q3: IPC 通信のパフォーマンスは？
invoke/handle は数百μs程度のオーバーヘッド。大量データの転送は MessagePort や SharedArrayBuffer の活用を検討。

---

## まとめ

| 概念 | ポイント |
|------|---------|
| プロセス分離 | メイン（バックエンド）とレンダラー（UI）を分離 |
| IPC | invoke/handle（Request-Response）が基本 |
| preload | contextBridge でホワイトリスト方式の API 公開 |
| セキュリティ | nodeIntegration:false + contextIsolation:true + sandbox:true |
| Tauri | Capabilities でAPI単位の権限管理 |

---

## 次に読むべきガイド
→ [[02-native-features.md]] — ネイティブ機能の活用

---

## 参考文献
1. Electron. "Security." electronjs.org/docs/tutorial/security, 2024.
2. Electron. "Context Isolation." electronjs.org/docs/tutorial/context-isolation, 2024.
3. Tauri. "Security." tauri.app/security, 2024.
