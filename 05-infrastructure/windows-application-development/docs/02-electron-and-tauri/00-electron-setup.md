# Electron セットアップ

> Vite + React + TypeScript 構成で Electron デスクトップアプリケーションの開発環境を構築し、ホットリロード・DevTools 統合まで完了させる。

---

## この章で学ぶこと

1. **Electron のアーキテクチャ**（Main / Renderer / Preload）を理解し、プロジェクトを正しく構成できるようになる
2. **Vite + React + TypeScript** を使った現代的な開発環境をゼロから構築できるようになる
3. **ホットリロードと DevTools** を活用した効率的な開発ワークフローを確立する

---

## 1. Electron のアーキテクチャ

### 1.1 プロセスモデル

```
+----------------------------------------------------------+
|                    Electron アプリ                         |
+----------------------------------------------------------+
|                                                          |
|  +------------------------+                              |
|  |    Main Process        |  ← Node.js ランタイム        |
|  |    (main.ts)           |                              |
|  |                        |                              |
|  |  - BrowserWindow 管理  |                              |
|  |  - システム API        |                              |
|  |  - メニュー/トレイ     |                              |
|  |  - IPC ハンドラ        |                              |
|  +--------+---+-----------+                              |
|           |   |                                          |
|     IPC   |   |  IPC                                     |
|           |   |                                          |
|  +--------v---+--------+   +-------------------------+   |
|  |  Renderer Process   |   |  Renderer Process       |   |
|  |  (ウィンドウ 1)     |   |  (ウィンドウ 2)         |   |
|  |                     |   |                         |   |
|  |  +---------------+  |   |  +------------------+   |   |
|  |  | Preload       |  |   |  | Preload          |   |   |
|  |  | (preload.ts)  |  |   |  | (preload.ts)     |   |   |
|  |  +-------+-------+  |   |  +--------+---------+   |   |
|  |          |           |   |           |             |   |
|  |  +-------v-------+  |   |  +--------v---------+   |   |
|  |  | Web ページ     |  |   |  | Web ページ        |   |   |
|  |  | (React App)   |  |   |  | (React App)      |   |   |
|  |  +---------------+  |   |  +------------------+   |   |
|  +---------------------+   +-------------------------+   |
+----------------------------------------------------------+
```

### 1.2 各プロセスの役割

| プロセス | 実行環境 | 役割 | セキュリティ |
|---|---|---|---|
| Main | Node.js | ウィンドウ管理、OS API、ファイル操作 | フルアクセス |
| Preload | Node.js (制限付き) | Main ↔ Renderer の橋渡し | contextBridge で制御 |
| Renderer | Chromium | UI レンダリング（React/Vue 等） | サンドボックス（Web と同等） |

---

## 2. プロジェクト作成

### 2.1 electron-vite による構築（推奨）

### コード例 1: プロジェクトの初期化

```bash
# electron-vite のスキャフォールディング（React + TypeScript テンプレート）
npm create @quick-start/electron@latest my-electron-app -- \
  --template react-ts

# ディレクトリに移動して依存関係をインストール
cd my-electron-app
npm install

# 開発サーバー起動（ホットリロード有効）
npm run dev
```

### 2.2 ディレクトリ構成

```
my-electron-app/
├── package.json
├── electron.vite.config.ts       ← Vite 設定（Main/Preload/Renderer 共通）
├── tsconfig.json                 ← TypeScript 設定（ルート）
├── tsconfig.node.json            ← TypeScript 設定（Main/Preload 用）
├── tsconfig.web.json             ← TypeScript 設定（Renderer 用）
│
├── src/
│   ├── main/                     ← Main プロセス
│   │   ├── index.ts              ← エントリポイント
│   │   └── ipc-handlers.ts       ← IPC ハンドラ定義
│   │
│   ├── preload/                  ← Preload スクリプト
│   │   ├── index.ts              ← contextBridge 定義
│   │   └── index.d.ts            ← 型定義
│   │
│   └── renderer/                 ← Renderer プロセス (React アプリ)
│       ├── index.html            ← HTML エントリポイント
│       ├── src/
│       │   ├── main.tsx          ← React エントリポイント
│       │   ├── App.tsx           ← ルートコンポーネント
│       │   ├── components/       ← UI コンポーネント
│       │   ├── hooks/            ← カスタムフック
│       │   └── assets/           ← 静的リソース
│       └── env.d.ts              ← Vite 環境型定義
│
├── resources/                    ← アイコン、ネイティブリソース
│   └── icon.png
├── build/                        ← ビルド設定
│   └── entitlements.mac.plist
└── out/                          ← ビルド出力
```

### コード例 2: Main プロセス（index.ts）

```typescript
// src/main/index.ts — Electron Main プロセスのエントリポイント
import { app, BrowserWindow, shell } from 'electron'
import { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'

// メインウィンドウの参照をモジュールスコープで保持
let mainWindow: BrowserWindow | null = null

function createWindow(): void {
  // ブラウザウィンドウを作成
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    minWidth: 800,
    minHeight: 600,
    // macOS 用: ネイティブタブ対応
    tabbingIdentifier: 'my-app',
    show: false, // 準備完了まで非表示にしてちらつきを防止
    webPreferences: {
      // Preload スクリプトのパス
      preload: join(__dirname, '../preload/index.js'),
      // サンドボックスを有効化（セキュリティ推奨）
      sandbox: true,
      // コンテキスト分離（必須: Renderer から Node.js を直接使えなくする）
      contextIsolation: true,
      // Node.js 統合を無効化（セキュリティ推奨）
      nodeIntegration: false,
    },
  })

  // ウィンドウの準備が完了したら表示
  mainWindow.on('ready-to-show', () => {
    mainWindow?.show()
  })

  // 外部リンクはデフォルトブラウザで開く
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })

  // 開発時は Vite Dev Server、本番時はビルド済み HTML を読み込む
  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

// Electron の初期化完了後にウィンドウを作成
app.whenReady().then(() => {
  // アプリ ID を設定（Windows の通知やタスクバーで使用）
  electronApp.setAppUserModelId('com.example.my-app')

  // 開発時: F12 で DevTools を開く、Ctrl+R でリロード
  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  createWindow()

  // macOS: Dock アイコンクリック時にウィンドウを再作成
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow()
    }
  })
})

// macOS 以外: 全ウィンドウ閉鎖でアプリ終了
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
```

### コード例 3: Preload スクリプト

```typescript
// src/preload/index.ts — Renderer に公開する API を定義
import { contextBridge, ipcRenderer } from 'electron'

// contextBridge で安全に API を公開
// Renderer から window.electronAPI でアクセス可能になる
contextBridge.exposeInMainWorld('electronAPI', {
  // プラットフォーム情報
  platform: process.platform,

  // ファイル操作: Main プロセスに委譲
  openFile: (): Promise<string | null> =>
    ipcRenderer.invoke('dialog:openFile'),

  saveFile: (content: string): Promise<boolean> =>
    ipcRenderer.invoke('dialog:saveFile', content),

  // ストア操作
  getStoreValue: (key: string): Promise<unknown> =>
    ipcRenderer.invoke('store:get', key),

  setStoreValue: (key: string, value: unknown): Promise<void> =>
    ipcRenderer.invoke('store:set', key, value),

  // Main → Renderer のイベント受信
  onUpdateAvailable: (callback: (version: string) => void): void => {
    ipcRenderer.on('update-available', (_event, version) => {
      callback(version)
    })
  },
})
```

```typescript
// src/preload/index.d.ts — Renderer 側で使う型定義
export interface ElectronAPI {
  platform: string
  openFile: () => Promise<string | null>
  saveFile: (content: string) => Promise<boolean>
  getStoreValue: (key: string) => Promise<unknown>
  setStoreValue: (key: string, value: unknown) => Promise<void>
  onUpdateAvailable: (callback: (version: string) => void) => void
}

declare global {
  interface Window {
    electronAPI: ElectronAPI
  }
}
```

### コード例 4: Renderer（React アプリ）

```tsx
// src/renderer/src/App.tsx — React ルートコンポーネント
import { useState } from 'react'
import './assets/main.css'

function App(): JSX.Element {
  const [fileContent, setFileContent] = useState<string | null>(null)
  const [platform] = useState(window.electronAPI.platform)

  // ファイルを開くボタンのハンドラ
  const handleOpenFile = async () => {
    // Preload で定義した API を呼び出す（型安全）
    const content = await window.electronAPI.openFile()
    if (content) {
      setFileContent(content)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Electron + React + TypeScript</h1>
        <p>プラットフォーム: {platform}</p>
      </header>

      <main className="app-main">
        <button onClick={handleOpenFile} className="btn-primary">
          ファイルを開く
        </button>

        {fileContent && (
          <pre className="file-preview">
            {fileContent}
          </pre>
        )}
      </main>
    </div>
  )
}

export default App
```

---

## 3. Vite 設定

### コード例 5: electron.vite.config.ts

```typescript
// electron.vite.config.ts — Main/Preload/Renderer 統合 Vite 設定
import { resolve } from 'path'
import { defineConfig, externalizeDepsPlugin } from 'electron-vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  // Main プロセス用設定
  main: {
    plugins: [
      // Node.js モジュールを外部化（バンドルに含めない）
      externalizeDepsPlugin()
    ],
    build: {
      rollupOptions: {
        input: {
          index: resolve(__dirname, 'src/main/index.ts')
        }
      }
    }
  },

  // Preload スクリプト用設定
  preload: {
    plugins: [externalizeDepsPlugin()],
    build: {
      rollupOptions: {
        input: {
          index: resolve(__dirname, 'src/preload/index.ts')
        }
      }
    }
  },

  // Renderer プロセス用設定（通常の Vite + React）
  renderer: {
    plugins: [react()],
    resolve: {
      alias: {
        // パスエイリアスの設定
        '@': resolve(__dirname, 'src/renderer/src')
      }
    },
    build: {
      rollupOptions: {
        input: {
          index: resolve(__dirname, 'src/renderer/index.html')
        }
      }
    }
  }
})
```

---

## 4. ホットリロードと DevTools

### 4.1 開発時の動作フロー

```
npm run dev 実行時:

  electron-vite dev
       |
       ├─→ Vite Dev Server 起動 (Renderer)
       |     localhost:5173
       |     HMR WebSocket 接続
       |
       ├─→ Main プロセスをビルド & 起動
       |     ファイル変更検知 → 自動再起動
       |
       └─→ Preload をビルド
             ファイル変更検知 → Renderer リロード

  変更の反映速度:
  ┌──────────────┬──────────────────────┐
  │ Renderer     │ ~50ms (HMR)          │
  │ Main         │ ~1s (プロセス再起動)  │
  │ Preload      │ ~500ms (リロード)     │
  └──────────────┴──────────────────────┘
```

### 4.2 DevTools の活用

```typescript
// 開発時のみ DevTools を自動で開く
if (is.dev) {
  mainWindow.webContents.openDevTools({ mode: 'right' })
}

// React DevTools の追加（開発時のみ）
// npm install --save-dev electron-devtools-installer
import installExtension, { REACT_DEVELOPER_TOOLS } from 'electron-devtools-installer'

app.whenReady().then(async () => {
  if (is.dev) {
    try {
      // React DevTools 拡張をインストール
      await installExtension(REACT_DEVELOPER_TOOLS)
      console.log('React DevTools をインストールしました')
    } catch (err) {
      console.error('DevTools インストールエラー:', err)
    }
  }
  createWindow()
})
```

---

## 5. IPC 通信のベストプラクティス

### 5.1 通信パターン

| パターン | API | 方向 | 用途 |
|---|---|---|---|
| invoke/handle | `ipcRenderer.invoke` → `ipcMain.handle` | Renderer → Main → 応答 | データ取得・ダイアログ |
| send/on | `ipcRenderer.send` → `ipcMain.on` | Renderer → Main (片方向) | ログ送信・イベント通知 |
| send/on | `webContents.send` → `ipcRenderer.on` | Main → Renderer (片方向) | 更新通知・状態変更 |

### IPC ハンドラの定義

```typescript
// src/main/ipc-handlers.ts — IPC ハンドラの集約定義
import { ipcMain, dialog, BrowserWindow } from 'electron'
import { readFile, writeFile } from 'fs/promises'

export function registerIpcHandlers(): void {
  // ファイルを開くダイアログ → ファイル内容を返す
  ipcMain.handle('dialog:openFile', async () => {
    const { canceled, filePaths } = await dialog.showOpenDialog({
      properties: ['openFile'],
      filters: [
        { name: 'テキストファイル', extensions: ['txt', 'md', 'json'] },
        { name: 'すべてのファイル', extensions: ['*'] },
      ],
    })

    if (canceled || filePaths.length === 0) return null

    // ファイルの内容を読み取って返す
    const content = await readFile(filePaths[0], 'utf-8')
    return content
  })

  // ファイルを保存
  ipcMain.handle('dialog:saveFile', async (_event, content: string) => {
    const { canceled, filePath } = await dialog.showSaveDialog({
      defaultPath: 'untitled.txt',
    })

    if (canceled || !filePath) return false

    await writeFile(filePath, content, 'utf-8')
    return true
  })
}
```

---

## 6. アンチパターン

### アンチパターン 1: nodeIntegration を有効にする

```typescript
// NG: Renderer で Node.js API に直接アクセス可能にする
const win = new BrowserWindow({
  webPreferences: {
    nodeIntegration: true,       // 危険: Renderer から fs, child_process 等が使える
    contextIsolation: false,     // 危険: Preload と Renderer のコンテキストが共有
  }
})
```

```typescript
// OK: contextIsolation + Preload で安全に API を公開
const win = new BrowserWindow({
  webPreferences: {
    nodeIntegration: false,      // Node.js 統合を無効化
    contextIsolation: true,      // コンテキスト分離を有効化
    sandbox: true,               // サンドボックスを有効化
    preload: join(__dirname, 'preload.js'),
  }
})
```

### アンチパターン 2: IPC チャネル名をハードコードで散在させる

```typescript
// NG: 文字列リテラルが Main/Preload/Renderer に散在 → タイポの温床
// main.ts
ipcMain.handle('get-user-data', ...)
// preload.ts
ipcRenderer.invoke('get-userData')  // タイポに気づけない
```

```typescript
// OK: チャネル名を定数として一元管理
// src/shared/ipc-channels.ts
export const IPC_CHANNELS = {
  GET_USER_DATA: 'user:getData',
  SET_USER_DATA: 'user:setData',
  OPEN_FILE: 'dialog:openFile',
  SAVE_FILE: 'dialog:saveFile',
} as const

// 型安全に使用
import { IPC_CHANNELS } from '../shared/ipc-channels'
ipcMain.handle(IPC_CHANNELS.GET_USER_DATA, ...)
ipcRenderer.invoke(IPC_CHANNELS.GET_USER_DATA)
```

---

## 7. FAQ

### Q1: electron-vite と electron-forge + Vite の違いは何か？

**A:** `electron-vite` は Vite を Electron 向けに最適化した統合ツールであり、Main/Preload/Renderer の3プロセスを1つの設定ファイルで管理できる。`electron-forge` は Electron 公式のビルドツールチェーンであり、パッケージング・署名・配布まで含むフルスタックツールである。新規プロジェクトでは開発体験の良い `electron-vite` で開発し、ビルド・配布には `electron-forge` または `electron-builder` を併用する構成が多い。

### Q2: Electron アプリのメモリ使用量が大きいのはなぜか？

**A:** Electron は Chromium を同梱しているため、最低でも約 80-100MB のメモリを消費する。各ウィンドウが独立した Renderer プロセスを持つことも要因の一つである。対策としては、(1) 不要なウィンドウの遅延生成、(2) バックグラウンドウィンドウの `backgroundThrottling` 有効化、(3) V8 スナップショットの活用が挙げられる。

### Q3: Electron で React 以外のフレームワーク（Vue, Svelte）は使えるか？

**A:** はい。Renderer プロセスは通常の Web アプリと同じであるため、任意のフレームワークが使用可能である。`electron-vite` は React / Vue / Svelte / Solid のテンプレートを公式に提供している。

---

## 8. まとめ

| トピック | キーポイント |
|---|---|
| アーキテクチャ | Main（Node.js）+ Renderer（Chromium）+ Preload（橋渡し） |
| プロジェクト作成 | `create @quick-start/electron` で React+TS テンプレートを生成 |
| Vite 統合 | `electron-vite` が Main/Preload/Renderer を一括管理 |
| ホットリロード | Renderer は HMR（~50ms）、Main は自動再起動（~1s） |
| IPC 通信 | invoke/handle パターンが推奨。チャネル名は定数化 |
| セキュリティ | contextIsolation: true + sandbox: true が必須 |
| DevTools | 開発時は自動オープン + React DevTools 拡張 |

---

## 次に読むべきガイド

- **[01-electron-advanced.md](./01-electron-advanced.md)** — マルチウィンドウ、ネイティブモジュール、パフォーマンス最適化
- **[02-tauri-setup.md](./02-tauri-setup.md)** — 軽量代替フレームワーク Tauri の入門

---

## 参考文献

1. Electron, "Official Documentation", https://www.electronjs.org/docs/latest/
2. electron-vite, "Getting Started", https://electron-vite.org/guide/
3. Electron, "Security Best Practices", https://www.electronjs.org/docs/latest/tutorial/security
4. Electron, "Process Model", https://www.electronjs.org/docs/latest/tutorial/process-model
