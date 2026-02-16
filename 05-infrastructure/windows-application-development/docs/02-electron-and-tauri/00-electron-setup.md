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

## 6. electron-store による設定管理

### 6.1 electron-store のセットアップ

```bash
# electron-store のインストール
npm install electron-store
```

```typescript
// src/main/store.ts — アプリケーション設定の永続化
import Store from 'electron-store'

// 設定のスキーマ定義（型安全）
interface AppConfig {
  window: {
    width: number
    height: number
    x?: number
    y?: number
    isMaximized: boolean
  }
  theme: 'light' | 'dark' | 'system'
  language: string
  recentFiles: string[]
  editor: {
    fontSize: number
    fontFamily: string
    tabSize: number
    wordWrap: boolean
    lineNumbers: boolean
    minimap: boolean
    autoSave: boolean
    autoSaveInterval: number
  }
  updates: {
    autoCheck: boolean
    channel: 'stable' | 'beta'
  }
}

// デフォルト値の定義
const defaults: AppConfig = {
  window: {
    width: 1200,
    height: 800,
    isMaximized: false,
  },
  theme: 'system',
  language: 'ja',
  recentFiles: [],
  editor: {
    fontSize: 14,
    fontFamily: 'Consolas, "Courier New", monospace',
    tabSize: 2,
    wordWrap: true,
    lineNumbers: true,
    minimap: true,
    autoSave: true,
    autoSaveInterval: 30000,
  },
  updates: {
    autoCheck: true,
    channel: 'stable',
  },
}

// 型安全なストアの作成
export const store = new Store<AppConfig>({
  defaults,
  // スキーマバリデーション（オプション）
  schema: {
    theme: {
      type: 'string',
      enum: ['light', 'dark', 'system'],
    },
    'editor.fontSize': {
      type: 'number',
      minimum: 8,
      maximum: 72,
    },
    'editor.tabSize': {
      type: 'number',
      enum: [2, 4, 8],
    },
  },
  // 暗号化（機密情報を保存する場合）
  // encryptionKey: 'your-encryption-key',
  // マイグレーション（バージョン間のスキーマ変更対応）
  migrations: {
    '1.0.0': (store) => {
      // v1.0.0 へのマイグレーション
      store.set('editor.minimap', true)
    },
    '2.0.0': (store) => {
      // v2.0.0 へのマイグレーション
      store.set('updates', { autoCheck: true, channel: 'stable' })
    },
  },
})
```

### 6.2 IPC 経由での設定アクセス

```typescript
// src/main/ipc-handlers.ts — 設定用 IPC ハンドラ
import { ipcMain } from 'electron'
import { store } from './store'

export function registerStoreHandlers(): void {
  // 設定値の取得
  ipcMain.handle('store:get', (_event, key: string) => {
    return store.get(key)
  })

  // 設定値の更新
  ipcMain.handle('store:set', (_event, key: string, value: unknown) => {
    store.set(key, value)
  })

  // 全設定の取得
  ipcMain.handle('store:getAll', () => {
    return store.store
  })

  // 設定のリセット
  ipcMain.handle('store:reset', () => {
    store.clear()
  })

  // 最近のファイル一覧に追加
  ipcMain.handle('store:addRecentFile', (_event, filePath: string) => {
    const recent = store.get('recentFiles', [])
    // 重複を除去し、先頭に追加、最大10件
    const updated = [filePath, ...recent.filter(f => f !== filePath)].slice(0, 10)
    store.set('recentFiles', updated)
    return updated
  })
}
```

```typescript
// src/preload/index.ts — Renderer に設定 API を公開（追加分）
contextBridge.exposeInMainWorld('electronAPI', {
  // ... 既存の API ...

  // 設定 API
  store: {
    get: (key: string) => ipcRenderer.invoke('store:get', key),
    set: (key: string, value: unknown) => ipcRenderer.invoke('store:set', key, value),
    getAll: () => ipcRenderer.invoke('store:getAll'),
    reset: () => ipcRenderer.invoke('store:reset'),
    addRecentFile: (path: string) => ipcRenderer.invoke('store:addRecentFile', path),
  },
})
```

```tsx
// src/renderer/src/hooks/useSettings.ts — React Hook で設定を管理
import { useState, useEffect, useCallback } from 'react'

interface EditorSettings {
  fontSize: number
  fontFamily: string
  tabSize: number
  wordWrap: boolean
  lineNumbers: boolean
  minimap: boolean
  autoSave: boolean
  autoSaveInterval: number
}

export function useSettings() {
  const [settings, setSettings] = useState<EditorSettings | null>(null)
  const [loading, setLoading] = useState(true)

  // 初期読み込み
  useEffect(() => {
    async function loadSettings() {
      const editor = await window.electronAPI.store.get('editor')
      setSettings(editor as EditorSettings)
      setLoading(false)
    }
    loadSettings()
  }, [])

  // 設定の更新
  const updateSetting = useCallback(async <K extends keyof EditorSettings>(
    key: K,
    value: EditorSettings[K]
  ) => {
    await window.electronAPI.store.set(`editor.${key}`, value)
    setSettings(prev => prev ? { ...prev, [key]: value } : null)
  }, [])

  return { settings, loading, updateSetting }
}
```

---

## 7. テスト環境の構築

### 7.1 テストツールの設定

```bash
# テストツールのインストール
npm install --save-dev vitest @testing-library/react @testing-library/jest-dom
npm install --save-dev @testing-library/user-event jsdom
npm install --save-dev @vitest/coverage-v8
```

```typescript
// vitest.config.ts — テスト設定
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/renderer/src/test/setup.ts'],
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'src/main/',      // Main プロセスは別途テスト
        'src/preload/',   // Preload は E2E テスト
      ],
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src/renderer/src'),
    },
  },
})
```

```typescript
// src/renderer/src/test/setup.ts — テストセットアップ
import '@testing-library/jest-dom'

// window.electronAPI のモック
const mockElectronAPI = {
  platform: 'win32',
  openFile: vi.fn().mockResolvedValue(null),
  saveFile: vi.fn().mockResolvedValue(true),
  getStoreValue: vi.fn().mockResolvedValue(null),
  setStoreValue: vi.fn().mockResolvedValue(undefined),
  onUpdateAvailable: vi.fn(),
  store: {
    get: vi.fn().mockResolvedValue(null),
    set: vi.fn().mockResolvedValue(undefined),
    getAll: vi.fn().mockResolvedValue({}),
    reset: vi.fn().mockResolvedValue(undefined),
    addRecentFile: vi.fn().mockResolvedValue([]),
  },
}

Object.defineProperty(window, 'electronAPI', {
  value: mockElectronAPI,
  writable: true,
})
```

### 7.2 コンポーネントテストの例

```tsx
// src/renderer/src/components/__tests__/FileExplorer.test.tsx
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { FileExplorer } from '../FileExplorer'

describe('FileExplorer', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('ファイルを開くボタンを表示する', () => {
    render(<FileExplorer />)
    expect(screen.getByText('ファイルを開く')).toBeInTheDocument()
  })

  it('ファイルを開くダイアログを呼び出す', async () => {
    const user = userEvent.setup()

    window.electronAPI.openFile = vi.fn().mockResolvedValue('テストコンテンツ')

    render(<FileExplorer />)
    await user.click(screen.getByText('ファイルを開く'))

    expect(window.electronAPI.openFile).toHaveBeenCalledTimes(1)
    await waitFor(() => {
      expect(screen.getByText('テストコンテンツ')).toBeInTheDocument()
    })
  })

  it('ファイル選択をキャンセルした場合は何も表示しない', async () => {
    const user = userEvent.setup()

    window.electronAPI.openFile = vi.fn().mockResolvedValue(null)

    render(<FileExplorer />)
    await user.click(screen.getByText('ファイルを開く'))

    expect(screen.queryByTestId('file-content')).not.toBeInTheDocument()
  })
})
```

### 7.3 E2E テスト（Playwright）

```typescript
// e2e/app.spec.ts — Electron E2E テスト
import { test, expect, _electron as electron } from '@playwright/test'
import { ElectronApplication, Page } from 'playwright'

let electronApp: ElectronApplication
let page: Page

test.beforeAll(async () => {
  // Electron アプリを起動
  electronApp = await electron.launch({
    args: ['.'],
    env: {
      ...process.env,
      NODE_ENV: 'test',
    },
  })

  // メインウィンドウを取得
  page = await electronApp.firstWindow()

  // ウィンドウの準備完了を待機
  await page.waitForLoadState('domcontentloaded')
})

test.afterAll(async () => {
  await electronApp.close()
})

test('アプリケーションが正常に起動する', async () => {
  const title = await page.title()
  expect(title).toBe('Electron + React + TypeScript')
})

test('ウィンドウのサイズが正しい', async () => {
  const windowState = await electronApp.evaluate(({ BrowserWindow }) => {
    const mainWindow = BrowserWindow.getAllWindows()[0]
    const { width, height } = mainWindow.getBounds()
    return { width, height }
  })

  expect(windowState.width).toBeGreaterThanOrEqual(800)
  expect(windowState.height).toBeGreaterThanOrEqual(600)
})

test('ファイルを開くボタンが機能する', async () => {
  await page.click('button:has-text("ファイルを開く")')

  // ダイアログはメインプロセスで処理されるため、
  // モックを使用するか、実際のファイルパスを注入する
})
```

---

## 8. ログ管理

```typescript
// src/main/logger.ts — 構造化ログ管理
import log from 'electron-log'
import { app } from 'electron'
import path from 'path'

// ログファイルのパス設定
log.transports.file.resolvePathFn = () =>
  path.join(app.getPath('logs'), 'main.log')

// ログのフォーマット設定
log.transports.file.format = '{y}-{m}-{d} {h}:{i}:{s}.{ms} [{level}] {text}'

// ファイルサイズの制限（5MB でローテーション）
log.transports.file.maxSize = 5 * 1024 * 1024

// ログレベルの設定
if (app.isPackaged) {
  // 本番環境: warn 以上のみ
  log.transports.console.level = 'warn'
  log.transports.file.level = 'info'
} else {
  // 開発環境: 全てのログ
  log.transports.console.level = 'debug'
  log.transports.file.level = 'debug'
}

// カスタムログ関数
export const logger = {
  info: (message: string, data?: Record<string, unknown>) => {
    log.info(message, data ? JSON.stringify(data) : '')
  },
  warn: (message: string, data?: Record<string, unknown>) => {
    log.warn(message, data ? JSON.stringify(data) : '')
  },
  error: (message: string, error?: Error) => {
    log.error(message, error?.stack || '')
  },
  debug: (message: string, data?: Record<string, unknown>) => {
    log.debug(message, data ? JSON.stringify(data) : '')
  },
}

// 未捕捉エラーのハンドリング
process.on('uncaughtException', (error) => {
  logger.error('未捕捉の例外', error)
})

process.on('unhandledRejection', (reason) => {
  logger.error('未処理のPromise拒否', reason instanceof Error ? reason : new Error(String(reason)))
})

export default log
```

---

## 9. package.json の詳細設定

```json
{
  "name": "my-electron-app",
  "version": "1.0.0",
  "description": "Electron + React + TypeScript デスクトップアプリ",
  "main": "./out/main/index.js",
  "author": "Your Name <your@email.com>",
  "license": "MIT",
  "homepage": "https://github.com/yourname/my-electron-app",
  "repository": {
    "type": "git",
    "url": "https://github.com/yourname/my-electron-app.git"
  },
  "scripts": {
    "dev": "electron-vite dev",
    "build": "electron-vite build",
    "preview": "electron-vite preview",
    "lint": "eslint src --ext .ts,.tsx",
    "lint:fix": "eslint src --ext .ts,.tsx --fix",
    "format": "prettier --write 'src/**/*.{ts,tsx,css}'",
    "typecheck": "tsc --noEmit",
    "test": "vitest",
    "test:coverage": "vitest --coverage",
    "test:e2e": "playwright test",
    "package:win": "electron-builder --win",
    "package:mac": "electron-builder --mac",
    "package:linux": "electron-builder --linux",
    "package:all": "electron-builder --win --mac --linux",
    "postinstall": "electron-builder install-app-deps"
  },
  "dependencies": {
    "electron-log": "^5.1.0",
    "electron-store": "^8.2.0",
    "electron-updater": "^6.1.0"
  },
  "devDependencies": {
    "@electron-toolkit/eslint-config-ts": "^1.0.0",
    "@electron-toolkit/utils": "^3.0.0",
    "@electron/notarize": "^2.3.0",
    "@quick-start/electron": "^2.0.0",
    "@testing-library/jest-dom": "^6.4.0",
    "@testing-library/react": "^14.2.0",
    "@testing-library/user-event": "^14.5.0",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "@vitejs/plugin-react": "^4.2.0",
    "@vitest/coverage-v8": "^1.3.0",
    "electron": "^28.0.0",
    "electron-builder": "^24.13.0",
    "electron-vite": "^2.0.0",
    "eslint": "^8.56.0",
    "prettier": "^3.2.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-router-dom": "^6.22.0",
    "typescript": "^5.3.0",
    "vitest": "^1.3.0"
  },
  "build": {
    "appId": "com.example.my-electron-app",
    "productName": "My Electron App",
    "copyright": "Copyright (C) 2024 Your Name",
    "directories": {
      "output": "dist",
      "buildResources": "build"
    },
    "files": [
      "out/**/*",
      "!node_modules/**/*"
    ],
    "win": {
      "target": ["nsis", "portable"],
      "icon": "resources/icon.ico"
    },
    "mac": {
      "target": ["dmg", "zip"],
      "icon": "resources/icon.icns",
      "category": "public.app-category.productivity"
    },
    "linux": {
      "target": ["AppImage", "deb"],
      "icon": "resources/icons",
      "category": "Utility"
    }
  }
}
```

---

## 10. セキュリティチェックリスト

```typescript
// セキュリティ設定の検証ユーティリティ
import { BrowserWindow } from 'electron'

function validateSecurityConfig(win: BrowserWindow): void {
  const webPreferences = win.webContents.getWebPreferences()

  // 必須: コンテキスト分離が有効であること
  if (!webPreferences.contextIsolation) {
    console.error('[セキュリティ] contextIsolation が無効です!')
  }

  // 必須: Node.js 統合が無効であること
  if (webPreferences.nodeIntegration) {
    console.error('[セキュリティ] nodeIntegration が有効です!')
  }

  // 推奨: サンドボックスが有効であること
  if (!webPreferences.sandbox) {
    console.warn('[セキュリティ] sandbox が無効です')
  }

  // 推奨: webSecurity が有効であること
  if (webPreferences.webSecurity === false) {
    console.error('[セキュリティ] webSecurity が無効です!')
  }
}

// CSP (Content Security Policy) の設定
function setupCSP(win: BrowserWindow): void {
  win.webContents.session.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [
          "default-src 'self'",
          "script-src 'self'",
          "style-src 'self' 'unsafe-inline'",
          "img-src 'self' data: https:",
          "font-src 'self' data:",
          "connect-src 'self' https://api.example.com",
        ].join('; '),
      },
    })
  })
}

// 外部リンクのナビゲーションを防止
function preventNavigation(win: BrowserWindow): void {
  // ウィンドウ内でのナビゲーションを制限
  win.webContents.on('will-navigate', (event, url) => {
    const appUrl = new URL(win.webContents.getURL())
    const targetUrl = new URL(url)

    // 異なるオリジンへのナビゲーションを防止
    if (targetUrl.origin !== appUrl.origin) {
      event.preventDefault()
      // 外部ブラウザで開く
      require('electron').shell.openExternal(url)
    }
  })

  // 新しいウィンドウの作成を制限
  win.webContents.setWindowOpenHandler(({ url }) => {
    require('electron').shell.openExternal(url)
    return { action: 'deny' }
  })
}
```

---

## 11. アンチパターン

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

## 12. デバッグとトラブルシューティング

### 12.1 Main プロセスのデバッグ

```typescript
// launch.json — VS Code でのデバッグ設定
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug Main Process",
      "type": "node",
      "request": "launch",
      "cwd": "${workspaceFolder}",
      "runtimeExecutable": "${workspaceFolder}/node_modules/.bin/electron-vite",
      "args": ["dev", "--inspect=5858"],
      "sourceMaps": true,
      "outFiles": ["${workspaceFolder}/out/**/*.js"],
      "console": "integratedTerminal",
      "env": {
        "NODE_ENV": "development"
      }
    },
    {
      "name": "Debug Renderer Process",
      "type": "chrome",
      "request": "attach",
      "port": 9222,
      "webRoot": "${workspaceFolder}/src/renderer/src",
      "sourceMapPathOverrides": {
        "webpack:///./src/*": "${webRoot}/*"
      }
    }
  ],
  "compounds": [
    {
      "name": "Debug All",
      "configurations": ["Debug Main Process", "Debug Renderer Process"]
    }
  ]
}
```

### 12.2 よくあるエラーと解決策

```typescript
// エラー 1: "Cannot use import statement outside a module"
// 原因: Main プロセスの ESM/CJS 設定の不整合
// 解決: electron.vite.config.ts で正しい設定を行う

// electron.vite.config.ts
import { defineConfig, externalizeDepsPlugin } from 'electron-vite'

export default defineConfig({
  main: {
    plugins: [externalizeDepsPlugin()],
    build: {
      rollupOptions: {
        output: {
          format: 'cjs', // Main プロセスは CJS を使用
        },
      },
    },
  },
  preload: {
    plugins: [externalizeDepsPlugin()],
    build: {
      rollupOptions: {
        output: {
          format: 'cjs', // Preload も CJS
        },
      },
    },
  },
  renderer: {
    // Renderer は ESM で問題なし
  },
})
```

```typescript
// エラー 2: "contextBridge API can only be used when contextIsolation is enabled"
// 原因: BrowserWindow の webPreferences で contextIsolation が false
// 解決: 必ず contextIsolation: true を設定する

// エラー 3: "Electron Security Warning (Insecure Content-Security-Policy)"
// 原因: CSP が設定されていない
// 解決: セクション10の CSP 設定を適用する

// エラー 4: IPC ハンドラが undefined を返す
// 原因: handle の登録前に invoke が呼ばれている
// 解決: app.whenReady() の中でハンドラを登録する
import { app, ipcMain } from 'electron'

app.whenReady().then(() => {
  // IPC ハンドラは app.whenReady() 内で登録する
  ipcMain.handle('channel', async (_event, ...args) => {
    // ハンドラの処理
    return result
  })

  // ウィンドウの作成もここで行う
  createWindow()
})
```

### 12.3 パフォーマンスプロファイリング

```typescript
// src/main/performance.ts — パフォーマンス計測ユーティリティ
import { performance, PerformanceObserver } from 'perf_hooks'
import { logger } from './logger'

// パフォーマンス計測の開始
export function startMeasure(name: string): void {
  performance.mark(`${name}-start`)
}

// パフォーマンス計測の終了とログ出力
export function endMeasure(name: string): number {
  performance.mark(`${name}-end`)
  performance.measure(name, `${name}-start`, `${name}-end`)

  const entries = performance.getEntriesByName(name)
  const duration = entries[entries.length - 1]?.duration ?? 0

  logger.info(`[Performance] ${name}: ${duration.toFixed(2)}ms`)

  // マークをクリーンアップ
  performance.clearMarks(`${name}-start`)
  performance.clearMarks(`${name}-end`)
  performance.clearMeasures(name)

  return duration
}

// 起動時間の計測例
export function measureStartupTime(): void {
  const observer = new PerformanceObserver((list) => {
    for (const entry of list.getEntries()) {
      logger.info(`[Startup] ${entry.name}: ${entry.duration.toFixed(2)}ms`)
    }
  })

  observer.observe({ entryTypes: ['measure'] })

  performance.mark('app-start')

  app.on('ready', () => {
    performance.mark('app-ready')
    performance.measure('App Ready Time', 'app-start', 'app-ready')
  })
}
```

---

## 13. 環境変数と設定の管理

```typescript
// src/main/env.ts — 環境変数の型安全な管理
import { app } from 'electron'
import path from 'path'

interface AppEnvironment {
  isDev: boolean
  isProd: boolean
  isTest: boolean
  appVersion: string
  platform: NodeJS.Platform
  arch: string
  userDataPath: string
  logPath: string
  tempPath: string
}

export function getAppEnvironment(): AppEnvironment {
  return {
    isDev: !app.isPackaged,
    isProd: app.isPackaged,
    isTest: process.env.NODE_ENV === 'test',
    appVersion: app.getVersion(),
    platform: process.platform,
    arch: process.arch,
    userDataPath: app.getPath('userData'),
    logPath: app.getPath('logs'),
    tempPath: app.getPath('temp'),
  }
}

// .env ファイルの読み込み（開発環境用）
import { config } from 'dotenv'

if (!app.isPackaged) {
  config({
    path: path.join(app.getAppPath(), '.env.development'),
  })
}

// 環境変数のバリデーション
function validateEnv(): void {
  const required = ['API_BASE_URL'] as const

  for (const key of required) {
    if (!process.env[key]) {
      throw new Error(`環境変数 ${key} が設定されていません`)
    }
  }
}
```

---

## 14. FAQ

### Q1: electron-vite と electron-forge + Vite の違いは何か？

**A:** `electron-vite` は Vite を Electron 向けに最適化した統合ツールであり、Main/Preload/Renderer の3プロセスを1つの設定ファイルで管理できる。`electron-forge` は Electron 公式のビルドツールチェーンであり、パッケージング・署名・配布まで含むフルスタックツールである。新規プロジェクトでは開発体験の良い `electron-vite` で開発し、ビルド・配布には `electron-forge` または `electron-builder` を併用する構成が多い。

### Q2: Electron アプリのメモリ使用量が大きいのはなぜか？

**A:** Electron は Chromium を同梱しているため、最低でも約 80-100MB のメモリを消費する。各ウィンドウが独立した Renderer プロセスを持つことも要因の一つである。対策としては、(1) 不要なウィンドウの遅延生成、(2) バックグラウンドウィンドウの `backgroundThrottling` 有効化、(3) V8 スナップショットの活用が挙げられる。

### Q3: Electron で React 以外のフレームワーク（Vue, Svelte）は使えるか？

**A:** はい。Renderer プロセスは通常の Web アプリと同じであるため、任意のフレームワークが使用可能である。`electron-vite` は React / Vue / Svelte / Solid のテンプレートを公式に提供している。

### Q4: Electron アプリの起動速度を改善するにはどうすればよいか？

**A:** 主な対策として以下が挙げられる。(1) Preload スクリプトの最小化 -- 不要なモジュールの読み込みを避ける。(2) メインウィンドウの `show: false` 設定と `ready-to-show` イベントでの表示 -- 白い画面のちらつきを防ぐ。(3) ネイティブモジュールの遅延読み込み -- 起動時に全モジュールをロードしない。(4) V8 コードキャッシュの活用 -- `v8-compile-cache` パッケージの使用。(5) スプラッシュスクリーンの活用 -- 体感速度の向上。

### Q5: Electron アプリのバイナリサイズを削減するには？

**A:** Electron アプリは Chromium を同梱するため、最低でも 50-80MB 程度のサイズになる。削減策としては、(1) `electron-builder` の asar アーカイブを有効化する、(2) 不要な `node_modules` を除外する（`files` オプションで制御）、(3) `devDependencies` がバンドルに含まれないことを確認する、(4) プラットフォーム固有のビルドで不要な OS のコードを排除する、(5) サイズが気になる場合は Tauri への移行を検討する（バイナリサイズが 2-10MB 程度）。

### Q6: 自動更新の仕組みはどうなっているか？

**A:** `electron-updater` パッケージを使用する。更新ファイルを GitHub Releases、S3、またはプライベートサーバーにホストし、アプリ起動時に更新チェックを行う。Windows では NSIS インストーラ、macOS では DMG/ZIP の差分更新に対応している。コード署名が正しく設定されていれば、ユーザーにセキュリティ警告を表示せずに更新が可能である。

---

## 15. まとめ

| トピック | キーポイント |
|---|---|
| アーキテクチャ | Main（Node.js）+ Renderer（Chromium）+ Preload（橋渡し） |
| プロジェクト作成 | `create @quick-start/electron` で React+TS テンプレートを生成 |
| Vite 統合 | `electron-vite` が Main/Preload/Renderer を一括管理 |
| ホットリロード | Renderer は HMR（~50ms）、Main は自動再起動（~1s） |
| IPC 通信 | invoke/handle パターンが推奨。チャネル名は定数化 |
| 設定管理 | electron-store でスキーマバリデーション付き永続化 |
| テスト | Vitest（Unit）+ Playwright（E2E）の二層構成 |
| ログ管理 | electron-log でファイルローテーション付きログ出力 |
| セキュリティ | contextIsolation: true + sandbox: true + CSP 設定が必須 |
| デバッグ | VS Code 統合デバッガで Main/Renderer 両プロセスをデバッグ |
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
