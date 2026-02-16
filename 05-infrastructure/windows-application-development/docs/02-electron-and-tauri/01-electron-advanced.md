# Electron 応用

> マルチウィンドウ管理、カスタムタイトルバー、ネイティブモジュール統合、SQLite データベース、パフォーマンス最適化など、本格的な Electron アプリ開発に必要な応用技術を習得する。

---

## この章で学ぶこと

1. **マルチウィンドウ管理**とカスタムタイトルバーの実装方法を習得する
2. **ネイティブモジュール（C++ アドオン）と SQLite** の統合手法を理解する
3. **パフォーマンスのボトルネック**を特定し、起動時間・メモリ使用量を最適化する

---

## 1. マルチウィンドウ管理

### 1.1 ウィンドウ管理アーキテクチャ

```
+----------------------------------------------------------+
|                    Main Process                           |
|                                                          |
|  WindowManager                                           |
|  ┌─────────────────────────────────────────────────────┐  |
|  │  windows: Map<string, BrowserWindow>                │  |
|  │                                                     │  |
|  │  ┌──────────┐  ┌──────────┐  ┌──────────┐         │  |
|  │  │ main     │  │ settings │  │ about    │         │  |
|  │  │ (メイン) │  │ (設定)   │  │ (概要)   │         │  |
|  │  └──────────┘  └──────────┘  └──────────┘         │  |
|  └─────────────────────────────────────────────────────┘  |
|                                                          |
|  ウィンドウ間通信: Main プロセス経由の IPC                 |
|  Window A  ───→  Main  ───→  Window B                    |
+----------------------------------------------------------+
```

### コード例 1: WindowManager クラス

```typescript
// src/main/window-manager.ts — ウィンドウの一元管理クラス
import { BrowserWindow, screen } from 'electron'
import { join } from 'path'
import { is } from '@electron-toolkit/utils'

// ウィンドウ設定の型定義
interface WindowConfig {
  width?: number
  height?: number
  minWidth?: number
  minHeight?: number
  parent?: BrowserWindow   // 親ウィンドウ（モーダル用）
  modal?: boolean          // モーダルウィンドウにするか
  route?: string           // Renderer 側のルートパス
  resizable?: boolean
}

class WindowManager {
  // ウィンドウ ID をキーとして管理
  private windows = new Map<string, BrowserWindow>()

  // ウィンドウを作成または既存ウィンドウにフォーカス
  createWindow(id: string, config: WindowConfig = {}): BrowserWindow {
    // 既にウィンドウが存在する場合はフォーカスして返す
    const existing = this.windows.get(id)
    if (existing && !existing.isDestroyed()) {
      existing.focus()
      return existing
    }

    const {
      width = 800,
      height = 600,
      minWidth = 400,
      minHeight = 300,
      parent,
      modal = false,
      route = '/',
      resizable = true,
    } = config

    const win = new BrowserWindow({
      width,
      height,
      minWidth,
      minHeight,
      parent,
      modal,
      resizable,
      show: false,
      webPreferences: {
        preload: join(__dirname, '../preload/index.js'),
        contextIsolation: true,
        sandbox: true,
      },
    })

    // 準備完了後に表示（ちらつき防止）
    win.once('ready-to-show', () => win.show())

    // ウィンドウ閉鎖時にマップから削除
    win.on('closed', () => {
      this.windows.delete(id)
    })

    // コンテンツの読み込み
    if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
      // 開発時: Vite Dev Server の URL + ルートパス
      win.loadURL(`${process.env['ELECTRON_RENDERER_URL']}#${route}`)
    } else {
      // 本番: ビルド済み HTML + ハッシュルーティング
      win.loadFile(join(__dirname, '../renderer/index.html'), {
        hash: route,
      })
    }

    this.windows.set(id, win)
    return win
  }

  // 全ウィンドウを取得
  getWindow(id: string): BrowserWindow | undefined {
    return this.windows.get(id)
  }

  // 特定ウィンドウにメッセージを送信
  sendTo(id: string, channel: string, ...args: unknown[]): void {
    const win = this.windows.get(id)
    if (win && !win.isDestroyed()) {
      win.webContents.send(channel, ...args)
    }
  }

  // 全ウィンドウにブロードキャスト
  broadcast(channel: string, ...args: unknown[]): void {
    for (const [, win] of this.windows) {
      if (!win.isDestroyed()) {
        win.webContents.send(channel, ...args)
      }
    }
  }

  // 全ウィンドウを閉じる
  closeAll(): void {
    for (const [, win] of this.windows) {
      if (!win.isDestroyed()) win.close()
    }
    this.windows.clear()
  }
}

// シングルトンとしてエクスポート
export const windowManager = new WindowManager()
```

---

## 2. カスタムタイトルバー

### 2.1 フレームレスウィンドウ構成

```
デフォルトタイトルバー:
+------------------------------------------------------+
| [icon] My App              [_] [□] [X]  ← OS ネイティブ|
+------------------------------------------------------+
| コンテンツ                                             |
+------------------------------------------------------+

カスタムタイトルバー:
+------------------------------------------------------+
| 🔍 検索...  |  ファイル  編集  表示  | ● ● ●  ← 独自UI |
+------------------------------------------------------+
| コンテンツ                                             |
+------------------------------------------------------+
```

### コード例 2: カスタムタイトルバーの実装

```typescript
// Main プロセス: フレームレスウィンドウの作成
const win = new BrowserWindow({
  frame: false,            // OS 標準のタイトルバーを非表示
  titleBarStyle: 'hidden', // macOS: ネイティブの信号ボタンは残す
  titleBarOverlay: {       // Windows: 最小化/最大化/閉じるボタンを残す
    color: '#1e1e2e',      // タイトルバーの背景色
    symbolColor: '#cdd6f4', // ボタンアイコンの色
    height: 40,            // タイトルバーの高さ
  },
  // Windows でのコンテンツ領域の調整
  ...(process.platform === 'win32' && {
    backgroundMaterial: 'mica',
  }),
})
```

```tsx
// src/renderer/src/components/TitleBar.tsx — カスタムタイトルバー
import { useState, useEffect } from 'react'
import './TitleBar.css'

export function TitleBar(): JSX.Element {
  const [isMaximized, setIsMaximized] = useState(false)

  useEffect(() => {
    // ウィンドウの最大化状態を監視
    window.electronAPI.onWindowStateChange((maximized: boolean) => {
      setIsMaximized(maximized)
    })
  }, [])

  return (
    <div className="titlebar">
      {/* ドラッグ可能領域（ウィンドウ移動用） */}
      <div className="titlebar-drag-region">
        <span className="titlebar-title">My App</span>
      </div>

      {/* メニュー領域（ドラッグ不可） */}
      <div className="titlebar-menu">
        <button className="menu-item">ファイル</button>
        <button className="menu-item">編集</button>
        <button className="menu-item">表示</button>
      </div>

      {/* ウィンドウ操作ボタン（macOS では非表示） */}
      {window.electronAPI.platform !== 'darwin' && (
        <div className="titlebar-controls">
          <button
            className="control-btn minimize"
            onClick={() => window.electronAPI.minimizeWindow()}
          >
            ─
          </button>
          <button
            className="control-btn maximize"
            onClick={() => window.electronAPI.maximizeWindow()}
          >
            {isMaximized ? '❐' : '□'}
          </button>
          <button
            className="control-btn close"
            onClick={() => window.electronAPI.closeWindow()}
          >
            ✕
          </button>
        </div>
      )}
    </div>
  )
}
```

```css
/* src/renderer/src/components/TitleBar.css */
.titlebar {
  display: flex;
  align-items: center;
  height: 40px;
  background: var(--bg-primary);
  user-select: none; /* テキスト選択を無効化 */
}

/* ドラッグ可能領域: ウィンドウの移動に使用 */
.titlebar-drag-region {
  flex: 1;
  height: 100%;
  display: flex;
  align-items: center;
  padding-left: 16px;
  -webkit-app-region: drag; /* この領域でウィンドウをドラッグ可能にする */
}

/* メニューやボタンはドラッグ不可にする */
.titlebar-menu,
.titlebar-controls {
  -webkit-app-region: no-drag;
}

/* 閉じるボタンのホバー効果 */
.control-btn.close:hover {
  background: #e81123;
  color: white;
}
```

---

## 3. ネイティブモジュール

### 3.1 ネイティブモジュールの種類

| 種類 | ビルドツール | 言語 | 用途 |
|---|---|---|---|
| N-API (node-addon-api) | node-gyp / cmake-js | C / C++ | 高速計算、OS API |
| Rust (napi-rs) | napi-rs | Rust | 安全な高速処理 |
| WASM | wasm-pack | Rust / C++ | ポータブルな計算 |
| FFI (ffi-napi) | なし（動的ロード） | C 互換 DLL | 既存 DLL の呼び出し |

### コード例 3: napi-rs による Rust ネイティブモジュール

```toml
# native-module/Cargo.toml — Rust プロジェクト設定
[package]
name = "my-native"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
napi = { version = "2", features = ["async"] }
napi-derive = "2"

[build-dependencies]
napi-build = "2"
```

```rust
// native-module/src/lib.rs — Rust で高速な画像処理を実装
use napi::bindgen_prelude::*;
use napi_derive::napi;

/// 画像のリサイズを高速に実行する関数
/// JavaScript から直接呼び出し可能
#[napi]
pub fn resize_image(
    input_path: String,
    output_path: String,
    width: u32,
    height: u32,
) -> Result<()> {
    let img = image::open(&input_path)
        .map_err(|e| Error::from_reason(format!("画像を開けません: {}", e)))?;

    let resized = img.resize_exact(
        width,
        height,
        image::imageops::FilterType::Lanczos3,
    );

    resized.save(&output_path)
        .map_err(|e| Error::from_reason(format!("保存に失敗: {}", e)))?;

    Ok(())
}

/// 非同期関数も定義可能
#[napi]
pub async fn hash_file(path: String) -> Result<String> {
    use sha2::{Sha256, Digest};
    use tokio::fs;

    let data = fs::read(&path).await
        .map_err(|e| Error::from_reason(format!("ファイル読み込みエラー: {}", e)))?;

    let mut hasher = Sha256::new();
    hasher.update(&data);
    let result = hasher.finalize();

    Ok(format!("{:x}", result))
}
```

```typescript
// TypeScript から Rust ネイティブモジュールを使用
import { resizeImage, hashFile } from 'my-native'

// 同期呼び出し（CPU バウンドの処理）
resizeImage('/path/to/input.jpg', '/path/to/output.jpg', 800, 600)

// 非同期呼び出し（I/O バウンドの処理）
const hash = await hashFile('/path/to/large-file.bin')
console.log(`ファイルハッシュ: ${hash}`)
```

---

## 4. SQLite 統合

### 4.1 SQLite ライブラリの比較

| ライブラリ | 種類 | 同期/非同期 | Electron 対応 |
|---|---|---|---|
| better-sqlite3 | ネイティブ (C) | 同期 | electron-rebuild 必要 |
| sql.js | WASM | 同期 | そのまま動作 |
| drizzle-orm + better-sqlite3 | ORM | 同期 | 型安全 |
| prisma | ORM | 非同期 | 設定が複雑 |

### コード例 4: better-sqlite3 + drizzle-orm

```typescript
// src/main/database/schema.ts — drizzle-orm でスキーマ定義
import { sqliteTable, text, integer, real } from 'drizzle-orm/sqlite-core'

// タスクテーブルの定義
export const tasks = sqliteTable('tasks', {
  id: integer('id').primaryKey({ autoIncrement: true }),
  title: text('title').notNull(),
  description: text('description'),
  priority: text('priority', { enum: ['low', 'medium', 'high'] })
    .notNull()
    .default('medium'),
  completed: integer('completed', { mode: 'boolean' })
    .notNull()
    .default(false),
  createdAt: integer('created_at', { mode: 'timestamp' })
    .notNull()
    .$defaultFn(() => new Date()),
  updatedAt: integer('updated_at', { mode: 'timestamp' })
    .notNull()
    .$defaultFn(() => new Date()),
})

// タスクの TypeScript 型を自動導出
export type Task = typeof tasks.$inferSelect
export type NewTask = typeof tasks.$inferInsert
```

```typescript
// src/main/database/index.ts — データベース接続と初期化
import Database from 'better-sqlite3'
import { drizzle } from 'drizzle-orm/better-sqlite3'
import { migrate } from 'drizzle-orm/better-sqlite3/migrator'
import { app } from 'electron'
import { join } from 'path'
import * as schema from './schema'

// データベースファイルのパス（ユーザーデータディレクトリに保存）
const DB_PATH = join(app.getPath('userData'), 'app-data.db')

// SQLite 接続を作成
const sqlite = new Database(DB_PATH)

// WAL モードを有効化（読み書きの並行性能向上）
sqlite.pragma('journal_mode = WAL')

// 外部キー制約を有効化
sqlite.pragma('foreign_keys = ON')

// drizzle ORM インスタンスを作成
export const db = drizzle(sqlite, { schema })

// マイグレーションの実行
export function runMigrations(): void {
  migrate(db, {
    migrationsFolder: join(__dirname, '../../drizzle'),
  })
}
```

```typescript
// src/main/database/task-repository.ts — リポジトリパターンの実装
import { eq, desc, and, like } from 'drizzle-orm'
import { db } from './index'
import { tasks, Task, NewTask } from './schema'

export class TaskRepository {
  // 全タスクを取得（新しい順）
  findAll(): Task[] {
    return db.select().from(tasks).orderBy(desc(tasks.createdAt)).all()
  }

  // ID でタスクを取得
  findById(id: number): Task | undefined {
    return db.select().from(tasks).where(eq(tasks.id, id)).get()
  }

  // タスクを検索
  search(query: string): Task[] {
    return db.select().from(tasks)
      .where(like(tasks.title, `%${query}%`))
      .all()
  }

  // タスクを作成
  create(task: NewTask): Task {
    return db.insert(tasks).values(task).returning().get()
  }

  // タスクを更新
  update(id: number, data: Partial<NewTask>): Task | undefined {
    return db.update(tasks)
      .set({ ...data, updatedAt: new Date() })
      .where(eq(tasks.id, id))
      .returning()
      .get()
  }

  // タスクを削除
  delete(id: number): void {
    db.delete(tasks).where(eq(tasks.id, id)).run()
  }

  // 完了済みタスクの一括削除
  deleteCompleted(): number {
    const result = db.delete(tasks)
      .where(eq(tasks.completed, true))
      .run()
    return result.changes
  }
}
```

---

## 5. パフォーマンス最適化

### 5.1 起動時間の最適化

```
典型的な Electron アプリの起動フロー:

  時間軸 (ms)
  0     200    400    600    800   1000   1200   1400
  |------|------|------|------|------|------|------|
  [== Electron 初期化 ==]
         [=== Main プロセス起動 ===]
                [== Preload 実行 ==]
                      [======= Renderer 読み込み =======]
                                    [=== React 初期化 ===]
                                                  [Ready!]

  最適化後:
  0     200    400    600    800
  |------|------|------|------|
  [= 初期化 =]
        [= Main =]
             [Preload]
               [=== Renderer ===]
                       [React]
                            [Ready!]
```

### コード例 5: 起動時間最適化テクニック集

```typescript
// src/main/index.ts — 起動時間の最適化

// 最適化1: 必要なモジュールを遅延インポート
// NG: import { autoUpdater } from 'electron-updater'
// OK: 必要になった時点でインポート
async function checkForUpdates(): Promise<void> {
  const { autoUpdater } = await import('electron-updater')
  autoUpdater.checkForUpdates()
}

// 最適化2: ウィンドウの事前ウォームアップ
let splashWindow: BrowserWindow | null = null

function createSplashScreen(): void {
  // 軽量なスプラッシュスクリーンを即座に表示
  splashWindow = new BrowserWindow({
    width: 400,
    height: 300,
    frame: false,
    transparent: true,
    resizable: false,
    webPreferences: { contextIsolation: true },
  })
  splashWindow.loadFile(join(__dirname, '../renderer/splash.html'))
  splashWindow.show()
}

async function createMainWindow(): Promise<void> {
  const mainWindow = new BrowserWindow({
    show: false, // メインウィンドウは裏で準備
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      sandbox: true,
    },
  })

  // 最適化3: V8 コードキャッシュの有効化
  mainWindow.webContents.session.setCodeCachePath(
    join(app.getPath('userData'), 'code-cache')
  )

  // Renderer の読み込みを開始
  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    await mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    await mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }

  // メインウィンドウの準備完了後にスプラッシュを閉じる
  mainWindow.show()
  splashWindow?.close()
  splashWindow = null
}

// 最適化4: アプリの初期化を並列実行
app.whenReady().then(async () => {
  // スプラッシュスクリーンを即座に表示
  createSplashScreen()

  // 並列で初期化を実行
  await Promise.all([
    createMainWindow(),
    runMigrations(),        // DB マイグレーション
    loadUserPreferences(),  // ユーザー設定読み込み
  ])
})
```

### 5.2 メモリ最適化

```typescript
// メモリ使用量の監視と最適化

// バックグラウンドウィンドウのスロットリング
mainWindow.on('blur', () => {
  // ウィンドウが非アクティブ時にフレームレートを下げる
  mainWindow.webContents.setFrameRate(5)
})

mainWindow.on('focus', () => {
  // アクティブ時は通常のフレームレートに戻す
  mainWindow.webContents.setFrameRate(60)
})

// 定期的なガベージコレクション（大量データ処理後）
function triggerGC(): void {
  if (global.gc) {
    global.gc()
  }
}

// メモリ使用量のログ出力
function logMemoryUsage(): void {
  const usage = process.memoryUsage()
  console.log({
    rss: `${(usage.rss / 1024 / 1024).toFixed(1)} MB`,
    heapUsed: `${(usage.heapUsed / 1024 / 1024).toFixed(1)} MB`,
    heapTotal: `${(usage.heapTotal / 1024 / 1024).toFixed(1)} MB`,
  })
}
```

---

## 6. 自動アップデート

### 6.1 electron-updater による自動更新

```typescript
// src/main/updater.ts — 自動アップデート管理
import { autoUpdater, UpdateCheckResult, UpdateInfo } from 'electron-updater'
import { BrowserWindow, dialog, app } from 'electron'
import { logger } from './logger'

interface UpdateState {
  checking: boolean
  available: boolean
  downloaded: boolean
  progress: number
  version: string | null
  error: Error | null
}

class AppUpdater {
  private state: UpdateState = {
    checking: false,
    available: false,
    downloaded: false,
    progress: 0,
    version: null,
    error: null,
  }

  private mainWindow: BrowserWindow | null = null

  constructor() {
    // ログの設定
    autoUpdater.logger = logger

    // 開発環境でもテスト可能にする
    autoUpdater.forceDevUpdateConfig = false

    // 自動ダウンロードを無効化（ユーザーの確認後にダウンロード）
    autoUpdater.autoDownload = false

    // プレリリースも含めるか
    autoUpdater.allowPrerelease = false

    // イベントハンドラの登録
    this.setupEventHandlers()
  }

  private setupEventHandlers(): void {
    autoUpdater.on('checking-for-update', () => {
      this.state.checking = true
      this.notifyRenderer('update:checking')
      logger.info('アップデートを確認中...')
    })

    autoUpdater.on('update-available', (info: UpdateInfo) => {
      this.state.checking = false
      this.state.available = true
      this.state.version = info.version
      this.notifyRenderer('update:available', info)
      logger.info(`アップデート利用可能: v${info.version}`)

      // ユーザーに確認ダイアログを表示
      this.promptUpdate(info)
    })

    autoUpdater.on('update-not-available', (info: UpdateInfo) => {
      this.state.checking = false
      this.state.available = false
      this.notifyRenderer('update:not-available', info)
      logger.info('最新バージョンです')
    })

    autoUpdater.on('download-progress', (progress) => {
      this.state.progress = progress.percent
      this.notifyRenderer('update:progress', {
        percent: progress.percent,
        bytesPerSecond: progress.bytesPerSecond,
        total: progress.total,
        transferred: progress.transferred,
      })

      // タスクバーのプログレス表示（Windows）
      this.mainWindow?.setProgressBar(progress.percent / 100)
    })

    autoUpdater.on('update-downloaded', (info: UpdateInfo) => {
      this.state.downloaded = true
      this.state.progress = 100
      this.notifyRenderer('update:downloaded', info)
      this.mainWindow?.setProgressBar(-1) // プログレスバーをリセット

      logger.info(`アップデートダウンロード完了: v${info.version}`)

      // 再起動の確認
      this.promptRestart(info)
    })

    autoUpdater.on('error', (error: Error) => {
      this.state.checking = false
      this.state.error = error
      this.notifyRenderer('update:error', error.message)
      this.mainWindow?.setProgressBar(-1)
      logger.error('アップデートエラー', error)
    })
  }

  private async promptUpdate(info: UpdateInfo): Promise<void> {
    if (!this.mainWindow) return

    const result = await dialog.showMessageBox(this.mainWindow, {
      type: 'info',
      title: 'アップデート利用可能',
      message: `新しいバージョン v${info.version} が利用可能です。`,
      detail: `現在のバージョン: v${app.getVersion()}\n\nダウンロードしますか？`,
      buttons: ['ダウンロード', '後で'],
      defaultId: 0,
      cancelId: 1,
    })

    if (result.response === 0) {
      autoUpdater.downloadUpdate()
    }
  }

  private async promptRestart(info: UpdateInfo): Promise<void> {
    if (!this.mainWindow) return

    const result = await dialog.showMessageBox(this.mainWindow, {
      type: 'info',
      title: 'アップデート準備完了',
      message: `v${info.version} のインストール準備が完了しました。`,
      detail: '今すぐ再起動してアップデートを適用しますか？',
      buttons: ['今すぐ再起動', '後で再起動'],
      defaultId: 0,
      cancelId: 1,
    })

    if (result.response === 0) {
      autoUpdater.quitAndInstall(false, true)
    }
  }

  private notifyRenderer(channel: string, data?: unknown): void {
    if (this.mainWindow && !this.mainWindow.isDestroyed()) {
      this.mainWindow.webContents.send(channel, data)
    }
  }

  setMainWindow(win: BrowserWindow): void {
    this.mainWindow = win
  }

  async checkForUpdates(): Promise<UpdateCheckResult | null> {
    return autoUpdater.checkForUpdates()
  }

  getState(): UpdateState {
    return { ...this.state }
  }
}

export const appUpdater = new AppUpdater()
```

### 6.2 更新配信サーバーの設定

```typescript
// electron-builder.yml での更新サーバー設定例

// パターン 1: GitHub Releases を利用（最も簡単）
// package.json の build セクション
const githubConfig = {
  publish: {
    provider: 'github',
    owner: 'your-org',
    repo: 'your-app',
    releaseType: 'release', // 'draft' | 'prerelease' | 'release'
  },
}

// パターン 2: S3 互換ストレージ
const s3Config = {
  publish: {
    provider: 's3',
    bucket: 'your-update-bucket',
    region: 'ap-northeast-1',
    path: '/releases/',
  },
}

// パターン 3: 汎用サーバー（社内配布向け）
const genericConfig = {
  publish: {
    provider: 'generic',
    url: 'https://updates.example.com/releases/',
    channel: 'latest',
  },
}
```

---

## 7. システムトレイとバックグラウンド動作

### 7.1 トレイアイコンの実装

```typescript
// src/main/tray.ts — システムトレイの管理
import { Tray, Menu, nativeImage, app, BrowserWindow } from 'electron'
import { join } from 'path'
import { windowManager } from './window-manager'

class TrayManager {
  private tray: Tray | null = null
  private isQuitting = false

  create(mainWindow: BrowserWindow): void {
    // プラットフォーム別のアイコン
    const iconPath = process.platform === 'win32'
      ? join(__dirname, '../../resources/tray-icon.ico')    // Windows: ICO
      : process.platform === 'darwin'
      ? join(__dirname, '../../resources/tray-iconTemplate.png') // macOS: Template
      : join(__dirname, '../../resources/tray-icon.png')    // Linux: PNG

    const icon = nativeImage.createFromPath(iconPath)

    // macOS のテンプレートイメージ設定
    if (process.platform === 'darwin') {
      icon.setTemplateImage(true)
    }

    this.tray = new Tray(icon)

    // ツールチップ
    this.tray.setToolTip(`${app.getName()} v${app.getVersion()}`)

    // コンテキストメニューの構築
    this.updateContextMenu(mainWindow)

    // ダブルクリックでウィンドウを表示（Windows/Linux）
    this.tray.on('double-click', () => {
      if (mainWindow.isVisible()) {
        mainWindow.focus()
      } else {
        mainWindow.show()
      }
    })

    // ウィンドウの閉じるボタンでトレイに格納（終了ではなく最小化）
    mainWindow.on('close', (event) => {
      if (!this.isQuitting) {
        event.preventDefault()
        mainWindow.hide()

        // Windows ではバルーン通知を表示
        if (process.platform === 'win32' && this.tray) {
          this.tray.displayBalloon({
            title: app.getName(),
            content: 'アプリはシステムトレイで実行中です',
            iconType: 'info',
          })
        }
      }
    })

    // app.quit() が呼ばれたら本当に終了
    app.on('before-quit', () => {
      this.isQuitting = true
    })
  }

  private updateContextMenu(mainWindow: BrowserWindow): void {
    const contextMenu = Menu.buildFromTemplate([
      {
        label: 'ウィンドウを表示',
        click: () => {
          mainWindow.show()
          mainWindow.focus()
        },
      },
      { type: 'separator' },
      {
        label: 'ステータス',
        submenu: [
          { label: 'オンライン', type: 'radio', checked: true },
          { label: '取り込み中', type: 'radio' },
          { label: 'オフライン', type: 'radio' },
        ],
      },
      { type: 'separator' },
      {
        label: '設定',
        click: () => {
          windowManager.createWindow('settings', {
            route: '/settings',
            width: 600,
            height: 500,
            parent: mainWindow,
            modal: true,
          })
        },
      },
      { type: 'separator' },
      {
        label: '終了',
        click: () => {
          this.isQuitting = true
          app.quit()
        },
      },
    ])

    this.tray?.setContextMenu(contextMenu)
  }

  // バッジ数の更新（通知数など）
  updateBadge(count: number): void {
    if (process.platform === 'darwin') {
      app.dock.setBadge(count > 0 ? String(count) : '')
    }

    // Windows: タスクバーのオーバーレイアイコン
    if (process.platform === 'win32') {
      const mainWindow = windowManager.getWindow('main')
      if (mainWindow && count > 0) {
        const badge = this.createBadgeImage(count)
        mainWindow.setOverlayIcon(badge, `${count} 件の通知`)
      } else if (mainWindow) {
        mainWindow.setOverlayIcon(null, '')
      }
    }
  }

  private createBadgeImage(count: number): Electron.NativeImage {
    // Canvas でバッジ画像を生成（16x16 px）
    const size = 16
    const canvas = new OffscreenCanvas(size, size)
    const ctx = canvas.getContext('2d')!
    ctx.fillStyle = '#e81123'
    ctx.beginPath()
    ctx.arc(size / 2, size / 2, size / 2, 0, Math.PI * 2)
    ctx.fill()
    ctx.fillStyle = '#ffffff'
    ctx.font = 'bold 10px sans-serif'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'
    ctx.fillText(count > 99 ? '99+' : String(count), size / 2, size / 2)

    const buffer = Buffer.from(canvas.transferToImageBitmap() as unknown as ArrayBuffer)
    return nativeImage.createFromBuffer(buffer, { width: size, height: size })
  }

  destroy(): void {
    this.tray?.destroy()
    this.tray = null
  }
}

export const trayManager = new TrayManager()
```

---

## 8. ファイル関連付けとプロトコルハンドラ

### 8.1 カスタムファイル拡張子の関連付け

```typescript
// electron-builder の設定でファイル関連付けを定義
// package.json の build セクション
const fileAssociations = {
  build: {
    fileAssociations: [
      {
        ext: 'myapp',             // 拡張子
        name: 'My App Document',  // ファイルタイプの表示名
        description: 'My App のドキュメントファイル',
        mimeType: 'application/x-myapp',
        icon: 'resources/file-icon', // .ico / .icns
        role: 'Editor',           // macOS: Editor | Viewer | Shell | None
      },
      {
        ext: ['json', 'yaml', 'yml'],
        name: 'Configuration File',
        role: 'Viewer',
      },
    ],
  },
}
```

```typescript
// src/main/file-handler.ts — ファイルを開く処理
import { app, ipcMain } from 'electron'
import { windowManager } from './window-manager'
import fs from 'fs'

// macOS: ファイルをドロップまたはダブルクリックで開いた時
app.on('open-file', (event, filePath) => {
  event.preventDefault()

  if (app.isReady()) {
    handleFileOpen(filePath)
  } else {
    // アプリ起動前にファイルが渡された場合はキューに入れる
    pendingFiles.push(filePath)
  }
})

// Windows/Linux: コマンドライン引数からファイルパスを取得
const pendingFiles: string[] = []

function processCommandLineArgs(argv: string[]): void {
  // 最初の引数はアプリのパスなのでスキップ
  const filePaths = argv.slice(1).filter(arg => {
    return !arg.startsWith('--') && fs.existsSync(arg)
  })

  for (const filePath of filePaths) {
    handleFileOpen(filePath)
  }
}

// 二重起動防止 + ファイルを既存インスタンスに渡す
const gotTheLock = app.requestSingleInstanceLock()

if (!gotTheLock) {
  app.quit()
} else {
  app.on('second-instance', (_event, argv) => {
    // 既存のインスタンスにフォーカスしてファイルを開く
    const mainWindow = windowManager.getWindow('main')
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore()
      mainWindow.focus()
      processCommandLineArgs(argv)
    }
  })
}

async function handleFileOpen(filePath: string): Promise<void> {
  try {
    const content = await fs.promises.readFile(filePath, 'utf-8')
    const mainWindow = windowManager.getWindow('main')

    if (mainWindow) {
      mainWindow.webContents.send('file:opened', {
        path: filePath,
        name: path.basename(filePath),
        content,
      })
    }
  } catch (error) {
    logger.error(`ファイルを開けません: ${filePath}`, error as Error)
  }
}
```

### 8.2 カスタムプロトコルハンドラ

```typescript
// src/main/protocol.ts — カスタムプロトコル (myapp://) の登録
import { app, protocol, net } from 'electron'
import { join } from 'path'
import { URL } from 'url'

// カスタムプロトコルを登録（app.whenReady() の前に呼ぶ必要あり）
if (process.defaultApp) {
  // 開発環境: コマンドライン引数でプロトコルを登録
  if (process.argv.length >= 2) {
    app.setAsDefaultProtocolClient('myapp', process.execPath, [
      join(__dirname, '..'),
    ])
  }
} else {
  // 本番環境: そのまま登録
  app.setAsDefaultProtocolClient('myapp')
}

// プロトコルリクエストのハンドリング
app.whenReady().then(() => {
  // myapp:// スキームの処理
  protocol.handle('myapp', (request) => {
    const url = new URL(request.url)

    switch (url.hostname) {
      case 'open':
        // myapp://open?file=path/to/file
        const filePath = url.searchParams.get('file')
        if (filePath) handleFileOpen(filePath)
        return new Response('OK')

      case 'settings':
        // myapp://settings
        windowManager.createWindow('settings', { route: '/settings' })
        return new Response('OK')

      default:
        return new Response('Not Found', { status: 404 })
    }
  })
})

// macOS: プロトコル URL でアプリが起動された時
app.on('open-url', (event, url) => {
  event.preventDefault()
  handleProtocolUrl(url)
})

// Windows/Linux: 二重起動時にプロトコル URL を受け取る
app.on('second-instance', (_event, argv) => {
  const url = argv.find(arg => arg.startsWith('myapp://'))
  if (url) handleProtocolUrl(url)
})

function handleProtocolUrl(url: string): void {
  try {
    const parsed = new URL(url)
    logger.info(`プロトコル URL を処理: ${parsed.hostname}${parsed.pathname}`)
    // URL に応じた処理を実行
  } catch (error) {
    logger.error('無効なプロトコル URL', error as Error)
  }
}
```

---

## 9. ドラッグ＆ドロップとクリップボード

### 9.1 ドラッグ＆ドロップの実装

```tsx
// src/renderer/src/components/DropZone.tsx — ファイルドロップ領域
import { useState, useCallback, DragEvent } from 'react'

interface DroppedFile {
  name: string
  path: string
  size: number
  type: string
}

export function DropZone(): JSX.Element {
  const [isDragging, setIsDragging] = useState(false)
  const [files, setFiles] = useState<DroppedFile[]>([])

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback(async (e: DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const droppedFiles: DroppedFile[] = []

    for (const file of Array.from(e.dataTransfer.files)) {
      droppedFiles.push({
        name: file.name,
        path: (file as File & { path: string }).path, // Electron 拡張
        size: file.size,
        type: file.type || 'application/octet-stream',
      })
    }

    setFiles(prev => [...prev, ...droppedFiles])

    // Main プロセスにファイルパスを送信して処理
    for (const file of droppedFiles) {
      await window.electronAPI.processDroppedFile(file.path)
    }
  }, [])

  return (
    <div
      className={`drop-zone ${isDragging ? 'dragging' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {isDragging ? (
        <p>ここにファイルをドロップ</p>
      ) : (
        <p>ファイルをドラッグ＆ドロップ</p>
      )}
      {files.length > 0 && (
        <ul className="file-list">
          {files.map((file, i) => (
            <li key={i}>
              <span>{file.name}</span>
              <span>{(file.size / 1024).toFixed(1)} KB</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
```

```css
/* ドロップゾーンのスタイル */
.drop-zone {
  border: 2px dashed var(--border-color, #ccc);
  border-radius: 8px;
  padding: 40px;
  text-align: center;
  transition: all 0.2s ease;
  cursor: pointer;
}

.drop-zone.dragging {
  border-color: var(--accent-color, #0078d4);
  background: rgba(0, 120, 212, 0.05);
}
```

### 9.2 アプリからのドラッグアウト（ファイルのエクスポート）

```typescript
// Main プロセス: ドラッグアウトのハンドリング
ipcMain.on('drag-out', (event, filePath: string) => {
  // ファイルをアプリからデスクトップやエクスプローラーにドラッグ
  event.sender.startDrag({
    file: filePath,
    icon: nativeImage.createFromPath(
      join(__dirname, '../../resources/file-drag-icon.png')
    ),
  })
})
```

```tsx
// Renderer 側: ドラッグ開始のハンドリング
function FileItem({ file }: { file: { name: string; path: string } }) {
  const handleDragStart = (e: React.DragEvent) => {
    e.preventDefault()
    // Main プロセスにドラッグ開始を通知
    window.electronAPI.startDrag(file.path)
  }

  return (
    <div draggable onDragStart={handleDragStart}>
      {file.name}
    </div>
  )
}
```

### 9.3 クリップボード操作

```typescript
// src/main/clipboard-handler.ts — クリップボードの高度な操作
import { clipboard, nativeImage, ipcMain } from 'electron'

// テキストの読み書き
ipcMain.handle('clipboard:readText', () => {
  return clipboard.readText()
})

ipcMain.handle('clipboard:writeText', (_event, text: string) => {
  clipboard.writeText(text)
})

// リッチテキスト（HTML）の読み書き
ipcMain.handle('clipboard:readHTML', () => {
  return clipboard.readHTML()
})

ipcMain.handle('clipboard:writeHTML', (_event, html: string) => {
  clipboard.writeText(html.replace(/<[^>]*>/g, '')) // プレーンテキストも同時に設定
  clipboard.writeHTML(html)
})

// 画像の読み書き
ipcMain.handle('clipboard:readImage', () => {
  const image = clipboard.readImage()
  if (image.isEmpty()) return null
  return image.toDataURL()
})

ipcMain.handle('clipboard:writeImage', (_event, dataUrl: string) => {
  const image = nativeImage.createFromDataURL(dataUrl)
  clipboard.writeImage(image)
})

// クリップボードの変更監視
let previousContent = ''
const CLIPBOARD_POLL_INTERVAL = 1000

function startClipboardWatcher(callback: (content: string) => void): NodeJS.Timer {
  return setInterval(() => {
    const current = clipboard.readText()
    if (current !== previousContent && current.length > 0) {
      previousContent = current
      callback(current)
    }
  }, CLIPBOARD_POLL_INTERVAL)
}
```

---

## 10. 通知とシステム連携

### 10.1 ネイティブ通知

```typescript
// src/main/notifications.ts — 通知管理
import { Notification, app, shell } from 'electron'

interface AppNotification {
  title: string
  body: string
  icon?: string
  urgency?: 'normal' | 'critical' | 'low'
  actions?: Array<{ type: 'button'; text: string }>
  silent?: boolean
  onClick?: () => void
}

class NotificationManager {
  private enabled = true

  async show(options: AppNotification): Promise<void> {
    if (!this.enabled) return

    // 通知がサポートされているか確認
    if (!Notification.isSupported()) {
      logger.warn('通知がサポートされていません')
      return
    }

    const notification = new Notification({
      title: options.title,
      body: options.body,
      icon: options.icon || join(__dirname, '../../resources/notification-icon.png'),
      urgency: options.urgency || 'normal',
      silent: options.silent || false,
      actions: options.actions,
    })

    if (options.onClick) {
      notification.on('click', options.onClick)
    }

    // アクションボタンのクリックハンドリング
    notification.on('action', (_event, index) => {
      logger.info(`通知アクション: インデックス ${index}`)
    })

    notification.show()
  }

  // 通知の有効/無効を切り替え
  setEnabled(enabled: boolean): void {
    this.enabled = enabled
  }

  // Windows: Focus Assist の状態を確認
  isDoNotDisturbEnabled(): boolean {
    // Windows 10+ の Focus Assist / Do Not Disturb の確認は
    // ネイティブモジュールが必要（electron-windows-notifications 等）
    return false
  }
}

export const notificationManager = new NotificationManager()
```

### 10.2 電源状態の監視

```typescript
// src/main/power-monitor.ts — 電源管理
import { powerMonitor, powerSaveBlocker, app } from 'electron'

class PowerManager {
  private saveBlockerId: number | null = null

  setup(): void {
    // スリープ/復帰の検知
    powerMonitor.on('suspend', () => {
      logger.info('システムがスリープします')
      // 保存されていないデータの自動保存
      this.autoSave()
    })

    powerMonitor.on('resume', () => {
      logger.info('システムがスリープから復帰しました')
      // ネットワーク接続の再確立
      this.reconnect()
    })

    // ロック/アンロックの検知
    powerMonitor.on('lock-screen', () => {
      logger.info('画面がロックされました')
    })

    powerMonitor.on('unlock-screen', () => {
      logger.info('画面がアンロックされました')
    })

    // AC/バッテリーの切り替え
    powerMonitor.on('on-ac', () => {
      logger.info('AC 電源に接続されました')
    })

    powerMonitor.on('on-battery', () => {
      logger.info('バッテリー駆動に切り替わりました')
      // バッテリー駆動時はバックグラウンド処理を制限
    })

    // シャットダウン検知
    powerMonitor.on('shutdown', () => {
      logger.info('システムがシャットダウンします')
      this.emergencySave()
    })
  }

  // スリープ防止（長時間処理の実行中に使用）
  preventSleep(reason: string): void {
    if (this.saveBlockerId !== null) return

    this.saveBlockerId = powerSaveBlocker.start('prevent-display-sleep')
    logger.info(`スリープ防止を開始: ${reason}`)
  }

  allowSleep(): void {
    if (this.saveBlockerId !== null) {
      powerSaveBlocker.stop(this.saveBlockerId)
      this.saveBlockerId = null
      logger.info('スリープ防止を解除')
    }
  }

  // バッテリー残量の取得（Electron 30+ で利用可能）
  getBatteryInfo(): { level: number; charging: boolean } {
    return {
      level: powerMonitor.isOnBatteryPower() ? -1 : 100,
      charging: !powerMonitor.isOnBatteryPower(),
    }
  }

  private autoSave(): void {
    // 保存処理の実装
  }

  private reconnect(): void {
    // 再接続処理の実装
  }

  private emergencySave(): void {
    // 緊急保存処理の実装
  }
}

export const powerManager = new PowerManager()
```

---

## 11. アンチパターン

### アンチパターン 1: 重い処理を Main プロセスで同期実行する

```typescript
// NG: Main プロセスで同期的に大量のファイルを処理
// → UI がフリーズし、ウィンドウが応答なしになる
ipcMain.handle('process-files', (_event, paths: string[]) => {
  const results = []
  for (const path of paths) {
    // 同期的に大量のファイルを読み込み・処理
    const data = fs.readFileSync(path)
    const processed = heavyComputation(data)
    results.push(processed)
  }
  return results
})
```

```typescript
// OK: Worker スレッドまたは UtilityProcess に委譲
import { utilityProcess } from 'electron'

ipcMain.handle('process-files', async (_event, paths: string[]) => {
  // UtilityProcess で重い処理を別プロセスで実行
  const worker = utilityProcess.fork(
    join(__dirname, 'workers/file-processor.js')
  )

  return new Promise((resolve) => {
    worker.postMessage({ type: 'process', paths })
    worker.on('message', (result) => {
      resolve(result)
      worker.kill()
    })
  })
})
```

### アンチパターン 2: BrowserWindow を無制限に作成する

```typescript
// NG: ユーザー操作のたびに新しいウィンドウを作成
ipcMain.handle('open-detail', (_event, itemId: string) => {
  // 100個のアイテムを開くと100個のウィンドウ → メモリ枯渇
  const win = new BrowserWindow({ width: 600, height: 400 })
  win.loadURL(`app://detail/${itemId}`)
})
```

```typescript
// OK: ウィンドウプールで上限管理
const MAX_WINDOWS = 10

ipcMain.handle('open-detail', (_event, itemId: string) => {
  const existing = windowManager.getWindow(`detail-${itemId}`)
  if (existing) {
    existing.focus()
    return
  }

  // ウィンドウ数の上限チェック
  if (windowManager.count() >= MAX_WINDOWS) {
    dialog.showMessageBox({
      type: 'warning',
      message: `ウィンドウは最大 ${MAX_WINDOWS} 個まで開けます`,
    })
    return
  }

  windowManager.createWindow(`detail-${itemId}`, {
    route: `/detail/${itemId}`,
    width: 600,
    height: 400,
  })
})
```

### アンチパターン 3: Renderer プロセスから直接ファイルシステムにアクセスする

```typescript
// NG: Renderer で fs を直接使う（nodeIntegration: true の状態）
// セキュリティリスクが非常に高い
import fs from 'fs'
const data = fs.readFileSync('/etc/passwd', 'utf-8') // 何でも読める
```

```typescript
// OK: IPC 経由で Main プロセスに委譲し、パスの検証を行う
// Renderer 側
const data = await window.electronAPI.readFile('data/config.json')

// Main 側（パスの検証付き）
ipcMain.handle('fs:readFile', (_event, relativePath: string) => {
  const safePath = join(app.getPath('userData'), relativePath)
  // パストラバーサル攻撃の防止
  if (!safePath.startsWith(app.getPath('userData'))) {
    throw new Error('不正なパスです')
  }
  return fs.readFileSync(safePath, 'utf-8')
})
```

---

## 12. FAQ

### Q1: Electron のバージョンを上げると better-sqlite3 が動かなくなる。どうすべきか？

**A:** ネイティブモジュールは Electron の Node.js バージョンに合わせてリビルドが必要である。`electron-rebuild` パッケージを使うと自動でリビルドされる。`package.json` の `scripts` に `"postinstall": "electron-rebuild"` を追加するのが定番である。あるいは `sql.js`（WASM ベース）に切り替えればリビルド不要になる。

### Q2: マルチウィンドウ間でデータを共有する最善の方法は？

**A:** Main プロセスをデータハブとして使い、IPC 経由でデータを配信するのが最も安全で管理しやすい。共有ストア（SQLite や electron-store）を Main プロセスに置き、各ウィンドウは IPC でデータを要求する設計が推奨される。`BrowserWindow.webContents.send()` で変更通知をブロードキャストすれば、全ウィンドウがリアルタイムに同期できる。

### Q3: Electron アプリのバイナリサイズを小さくするには？

**A:** 以下の手法を組み合わせる。(1) `electron-builder` の `asar` パッキングを有効化する、(2) `devDependencies` を正しく分離し、本番ビルドに含めない、(3) 未使用の `node_modules` を `files` 設定で除外する、(4) UPX 圧縮を適用する（Windows/Linux）。通常 150-200MB から 80-100MB 程度まで削減可能である。

---

### Q4: UtilityProcess と Worker Threads の使い分けはどうすべきか？

**A:** `UtilityProcess` は Electron 独自の API で、完全に独立したプロセスとして動作する。Node.js の全 API が利用可能であり、クラッシュしてもメインプロセスに影響しない。一方、`Worker Threads` は Node.js 標準のスレッド機能で、メモリをメインプロセスと共有できる（SharedArrayBuffer）。CPU バウンドの重い計算には `UtilityProcess`、比較的軽い非同期タスクには `Worker Threads` が適している。

### Q5: Electron アプリでのデータベースのバックアップ戦略は？

**A:** SQLite の場合、以下の戦略を推奨する。(1) `VACUUM INTO` コマンドで定期的にバックアップファイルを作成する、(2) WAL モードを有効にして書き込み中でも安全にコピーできるようにする、(3) `app.getPath('userData')` 内にバックアップディレクトリを作り、世代管理する（最新5件など）、(4) ファイル名にタイムスタンプを含める（`backup-2024-01-15T10-30-00.db`）、(5) アプリ起動時に自動バックアップを実行する。

### Q6: カスタムタイトルバーを実装するとアクセシビリティに影響はあるか？

**A:** Windows の場合、`titleBarOverlay` オプションを使えばネイティブのウィンドウ操作ボタン（最小化、最大化、閉じる）が残るため、アクセシビリティへの影響は最小限である。ただし、カスタムメニュー領域にはキーボードナビゲーション（Tab/Enter/Escape）を適切に実装する必要がある。macOS ではネイティブの信号ボタン（赤黄緑）を `titleBarStyle: 'hidden'` で残すことが推奨される。完全なフレームレス（`frame: false`）は非推奨である。

---

## 13. まとめ

| トピック | キーポイント |
|---|---|
| マルチウィンドウ | WindowManager で一元管理。ウィンドウ数に上限を設ける |
| カスタムタイトルバー | `titleBarOverlay`（Windows）+ `-webkit-app-region: drag` |
| ネイティブモジュール | napi-rs (Rust) が安全性と性能のバランスに優れる |
| SQLite | better-sqlite3 + drizzle-orm で型安全な DB 操作 |
| 起動時間 | スプラッシュスクリーン + 遅延インポート + 並列初期化 |
| メモリ最適化 | バックグラウンドスロットリング + UtilityProcess |
| 自動更新 | electron-updater でダイアログ確認 + 差分ダウンロード |
| システムトレイ | TrayManager でバックグラウンド常駐 + バッジ通知 |
| ファイル関連付け | electron-builder 設定 + protocol.handle でカスタムスキーム |
| ドラッグ＆ドロップ | Renderer のドロップ受信 + Main の startDrag でエクスポート |
| セキュリティ | 全てのファイル操作は Main プロセス経由 + パス検証 |

---

## 次に読むべきガイド

- **[02-tauri-setup.md](./02-tauri-setup.md)** — 軽量な代替フレームワーク Tauri の入門
- **[00-packaging-and-signing.md](../03-distribution/00-packaging-and-signing.md)** — Electron アプリのパッケージングと署名

---

## 参考文献

1. Electron, "Performance", https://www.electronjs.org/docs/latest/tutorial/performance
2. Electron, "UtilityProcess", https://www.electronjs.org/docs/latest/api/utility-process
3. napi-rs, "Getting Started", https://napi.rs/docs/introduction/getting-started
4. better-sqlite3, "API Documentation", https://github.com/WiseLibs/better-sqlite3/blob/master/docs/api.md
