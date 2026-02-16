# クロスプラットフォーム対応

> 1つのコードベースで Windows・macOS・Linux をサポートする。プラットフォーム検出、OS 固有 API の抽象化、パス処理、UI/UX の差異対応まで、クロスプラットフォーム設計を解説する。

## この章で学ぶこと

- [ ] プラットフォーム検出と条件分岐を実装できる
- [ ] OS 固有の UI/UX 差異に対応できる
- [ ] パス・改行コード等の環境差異を適切に処理できる
- [ ] フォント・レンダリング・ファイルシステムの差異を理解し対処できる
- [ ] CI/CD でマルチプラットフォームビルドを構築できる
- [ ] 各 OS のネイティブ機能（通知・トレイ・ショートカット）に適切に対応できる

---

## 1. プラットフォーム検出

### 1.1 基本的なプラットフォーム検出

```typescript
// プラットフォーム検出
const platform = process.platform;
// 'win32' | 'darwin' | 'linux'

// OS 別処理の抽象化
function getPlatformConfig() {
  switch (process.platform) {
    case 'win32':
      return {
        appDataPath: process.env.APPDATA!,
        separator: '\\',
        lineEnding: '\r\n',
        shortcutModifier: 'Ctrl',
      };
    case 'darwin':
      return {
        appDataPath: `${process.env.HOME}/Library/Application Support`,
        separator: '/',
        lineEnding: '\n',
        shortcutModifier: 'Cmd',
      };
    case 'linux':
      return {
        appDataPath: process.env.XDG_CONFIG_HOME || `${process.env.HOME}/.config`,
        separator: '/',
        lineEnding: '\n',
        shortcutModifier: 'Ctrl',
      };
    default:
      throw new Error(`Unsupported platform: ${process.platform}`);
  }
}
```

### 1.2 詳細なプラットフォーム情報の取得

```typescript
import os from 'os';
import { app } from 'electron';

// 詳細なシステム情報を取得するユーティリティクラス
class SystemInfo {
  // OS のバージョン情報
  static getOSVersion(): string {
    return os.release(); // 例: '10.0.22631' (Windows 11)
  }

  // OS の種類をフレンドリー名で返す
  static getOSName(): string {
    switch (process.platform) {
      case 'win32': return `Windows ${this.getWindowsVersion()}`;
      case 'darwin': return `macOS ${this.getMacOSVersion()}`;
      case 'linux': return this.getLinuxDistro();
      default: return 'Unknown';
    }
  }

  // Windows のバージョンを判定
  private static getWindowsVersion(): string {
    const release = os.release();
    const build = parseInt(release.split('.')[2] || '0');
    if (build >= 22000) return '11';
    if (build >= 10240) return '10';
    return release;
  }

  // macOS のバージョンを判定
  private static getMacOSVersion(): string {
    const release = os.release();
    const major = parseInt(release.split('.')[0]);
    // Darwin カーネルバージョンと macOS バージョンの対応
    const macVersionMap: Record<number, string> = {
      23: '14 (Sonoma)',
      22: '13 (Ventura)',
      21: '12 (Monterey)',
      20: '11 (Big Sur)',
    };
    return macVersionMap[major] || release;
  }

  // Linux ディストリビューションの判定
  private static getLinuxDistro(): string {
    try {
      const osRelease = require('fs').readFileSync('/etc/os-release', 'utf-8');
      const match = osRelease.match(/PRETTY_NAME="(.+)"/);
      return match ? match[1] : 'Linux';
    } catch {
      return 'Linux';
    }
  }

  // アーキテクチャの取得
  static getArch(): string {
    return process.arch; // 'x64' | 'arm64' | 'ia32'
  }

  // メモリ情報
  static getMemoryInfo(): { total: number; free: number; used: number } {
    const total = os.totalmem();
    const free = os.freemem();
    return {
      total: Math.round(total / (1024 * 1024)),
      free: Math.round(free / (1024 * 1024)),
      used: Math.round((total - free) / (1024 * 1024)),
    };
  }

  // CPU 情報
  static getCPUInfo(): { model: string; cores: number; speed: number } {
    const cpus = os.cpus();
    return {
      model: cpus[0]?.model || 'Unknown',
      cores: cpus.length,
      speed: cpus[0]?.speed || 0,
    };
  }

  // ユーザーのロケール情報
  static getLocale(): string {
    return app.getLocale(); // 例: 'ja', 'en-US'
  }

  // ダークモードの判定
  static isDarkMode(): boolean {
    const { nativeTheme } = require('electron');
    return nativeTheme.shouldUseDarkColors;
  }
}
```

### 1.3 機能ベースの検出パターン

```typescript
// OS ではなく機能の有無で分岐する設計（推奨）
class FeatureDetector {
  // システムトレイがサポートされているか
  static supportsTray(): boolean {
    // Linux の一部環境ではトレイがサポートされない
    if (process.platform === 'linux') {
      // Wayland 環境ではシステムトレイの挙動が異なる
      return process.env.XDG_SESSION_TYPE !== 'wayland' ||
             !!process.env.DBUS_SESSION_BUS_ADDRESS;
    }
    return true;
  }

  // ネイティブ通知がサポートされているか
  static supportsNotification(): boolean {
    const { Notification } = require('electron');
    return Notification.isSupported();
  }

  // タッチスクリーンが利用可能か
  static hasTouchScreen(): boolean {
    // Renderer プロセスで使用
    return 'ontouchstart' in window || navigator.maxTouchPoints > 0;
  }

  // ハイコントラストモードが有効か
  static isHighContrast(): boolean {
    const { nativeTheme } = require('electron');
    return nativeTheme.shouldUseHighContrastColors;
  }

  // Apple Silicon (ARM) かどうか
  static isAppleSilicon(): boolean {
    return process.platform === 'darwin' && process.arch === 'arm64';
  }

  // Windows の特定バージョン以降かチェック
  static isWindows11OrLater(): boolean {
    if (process.platform !== 'win32') return false;
    const build = parseInt(require('os').release().split('.')[2] || '0');
    return build >= 22000;
  }

  // Mica / Acrylic が利用可能か（Windows 11 以降）
  static supportsMica(): boolean {
    return this.isWindows11OrLater();
  }
}
```

---

## 2. パス処理

### 2.1 基本的なパス処理

```
パス処理の注意点:

  Windows: C:\Users\gaku\Documents\file.txt
  macOS:   /Users/gaku/Documents/file.txt
  Linux:   /home/gaku/Documents/file.txt

  解決策: path モジュールを常に使用する
```

```typescript
import path from 'path';
import { app } from 'electron';

// 正しい: path.join を使用
const configPath = path.join(app.getPath('userData'), 'config.json');

// 間違い: 文字列結合
const badPath = app.getPath('userData') + '/config.json'; // Windows で壊れる

// アプリデータの保存先
const paths = {
  userData: app.getPath('userData'),      // アプリ設定
  documents: app.getPath('documents'),    // ユーザードキュメント
  downloads: app.getPath('downloads'),    // ダウンロード
  temp: app.getPath('temp'),              // 一時ファイル
  home: app.getPath('home'),              // ホーム
  desktop: app.getPath('desktop'),        // デスクトップ
};
```

### 2.2 高度なパス処理ユーティリティ

```typescript
import path from 'path';
import fs from 'fs/promises';
import { app } from 'electron';

class PathUtils {
  // アプリケーション固有のパスを安全に構築
  static getAppPath(...segments: string[]): string {
    const basePath = app.getPath('userData');
    const resolved = path.resolve(basePath, ...segments);
    // パストラバーサル攻撃の防止
    if (!resolved.startsWith(basePath)) {
      throw new Error(`不正なパス: ${resolved}`);
    }
    return resolved;
  }

  // ファイル名をプラットフォームに合わせてサニタイズ
  static sanitizeFileName(name: string): string {
    // Windows で許可されない文字を除去
    const windowsForbidden = /[<>:"/\\|?*\x00-\x1f]/g;
    let sanitized = name.replace(windowsForbidden, '_');

    // Windows の予約名を回避
    const windowsReserved = /^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\.|$)/i;
    if (windowsReserved.test(sanitized)) {
      sanitized = `_${sanitized}`;
    }

    // macOS ではコロンも問題になる
    if (process.platform === 'darwin') {
      sanitized = sanitized.replace(/:/g, '_');
    }

    // ファイル名の長さを制限（ext4: 255バイト, NTFS: 255文字）
    if (sanitized.length > 200) {
      const ext = path.extname(sanitized);
      sanitized = sanitized.substring(0, 200 - ext.length) + ext;
    }

    return sanitized;
  }

  // パスの最大長チェック
  static validatePathLength(filePath: string): boolean {
    if (process.platform === 'win32') {
      // Windows: MAX_PATH は 260 文字（長いパスを有効化していない場合）
      return filePath.length < 260;
    }
    // macOS / Linux: PATH_MAX は通常 4096
    return filePath.length < 4096;
  }

  // UNC パスの処理（Windows ネットワークドライブ）
  static isUNCPath(filePath: string): boolean {
    return filePath.startsWith('\\\\') || filePath.startsWith('//');
  }

  // ホームディレクトリの展開（~/ → 実際のパス）
  static expandHome(filePath: string): string {
    if (filePath.startsWith('~/') || filePath === '~') {
      return filePath.replace('~', app.getPath('home'));
    }
    return filePath;
  }

  // クロスプラットフォームな一時ファイルの作成
  static async createTempFile(prefix: string, extension: string): Promise<string> {
    const tempDir = app.getPath('temp');
    const fileName = `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}.${extension}`;
    const tempPath = path.join(tempDir, fileName);
    await fs.writeFile(tempPath, '');
    return tempPath;
  }

  // ディレクトリの再帰的な作成（存在しない場合のみ）
  static async ensureDirectory(dirPath: string): Promise<void> {
    try {
      await fs.mkdir(dirPath, { recursive: true });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'EEXIST') {
        throw error;
      }
    }
  }

  // シンボリックリンクの解決（Linux/macOS でよく使われる）
  static async resolveSymlinks(filePath: string): Promise<string> {
    try {
      return await fs.realpath(filePath);
    } catch {
      return filePath;
    }
  }
}
```

### 2.3 ファイルシステムの差異対応

```typescript
import fs from 'fs/promises';
import path from 'path';

class FileSystemCompat {
  // ファイルシステムの大文字小文字の区別
  // Windows: 区別なし（NTFS）
  // macOS: デフォルトでは区別なし（APFS は大文字小文字を保持するが検索では無視）
  // Linux: 区別あり（ext4）
  static isCaseSensitive(): boolean {
    return process.platform === 'linux';
  }

  // ファイルの権限設定（Unix 系と Windows で異なる）
  static async setFilePermissions(
    filePath: string,
    mode: 'readable' | 'writable' | 'executable'
  ): Promise<void> {
    if (process.platform === 'win32') {
      // Windows では POSIX パーミッションが限定的にしか機能しない
      // icacls コマンドを使用する場合もある
      return;
    }

    const modeMap = {
      readable: 0o644,    // rw-r--r--
      writable: 0o666,    // rw-rw-rw-
      executable: 0o755,  // rwxr-xr-x
    };

    await fs.chmod(filePath, modeMap[mode]);
  }

  // ファイルロックの処理（排他制御）
  static async acquireFileLock(lockPath: string): Promise<boolean> {
    try {
      // O_EXCL フラグでアトミックに作成（既に存在する場合はエラー）
      const fd = await fs.open(lockPath, 'wx');
      await fd.writeFile(String(process.pid));
      await fd.close();
      return true;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === 'EEXIST') {
        return false; // 既にロックされている
      }
      throw error;
    }
  }

  // ファイルロックの解放
  static async releaseFileLock(lockPath: string): Promise<void> {
    try {
      await fs.unlink(lockPath);
    } catch {
      // ロックファイルが存在しない場合は無視
    }
  }

  // クロスプラットフォームなファイル監視
  static watchFile(
    filePath: string,
    callback: (eventType: string, filename: string) => void
  ): fs.FileHandle | null {
    // fs.watch の挙動が OS によって異なるため、ラッパーを使用
    // 注意: macOS では rename イベントが多発する場合がある
    try {
      const watcher = require('fs').watch(filePath, (eventType: string, filename: string) => {
        callback(eventType, filename || path.basename(filePath));
      });
      return watcher;
    } catch {
      return null;
    }
  }

  // 改行コードの正規化
  static normalizeLineEndings(content: string): string {
    // Windows の CRLF → LF に統一
    return content.replace(/\r\n/g, '\n');
  }

  // プラットフォームに合わせた改行コードに変換
  static toPlatformLineEndings(content: string): string {
    const normalized = content.replace(/\r\n/g, '\n');
    if (process.platform === 'win32') {
      return normalized.replace(/\n/g, '\r\n');
    }
    return normalized;
  }
}
```

---

## 3. メニューバーの OS 差異

### 3.1 メニューバーの構造的な違い

```
メニューバーの違い:

  macOS:   画面上部にグローバルメニューバー
           アプリ名メニュー（About/Preferences/Quit）が必須
           Cmd+Q で終了

  Windows: ウィンドウ上部にメニューバー
           File メニューに Exit
           Alt+F4 で終了

  Linux:   ウィンドウ上部（デスクトップ環境による）
           File メニューに Quit
           GNOME ではグローバルメニューバーの場合もある
```

### 3.2 完全なメニュー実装例

```typescript
import { Menu, app, shell, dialog, BrowserWindow } from 'electron';

function createMenu() {
  const isMac = process.platform === 'darwin';

  const template: Electron.MenuItemConstructorOptions[] = [
    // macOS: アプリ名メニュー
    ...(isMac ? [{
      label: app.name,
      submenu: [
        { role: 'about' as const },
        { type: 'separator' as const },
        { label: '設定...', accelerator: 'Cmd+,', click: openSettings },
        { type: 'separator' as const },
        { role: 'services' as const },
        { type: 'separator' as const },
        { role: 'hide' as const },
        { role: 'hideOthers' as const },
        { role: 'unhide' as const },
        { type: 'separator' as const },
        { role: 'quit' as const },
      ],
    }] : []),
    // File メニュー
    {
      label: 'ファイル',
      submenu: [
        { label: '新規', accelerator: 'CmdOrCtrl+N', click: newFile },
        { label: '開く', accelerator: 'CmdOrCtrl+O', click: openFile },
        { type: 'separator' },
        { label: '保存', accelerator: 'CmdOrCtrl+S', click: saveFile },
        { label: '名前を付けて保存...', accelerator: 'CmdOrCtrl+Shift+S', click: saveFileAs },
        { type: 'separator' },
        ...(isMac ? [] : [
          { label: '設定', accelerator: 'Ctrl+,', click: openSettings },
          { type: 'separator' as const },
          { label: '終了', accelerator: 'Alt+F4', click: () => app.quit() },
        ]),
      ],
    },
    // Edit メニュー
    {
      label: '編集',
      submenu: [
        { role: 'undo' as const, label: '元に戻す' },
        { role: 'redo' as const, label: 'やり直し' },
        { type: 'separator' as const },
        { role: 'cut' as const, label: '切り取り' },
        { role: 'copy' as const, label: 'コピー' },
        { role: 'paste' as const, label: '貼り付け' },
        ...(isMac ? [
          { role: 'pasteAndMatchStyle' as const, label: 'スタイルを合わせて貼り付け' },
          { role: 'delete' as const, label: '削除' },
          { role: 'selectAll' as const, label: 'すべて選択' },
          { type: 'separator' as const },
          {
            label: 'スピーチ',
            submenu: [
              { role: 'startSpeaking' as const, label: '読み上げ開始' },
              { role: 'stopSpeaking' as const, label: '読み上げ停止' },
            ],
          },
        ] : [
          { role: 'delete' as const, label: '削除' },
          { type: 'separator' as const },
          { role: 'selectAll' as const, label: 'すべて選択' },
        ]),
      ],
    },
    // View メニュー
    {
      label: '表示',
      submenu: [
        { role: 'reload' as const, label: '再読み込み' },
        { role: 'forceReload' as const, label: '強制再読み込み' },
        { role: 'toggleDevTools' as const, label: '開発者ツール' },
        { type: 'separator' as const },
        { role: 'resetZoom' as const, label: '拡大率をリセット' },
        { role: 'zoomIn' as const, label: '拡大' },
        { role: 'zoomOut' as const, label: '縮小' },
        { type: 'separator' as const },
        { role: 'togglefullscreen' as const, label: 'フルスクリーン' },
      ],
    },
    // Window メニュー
    {
      label: 'ウィンドウ',
      submenu: [
        { role: 'minimize' as const, label: '最小化' },
        { role: 'zoom' as const, label: 'ズーム' },
        ...(isMac ? [
          { type: 'separator' as const },
          { role: 'front' as const, label: '手前に表示' },
          { type: 'separator' as const },
          { role: 'window' as const },
        ] : [
          { role: 'close' as const, label: '閉じる' },
        ]),
      ],
    },
    // Help メニュー
    {
      label: 'ヘルプ',
      submenu: [
        {
          label: 'ドキュメント',
          click: async () => {
            await shell.openExternal('https://example.com/docs');
          },
        },
        { type: 'separator' },
        {
          label: 'バージョン情報',
          click: () => {
            dialog.showMessageBox({
              type: 'info',
              title: 'バージョン情報',
              message: `${app.name} v${app.getVersion()}`,
              detail: `Electron: ${process.versions.electron}\nNode.js: ${process.versions.node}\nChromium: ${process.versions.chrome}\nOS: ${process.platform} ${process.arch}`,
            });
          },
        },
      ],
    },
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}
```

### 3.3 コンテキストメニュー（右クリックメニュー）の実装

```typescript
import { Menu, MenuItem, BrowserWindow } from 'electron';

// Renderer から呼び出されるコンテキストメニュー
function showContextMenu(
  window: BrowserWindow,
  params: { x: number; y: number; isEditable: boolean; selectedText: string }
): void {
  const menu = new Menu();

  if (params.isEditable) {
    // テキスト編集可能な要素の場合
    menu.append(new MenuItem({
      label: '元に戻す',
      role: 'undo',
      accelerator: 'CmdOrCtrl+Z',
    }));
    menu.append(new MenuItem({
      label: 'やり直し',
      role: 'redo',
      accelerator: 'CmdOrCtrl+Shift+Z',
    }));
    menu.append(new MenuItem({ type: 'separator' }));
    menu.append(new MenuItem({
      label: '切り取り',
      role: 'cut',
      accelerator: 'CmdOrCtrl+X',
    }));
    menu.append(new MenuItem({
      label: 'コピー',
      role: 'copy',
      accelerator: 'CmdOrCtrl+C',
    }));
    menu.append(new MenuItem({
      label: '貼り付け',
      role: 'paste',
      accelerator: 'CmdOrCtrl+V',
    }));
    menu.append(new MenuItem({
      label: 'すべて選択',
      role: 'selectAll',
      accelerator: 'CmdOrCtrl+A',
    }));
  } else if (params.selectedText) {
    // テキストが選択されている場合
    menu.append(new MenuItem({
      label: 'コピー',
      role: 'copy',
      accelerator: 'CmdOrCtrl+C',
    }));
    menu.append(new MenuItem({ type: 'separator' }));
    menu.append(new MenuItem({
      label: `"${params.selectedText.substring(0, 20)}..." で検索`,
      click: () => {
        const { shell } = require('electron');
        shell.openExternal(
          `https://www.google.com/search?q=${encodeURIComponent(params.selectedText)}`
        );
      },
    }));
  }

  menu.popup({ window, x: params.x, y: params.y });
}
```

---

## 4. ウィンドウ管理の差異

### 4.1 基本的なウィンドウライフサイクル

```typescript
import { app, BrowserWindow } from 'electron';

// macOS: ウィンドウを閉じてもアプリは終了しない
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// macOS: Dock アイコンクリックでウィンドウ再作成
app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

// タイトルバーのカスタマイズ
const win = new BrowserWindow({
  titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
  // macOS: トラフィックライト（閉じる/最小化/最大化）の位置
  trafficLightPosition: { x: 15, y: 15 },
  // Windows: タイトルバー非表示時のフレーム
  frame: process.platform === 'darwin' ? true : true,
});
```

### 4.2 ウィンドウ位置・サイズの保存と復元

```typescript
import { BrowserWindow, screen } from 'electron';
import Store from 'electron-store';

interface WindowState {
  x: number;
  y: number;
  width: number;
  height: number;
  isMaximized: boolean;
  isFullScreen: boolean;
}

class WindowStateManager {
  private store: Store;
  private windowId: string;
  private defaultState: WindowState;

  constructor(windowId: string, defaults: Partial<WindowState> = {}) {
    this.store = new Store({ name: 'window-state' });
    this.windowId = windowId;
    this.defaultState = {
      x: 0,
      y: 0,
      width: defaults.width || 1200,
      height: defaults.height || 800,
      isMaximized: false,
      isFullScreen: false,
    };
  }

  // 保存された状態を取得
  getState(): WindowState {
    const saved = this.store.get(`windows.${this.windowId}`) as WindowState | undefined;
    if (!saved) return this.defaultState;

    // 保存された位置が現在のディスプレイに収まるか検証
    const displays = screen.getAllDisplays();
    const isVisible = displays.some(display => {
      const bounds = display.bounds;
      return (
        saved.x >= bounds.x &&
        saved.y >= bounds.y &&
        saved.x + saved.width <= bounds.x + bounds.width &&
        saved.y + saved.height <= bounds.y + bounds.height
      );
    });

    return isVisible ? saved : this.defaultState;
  }

  // ウィンドウの状態変更を監視して自動保存
  track(window: BrowserWindow): void {
    const saveState = (): void => {
      if (window.isDestroyed()) return;

      const bounds = window.getBounds();
      const state: WindowState = {
        x: bounds.x,
        y: bounds.y,
        width: bounds.width,
        height: bounds.height,
        isMaximized: window.isMaximized(),
        isFullScreen: window.isFullScreen(),
      };

      this.store.set(`windows.${this.windowId}`, state);
    };

    // 各種イベントで状態を保存
    window.on('resize', saveState);
    window.on('move', saveState);
    window.on('maximize', saveState);
    window.on('unmaximize', saveState);
    window.on('enter-full-screen', saveState);
    window.on('leave-full-screen', saveState);
    window.on('close', saveState);
  }

  // 保存された状態でウィンドウを復元
  restore(window: BrowserWindow): void {
    const state = this.getState();

    if (state.isMaximized) {
      window.maximize();
    } else if (state.isFullScreen) {
      window.setFullScreen(true);
    }
  }
}

// 使用例
function createWindow(): BrowserWindow {
  const stateManager = new WindowStateManager('main', {
    width: 1200,
    height: 800,
  });

  const state = stateManager.getState();

  const win = new BrowserWindow({
    x: state.x,
    y: state.y,
    width: state.width,
    height: state.height,
    show: false,
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      sandbox: true,
    },
  });

  stateManager.track(win);
  stateManager.restore(win);

  win.once('ready-to-show', () => win.show());

  return win;
}
```

### 4.3 マルチディスプレイ対応

```typescript
import { screen, BrowserWindow } from 'electron';

class DisplayManager {
  // 全ディスプレイの情報を取得
  static getAllDisplays() {
    return screen.getAllDisplays().map(display => ({
      id: display.id,
      label: display.label,
      bounds: display.bounds,
      workArea: display.workArea,
      scaleFactor: display.scaleFactor,
      isPrimary: display.id === screen.getPrimaryDisplay().id,
    }));
  }

  // カーソル位置のディスプレイを取得
  static getDisplayAtCursor() {
    const cursor = screen.getCursorScreenPoint();
    return screen.getDisplayNearestPoint(cursor);
  }

  // ウィンドウを特定のディスプレイの中央に配置
  static centerOnDisplay(window: BrowserWindow, displayId?: number): void {
    const display = displayId
      ? screen.getAllDisplays().find(d => d.id === displayId) || screen.getPrimaryDisplay()
      : screen.getPrimaryDisplay();

    const { x, y, width, height } = display.workArea;
    const [winWidth, winHeight] = window.getSize();

    window.setPosition(
      Math.round(x + (width - winWidth) / 2),
      Math.round(y + (height - winHeight) / 2)
    );
  }

  // DPI スケーリングの処理
  static getScaleFactor(): number {
    const primaryDisplay = screen.getPrimaryDisplay();
    return primaryDisplay.scaleFactor;
  }
}
```

---

## 5. システムトレイの OS 差異

### 5.1 トレイの実装

```typescript
import { Tray, Menu, nativeImage, app } from 'electron';
import path from 'path';

class TrayManager {
  private tray: Tray | null = null;

  create(): void {
    // OS ごとにアイコンサイズが異なる
    const iconPath = this.getIconPath();
    const icon = nativeImage.createFromPath(iconPath);

    // macOS: テンプレートイメージを使用すると自動でダークモード対応
    if (process.platform === 'darwin') {
      icon.setTemplateImage(true);
    }

    this.tray = new Tray(icon);

    // ツールチップの設定
    this.tray.setToolTip(app.name);

    // コンテキストメニューの設定
    const contextMenu = Menu.buildFromTemplate([
      {
        label: 'ウィンドウを表示',
        click: () => {
          const windows = BrowserWindow.getAllWindows();
          if (windows.length > 0) {
            windows[0].show();
            windows[0].focus();
          }
        },
      },
      { type: 'separator' },
      {
        label: 'ステータス',
        enabled: false,
        // アイコンの表示（macOS では自動でリサイズ）
        icon: nativeImage.createFromPath(
          path.join(__dirname, '../../resources/status-ok.png')
        ).resize({ width: 16, height: 16 }),
      },
      { type: 'separator' },
      {
        label: '終了',
        click: () => app.quit(),
      },
    ]);

    this.tray.setContextMenu(contextMenu);

    // Windows / Linux: クリックでウィンドウ表示
    // macOS: クリックでメニュー表示（デフォルト動作）
    if (process.platform !== 'darwin') {
      this.tray.on('click', () => {
        const windows = BrowserWindow.getAllWindows();
        if (windows.length > 0) {
          if (windows[0].isVisible()) {
            windows[0].hide();
          } else {
            windows[0].show();
            windows[0].focus();
          }
        }
      });
    }

    // macOS: ダブルクリックでウィンドウ表示
    if (process.platform === 'darwin') {
      this.tray.on('double-click', () => {
        const windows = BrowserWindow.getAllWindows();
        if (windows.length > 0) {
          windows[0].show();
          windows[0].focus();
        }
      });
    }
  }

  // プラットフォーム別のアイコンパス
  private getIconPath(): string {
    const resourcesPath = path.join(__dirname, '../../resources');

    switch (process.platform) {
      case 'win32':
        // Windows: .ico ファイルを使用（16x16, 32x32, 48x48 を含む）
        return path.join(resourcesPath, 'tray-icon.ico');
      case 'darwin':
        // macOS: @2x を含む PNG テンプレートイメージ（16x16 + 32x32）
        return path.join(resourcesPath, 'tray-iconTemplate.png');
      case 'linux':
        // Linux: PNG ファイル（24x24 推奨）
        return path.join(resourcesPath, 'tray-icon.png');
      default:
        return path.join(resourcesPath, 'tray-icon.png');
    }
  }

  // トレイのバッジ（未読数など）を更新
  updateBadge(count: number): void {
    if (!this.tray) return;

    if (process.platform === 'darwin') {
      // macOS: Dock にバッジを表示
      app.dock.setBadge(count > 0 ? String(count) : '');
    }

    // トレイのツールチップを更新
    this.tray.setToolTip(
      count > 0 ? `${app.name} (${count} 件の通知)` : app.name
    );
  }

  destroy(): void {
    this.tray?.destroy();
    this.tray = null;
  }
}
```

---

## 6. 通知の OS 差異

```typescript
import { Notification, app } from 'electron';

class NotificationManager {
  // クロスプラットフォームな通知の送信
  static send(options: {
    title: string;
    body: string;
    icon?: string;
    urgency?: 'normal' | 'critical' | 'low';
    silent?: boolean;
    actions?: Array<{ text: string; type: string }>;
  }): void {
    if (!Notification.isSupported()) {
      console.warn('通知はこの環境ではサポートされていません');
      return;
    }

    const notification = new Notification({
      title: options.title,
      body: options.body,
      icon: options.icon,
      silent: options.silent || false,
      // macOS: アクションボタン
      ...(process.platform === 'darwin' && options.actions && {
        actions: options.actions.map(a => ({
          text: a.text,
          type: a.type as 'button',
        })),
        hasReply: false,
      }),
      // Linux: 緊急度の設定
      ...(process.platform === 'linux' && {
        urgency: options.urgency || 'normal',
      }),
    });

    // 通知クリック時の処理
    notification.on('click', () => {
      const windows = BrowserWindow.getAllWindows();
      if (windows.length > 0) {
        windows[0].show();
        windows[0].focus();
      }
    });

    // macOS: アクションボタンクリック時の処理
    notification.on('action', (_event, index) => {
      console.log(`アクション ${index} がクリックされました`);
    });

    notification.show();
  }

  // Windows 特有: トースト通知のための設定
  static setupWindowsNotifications(): void {
    if (process.platform !== 'win32') return;

    // AppUserModelId を設定（スタートメニューのショートカットと一致させる必要がある）
    app.setAppUserModelId('com.example.myapp');
  }
}
```

---

## 7. キーボードショートカットの統一

### 7.1 CmdOrCtrl の使用

```typescript
import { globalShortcut, app } from 'electron';

// グローバルショートカットの登録
function registerGlobalShortcuts(): void {
  // CmdOrCtrl を使用すると OS に応じて自動的に切り替わる
  globalShortcut.register('CmdOrCtrl+Shift+Space', () => {
    // クイックアクション（全 OS 共通）
    showQuickAction();
  });

  // OS 固有のショートカット
  if (process.platform === 'darwin') {
    // macOS 固有: Cmd+Option+I は Safari のインスペクタと同じ
    globalShortcut.register('Cmd+Option+I', () => {
      toggleDevTools();
    });
  }
}

// アプリ終了時にグローバルショートカットを解除
app.on('will-quit', () => {
  globalShortcut.unregisterAll();
});
```

### 7.2 キーボードショートカットの一覧表示

```typescript
// Renderer 側のショートカット一覧コンポーネント
interface ShortcutEntry {
  action: string;
  windows: string;
  mac: string;
  linux: string;
}

const shortcuts: ShortcutEntry[] = [
  { action: '新規ファイル', windows: 'Ctrl+N', mac: 'Cmd+N', linux: 'Ctrl+N' },
  { action: '開く', windows: 'Ctrl+O', mac: 'Cmd+O', linux: 'Ctrl+O' },
  { action: '保存', windows: 'Ctrl+S', mac: 'Cmd+S', linux: 'Ctrl+S' },
  { action: '閉じる', windows: 'Ctrl+W', mac: 'Cmd+W', linux: 'Ctrl+W' },
  { action: '設定', windows: 'Ctrl+,', mac: 'Cmd+,', linux: 'Ctrl+,' },
  { action: '検索', windows: 'Ctrl+F', mac: 'Cmd+F', linux: 'Ctrl+F' },
  { action: '置換', windows: 'Ctrl+H', mac: 'Cmd+Option+F', linux: 'Ctrl+H' },
  { action: '全画面', windows: 'F11', mac: 'Ctrl+Cmd+F', linux: 'F11' },
  { action: '終了', windows: 'Alt+F4', mac: 'Cmd+Q', linux: 'Ctrl+Q' },
  { action: 'DevTools', windows: 'F12', mac: 'Cmd+Option+I', linux: 'F12' },
];

// 現在のプラットフォームに合わせたショートカットを取得
function getShortcutForPlatform(entry: ShortcutEntry): string {
  switch (process.platform) {
    case 'darwin': return entry.mac;
    case 'win32': return entry.windows;
    case 'linux': return entry.linux;
    default: return entry.windows;
  }
}
```

---

## 8. フォントとレンダリングの差異

```typescript
// クロスプラットフォームなフォント設定
const fontFamilies = {
  win32: {
    sansSerif: '"Segoe UI", "Yu Gothic UI", "Meiryo", sans-serif',
    monospace: '"Cascadia Code", "Consolas", "MS Gothic", monospace',
    system: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
  },
  darwin: {
    sansSerif: '-apple-system, BlinkMacSystemFont, "Hiragino Sans", sans-serif',
    monospace: '"SF Mono", "Menlo", "Osaka-Mono", monospace',
    system: '-apple-system, BlinkMacSystemFont, "Hiragino Sans", sans-serif',
  },
  linux: {
    sansSerif: '"Noto Sans CJK JP", "Ubuntu", sans-serif',
    monospace: '"Ubuntu Mono", "DejaVu Sans Mono", "Noto Sans Mono CJK JP", monospace',
    system: '"Noto Sans CJK JP", "Ubuntu", sans-serif',
  },
};

// CSS でのクロスプラットフォーム対応
const crossPlatformCSS = `
/* システムフォントを使用する推奨設定 */
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               "Hiragino Sans", "Noto Sans CJK JP", "Yu Gothic UI",
               "Meiryo", sans-serif;

  /* OS ごとに異なるフォントレンダリングの調整 */
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-rendering: optimizeLegibility;
}

/* macOS 特有: サブピクセルレンダリング */
@media screen and (-webkit-min-device-pixel-ratio: 2) {
  body {
    -webkit-font-smoothing: subpixel-antialiased;
  }
}

/* Windows 特有: ClearType の最適化 */
@media screen and (-ms-high-contrast: none) {
  body {
    font-feature-settings: "liga" 0; /* リガチャの無効化（ClearType との相性） */
  }
}

/* スクロールバーのカスタマイズ（OS 間の統一） */
::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}

::-webkit-scrollbar-track {
  background: transparent;
}

::-webkit-scrollbar-thumb {
  background: rgba(128, 128, 128, 0.4);
  border-radius: 5px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(128, 128, 128, 0.6);
}

/* macOS: オーバーレイスクロールバーの場合は非表示 */
@supports (-webkit-overflow-scrolling: touch) {
  ::-webkit-scrollbar {
    display: none;
  }
}
`;
```

---

## 9. ネイティブダイアログの差異

```typescript
import { dialog, BrowserWindow } from 'electron';

class DialogManager {
  // ファイル選択ダイアログ（OS によって外観が異なる）
  static async openFile(parentWindow: BrowserWindow): Promise<string | null> {
    const result = await dialog.showOpenDialog(parentWindow, {
      title: 'ファイルを選択',
      // macOS: ファイルとフォルダの同時選択が可能
      properties: [
        'openFile',
        ...(process.platform === 'darwin' ? ['treatPackageAsDirectory' as const] : []),
      ],
      filters: [
        { name: 'テキストファイル', extensions: ['txt', 'md', 'json'] },
        { name: '画像', extensions: ['png', 'jpg', 'gif', 'svg'] },
        { name: 'すべてのファイル', extensions: ['*'] },
      ],
      // macOS: シートダイアログとして表示（親ウィンドウに紐づく）
      // Windows / Linux: 独立したダイアログ
    });

    return result.canceled ? null : result.filePaths[0];
  }

  // 確認ダイアログ
  static async confirm(
    parentWindow: BrowserWindow,
    message: string,
    detail?: string
  ): Promise<boolean> {
    const result = await dialog.showMessageBox(parentWindow, {
      type: 'question',
      title: '確認',
      message,
      detail,
      buttons: process.platform === 'darwin'
        ? ['キャンセル', 'OK'] // macOS: 右側が肯定
        : ['OK', 'キャンセル'], // Windows/Linux: 左側が肯定
      defaultId: process.platform === 'darwin' ? 1 : 0,
      cancelId: process.platform === 'darwin' ? 0 : 1,
      // macOS: チェックボックスの追加が可能
      checkboxLabel: process.platform === 'darwin' ? '次回から表示しない' : undefined,
    });

    return result.response === (process.platform === 'darwin' ? 1 : 0);
  }

  // エラーダイアログ
  static showError(title: string, content: string): void {
    dialog.showErrorBox(title, content);
  }
}
```

---

## 10. 自動更新のプラットフォーム差異

```typescript
import { autoUpdater, UpdateCheckResult } from 'electron-updater';
import { app, BrowserWindow, dialog } from 'electron';
import log from 'electron-log';

class UpdateManager {
  private mainWindow: BrowserWindow;

  constructor(mainWindow: BrowserWindow) {
    this.mainWindow = mainWindow;
    this.configure();
  }

  private configure(): void {
    // ログの設定
    autoUpdater.logger = log;

    // 自動ダウンロードの設定
    autoUpdater.autoDownload = false;
    autoUpdater.autoInstallOnAppQuit = true;

    // macOS: コード署名の検証を要求
    if (process.platform === 'darwin') {
      autoUpdater.autoRunAppAfterInstall = true;
    }

    // イベントリスナーの設定
    autoUpdater.on('update-available', async (info) => {
      const result = await dialog.showMessageBox(this.mainWindow, {
        type: 'info',
        title: '更新があります',
        message: `バージョン ${info.version} が利用可能です。ダウンロードしますか？`,
        buttons: ['ダウンロード', '後で'],
        defaultId: 0,
      });

      if (result.response === 0) {
        autoUpdater.downloadUpdate();
      }
    });

    autoUpdater.on('update-downloaded', async () => {
      const result = await dialog.showMessageBox(this.mainWindow, {
        type: 'info',
        title: '更新の準備完了',
        message: '再起動して更新を適用しますか？',
        buttons: ['今すぐ再起動', '後で'],
        defaultId: 0,
      });

      if (result.response === 0) {
        autoUpdater.quitAndInstall();
      }
    });

    autoUpdater.on('error', (error) => {
      log.error('自動更新エラー:', error);
    });
  }

  // 更新チェックの実行
  async checkForUpdates(): Promise<void> {
    try {
      await autoUpdater.checkForUpdates();
    } catch (error) {
      log.error('更新チェックに失敗:', error);
    }
  }
}
```

---

## 11. CI/CD マルチプラットフォームビルド

### 11.1 GitHub Actions の設定

```yaml
# .github/workflows/build.yml
name: Build
on:
  push:
    tags: ['v*']

jobs:
  build:
    strategy:
      matrix:
        include:
          - os: windows-latest
            target: win
          - os: macos-latest
            target: mac
          - os: ubuntu-latest
            target: linux

    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: pnpm install
      - run: pnpm build
      - run: pnpm package:${{ matrix.target }}
      - uses: actions/upload-artifact@v4
        with:
          name: app-${{ matrix.target }}
          path: out/make/**/*
```

### 11.2 electron-builder のマルチプラットフォーム設定

```yaml
# electron-builder.yml
appId: com.example.myapp
productName: My App

# Windows 固有の設定
win:
  target:
    - target: nsis
      arch: [x64, arm64]
    - target: portable
      arch: [x64]
  icon: resources/icon.ico
  # コード署名
  certificateFile: ${env.WIN_CERT_FILE}
  certificatePassword: ${env.WIN_CERT_PASSWORD}

nsis:
  oneClick: false
  perMachine: false
  allowToChangeInstallationDirectory: true
  createDesktopShortcut: true
  createStartMenuShortcut: true
  shortcutName: My App
  # 日本語のインストーラ UI
  language: 1041

# macOS 固有の設定
mac:
  target:
    - target: dmg
      arch: [universal]
    - target: zip
      arch: [universal]
  icon: resources/icon.icns
  category: public.app-category.productivity
  hardenedRuntime: true
  gatekeeperAssess: false
  entitlements: build/entitlements.mac.plist
  entitlementsInherit: build/entitlements.mac.plist
  # 公証（Notarization）
  notarize:
    teamId: ${env.APPLE_TEAM_ID}

dmg:
  sign: false
  contents:
    - x: 130
      y: 220
    - x: 410
      y: 220
      type: link
      path: /Applications

# Linux 固有の設定
linux:
  target:
    - target: AppImage
      arch: [x64]
    - target: deb
      arch: [x64]
    - target: rpm
      arch: [x64]
  icon: resources/icons
  category: Utility
  maintainer: developer@example.com
  synopsis: A cross-platform desktop application
  description: |
    My App は Windows、macOS、Linux に対応した
    クロスプラットフォームデスクトップアプリケーションです。

deb:
  depends:
    - gconf2
    - gconf-service
    - libnotify4
    - libappindicator1
    - libxtst6
    - libnss3

# 自動更新の設定
publish:
  provider: github
  owner: myorg
  repo: myapp
```

### 11.3 macOS 公証（Notarization）のスクリプト

```bash
#!/bin/bash
# scripts/notarize.sh — macOS アプリの公証スクリプト

set -e

APP_PATH="$1"
APPLE_ID="${APPLE_ID}"
APPLE_PASSWORD="${APPLE_APP_SPECIFIC_PASSWORD}"
TEAM_ID="${APPLE_TEAM_ID}"

echo "公証を開始: $APP_PATH"

# アプリを zip に圧縮
ditto -c -k --keepParent "$APP_PATH" "$APP_PATH.zip"

# Apple に送信して公証を要求
xcrun notarytool submit "$APP_PATH.zip" \
  --apple-id "$APPLE_ID" \
  --password "$APPLE_PASSWORD" \
  --team-id "$TEAM_ID" \
  --wait

# 公証結果をアプリにステープル
xcrun stapler staple "$APP_PATH"

echo "公証が完了しました"
```

---

## 12. テストのクロスプラットフォーム対応

```typescript
import { describe, it, expect, beforeAll } from 'vitest';

// プラットフォーム依存のテストをスキップするヘルパー
const onlyOnWindows = process.platform === 'win32' ? describe : describe.skip;
const onlyOnMac = process.platform === 'darwin' ? describe : describe.skip;
const onlyOnLinux = process.platform === 'linux' ? describe : describe.skip;
const skipOnCI = process.env.CI ? describe.skip : describe;

describe('PathUtils', () => {
  it('sanitizeFileName は Windows の予約名を処理する', () => {
    expect(PathUtils.sanitizeFileName('CON')).toBe('_CON');
    expect(PathUtils.sanitizeFileName('NUL.txt')).toBe('_NUL.txt');
    expect(PathUtils.sanitizeFileName('normal.txt')).toBe('normal.txt');
  });

  it('sanitizeFileName は不正な文字を除去する', () => {
    expect(PathUtils.sanitizeFileName('file<>:name.txt')).toBe('file___name.txt');
    expect(PathUtils.sanitizeFileName('file|name?.txt')).toBe('file_name_.txt');
  });

  it('expandHome はホームディレクトリを展開する', () => {
    const expanded = PathUtils.expandHome('~/Documents/test.txt');
    expect(expanded).not.toContain('~');
    expect(expanded).toContain('Documents/test.txt');
  });
});

onlyOnWindows('Windows 固有テスト', () => {
  it('UNC パスを正しく検出する', () => {
    expect(PathUtils.isUNCPath('\\\\server\\share')).toBe(true);
    expect(PathUtils.isUNCPath('C:\\Users')).toBe(false);
  });

  it('Windows のパス長制限を検証する', () => {
    const longPath = 'C:\\' + 'a'.repeat(260);
    expect(PathUtils.validatePathLength(longPath)).toBe(false);
  });
});

onlyOnMac('macOS 固有テスト', () => {
  it('macOS のバージョンを正しく検出する', () => {
    const version = SystemInfo.getOSName();
    expect(version).toContain('macOS');
  });
});

onlyOnLinux('Linux 固有テスト', () => {
  it('ファイルシステムが大文字小文字を区別する', () => {
    expect(FileSystemCompat.isCaseSensitive()).toBe(true);
  });
});
```

---

## 13. Tauri でのクロスプラットフォーム対応

```rust
// src-tauri/src/platform.rs — Tauri でのプラットフォーム固有処理

use std::env;

/// プラットフォーム固有の設定ディレクトリを取得
pub fn get_config_dir() -> std::path::PathBuf {
    #[cfg(target_os = "windows")]
    {
        let appdata = env::var("APPDATA").expect("APPDATA not set");
        std::path::PathBuf::from(appdata).join("com.example.myapp")
    }

    #[cfg(target_os = "macos")]
    {
        let home = env::var("HOME").expect("HOME not set");
        std::path::PathBuf::from(home)
            .join("Library")
            .join("Application Support")
            .join("com.example.myapp")
    }

    #[cfg(target_os = "linux")]
    {
        let config_dir = env::var("XDG_CONFIG_HOME")
            .unwrap_or_else(|_| {
                let home = env::var("HOME").expect("HOME not set");
                format!("{}/.config", home)
            });
        std::path::PathBuf::from(config_dir).join("com.example.myapp")
    }
}

/// プラットフォーム固有の初期化処理
pub fn platform_init() {
    #[cfg(target_os = "windows")]
    {
        // Windows: DPI 対応の設定
        unsafe {
            winapi::um::shellscalingapi::SetProcessDpiAwareness(
                winapi::um::shellscalingapi::PROCESS_PER_MONITOR_DPI_AWARE,
            );
        }
    }

    #[cfg(target_os = "macos")]
    {
        // macOS: 特別な初期化は不要（Tauri が処理）
    }

    #[cfg(target_os = "linux")]
    {
        // Linux: GTK テーマの設定
        env::set_var("GTK_THEME", "Adwaita:dark");
    }
}
```

```toml
# src-tauri/Cargo.toml — プラットフォーム固有の依存関係

[target.'cfg(target_os = "windows")'.dependencies]
winapi = { version = "0.3", features = ["shellscalingapi", "winuser"] }
windows-sys = "0.52"

[target.'cfg(target_os = "macos")'.dependencies]
cocoa = "0.25"
objc = "0.2"

[target.'cfg(target_os = "linux")'.dependencies]
gtk = "0.18"
```

---

## FAQ

### Q1: Windows と macOS でキーボードショートカットをどう統一する？
`CmdOrCtrl` を使用。macOS では Cmd、Windows/Linux では Ctrl に自動マッピングされる。ただし、`Cmd+Option` のような macOS 固有の修飾キーの組み合わせは別途処理が必要。Electron の `accelerator` 文字列で `CmdOrCtrl` を使うのが最も簡便な方法である。

### Q2: macOS のダークモードにどう対応する？
`nativeTheme.shouldUseDarkColors` で現在のテーマを検出。`nativeTheme.on('updated', ...)` で変更を監視。CSS では `prefers-color-scheme` メディアクエリを使用する。Electron 側では `nativeTheme.themeSource` を `'system'`、`'dark'`、`'light'` のいずれかに設定することでテーマを制御できる。

### Q3: Apple Silicon (arm64) と Intel (x64) の両方に対応するには？
Electron: `--arch=universal` でユニバーサルバイナリを作成。Tauri: `--target aarch64-apple-darwin` と `--target x86_64-apple-darwin` で個別ビルド後、`lipo` で結合。CI/CD では `macos-latest` ランナー（Apple Silicon）と `macos-13`（Intel）の両方でビルドし、`lipo -create` で統合する方法が確実。

### Q4: Linux の複数ディストリビューションにどう対応する？
AppImage 形式が最もポータブルで、ほぼ全ての Linux ディストリビューションで動作する。Debian/Ubuntu 向けには `.deb`、Fedora/RHEL 向けには `.rpm` を追加で提供するのが一般的。Snap や Flatpak は配布の仕組みとしては優れるが、サンドボックスの制限に注意が必要。

### Q5: Windows でのネイティブ通知が表示されない場合は？
Windows 10/11 では `app.setAppUserModelId()` でアプリケーション ID を正しく設定する必要がある。また、スタートメニューにショートカットが存在しないとトースト通知が表示されない場合がある。NSIS インストーラーでショートカットを作成するか、開発時は `Notification.isSupported()` で事前にチェックする。

### Q6: ファイルの drag & drop でパスが OS ごとに異なる問題は？
Electron の `webContents` で `will-navigate` イベントを監視し、ドロップされたファイルのパスを `path.normalize()` で正規化する。macOS では `file://` プロトコルが付与される場合があるため、`new URL(path).pathname` でパスを抽出する。Windows ではバックスラッシュをフォワードスラッシュに変換する必要が生じることがある。

---

## まとめ

| 項目 | Windows | macOS | Linux |
|------|---------|-------|-------|
| パス区切り | `\` | `/` | `/` |
| 改行コード | `\r\n` | `\n` | `\n` |
| 修飾キー | Ctrl | Cmd | Ctrl |
| メニュー位置 | ウィンドウ内 | 画面上部 | ウィンドウ内 |
| 終了動作 | 全Window閉じ→終了 | Window閉じ→常駐 | 全Window閉じ→終了 |
| 大文字小文字 | 区別なし (NTFS) | 区別なし (APFS) | 区別あり (ext4) |
| アイコン形式 | .ico | .icns | .png |
| インストーラ | NSIS / MSI / MSIX | DMG / PKG | AppImage / deb / rpm |
| 通知 | トースト (Win10+) | Notification Center | libnotify |
| トレイ | 常にサポート | サポート | DE 依存 |
| 自動更新 | NSIS / Squirrel | DMG / ZIP | AppImage のみ |
| DPI対応 | 手動設定が必要な場合あり | 自動（Retina） | 手動設定が必要な場合あり |

---

## 次に読むべきガイド
→ [[../01-wpf-and-winui/00-windows-ui-frameworks.md]] — Windows UI フレームワーク

---

## 参考文献
1. Electron. "Platform Considerations." electronjs.org/docs, 2024.
2. Apple. "Human Interface Guidelines." developer.apple.com/design, 2024.
3. Microsoft. "Windows App Design." learn.microsoft.com, 2024.
4. Tauri. "Cross-Platform Development." tauri.app/guides, 2024.
5. Electron. "Notifications." electronjs.org/docs/latest/tutorial/notifications, 2024.
6. Electron Builder. "Multi-Platform Build." electron.build/multi-platform-build, 2024.
7. freedesktop.org. "Desktop Entry Specification." specifications.freedesktop.org, 2024.
